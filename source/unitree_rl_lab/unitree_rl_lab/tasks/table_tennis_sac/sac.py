from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SACConfig:
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    gamma: float = 0.98
    tau: float = 0.005
    # Exploration floor raised again for the sparse/event-heavy reward pass: with the dense
    # approach-window proxies removed, the policy needs broader stochastic coverage to discover
    # contact-frame swings before valid-return events are common.
    initial_alpha: float = 0.10
    min_alpha: float = 0.10
    target_entropy: float | None = None
    aux_reconstruction_coef: float = 0.05


def _activation(name: str) -> type[nn.Module]:
    if name == "elu":
        return nn.ELU
    if name == "relu":
        return nn.ReLU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


def mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    act_cls = _activation(activation)
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, hidden))
        layers.append(act_cls())
        last = hidden
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


def mlp_body(input_dim: int, hidden_dims: Iterable[int], activation: str) -> tuple[nn.Sequential, int]:
    layers: list[nn.Module] = []
    last = input_dim
    act_cls = _activation(activation)
    for hidden in hidden_dims:
        layers.append(nn.Linear(last, hidden))
        layers.append(act_cls())
        last = hidden
    return nn.Sequential(*layers), last


def _grad_l2_norm(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    total = None
    for param in parameters:
        if param.grad is None:
            continue
        value = param.grad.detach().pow(2).sum()
        total = value if total is None else total + value
    if total is None:
        return torch.tensor(0.0)
    return total.sqrt()


class TanhGaussianActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        aux_target_dim: int = 0,
    ):
        super().__init__()
        self.trunk, trunk_dim = mlp_body(obs_dim, hidden_dims, activation)
        self.action_head = nn.Linear(trunk_dim, action_dim * 2)
        self.aux_head = nn.Linear(trunk_dim, aux_target_dim) if aux_target_dim > 0 else None
        self.action_dim = action_dim
        self._legacy_final_index = 2 * len(hidden_dims)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        mean, log_std = self.action_head(features).chunk(2, dim=-1)
        log_std = log_std.clamp(-5.0, 2.0)
        return mean, log_std

    def reconstruct_aux(self, obs: torch.Tensor) -> torch.Tensor | None:
        if self.aux_head is None:
            return None
        return self.aux_head(self.trunk(obs))

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros(action.shape[0], 1, device=action.device)
            return action, log_prob

        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1.0e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> bool:
        """Load current actor weights, or convert the pre-auxiliary ``net.*`` layout.

        Returns True for an exact current-layout load. Returns False when the legacy policy
        trunk/action weights were converted and the auxiliary head stayed freshly initialized.
        """
        try:
            self.load_state_dict(state_dict)
            return True
        except RuntimeError:
            pass
        if not any(key.startswith("net.") for key in state_dict):
            self.load_state_dict(state_dict)
            return True

        converted: dict[str, torch.Tensor] = {}
        final_prefix = f"net.{self._legacy_final_index}."
        for key, value in state_dict.items():
            if key.startswith(final_prefix):
                converted["action_head." + key[len(final_prefix) :]] = value
            elif key.startswith("net."):
                converted["trunk." + key[len("net.") :]] = value
        missing, unexpected = self.load_state_dict(converted, strict=False)
        allowed_missing = {"aux_head.weight", "aux_head.bias"}
        if unexpected or any(key not in allowed_missing for key in missing):
            details = f"missing={missing}, unexpected={unexpected}"
            raise RuntimeError(f"Could not convert legacy actor checkpoint: {details}")
        return False


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...], activation: str):
        super().__init__()
        self.q = mlp(obs_dim + action_dim, hidden_dims, 1, activation)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q(torch.cat([obs, action], dim=-1))


class SACAgent:
    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        *,
        config: SACConfig | None = None,
        device: str | torch.device = "cuda",
    ):
        self.config = config or SACConfig()
        self.device = torch.device(device)
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.action_dim = int(action_dim)
        self.checkpoint_step = 0
        target_entropy = self.config.target_entropy
        self.target_entropy = -float(action_dim) if target_entropy is None else float(target_entropy)
        self._min_log_alpha = math.log(self.config.min_alpha) if self.config.min_alpha > 0.0 else None

        self.actor = TanhGaussianActor(
            actor_obs_dim,
            action_dim,
            self.config.actor_hidden_dims,
            self.config.activation,
            aux_target_dim=self._aux_target_dim(),
        ).to(self.device)
        self.critic1 = Critic(
            critic_obs_dim, action_dim, self.config.critic_hidden_dims, self.config.activation
        ).to(self.device)
        self.critic2 = Critic(
            critic_obs_dim, action_dim, self.config.critic_hidden_dims, self.config.activation
        ).to(self.device)
        self.target_critic1 = Critic(
            critic_obs_dim, action_dim, self.config.critic_hidden_dims, self.config.activation
        ).to(self.device)
        self.target_critic2 = Critic(
            critic_obs_dim, action_dim, self.config.critic_hidden_dims, self.config.activation
        ).to(self.device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_opt = torch.optim.Adam(critic_params, lr=self.config.critic_lr)
        self.log_alpha = torch.tensor(
            float(self.config.initial_alpha), device=self.device
        ).log().detach().requires_grad_(True)
        self._clamp_log_alpha()
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs_actor: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        obs_actor = obs_actor.to(self.device)
        action, _ = self.actor.sample(obs_actor, deterministic=deterministic)
        return action

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs_actor = batch["obs_actor"].to(self.device).float()
        obs_critic = batch["obs_critic"].to(self.device).float()
        action = batch["action"].to(self.device).float()
        reward = batch["reward"].to(self.device).float()
        next_obs_actor = batch["next_obs_actor"].to(self.device).float()
        next_obs_critic = batch["next_obs_critic"].to(self.device).float()
        done = batch["done"].to(self.device).float()

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_obs_actor)
            target_q1 = self.target_critic1(next_obs_critic, next_action)
            target_q2 = self.target_critic2(next_obs_critic, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            backup = reward + (1.0 - done) * self.config.gamma * target_q

        q1 = self.critic1(obs_critic, action)
        q2 = self.critic2(obs_critic, action)
        critic_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)

        critic_params = list(self.critic1.parameters()) + list(self.critic2.parameters())
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = _grad_l2_norm(critic_params)
        self.critic_opt.step()

        new_action, log_prob = self.actor.sample(obs_actor)
        q1_pi = self.critic1(obs_critic, new_action)
        q2_pi = self.critic2(obs_critic, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha.detach() * log_prob - q_pi).mean()
        aux_loss = self._aux_reconstruction_loss(obs_actor, obs_critic)
        actor_loss = policy_loss + self.config.aux_reconstruction_coef * aux_loss

        actor_params = list(self.actor.parameters())
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = _grad_l2_norm(actor_params)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self._clamp_log_alpha()

        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "aux_reconstruction_loss": float(aux_loss.detach().cpu()),
            "actor_grad_norm": float(actor_grad_norm.detach().cpu()),
            "critic_grad_norm": float(critic_grad_norm.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "q1_mean": float(q1.detach().mean().cpu()),
            "q2_mean": float(q2.detach().mean().cpu()),
        }

    def _clamp_log_alpha(self):
        if self._min_log_alpha is None:
            return
        with torch.no_grad():
            self.log_alpha.clamp_(min=self._min_log_alpha)

    def _aux_target_dim(self) -> int:
        if self.config.aux_reconstruction_coef <= 0.0:
            return 0
        return max(0, self.critic_obs_dim - self.actor_obs_dim)

    def _aux_target_scale(self, target: torch.Tensor) -> torch.Tensor:
        if target.shape[-1] == 32:
            scale = target.new_tensor(
                [8.0] * 7
                + [15.0] * 3
                + [10.0] * 3
                + [20.0] * 3
                + [15.0] * 3
                + [1.0] * 9
                + [3.0, 1.5, 1.5, 1.0]
            )
            return scale.unsqueeze(0)
        return target.detach().std(dim=0, keepdim=True).clamp(min=1.0)

    def _aux_reconstruction_loss(self, obs_actor: torch.Tensor, obs_critic: torch.Tensor) -> torch.Tensor:
        if self.config.aux_reconstruction_coef <= 0.0 or self.critic_obs_dim <= self.actor_obs_dim:
            return obs_actor.new_zeros(())
        pred = self.actor.reconstruct_aux(obs_actor)
        if pred is None:
            return obs_actor.new_zeros(())
        target = obs_critic[:, self.actor_obs_dim :].detach()
        target = target / self._aux_target_scale(target)
        return F.smooth_l1_loss(pred, target)

    def _soft_update(self, source: nn.Module, target: nn.Module):
        with torch.no_grad():
            for src, dst in zip(source.parameters(), target.parameters()):
                dst.data.mul_(1.0 - self.config.tau).add_(src.data, alpha=self.config.tau)

    def save(self, path: str | Path, *, step: int = 0):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "step": step,
                "actor_obs_dim": self.actor_obs_dim,
                "critic_obs_dim": self.critic_obs_dim,
                "action_dim": self.action_dim,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "target_critic1": self.target_critic1.state_dict(),
                "target_critic2": self.target_critic2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cuda") -> "SACAgent":
        checkpoint = torch.load(path, map_location=device)
        config = SACConfig(**checkpoint["config"])
        agent = cls(
            checkpoint["actor_obs_dim"],
            checkpoint["critic_obs_dim"],
            checkpoint["action_dim"],
            config=config,
            device=device,
        )
        actor_exact = agent.actor.load_compatible_state_dict(checkpoint["actor"])
        agent.critic1.load_state_dict(checkpoint["critic1"])
        agent.critic2.load_state_dict(checkpoint["critic2"])
        agent.target_critic1.load_state_dict(checkpoint["target_critic1"])
        agent.target_critic2.load_state_dict(checkpoint["target_critic2"])
        agent.log_alpha.data.copy_(checkpoint["log_alpha"].to(agent.device))
        agent._clamp_log_alpha()
        if actor_exact:
            agent.actor_opt.load_state_dict(checkpoint["actor_opt"])
        agent.critic_opt.load_state_dict(checkpoint["critic_opt"])
        agent.alpha_opt.load_state_dict(checkpoint["alpha_opt"])
        agent.checkpoint_step = int(checkpoint.get("step", 0))
        return agent

    @classmethod
    def load_actor_only(
        cls,
        path: str | Path,
        *,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: str | torch.device = "cuda",
    ) -> "SACAgent":
        checkpoint = torch.load(path, map_location=device)
        if int(checkpoint["actor_obs_dim"]) != int(actor_obs_dim):
            raise ValueError(f"Actor obs dim mismatch: checkpoint={checkpoint['actor_obs_dim']}, env={actor_obs_dim}")
        if int(checkpoint["action_dim"]) != int(action_dim):
            raise ValueError(f"Action dim mismatch: checkpoint={checkpoint['action_dim']}, env={action_dim}")

        config = SACConfig(**checkpoint["config"])
        agent = cls(
            actor_obs_dim,
            critic_obs_dim,
            action_dim,
            config=config,
            device=device,
        )
        agent.actor.load_compatible_state_dict(checkpoint["actor"])
        agent.log_alpha.data.copy_(checkpoint["log_alpha"].to(agent.device))
        agent._clamp_log_alpha()
        agent.checkpoint_step = int(checkpoint.get("step", 0))
        return agent
