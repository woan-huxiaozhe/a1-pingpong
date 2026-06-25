from __future__ import annotations

from dataclasses import dataclass

import torch

from unitree_rl_lab.tasks.table_tennis_sac.event_tags import EVENT_TAGS, EVENT_TO_BIT


def _as_buffer_tensor(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.detach().to(device=device)


class UniformReplayBuffer:
    """Circular tensor replay buffer with actor and critic observations."""

    def __init__(self, capacity: int, *, device: str | torch.device = "cpu"):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.pos = 0
        self.size = 0
        self._next_version = 1
        self._data: dict[str, torch.Tensor] = {}
        self._slot_versions = torch.zeros(self.capacity, dtype=torch.long, device=self.device)

    def __len__(self) -> int:
        return self.size

    @property
    def slot_versions(self) -> torch.Tensor:
        return self._slot_versions

    def _allocate(self, sample: dict[str, torch.Tensor]):
        if self._data:
            return
        for key, value in sample.items():
            shape = (self.capacity, *value.shape[1:])
            self._data[key] = torch.empty(shape, dtype=value.dtype, device=self.device)
        self._data["event_mask"] = torch.zeros(self.capacity, dtype=torch.long, device=self.device)

    def add_batch(
        self,
        *,
        obs_actor: torch.Tensor,
        obs_critic: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs_actor: torch.Tensor,
        next_obs_critic: torch.Tensor,
        done: torch.Tensor,
        episode_id: torch.Tensor,
        step_index: torch.Tensor,
        event_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = {
            "obs_actor": _as_buffer_tensor(obs_actor, self.device),
            "obs_critic": _as_buffer_tensor(obs_critic, self.device),
            "action": _as_buffer_tensor(action, self.device),
            "reward": _as_buffer_tensor(reward.reshape(-1, 1), self.device).float(),
            "next_obs_actor": _as_buffer_tensor(next_obs_actor, self.device),
            "next_obs_critic": _as_buffer_tensor(next_obs_critic, self.device),
            "done": _as_buffer_tensor(done.reshape(-1, 1), self.device).float(),
            "episode_id": _as_buffer_tensor(episode_id.reshape(-1), self.device).long(),
            "step_index": _as_buffer_tensor(step_index.reshape(-1), self.device).long(),
        }
        self._allocate(batch)

        n = batch["action"].shape[0]
        if n > self.capacity:
            start = n - self.capacity
            batch = {k: v[start:] for k, v in batch.items()}
            n = self.capacity
            if event_mask is not None:
                event_mask = event_mask[start:]

        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        versions = torch.arange(self._next_version, self._next_version + n, device=self.device)
        self._next_version += n

        for key, value in batch.items():
            self._data[key][idx] = value
        self._data["event_mask"][idx] = 0 if event_mask is None else _as_buffer_tensor(event_mask.reshape(-1), self.device)
        self._slot_versions[idx] = versions

        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        return idx.detach().cpu(), versions.detach().cpu()

    def is_valid(self, indices: torch.Tensor, versions: torch.Tensor) -> torch.Tensor:
        indices = indices.to(self.device)
        versions = versions.to(self.device)
        in_range = (indices >= 0) & (indices < self.capacity)
        valid = torch.zeros_like(in_range, dtype=torch.bool)
        if torch.any(in_range):
            valid[in_range] = self._slot_versions[indices[in_range]] == versions[in_range]
        return valid.cpu()

    def sample_indices(self, batch_size: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        return torch.randint(self.size, (batch_size,), generator=generator, device=self.device).cpu()

    def sample(self, batch_size: int, *, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
        return self.get_batch(self.sample_indices(batch_size), device=device)

    def get_batch(self, indices: torch.Tensor, *, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
        out_device = self.device if device is None else torch.device(device)
        idx = indices.to(self.device).long()
        return {key: value[idx].to(out_device) for key, value in self._data.items()}

    def or_event_masks(self, indices: torch.Tensor, mask: int | torch.Tensor):
        if len(indices) == 0:
            return
        idx = indices.to(self.device).long()
        if isinstance(mask, torch.Tensor):
            event_mask = mask.to(self.device).long()
            self._data["event_mask"][idx] |= event_mask
        else:
            self._data["event_mask"][idx] |= int(mask)


class EventReplayTable:
    """Replay-index side table for rare event windows."""

    def __init__(self, name: str, capacity: int = 250_000):
        self.name = name
        self.capacity = int(capacity)
        self.pos = 0
        self.size = 0
        self.indices = torch.full((self.capacity,), -1, dtype=torch.long)
        self.versions = torch.zeros(self.capacity, dtype=torch.long)

    def __len__(self) -> int:
        return self.size

    def add(self, indices: torch.Tensor, versions: torch.Tensor):
        if indices.numel() == 0:
            return
        indices = indices.detach().cpu().long().reshape(-1)
        versions = versions.detach().cpu().long().reshape(-1)
        if indices.numel() > self.capacity:
            indices = indices[-self.capacity :]
            versions = versions[-self.capacity :]
        n = indices.numel()
        slots = (torch.arange(n) + self.pos) % self.capacity
        self.indices[slots] = indices
        self.versions[slots] = versions
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def valid_size(self, replay: UniformReplayBuffer) -> int:
        if self.size == 0:
            return 0
        idx = torch.arange(self.size)
        return int(replay.is_valid(self.indices[idx], self.versions[idx]).sum().item())

    def sample(
        self,
        batch_size: int,
        replay: UniformReplayBuffer,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.size == 0:
            return torch.empty(0, dtype=torch.long)
        sampled: list[torch.Tensor] = []
        attempts = 0
        while sum(t.numel() for t in sampled) < batch_size and attempts < 8:
            draw = torch.randint(self.size, (batch_size,), generator=generator)
            idx = self.indices[draw]
            ver = self.versions[draw]
            valid = replay.is_valid(idx, ver)
            if torch.any(valid):
                sampled.append(idx[valid])
            attempts += 1
        if not sampled:
            return torch.empty(0, dtype=torch.long)
        return torch.cat(sampled)[:batch_size]


DEFAULT_EVENT_RATIOS = {
    "uniform": 0.30,
    "near_miss": 0.10,
    "hit": 0.10,
    "return": 0.15,
    "valid_return": 0.25,
    "miss": 0.05,
    "bad_hit": 0.05,
}


class StratifiedReplaySampler:
    def __init__(
        self,
        replay: UniformReplayBuffer,
        event_tables: dict[str, EventReplayTable],
        ratios: dict[str, float] | None = None,
    ):
        self.replay = replay
        self.event_tables = event_tables
        self.ratios = ratios or DEFAULT_EVENT_RATIOS

    def sample(
        self,
        batch_size: int,
        *,
        device: str | torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        if len(self.replay) == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        requested = {name: int(batch_size * ratio) for name, ratio in self.ratios.items()}
        requested["uniform"] = requested.get("uniform", 0) + batch_size - sum(requested.values())

        sampled_indices: list[torch.Tensor] = []
        composition: dict[str, int] = {name: 0 for name in self.ratios}
        uniform_fallback = 0

        for name, count in requested.items():
            if count <= 0:
                continue
            if name == "uniform":
                idx = self.replay.sample_indices(count, generator=generator)
                sampled_indices.append(idx)
                composition["uniform"] = composition.get("uniform", 0) + idx.numel()
                continue
            table = self.event_tables.get(name)
            idx = table.sample(count, self.replay, generator=generator) if table is not None else torch.empty(0, dtype=torch.long)
            if idx.numel() < count:
                uniform_fallback += count - idx.numel()
            if idx.numel() > 0:
                sampled_indices.append(idx)
                composition[name] = composition.get(name, 0) + idx.numel()

        if uniform_fallback > 0:
            idx = self.replay.sample_indices(uniform_fallback, generator=generator)
            sampled_indices.append(idx)
            composition["uniform"] = composition.get("uniform", 0) + idx.numel()

        indices = torch.cat(sampled_indices)
        if indices.numel() > batch_size:
            indices = indices[:batch_size]
        elif indices.numel() < batch_size:
            extra = self.replay.sample_indices(batch_size - indices.numel(), generator=generator)
            indices = torch.cat([indices, extra])
            composition["uniform"] = composition.get("uniform", 0) + extra.numel()
        return self.replay.get_batch(indices, device=device), composition


@dataclass
class EpisodeInfo:
    event_mask: int = 0
    min_dist: float = float("nan")
    closest_step: int = -1
    hit_step: int = -1
    return_step: int = -1
    valid_return_step: int = -1
    bad_hit_step: int = -1
    miss_step: int = -1
    landing_x: float = float("nan")
    landing_y: float = float("nan")
    hit_center_offset: float = float("nan")
    hit_outgoing_speed: float = float("nan")
    hit_up_speed: float = float("nan")
    post_hit_max_outgoing_speed: float = float("nan")
    post_hit_max_height: float = float("nan")


def _window(center_start: int, center_end: int, length: int) -> range:
    start = max(0, center_start)
    end = min(length - 1, center_end)
    if start > end:
        return range(0, 0)
    return range(start, end + 1)


def extract_event_windows(length: int, info: EpisodeInfo) -> dict[str, list[int]]:
    windows: dict[str, list[int]] = {name: [] for name in EVENT_TAGS}
    if length <= 0:
        return windows
    if info.event_mask & EVENT_TO_BIT["near_miss"] and info.closest_step >= 0:
        windows["near_miss"] = list(_window(info.closest_step - 20, info.closest_step, length))
    if info.event_mask & EVENT_TO_BIT["miss"] and info.closest_step >= 0:
        windows["miss"] = list(_window(info.closest_step - 20, info.closest_step, length))
    if info.event_mask & EVENT_TO_BIT["hit"] and info.hit_step >= 0:
        windows["hit"] = list(_window(info.hit_step - 20, info.hit_step + 10, length))
    if info.event_mask & EVENT_TO_BIT["return"] and info.hit_step >= 0:
        end = info.valid_return_step if info.valid_return_step >= 0 else info.return_step
        windows["return"] = list(_window(info.hit_step - 20, end, length))
    if info.event_mask & EVENT_TO_BIT["valid_return"] and info.hit_step >= 0 and info.valid_return_step >= 0:
        windows["valid_return"] = list(_window(info.hit_step - 20, info.valid_return_step, length))
    if info.event_mask & EVENT_TO_BIT["bad_hit"] and info.hit_step >= 0:
        end = info.bad_hit_step if info.bad_hit_step >= 0 else info.hit_step
        windows["bad_hit"] = list(_window(info.hit_step - 20, end, length))
    return windows


class EpisodeTraceBuffer:
    """Keeps replay slot ids per vector env until episode finalization."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.episode_ids = torch.arange(num_envs, dtype=torch.long)
        self._next_episode_id = int(num_envs)
        self.indices: list[list[int]] = [[] for _ in range(num_envs)]
        self.versions: list[list[int]] = [[] for _ in range(num_envs)]

    def append(self, env_ids: torch.Tensor, indices: torch.Tensor, versions: torch.Tensor):
        for env_id, idx, ver in zip(env_ids.cpu().tolist(), indices.cpu().tolist(), versions.cpu().tolist()):
            self.indices[env_id].append(int(idx))
            self.versions[env_id].append(int(ver))

    def finalize(self, env_id: int, info: EpisodeInfo) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        idx = self.indices[env_id]
        ver = self.versions[env_id]
        windows = extract_event_windows(len(idx), info)
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for event, positions in windows.items():
            if not positions:
                continue
            out[event] = (
                torch.tensor([idx[p] for p in positions], dtype=torch.long),
                torch.tensor([ver[p] for p in positions], dtype=torch.long),
            )
        self.indices[env_id] = []
        self.versions[env_id] = []
        self.episode_ids[env_id] = self._next_episode_id
        self._next_episode_id += 1
        return out


def make_event_tables(capacity: int = 250_000) -> dict[str, EventReplayTable]:
    return {name: EventReplayTable(name, capacity=capacity) for name in EVENT_TAGS}


def recompute_landing_y_reward(
    achieved_y: torch.Tensor,
    target_y: torch.Tensor,
    *,
    sigma: float = 0.15,
    invalid_reward: float = 0.0,
) -> torch.Tensor:
    valid = torch.isfinite(achieved_y)
    reward = torch.exp(-((achieved_y - target_y) ** 2) / (2.0 * sigma**2))
    return torch.where(valid, reward, torch.full_like(reward, invalid_reward))


class HERRelabeler:
    """Minimal HER scaffold for the later target_y curriculum stage."""

    def __init__(self, ratio: float = 0.30, landing_sigma: float = 0.15):
        self.ratio = ratio
        self.landing_sigma = landing_sigma

    def relabel_reward(self, achieved_y: torch.Tensor, new_target_y: torch.Tensor) -> torch.Tensor:
        return recompute_landing_y_reward(achieved_y, new_target_y, sigma=self.landing_sigma)


def table_sizes(event_tables: dict[str, EventReplayTable], replay: UniformReplayBuffer | None = None) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for name, table in event_tables.items():
        sizes[name] = table.valid_size(replay) if replay is not None else len(table)
    return sizes
