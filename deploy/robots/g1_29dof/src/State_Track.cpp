#include "State_Track.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/observations/motion_observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_Track::State_Track(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    spdlog::info("Initializing State_{}...", state_string);
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate);
    articulation->data.motion_loader = new isaaclab::MotionLoader(
        cfg["motion_file"].as<std::string>(),
        cfg["fps"].as<float>(50.0f)
    );
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        articulation
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    auto & joy = FSMState::lowstate->joystick;
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return time > env->robot->data.motion_loader->duration; }, // time out
            (int)FSMMode::Velocity
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); }, // bad orientation
            (int)FSMMode::Passive
        )
    );
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return joy.RB.pressed && joy.X.on_pressed; }, // R1 + X
            (int)FSMMode::Velocity
        )
    );
}

void State_Track::enter()
{
    // set gain
    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    time = 0.0f;
    env->robot->update();
    env->reset();

    // Start policy thread
    policy_thread_running = true;
    policy_thread = std::thread([this]{
        using clock = std::chrono::high_resolution_clock;
        const std::chrono::duration<double> desiredDuration(env->step_dt);
        const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

        // Initialize timing
        const auto start = clock::now();
        auto sleepTill = start + dt;

        while (policy_thread_running)
        {
            time += env->step_dt;
            env->robot->data.motion_loader->sample(time);
            env->step();

            // Sleep
            std::this_thread::sleep_until(sleepTill);
            sleepTill += dt;
        }
    });
}


void State_Track::run()
{
    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}