// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(motion_joint_pos)
{
    auto & robot = env->robot;
    auto & loader = robot->data.motion_loader;
    auto data = loader->joint_pos();
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(motion_root_pos)
{
    auto & robot = env->robot;
    auto & loader = robot->data.motion_loader;
    auto data = loader->root_position();
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(motion_root_quat)
{
    auto & robot = env->robot;
    auto & loader = robot->data.motion_loader;
    auto quat = loader->root_quaternion();
    return std::vector<float>({quat.w(), quat.x(), quat.y(), quat.z()});
}

REGISTER_OBSERVATION(motion_root_ori_b)
{
    auto & robot = env->robot;
    auto & loader = robot->data.motion_loader;
    auto quat = loader->root_quaternion();
    Eigen::Quaternionf quat_b = robot->data.root_quat_b.conjugate() * quat;
    auto mat = quat_b.toRotationMatrix().block<2,3>(0,0);
    auto data = Eigen::Map<Eigen::VectorXf>(mat.data(), mat.size());
    return std::vector<float>(data.data(), data.data() + data.size());
}

} // namespace mdp
} // namespace isaaclab
