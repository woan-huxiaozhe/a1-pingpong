#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <eigen3/Eigen/Dense>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

namespace isaaclab
{

/* Unitree CSV Motion File */
class MotionLoader
{
public:
    MotionLoader(std::string motion_file, float fps = 50.0f)
    : dt(1.0f / fps)
    {
        auto data = _read_csv(motion_file);
        
        num_frames = data.size();
        duration = num_frames * dt;
        
        for(int i(0); i < num_frames; ++i)
        {
            root_positions.push_back(Eigen::VectorXf::Map(data[i].data(), 3));
            root_quaternions.push_back(Eigen::Quaternionf(data[i][3], data[i][4], data[i][5], data[i][6]));
            dof_positions.push_back(Eigen::VectorXf::Map(data[i].data() + 7, data[i].size() - 7));
        }

        sample(0.0f);
    }

    void sample(float time) 
    {
        float phase = std::clamp(time / duration, 0.0f, 1.0f);
        index_0_ = std::round(phase * (num_frames - 1));
        index_1_ = std::min(index_0_ + 1, num_frames - 1);
        blend_ = std::round((time - index_0_ * dt) / dt * 1e5f) / 1e5f;
    }

    Eigen::VectorXf joint_pos() {
        return dof_positions[index_0_] * (1 - blend_) + dof_positions[index_1_] * blend_;
    }

    Eigen::VectorXf root_position() {
        return root_positions[index_0_] * (1 - blend_) + root_positions[index_1_] * blend_;
    }

    Eigen::Quaternionf root_quaternion() {
        return root_quaternions[index_0_].slerp(blend_, root_quaternions[index_1_]);
    }

    float dt;
    int num_frames;
    float duration;

    std::vector<Eigen::VectorXf> root_positions;
    std::vector<Eigen::Quaternionf> root_quaternions;
    std::vector<Eigen::VectorXf> dof_positions;

private:
    int index_0_;
    int index_1_;
    float blend_;

    std::vector<std::vector<float>> _read_csv(const std::string& filename)
    {
        std::vector<std::vector<float>> data;
        std::ifstream file(filename);
        if (!file.is_open())
        {
            spdlog::error("Error opening file: {}", filename);
            return data;
        }

        std::string line;
        while (std::getline(file, line))
        {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ','))
            {
                try
                {
                    row.push_back(std::stof(value));
                }
                catch (const std::invalid_argument& e)
                {
                    spdlog::error("Invalid value in file: {}", value);
                }
            }
            data.push_back(row);
        }
        file.close();
        return data;
    }
};

};