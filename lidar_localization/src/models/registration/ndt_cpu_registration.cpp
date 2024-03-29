/*
 * @Description: NDT 匹配模块
 * @Author: Ren Qian
 * @Date: 2020-02-08 21:46:45
 */
#include "lidar_localization/models/registration/ndt_cpu_registration.hpp"

#include "glog/logging.h"

namespace lidar_localization {

NDTCpuRegistration::NDTCpuRegistration(const YAML::Node& node)
{
    
    float res = node["res"].as<float>();
    float step_size = node["step_size"].as<float>();
    float trans_eps = node["trans_eps"].as<float>();
    int max_iter = node["max_iter"].as<int>();

    SetRegistrationParam(res, step_size, trans_eps, max_iter);
}

NDTCpuRegistration::NDTCpuRegistration(float res, float step_size, float trans_eps, int max_iter)
{

    SetRegistrationParam(res, step_size, trans_eps, max_iter);
}

bool NDTCpuRegistration::SetRegistrationParam(float res, float step_size, float trans_eps, int max_iter) {
    ndt_cpu_.setResolution(res);
    ndt_cpu_.setStepSize(step_size);
    ndt_cpu_.setTransformationEpsilon(trans_eps);
    ndt_cpu_.setMaximumIterations(max_iter);

    LOG(INFO) << "NDT 的匹配参数为：" << std::endl
              << "res: " << res << ", "
              << "step_size: " << step_size << ", "
              << "trans_eps: " << trans_eps << ", "
              << "max_iter: " << max_iter 
              << std::endl << std::endl;

    return true;
}

bool NDTCpuRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target) {
    ndt_cpu_.setInputTarget(input_target);
    return true;
}

bool NDTCpuRegistration::ScanMatch(const CloudData::CLOUD_PTR& input_source, 
                                const Eigen::Matrix4f& predict_pose, 
                                CloudData::CLOUD_PTR& result_cloud_ptr,
                                Eigen::Matrix4f& result_pose) {
    ndt_cpu_.setInputSource(input_source);
    ndt_cpu_.align(*result_cloud_ptr, predict_pose);
    result_pose = ndt_cpu_.getFinalTransformation();
    
    return true;
}
}