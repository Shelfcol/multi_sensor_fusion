#include "lidar_localization/models/registration/icp_registration.hpp"
#include "glog/logging.h"

namespace lidar_localization{

// ICPRegistration::ICPRegistration(const YAML::Node& node)
// {
//     icp_ptr_ = (new pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT>())；
//     float max_corres_dist = node["max_corres_dist"].as<float>();
//     float euclidean_fitness_eps = node["euclidean_fitness_eps"].as<float>();
//     float trans_eps = node["trans_eps"].as<float>();
//     int max_iter = node["max_iter"].as<int>();
//     SetRegistrationParam( max_corres_dist, euclidean_fitness_eps,  trans_eps, max_iter);
// }

bool ICPRegistration::SetRegistrationParam(float max_corres_dist, float euclidean_fitness_eps, float trans_eps, int max_iter)
{
    icp_ptr_->setMaxCorrespondenceDistance(max_corres_dist);
    icp_ptr_->setEuclideanFitnessEpsilon(euclidean_fitness_eps);
    icp_ptr_->setTransformationEpsilon(trans_eps);
    icp_ptr_->setMaximumIterations(max_iter);

    LOG(INFO) << "ICP 的匹配参数为：" << std::endl
            << "max_corres_dist: " << max_corres_dist << ", "
            << "euclidean_fitness_eps: " << euclidean_fitness_eps << ", "
            << "trans_eps: " << trans_eps << ", "
            << "max_iter: " << max_iter 
            << std::endl << std::endl;
    return true;
}

bool ICPRegistration::SetInputTarget(const CloudData::CLOUD_PTR& input_target)
{
    icp_ptr_->setInputTarget(input_target);
    return true;
}

bool ICPRegistration::ScanMatch(const CloudData::CLOUD_PTR& input_source,
                const Eigen::Matrix4f& predict_pose,
                CloudData::CLOUD_PTR& result_cloud_ptr,
                Eigen::Matrix4f& result_pose)
{
    icp_ptr_->setInputSource(input_source);
    icp_ptr_->align(*result_cloud_ptr, predict_pose);
    result_pose = icp_ptr_->getFinalTransformation();
    printf("icp fitness score = %f\n", icp_ptr_->getFitnessScore());
    if(!icp_ptr_->hasConverged()){
        printf("ICP not converged\n");
    }
    return true;
}
}

