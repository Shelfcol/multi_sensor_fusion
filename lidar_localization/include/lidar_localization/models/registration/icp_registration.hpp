#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_ICP_REGISTRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_ICP_REGISTRATION_HPP_

#include <pcl/registration/icp.h>
#include "lidar_localization/models/registration/registration_interface.hpp"
#include "lidar_localization/tools/tic_toc.hpp"

namespace lidar_localization{
class ICPRegistration: public RegistrationInterface{
public:
    // ICPRegistration(const YAML::Node& node);
    ICPRegistration(const YAML::Node& node):
    icp_ptr_(new pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT>())
    {
        
        float max_corres_dist = node["max_corres_dist"].as<float>();
        float euclidean_fitness_eps = node["euclidean_fitness_eps"].as<float>();
        float trans_eps = node["trans_eps"].as<float>();
        int max_iter = node["max_iter"].as<int>();
        SetRegistrationParam( max_corres_dist, euclidean_fitness_eps,  trans_eps, max_iter);
    }
    
    bool SetInputTarget(const CloudData::CLOUD_PTR& input_target) override;
    bool ScanMatch(const CloudData::CLOUD_PTR& input_source,
                   const Eigen::Matrix4f& predict_pose,
                   CloudData::CLOUD_PTR& result_cloud_ptr,
                   Eigen::Matrix4f& result_pose) override;
private:
    bool SetRegistrationParam(float max_corres_dist, float euclidean_fitness_eps, float trans_eps, int max_iter);

private:
    pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT>::Ptr icp_ptr_;
};
}

#endif