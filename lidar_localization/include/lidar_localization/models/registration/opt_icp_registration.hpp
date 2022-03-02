#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_OPT_ICP_REGISTRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_OPT_ICP_REGISTRATION_HPP_
// #include "sophus/so3.h"
#include "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/third_party/sophus/so3.hpp"
#include "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/third_party/sophus/se3.hpp"
#include "lidar_localization/models/registration/registration_interface.hpp"
#include "lidar_localization/tools/tic_toc.hpp"
#include <pcl/kdtree/kdtree_flann.h>

namespace lidar_localization{
class OptICPRegistration: public RegistrationInterface{
public:
    // ICPRegistration(const YAML::Node& node);
    // OptICPRegistration(const YAML::Node& node):
    // icp_ptr_(new pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT>())
    // {
        
    //     float max_corres_dist = node["max_corres_dist"].as<float>();
    //     float euclidean_fitness_eps = node["euclidean_fitness_eps"].as<float>();
    //     float trans_eps = node["trans_eps"].as<float>();
    //     int max_iter = node["max_iter"].as<int>();
    //     SetRegistrationParam( max_corres_dist, euclidean_fitness_eps,  trans_eps, max_iter);
    // }
    OptICPRegistration(const YAML::Node& node):kdtree_ptr_(new pcl::KdTreeFLANN<CloudData::POINT>()),H_(6,6),B_(6,1){
    }
    
    bool SetInputTarget(const CloudData::CLOUD_PTR& input_target) override
    {
        kdtree_ptr_->setInputCloud(input_target);
        input_target_ptr_ = input_target;
    }
    bool ScanMatch(const CloudData::CLOUD_PTR& input_source,
                   const Eigen::Matrix4f& predict_pose,
                   CloudData::CLOUD_PTR& result_cloud_ptr,
                   Eigen::Matrix4f& result_pose) override
    {
        // 置零
        H_ = Eigen::MatrixXf::Zero(6,6);
        B_ = Eigen::MatrixXf::Zero(6,1);
        // 将点云根据预测位姿转到
        double trans_eps = 0.01;
        double this_trans_eps = 1;
        int max_iter = 30;
        Eigen::Matrix4f predict_T = predict_pose;
        while((max_iter--)>0 && this_trans_eps>trans_eps)
        {   
            result_cloud_ptr.reset(new CloudData::CLOUD(*input_source)); //  结果点云用source点云重置
            pcl::transformPointCloud(*result_cloud_ptr, *result_cloud_ptr, predict_T);

            // 寻找最近点，求雅可比矩阵
            for(size_t i  =0; i<input_source->points.size();++i)
            {
                int nearestIdx = GetNearestPoint(result_cloud_ptr->points[i]);
                if(nearestIdx>=0)
                {
                    // pi使用的是原始点云，在qi中找最近点使用的是转换后的点云
                    Eigen::Vector3f pi = Eigen::Vector3f(input_source->points[i].x,input_source->points[i].y,input_source->points[i].z);
                    Eigen::Vector3f pi_trans = Eigen::Vector3f(result_cloud_ptr->points[i].x,result_cloud_ptr->points[i].y,result_cloud_ptr->points[i].z);
                    Eigen::Vector3f qi = Eigen::Vector3f(input_target_ptr_->points[nearestIdx].x,input_target_ptr_->points[nearestIdx].y,input_target_ptr_->points[nearestIdx].z);
                    UpdateHBSE3(pi, pi_trans, qi, predict_T);
                }
            }
            Eigen::VectorXf dx(6,1);
            dx = H_.inverse()*B_;
            UpdateTransSE3(dx, predict_T);
            this_trans_eps = fabs(dx(0,0))+fabs(dx(1,0))+fabs(dx(2,0));
            std::cout<< "dx = "<< dx << std::endl;
        }
        result_pose = predict_T;
        std::cout<<"left_iter = "<<max_iter<<std::endl;
    }
private:
    // bool SetRegistrationParam(float max_corres_dist, float euclidean_fitness_eps, float trans_eps, int max_iter);
    int GetNearestPoint(CloudData::POINT p) // 找到满足条件的，返回下标，否则返回-1
    {
        int K=1;
        std::vector<int> searchIdx(K);
        std::vector<float> searchDist(K);
        if(kdtree_ptr_->nearestKSearch(p,K,searchIdx,searchDist)>0)
        {
            if(searchDist[0]<1.0)  return searchIdx[0]; 
        }
        return -1;
    }

    void UpdateHBSO3(Eigen::Vector3f pi, Eigen::Vector3f pi_trans, Eigen::Vector3f qi,Eigen::Matrix4f& T)
    {
        Eigen::Matrix3f R = T.block<3,3>(0,0);
        Eigen::Vector3f t = T.block<3,1>(0,3);
        Eigen::Vector3f fi = pi_trans-qi; // 误差使用的是转换后的点云
        
        Eigen::MatrixXf Ja(3,6);
        Ja = Eigen::MatrixXf::Zero(3,6);

        /* 矩阵运算
        Ja(0,0)=Ja(1,1)=Ja(2,2)=1;
        Eigen::Matrix3f pi_hat  = (Eigen::Matrix3f()<< 0,-pi[2],pi[1],
                                                    pi[2],0,-pi[0],
                                                    -pi[1],pi[0],0).finished();
        Eigen::Matrix3f df_dR = -R*pi_hat;
        Ja.block<3,3>(0,3) = df_dR;
        */

        // Sophus运算
        Ja.leftCols<3>() = Eigen::Matrix3f::Identity();
        Ja.rightCols<3>() = -R*Sophus::SO3f::hat(pi).matrix();

        H_ += Ja.transpose()*Ja;
        B_-=Ja.transpose()*fi;
    }

    void UpdateHBSE3(Eigen::Vector3f pi, Eigen::Vector3f pi_trans, Eigen::Vector3f qi,Eigen::Matrix4f& T)
    {
        Eigen::Matrix3f R = T.block<3,3>(0,0);
        Eigen::Vector4f fi = Eigen::Vector4f::Zero(); // 坐标点应该用齐次坐标表示，所以第四维为0
        fi[0] = (pi_trans - qi)[0];
        fi[1] = (pi_trans - qi)[1];
        fi[2] = (pi_trans - qi)[2];
        Eigen::Matrix<float,4,6> Ja = Eigen::MatrixXf::Zero(4,6);
        Ja.block<3,3>(0,0) = R;
        Ja.block<3,3>(0,3) = -R*Sophus::SO3f::hat(pi).matrix();
        H_ += Ja.transpose()*Ja;
        B_ -= Ja.transpose()*fi;
    }

    // 右扰动模型 dx(tx,ty,tz,fai1,fai2,fai3)
    void UpdateTransSE3(const Eigen::VectorXf& dx, Eigen::Matrix4f& T)
    {
        T *= Sophus::SE3f::exp(dx).matrix();
    }

    // 右扰动模型 dx(tx,ty,tz,fai1,fai2,fai3)
    void UpdateTransSO3(const Eigen::VectorXf& dx, Eigen::Matrix4f& T)
    {
        /*
        Eigen::Matrix3f new_R = T.block<3,3>(0,0); // 原来的R
        Eigen::Vector3f new_t = T.block<3,1>(0,3); // 原来的t
        new_t += dx.block<3,1>(0,0);
        double fai1 = dx[3];
        double fai2 = dx[4];
        double fai3 = dx[5];
        Eigen::Vector3f fai = dx.block<3,1>(3,0);
        double theta = fai.norm(); //模长
        Eigen::Vector3f a = fai/theta; // fai = theta*a
        Eigen::Matrix3f aaT = a*a.transpose();
        Eigen::Matrix3f a_hat = (Eigen::Matrix3f()<<0,-a[2],a[1],
                                a[2],0,-a[0],
                                -a[1],a[0],0).finished();
        Eigen::Matrix3f dR = cos(theta)*Eigen::Matrix3f::Identity()+(1-cos(theta))*aaT+sin(theta)*a_hat;
        new_R = new_R*dR; // 右扰动，所以右乘
        T.block<3,3>(0,0) = new_R;
        T.block<3,1>(0,3) = new_t;
        */
        // Sophus更新
        T.block<3,1>(0,3) += dx.head<3>();
        Eigen::Matrix3f rotation_matrix = T.block<3,3>(0,0);
        T.block<3,3>(0,0) = rotation_matrix * Sophus::SO3f::exp(dx.tail<3>()).matrix();
    }

private:
    pcl::KdTreeFLANN<CloudData::POINT>::Ptr kdtree_ptr_;
    CloudData::CLOUD_PTR input_target_ptr_;
    Eigen::Matrix3f predict_R_;
    Eigen::Vector3f predict_t_;
    Eigen::Matrix<float,6,6>  H_;
    Eigen::Matrix<float,6,1>  B_;
};
}

#endif