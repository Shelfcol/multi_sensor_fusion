#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_CERES_QT_REGISTRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_CERES_QT_REGISTRATION_HPP_
// #include "sophus/so3.h"
#include "../../../../third_party/Sophus-master/sophus/so3.hpp"
#include "../../../../third_party/Sophus-master/sophus/se3.hpp"
#include "lidar_localization/models/registration/registration_interface.hpp"
#include "lidar_localization/tools/tic_toc.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/local_parameterization.h>
#include <ceres/ceres.h>

namespace lidar_localization{

// SE3上的右乘损失函数
// 
class QtCost
{
public:
    // 输入的参数
    QtCost(const Eigen::Vector3d& pi, const Eigen::Vector3d& qi){
        p = pi;
        q = qi;
    }

    template<typename T>
    bool operator()(const T* const q_para, const T* const t_para, T* residuals) const{ //输入参数维度(7维，四元数+平移)，需要设置一个初始值，残差计算(3*1)
        Eigen::Map<const Eigen::Quaternion<T>> const q_trans(q_para);
        Eigen::Map<const Eigen::Matrix<T,3,1>> const t_trans(t_para);
        
        Eigen::Map<Eigen::Matrix<T,3,1>> res(residuals);
        res = q.cast<T>()-(q_trans*p.cast<T>()+t_trans);
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pi, const Eigen::Vector3d& qi){
        return new ceres::AutoDiffCostFunction<QtCost,3,4,3>(new QtCost(pi,qi));
    }

private:
    Eigen::Vector3d p; // 求解p->q的转换矩阵
    Eigen::Vector3d q;
};



class CeresQtICPRegistration: public RegistrationInterface{
public:
    CeresQtICPRegistration(const YAML::Node& node):kdtree_ptr_(new pcl::KdTreeFLANN<CloudData::POINT>()){
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

        // 将点云根据预测位姿转到
        double trans_eps = 0.01;
        double this_trans_eps = 1;
        int max_iter = 30;

        // 待优化的四元数，初始值为predict_pose
        Eigen::Matrix4d predict_T = predict_pose.cast<double>();       

        while((max_iter--)>0 && this_trans_eps>trans_eps)
        {   


            result_cloud_ptr.reset(new CloudData::CLOUD(*input_source)); //  结果点云用source点云重置
            pcl::transformPointCloud(*result_cloud_ptr, *result_cloud_ptr, predict_T);
            
            
            ceres::Problem problem;
            ceres::LocalParameterization* q_local_param = new ceres::EigenQuaternionParameterization;
            double q_param[]={0,0,0,1};//实部在后
            double t_param[]={0,0,0};
            problem.AddParameterBlock(q_param,4,q_local_param);
            problem.AddParameterBlock(t_param,3);

            // 寻找最近点，求雅可比矩阵
            for(size_t i  =0; i<input_source->points.size();++i)
            {
                int nearestIdx = GetNearestPoint(result_cloud_ptr->points[i]);
                if(nearestIdx >=0 )
                {
                    // pi使用的是原始点云，在qi中找最近点使用的是转换后的点云
                    Eigen::Vector3d pi = Eigen::Vector3d(input_source->points[i].x,input_source->points[i].y,input_source->points[i].z);
                    Eigen::Vector3d pi_trans = Eigen::Vector3d(result_cloud_ptr->points[i].x,result_cloud_ptr->points[i].y,result_cloud_ptr->points[i].z);
                    Eigen::Vector3d qi = Eigen::Vector3d(input_target_ptr_->points[nearestIdx].x,input_target_ptr_->points[nearestIdx].y,input_target_ptr_->points[nearestIdx].z);
                    ceres::CostFunction* cost_func = QtCost::Create(pi_trans, qi);// 残差维度，输入参数维度
                    problem.AddResidualBlock(cost_func, nullptr, q_param,t_param); // 损失函数，核函数，优化变量
                }
            }
            ceres::Solver::Summary summary;
              // 配置求解器
            ceres::Solver::Options options;     // 这里有很多配置项可以填
            options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
            options.minimizer_progress_to_stdout = true;   // 输出到cout
            ceres::Solve(options, &problem, &summary);
      
            UpdateTrans(q_param,t_param, predict_T);
        }
        result_pose = predict_T.cast<float>();
        std::cout<<"left_iter = "<<max_iter<<std::endl;
    }
private:
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

    // 右扰动模型 dx(tx,ty,tz,fai1,fai2,fai3)
    void UpdateTrans(const double* q_param, const double* t_param, Eigen::Matrix4d& T)
    {
        Eigen::Map<const Eigen::Quaternion<double>> const q_trans(q_param);
        Eigen::Map<const Eigen::Matrix<double,3,1>> const t_trans(t_param);
        Eigen::Matrix3d R(q_trans);
        Eigen::Matrix<double,4,4> dt = Eigen::Matrix4d::Identity();
        dt.block<3,3>(0,0) = R;
        dt.block<3,1>(0,3) = t_trans;
        T = dt*T;
    }

   
private:
    pcl::KdTreeFLANN<CloudData::POINT>::Ptr kdtree_ptr_;
    CloudData::CLOUD_PTR input_target_ptr_;
};
}

#endif