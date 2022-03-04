#ifndef LIDAR_LOCALIZATION_MODELS_REGISTRATION_CERES_SE3_ICP_REGISTRATION_HPP_LIDAR_LOCALIZATION_MODELS_REGISTRATION_CERES_SE3_ICP_REGISTRATION_HPP_
#define LIDAR_LOCALIZATION_MODELS_REGISTRATION_CERES_SE3_ICP_REGISTRATION_HPP_

#include "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/third_party/sophus/so3.hpp"
#include "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/third_party/sophus/se3.hpp"
#include "lidar_localization/models/registration/registration_interface.hpp"
#include "lidar_localization/tools/tic_toc.hpp"
#include <pcl/kdtree/kdtree_flann.h>
#include <ceres/local_parameterization.h>
#include <ceres/ceres.h>

namespace lidar_localization{

// 因为李代数不符合加法运算，所以需要定义广义加法
class CERES_EXPORT SE3Parameterization : public ceres::LocalParameterization {
public:
    SE3Parameterization() {}
    virtual ~SE3Parameterization() {}
    // SE3 se3 SE3
    virtual bool Plus(const double* T_raw,
                        const double* delta_raw,
                    double* T_plus_delta_raw) const // 广义的加法
    {
        Eigen::Map<Sophus::SE3d const> const T(T_raw);
        Eigen::Map<Sophus::SE3d> T_plus_delta(T_plus_delta_raw);
        Eigen::Map<Eigen::Vector3d const> rotation(delta_raw);
        Eigen::Map<Eigen::Vector3d const> translation(delta_raw+3);
        Eigen::Matrix<double,6,1> delta;
        delta.block<3,1>(0,0) = translation;
        delta.block<3,1>(3,0) = rotation;
        T_plus_delta = T* Sophus::SE3d::exp(delta); // 右扰动
        // Sophus::SE3f::exp(const Eigen::Vector6f& a)中a的前三维对应SE3中的translation，后三维对应SO3部分
        // Sophus中存储的顺序刚好相反，前三维对应rotation，后三维对应translation
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    // Dx T * exp(x)  with  x=0
    virtual bool ComputeJacobian(const double* T_raw,
                                double* jacobian_raw) const // 参数有7维（q,t），状态量有6维
    {
        Eigen::Map<Sophus::SE3d const> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6,Eigen::RowMajor>> jacobian(jacobian_raw);
        jacobian.setZero();
        jacobian.block<3,3>(4,0) = Eigen::Matrix3d::Identity();
        jacobian.block<3,3>(0,3) = Eigen::Matrix3d::Identity();//注意需要一一对应
        
        return true;
    }
    virtual int GlobalSize() const { return Sophus::SE3d::num_parameters; } // 参数自由度，可能有冗余
    virtual int LocalSize() const { return Sophus::SE3d::DoF; } // dx的局部正切空间的自由度
    static ceres::LocalParameterization* Create(){
        return new SE3Parameterization;
    }
};

// // 定义加法，然后使用AutoDiffParam
// class SE3LocalParameterizarion{
// public:
//     template<typename T>
//     bool operator()(const T* const se3_T_, const T* const delta_se3_T_, T* se3_plus_delta_T_) const {
//         //se3中的顺序是 qx,qy,qz,qw,tx.ty,tz ，delta_se3_T_为小量phi1,phi2,phi3,tx,ty,tz  而扰动时exp(const Eigen::Vector3d& a)中a的顺序是tx,ty,tz,phi1,phi2,phi3
//         Eigen::Map<const Sophus::SE3<T>> const se3_T(se3_T_);
//         Eigen::Map<Sophus::SE3<T>>  se3_plus_delta_T(se3_plus_delta_T_);
//         Eigen::Map<const Eigen::Matrix<T,3,1>> const delta_rot(delta_se3_T_);
//         Eigen::Map<const Eigen::Matrix<T,3,1>> const delta_trans(delta_se3_T_+3);
//         Eigen::Matrix<T,6,1> delta_se3_adjust;
//         delta_se3_adjust[0] = delta_trans[0];
//         delta_se3_adjust[1] = delta_trans[1];
//         delta_se3_adjust[2] = delta_trans[2];
//         delta_se3_adjust[3] = delta_rot[0];
//         delta_se3_adjust[4] = delta_rot[1];
//         delta_se3_adjust[5] = delta_rot[2];
//         // delta_se3_adjust.block<3,1>(0,0) = delta_trans;
//         // delta_se3_adjust.block<3,1>(3,0) = delta_rot;
//         // const Eigen::Matrix<double,6,1> delta_se3_adjust_double = delta_se3_adjust.cast<double>();
//         se3_plus_delta_T = se3_T*(Sophus::SE3<T>::exp(delta_se3_adjust));
//         return true;
//     }
    
//     static ceres::LocalParameterization* Create(){
//         return new ceres::AutoDiffLocalParameterization<SE3LocalParameterizarion,7,6>;
//     }
// };

// SE3上的右乘损失函数
struct SE3_COST
{
    // 输入的参数
    SE3_COST(const Eigen::Vector3d& pi, const Eigen::Vector3d& qi){
        p = pi;
        q = qi;
    }

    template<typename T>
    bool operator()(const T* const T_opt, T* residual) const{ //输入参数维度(7维，四元数+平移)，需要设置一个初始值，残差计算(3*1)
        Eigen::Map<const Sophus::SE3<T>> const T_new(T_opt); // 需要计算得到的参数
        Eigen::Map<Eigen::Matrix<T,3,1>> res(residual);
        res = T_new*p.cast<T>()-q.cast<T>();
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pi, const Eigen::Vector3d& qi){
        return new ceres::AutoDiffCostFunction<SE3_COST,3,7>(new SE3_COST(pi,qi));
    }

    Eigen::Vector3d p; // 求解p->q的转换矩阵
    Eigen::Vector3d q;
};



class CeresSE3ICPRegistration: public RegistrationInterface{
public:
    CeresSE3ICPRegistration(const YAML::Node& node):kdtree_ptr_(new pcl::KdTreeFLANN<CloudData::POINT>()){
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
            ceres::LocalParameterization* SE3_local_param = SE3Parameterization::Create();
            // SE3 xyzw tx ty tz
            double SE3_opt_param[7]={0,0,0,1,0,0,0}; //旋转在前，平移在后
            
            ceres::Problem problem;
            problem.AddParameterBlock(SE3_opt_param, 7, SE3_local_param); // SE3参数化，维度

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
                    ceres::CostFunction* cost_func = SE3_COST::Create(pi_trans, qi);// 残差维度，输入参数维度
                    problem.AddResidualBlock(cost_func, nullptr, SE3_opt_param); // 损失函数，核函数，优化变量
                }
            }
            ceres::Solver::Summary summary;
              // 配置求解器
            ceres::Solver::Options options;     // 这里有很多配置项可以填
            options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
            //options.linear_solver_type = ceres::DENSE_QR;
            options.minimizer_progress_to_stdout = true;   // 输出到cout
            ceres::Solve(options, &problem, &summary);
            Eigen::Map<Sophus::SE3d> delta_SE3(SE3_opt_param);
            UpdateTransSE3(delta_SE3, predict_T);
        }
        result_pose = predict_T.cast<float>();
        std::cout<<"left_iter = "<<max_iter<<std::endl;
    }
private:
    // bool SetRegistrationParam(double max_corres_dist, double euclidean_fitness_eps, double trans_eps, int max_iter);
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
    void UpdateTransSE3(const Sophus::SE3d& delta_SE3, Eigen::Matrix4d& T)
    {
        // Eigen::Matrix3d deltaR = delta_SE3.rotationMatrix();
        // Eigen::Vector3d deltat = delta_SE3.translation();
        // Eigen::Matrix4d deltaT = Eigen::Matrix4d::Identity();
        // deltaT.block<3,3>(0,0) = deltaR;
        // deltaT.block<3,1>(0,3) = deltat;
        auto deltaT = delta_SE3.matrix();
        T =deltaT* T;
    }
private:
    pcl::KdTreeFLANN<CloudData::POINT>::Ptr kdtree_ptr_;
    CloudData::CLOUD_PTR input_target_ptr_;
};
}

#endif