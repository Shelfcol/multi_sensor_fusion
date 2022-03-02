#ifndef SLAM_THREED_HPP_
#define SLAM_THREED_HPP_

#include "../common/read_g2o.h"
#include "../common/slam_type.hpp"
#include "lidar_localization/global_defination/global_defination.h"


// 构造使用的是边信息，计算operator()时利用的是节点，代表优化变量
struct Slam3dErr
{
public:
    Slam3dErr(const Pose3d& pose_ab_,  const Eigen::Matrix<double,6,6>& inv_sqrt_information_):
    pose_ab(pose_ab_),inv_sqrt_information(inv_sqrt_information_){}
    
    template<typename T>
    bool operator()(const T* const p_a_, const T* const q_a_ , const T* const p_b_,const T* const q_b_, T* residuals) const { // 
        Eigen::Map<const Eigen::Matrix<T,3,1>> const p_a(p_a_);
        Eigen::Map<const Eigen::Quaternion<T>> const q_a(q_a_);
        Eigen::Map<const Eigen::Matrix<T,3,1>> const p_b(p_b_);
        Eigen::Map<const Eigen::Quaternion<T>> const q_b(q_b_); 

        Eigen::Matrix<T,3,1> res_trans = q_a.conjugate()*(p_b-p_a)-pose_ab.p.cast<T>();
        Eigen::Quaternion<T>  res_qua = (q_a.conjugate()*q_b)*pose_ab.q.cast<T>().conjugate();

        residuals[0] = res_trans[0];
        residuals[1] = res_trans[1];
        residuals[2] = res_trans[2];
        residuals[3] = res_qua.x();
        residuals[4] = res_qua.y();
        residuals[5] = res_qua.z();
        Eigen::Map<Eigen::Matrix<T,6,1>> res(residuals);
        res = inv_sqrt_information.cast<T>()*res;
        return true;
    }

    static ceres::CostFunction* Create(const Pose3d& pose_ab_,  const Eigen::Matrix<double,6,6>& inv_sqrt_information_){
        return new ceres::AutoDiffCostFunction<Slam3dErr,6,3,4,3,4>(new Slam3dErr(pose_ab_,inv_sqrt_information_));
    }

private:
    const Pose3d pose_ab;
    const Eigen::Matrix<double,6,6> inv_sqrt_information;
};


bool BuildOptimizationProblem(std::vector<Constraint3d>& constraints, std::map<int, Pose3d>& poses, ceres::Problem& problem)
{
    ceres::LocalParameterization* local_quat = new ceres::EigenQuaternionParameterization;
    for(auto iter = constraints.begin(); iter!=constraints.end();++iter){
        int id_begin = iter->id_begin;
        int id_end = iter->id_end;
        if(poses.find(id_begin)==poses.end()){CHECK(false);}
        if(poses.find(id_end)==poses.end()){CHECK(false);}
        
        // 添加残差项
        Eigen::Matrix<double,6,6> inv_sqrt_information = iter->information.llt().matrixL(); // 使用LLT分解，表示开根号
        ceres::CostFunction* cost_func = Slam3dErr::Create(iter->pose_ab,inv_sqrt_information);

        problem.AddParameterBlock(poses[id_begin].p.data(),3);
        problem.AddParameterBlock(poses[id_begin].q.coeffs().data(),4,local_quat);
        problem.AddParameterBlock(poses[id_end].p.data(),3);
        problem.AddParameterBlock(poses[id_end].q.coeffs().data(),4,local_quat);

        problem.AddResidualBlock(cost_func,nullptr,poses[id_begin].p.data(),poses[id_begin].q.coeffs().data(),poses[id_end].p.data(),poses[id_end].q.coeffs().data());
    }
    int id_begin = constraints.begin()->id_begin;
    problem.SetParameterBlockConstant(poses[id_begin].p.data());
    problem.SetParameterBlockConstant(poses[id_begin].q.coeffs().data());
    return true;
}


void TestSlam3d(const std::string& g2o_file_path)
{
    std::map<int, Pose3d> poses;
    std::vector<Constraint3d> constraints;

    CHECK(ReadG2oFile(g2o_file_path, &poses, &constraints));

    std::cout << "Number of poses: " << poses.size() << '\n';
    std::cout << "Number of constraints: " << constraints.size() << '\n';

    CHECK(OutputPoses(lidar_localization::WORK_SPACE_PATH+"/include/lidar_localization/ceres_tutorial/slam/data/3d_poses_original.txt", poses))
        << "Error outputting to poses_original.txt";

    ceres::Problem problem;
    BuildOptimizationProblem(constraints, poses, problem);

    CHECK(SolveOptimizationProblem(problem))
        << "The solve was not successful, exiting.";

    CHECK(OutputPoses( lidar_localization::WORK_SPACE_PATH+"/include/lidar_localization/ceres_tutorial/slam/data/3d_poses_optimized.txt", poses))
        << "Error outputting to poses_original.txt";
}


#endif