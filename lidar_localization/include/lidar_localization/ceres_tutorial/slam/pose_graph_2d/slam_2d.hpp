#ifndef SLMA_TWOD_HPP_
#define SLMA_TWOD_HPP_

#include "../common/read_g2o.h"
#include "../common/slam_type.hpp"
#include "lidar_localization/global_defination/global_defination.h"


// 构造使用的是边信息，计算operator()时利用的是节点，代表优化变量
struct Slam2dErr
{
public:
    Slam2dErr(const Pose2d& pose_ab_,  const Eigen::Matrix3d& inv_sqrt_information_):
    pose_ab(pose_ab_),inv_sqrt_information(inv_sqrt_information_){}
    
    template<typename T>
    bool operator()(const T* const pose_a, const T* const pose_b, T* residuals) const { // 
        Eigen::Map<const Eigen::Matrix<T,2,1>> const pose_a_trans(pose_a);
        T pose_a_yaw = pose_a[2];

        Eigen::Map<const Eigen::Matrix<T,2,1>> const pose_b_trans(pose_b);
        T pose_b_yaw = pose_b[2];

        Eigen::Matrix<T,2,2> R_a = (Eigen::Matrix<T,2,2>()<<ceres::cos(pose_a_yaw),-ceres::sin(pose_a_yaw),
                                                            ceres::sin(pose_a_yaw),ceres::cos(pose_a_yaw)).finished();
        Eigen::Matrix<T,2,2> R_a_T = R_a.transpose();

        Eigen::Matrix<T,2,1>  pose_ab_trans = (Eigen::Matrix<T,2,1>()<<T(pose_ab.p[0]),T(pose_ab.p[1])).finished();
        Eigen::Matrix<T,2,1> delta_trans = R_a_T*(pose_b_trans-pose_a_trans)-pose_ab_trans;
        residuals[0] = delta_trans[0];
        residuals[1] = delta_trans[1];
        residuals[2] = NormalizeAngle(pose_b_yaw-pose_a_yaw-T(pose_ab.p[2]));
        Eigen::Map<Eigen::Matrix<T,3,1>> res(residuals);
        res = inv_sqrt_information.cast<T>()*res;
        // res[2]=NormalizeAngle(res[2]);
        return true;
    }

    static ceres::CostFunction* Create(const Pose2d& pose_ab_,  const Eigen::Matrix3d& inv_sqrt_information_){
        return new ceres::AutoDiffCostFunction<Slam2dErr,3,3,3>(new Slam2dErr(pose_ab_,inv_sqrt_information_));
    }

private:
    const Pose2d pose_ab;
    const Eigen::Matrix3d inv_sqrt_information;
};


bool BuildOptimizationProblem(std::vector<Constraint2d>& constraints, std::map<int, Pose2d>& poses, ceres::Problem& problem)
{
    ceres::LocalParameterization* pose2d_param = Pose2dParameterization::Create();
    for(auto iter = constraints.begin(); iter!=constraints.end();++iter){
        int id_begin = iter->id_begin;
        int id_end = iter->id_end;
        if(poses.find(id_begin)==poses.end()){CHECK(false);}
        if(poses.find(id_end)==poses.end()){CHECK(false);}
        
        // 添加残差项
        Eigen::Matrix3d inv_sqrt_information = iter->information.llt().matrixL(); // 使用LLT分解，表示开根号
        ceres::CostFunction* cost_func = Slam2dErr::Create(iter->pose_ab,inv_sqrt_information);
        problem.AddParameterBlock(poses[id_begin].p.data(),3,pose2d_param);
        problem.AddParameterBlock(poses[id_end].p.data(),3,pose2d_param);
        problem.AddResidualBlock(cost_func,nullptr,poses[id_begin].p.data(),poses[id_end].p.data());
    }
    int id_begin = constraints.begin()->id_begin;
    int id_end = constraints.begin()->id_end;
    problem.SetParameterBlockConstant(poses[id_begin].p.data());
    // problem.SetParameterBlockConstant(poses[id_end].p.data());

    return true;
}


void TestSlam2d(const std::string& g2o_file_path)
{
    std::map<int, Pose2d> poses;
    std::vector<Constraint2d> constraints;

    CHECK(ReadG2oFile(g2o_file_path, &poses, &constraints));

    std::cout << "Number of poses: " << poses.size() << '\n';
    std::cout << "Number of constraints: " << constraints.size() << '\n';

    CHECK(OutputPoses(lidar_localization::WORK_SPACE_PATH+"/include/lidar_localization/ceres_tutorial/slam/data/2d_poses_original.txt", poses))
        << "Error outputting to poses_original.txt";

    ceres::Problem problem;
    BuildOptimizationProblem(constraints, poses, problem);

    CHECK(SolveOptimizationProblem(problem))
        << "The solve was not successful, exiting.";

    CHECK(OutputPoses( lidar_localization::WORK_SPACE_PATH+"/include/lidar_localization/ceres_tutorial/slam/data/2d_poses_optimized.txt", poses))
        << "Error outputting to poses_original.txt";
}


#endif