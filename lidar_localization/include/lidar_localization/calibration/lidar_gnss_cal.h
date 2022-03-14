#ifndef LIDAR_GNSS_CAL_H_
#define LIDAR_GNSS_CAL_H_
#include "lidar_localization/global_definition/global_definition.h"
#include "lidar_localization/sensor_data/pose_data.hpp"
#include <vector>
#include <utility>
#include <deque>
#include <iostream>
#include <string>
#include <iostream>
#include <ceres/ceres.h>
#include "ceres/local_parameterization.h"

namespace lidar_localization{
// 使用手眼标定和ceres标定激光雷达和组合导航
class LidarGnssCal{
public:
    LidarGnssCal()=delete;
    LidarGnssCal(const std::string& lidar_file_path, const std::string& gnss_file_path)
    :lidar_file_path_(lidar_file_path),gnss_file_path_(gnss_file_path){
        cal_res_.setIdentity(); // 初始化为单位矩阵
    } 
    bool Run();


    // ceres 优化
    class CalibErr{
    public:
        CalibErr(const Eigen::Matrix4d& A,const Eigen::Matrix4d& B):
        A_(A),B_(B){}

        template<typename T>
        bool operator()(const T* const trans, const T* const qua, T* res) const // 三维向量和四元数
        {
            Eigen::Map<const Eigen::Matrix<T,3,1>> t(trans);
            Eigen::Map<const Eigen::Quaternion<T>> q(qua);
            Eigen::Quaternion<T> q_A(A_.block<3,3>(0,0).cast<T>());
            Eigen::Quaternion<T> q_B(B_.block<3,3>(0,0).cast<T>());
            Eigen::Matrix<T,3,1> t_A(A_.block<3,1>(0,3).cast<T>());
            Eigen::Matrix<T,3,1> t_B(B_.block<3,1>(0,3).cast<T>());
            Eigen::Matrix<T,3,3> T1 =(q_A*q).toRotationMatrix();
            Eigen::Matrix<T,3,3> T2 =(q*q_B).toRotationMatrix();
            Eigen::Quaternion<T> res_q = Eigen::Quaternion<T>(T1.inverse()*T2);
            Eigen::Matrix<T,3,1> res_t = q_A*t+t_A-q*t_B-t;
            res[0] = res_q.x();
            res[1] = res_q.y();
            res[2] = res_q.z();
            res[3] = res_t[0];
            res[4] = res_t[1];
            res[4] = res_t[2];
            return true;
        }

        static ceres::CostFunction* Create(const Eigen::Matrix4d& A,const Eigen::Matrix4d& B)
        {
            return new ceres::AutoDiffCostFunction<CalibErr,6,3,4>(new CalibErr(A,B));
        }
    private:
        Eigen::Matrix4d A_;
        Eigen::Matrix4d B_;
    };

    bool BuildAndSolveWithCeres();

private:
    PoseData GetT(std::string& data_file_line);
    bool ReadTumFormData(const std::string& file_path, std::deque<PoseData>& pose_dep);
    bool ValidateData(); // 将gnss_pose_deq_的数据使用插值于lidar_pose_deq_进行插值对齐，最后两个的时间戳时一致的，数量也是一致的
    bool GetHandEyeABMatrix();
private:
    std::deque<PoseData>  lidar_pose_deq_;
    std::deque<PoseData>  gnss_pose_deq_;

    std::vector<Eigen::Matrix4d>  A_vec_;
    std::vector<Eigen::Matrix4d>  B_vec_;

    Eigen::Matrix4d cal_res_;// 标定结果

    std::string lidar_file_path_;
    std::string gnss_file_path_;
    int window_size_{3}; // 每隔window_size选择相邻的位姿
};
}
#endif