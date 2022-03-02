#ifndef SLAM_TYPE_HPP_
#define SLAM_TYPE_HPP__


#include <ceres/ceres.h>
#include <istream>
#include <map>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "ceres/local_parameterization.h"

template<typename T>
inline T NormalizeAngle(const T& angle_radians){
    // Use ceres::floor because it is specialized for double and Jet types.
    T two_pi(2.0 * M_PI);
    return angle_radians - two_pi * ceres::floor((angle_radians + T(M_PI)) / two_pi);
}

struct Pose2d
{
    Eigen::Vector3d p; // x,y,yaw
    static std::string name(){
        return "VERTEX_SE2";
    }
};

// 重载>>
std::istream& operator>>(std::istream& input, Pose2d& pose){
    input>>pose.p[0]>>pose.p[1]>>pose.p[2];
    pose.p[2] = NormalizeAngle(pose.p[2]);
    return input;
}

// VERTEX_SE3:QUAT ID x y z q_x q_y q_z q_w
struct Pose3d
{
    Eigen::Vector3d p;
    Eigen::Quaterniond q;
    static std::string name(){
        return "VERTEX_SE3:QUAT";
    }
};

std::istream& operator>>(std::istream& input, Pose3d& pose){
    input>>pose.p.x()>>pose.p.y()>>pose.p.z()
            >>pose.q.x()>>pose.q.y()>>pose.q.z()>>pose.q.w();
    pose.q.normalize();
    return input;
}



struct Constraint2d{
    int id_begin;
    int id_end;

    // id_begin -> id_end
    Pose2d pose_ab;

    Eigen::Matrix3d information;

    static std::string name(){
        return "EDGE_SE2";
    }
};

std::istream& operator>>(std::istream& input, Constraint2d& constraint){
    input   >>constraint.id_begin>>constraint.id_end
            >>constraint.pose_ab.p[0]>>constraint.pose_ab.p[1]>>constraint.pose_ab.p[2]
            >>constraint.information(0,0)>>constraint.information(0,1)
            >>constraint.information(0,2)>>constraint.information(1,1)
            >>constraint.information(1,2)>>constraint.information(2,2);
    // Set the lower triangular part of the information matrix.
    constraint.information(1, 0) = constraint.information(0, 1);
    constraint.information(2, 0) = constraint.information(0, 2);
    constraint.information(2, 1) = constraint.information(1, 2);

    // Normalize the angle between -pi to pi.
    constraint.pose_ab.p[2] = NormalizeAngle(constraint.pose_ab.p[2]);
    return input;
}

// EDGE_SE3:QUAT ID_a ID_b x_ab y_ab z_ab q_x_ab q_y_ab q_z_ab q_w_ab I_11 I_12 I_13 ... I_16 I_22 I_23 ... I_26 ... I_66 // NOLINT
struct Constraint3d{
    int id_begin;
    int id_end;
    Pose3d pose_ab;
    Eigen::Matrix<double,6,6> information;
    static std::string name(){
        return "EDGE_SE3:QUAT";
    }
};

std::istream& operator>>(std::istream& input, Constraint3d& constraint){
    input>>constraint.id_begin>>constraint.id_end>>constraint.pose_ab;
    Eigen::Matrix<double,6,6>& information = constraint.information;
    for(int i=0;i<6&&input.good();++i){
        for(int j=i;j<6&&input.good();++j){
            input>>information(i,j);
            information(j,i) = information(i,j);
        }
    }
    return input;
}


// 因为需要对角度进行约束限制，所以需要定义角度的加法，以免超出范围
// 定义LocalParameterization的第二种方法
struct Pose2dParameterization{
public:
    template<typename T>
    bool operator()(const T* const pose_2d, const T* const delta_pose_2d, T* pose_plus_delta_pose_2d) const {
        pose_plus_delta_pose_2d[0] = pose_2d[0] + delta_pose_2d[0];
        pose_plus_delta_pose_2d[1] = pose_2d[1] + delta_pose_2d[1];
        pose_plus_delta_pose_2d[2] = NormalizeAngle(pose_2d[2] + delta_pose_2d[2]);
        return true;
    }
    static ceres::LocalParameterization* Create(){
        return new ceres::AutoDiffLocalParameterization<Pose2dParameterization,3,3>; // 参数 Global Size和 Local Size
    }
};

// Output the poses to the file with format: ID x y yaw_radians.
bool OutputPoses(const std::string& filename,
                 const std::map<int, Pose2d>& poses) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    std::cerr << "Error opening the file: " << filename << '\n';
    return false;
  }
  for (std::map<int, Pose2d>::const_iterator poses_iter = poses.begin();
       poses_iter != poses.end(); ++poses_iter) {
    const std::map<int, Pose2d>::value_type& pair = *poses_iter;
    outfile <<  pair.first << " " << pair.second.p[0] << " " << pair.second.p[1]
            << ' ' << pair.second.p[2] << '\n';
  }
  return true;
}

// Output the poses to the file with format: id x y z q_x q_y q_z q_w.
bool OutputPoses(const std::string& filename, const std::map<int, Pose3d>& poses) {
  std::fstream outfile;
  outfile.open(filename.c_str(), std::istream::out);
  if (!outfile) {
    LOG(ERROR) << "Error opening the file: " << filename;
    return false;
  }
  for (std::map<int, Pose3d>::const_iterator poses_iter = poses.begin();poses_iter != poses.end(); ++poses_iter) {
    const std::map<int, Pose3d, std::less<int>,
                   Eigen::aligned_allocator<std::pair<const int, Pose3d> > >::
        value_type& pair = *poses_iter; //first代表key
    outfile << pair.first << " " << pair.second.p.transpose() << " "
            << pair.second.q.x() << " " << pair.second.q.y() << " "
            << pair.second.q.z() << " " << pair.second.q.w() << '\n';
  }
  return true;
}

bool SolveOptimizationProblem(ceres::Problem& problem){
    ceres::Solver::Options options;
    options.num_threads = 8;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 30000;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    std::cout<<summary.FullReport()<<std::endl;
    return true;
}

#endif

