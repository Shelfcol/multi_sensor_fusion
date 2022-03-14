#ifndef POSE_DATA_H_
#define POSE_DATA_H_
#include <iostream>
#include <Eigen/Dense>

namespace lidar_localization{
class PoseData{
public:
    PoseData(double time_, Eigen::Matrix4d T_)
    :time(time_),T(T_){}
    PoseData(){
        time = 0;
        T =Eigen::Matrix4d::Identity();
    }
    friend PoseData  PoseInterp(const PoseData& pose1, const PoseData& pose2, double sync_time);
    friend PoseData  GetDeltaPose(const PoseData& pose_front,const PoseData& pose_back);
public:
    double time;
    Eigen::Matrix4d T;
};
}


#endif