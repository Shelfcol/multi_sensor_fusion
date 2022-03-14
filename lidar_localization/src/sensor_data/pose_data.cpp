#include "lidar_localization/sensor_data/pose_data.hpp"
#include <Eigen/Dense>

namespace lidar_localization{

    PoseData  PoseInterp(const PoseData& pose1,const PoseData& pose2, double sync_time){
        if(!((pose1.time<=sync_time&&pose2.time>=sync_time)||(pose1.time>=sync_time&&pose2.time<=sync_time)) ) {
            std::cout<<"time serilization: "<<pose1.time<<", "<<sync_time<<", "<<pose2.time<<std::endl;
            std::cout<<"sync time false"<<std::endl;
            exit(-1);
            return PoseData();
        }
        PoseData  sync_pose;
        PoseData front_data = pose1;
        PoseData back_data = pose2;
        double front_scale = (back_data.time - sync_time) / (back_data.time - front_data.time);
        double back_scale = (sync_time - front_data.time) / (back_data.time - front_data.time);

        sync_pose.time = sync_time;
        sync_pose.T(0,3) = front_data.T(0,3)*front_scale + back_data.T(0,3)*back_scale ;
        sync_pose.T(1,3) = front_data.T(1,3)*front_scale + back_data.T(1,3)*back_scale ;
        sync_pose.T(2,3) = front_data.T(2,3)*front_scale + back_data.T(2,3)*back_scale ;

        Eigen::Matrix3d front_trans = front_data.T.block<3,3>(0,0);
        Eigen::Matrix3d back_trans = back_data.T.block<3,3>(0,0);
        Eigen::Quaterniond front_q(front_trans);
        Eigen::Quaterniond back_q(back_trans);
        Eigen::Quaterniond sync_q;
        sync_q.x() = front_q.x()*front_scale + back_q.x()*back_scale ;
        sync_q.y() = front_q.y()*front_scale + back_q.y()*back_scale ;
        sync_q.z() = front_q.z()*front_scale + back_q.z()*back_scale ;
        sync_q.w() = front_q.w()*front_scale + back_q.w()*back_scale ;
        sync_q.normalize();
        Eigen::Matrix3d sync_trans(sync_q);

        sync_pose.T.block<3,3>(0,0) = sync_trans;
        return sync_pose;
    }

    PoseData  GetDeltaPose(const PoseData& pose_front,const PoseData& pose_back)
    {
        if(abs(pose_front.time-pose_back.time)>1e-1){
            std::cerr<<"time diff too much"<<std::endl;
            exit(0);
        }
        Eigen::Matrix4d T1 = pose_front.T;
        Eigen::Matrix4d T2 = pose_back.T;
        Eigen::Matrix4d dT = T1.inverse()*T2;
        PoseData tmp  = pose_front;
        tmp.T = dT;
        return tmp;
    }



}