#ifndef POSE_MANAGER_HPP_
#define POSE_MANAGER_HPP_
#include <iostream>
#include <vector>
#include <Eigen/Dense>

#include "lidar_localization/sensor_data/pose_data.hpp"
using namespace std;
namespace lidar_localization{
class PoseReadSave{
public:
    bool SaveTum(const std::string& file_path, const std::vector<PoseData>&  pose_vec)
    {
        FILE *fp = NULL;
        char end1 = 0x0d; // "/n"
        char end2 = 0x0a;

        // lidar odometry
        string lidar_tum_file = file_path;
        fp = fopen(lidar_tum_file.c_str(), "w+");

        if (fp == NULL)
        {
            printf("fail to open file %s ! \n", lidar_tum_file.c_str());
            exit(1);
        }
        else
            printf("TUM : write lidar data to %s \n", lidar_tum_file.c_str());

        for (int i = 0; i < pose_vec.size(); i++)
        {
            Eigen::Matrix3d Rot;
            Rot = pose_vec[i].T.block<3,3>(0,0);
            Eigen::Quaterniond q = Eigen::Quaterniond(Rot);
            Eigen::Vector3d t = pose_vec[i].T.block<3,1>(0,3);
            double time = pose_vec[i].time;
            fprintf(fp, "%.3lf %.3lf %.3lf %.3lf %.5lf %.5lf %.5lf %.5lf%c",
                    time, t(0), t(1), t(2),
                    q.x(), q.y(), q.z(), q.w(), end2);
        }
        fclose(fp);
    }
};
}

#endif /* POSE_MANAGER_HPP_ */
