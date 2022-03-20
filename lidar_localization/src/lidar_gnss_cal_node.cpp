#include "lidar_localization/calibration/lidar_gnss_cal.h"
#include <ceres/ceres.h>
#include <iostream>
#include <memory>

using namespace std;;

using namespace lidar_localization;

int main(int argc, char** argv)
{
    std::string lidar_pose_path = WORK_SPACE_PATH + "/traj/laser_odom_tum.txt";
    std::string gnss_pose_path = WORK_SPACE_PATH + "/traj/ground_truth_tum.txt";
    std::shared_ptr<LidarGnssCal> lidar_gnss_cal_ptr = std::make_shared<LidarGnssCal>(lidar_pose_path,gnss_pose_path);
    lidar_gnss_cal_ptr->Run();
    return 0;
}