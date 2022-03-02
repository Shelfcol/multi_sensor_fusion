/*
 * @Description: front end 任务管理， 放在类里使代码更清晰
 * @Author: Ren Qian
 * @Date: 2020-02-10 08:38:42
 */
#include "lidar_localization/front_end/front_end_flow.hpp"

#include "glog/logging.h"

#include "lidar_localization/tools/file_manager.hpp"
#include "lidar_localization/global_defination/global_defination.h"

namespace lidar_localization {
FrontEndFlow::FrontEndFlow(ros::NodeHandle& nh) {
    cloud_sub_ptr_ = std::make_shared<CloudSubscriber>(nh, "/kitti/velo/pointcloud", 100000);
    imu_sub_ptr_ = std::make_shared<IMUSubscriber>(nh, "/kitti/oxts/imu", 1000000);
    velocity_sub_ptr_ = std::make_shared<VelocitySubscriber>(nh, "/kitti/oxts/gps/vel", 1000000);
    gnss_sub_ptr_ = std::make_shared<GNSSSubscriber>(nh, "/kitti/oxts/gps/fix", 1000000);
    lidar_to_imu_ptr_ = std::make_shared<TFListener>(nh, "imu_link", "velo_link");

    cloud_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "current_scan", 100, "map");
    local_map_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "local_map", 100, "map");
    global_map_pub_ptr_ = std::make_shared<CloudPublisher>(nh, "global_map", 100, "map");
    laser_odom_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "laser_odom", "map", "lidar", 100);
    gnss_pub_ptr_ = std::make_shared<OdometryPublisher>(nh, "gnss", "map", "lidar", 100);

    front_end_ptr_ = std::make_shared<FrontEnd>();

    local_map_ptr_.reset(new CloudData::CLOUD());
    global_map_ptr_.reset(new CloudData::CLOUD());
    current_scan_ptr_.reset(new CloudData::CLOUD());
}

bool FrontEndFlow::Run() {
    if (!ReadData()) // 读数据
        return false;

    if (!InitCalibration()) 
        return false;

    if (!InitGNSS())
        return false;

    while(HasData()) {
        if (!ValidData()) // 只要没找到合适的时间对应的数据，就进行while循环查找
            continue;

        UpdateGNSSOdometry();
        if (UpdateLaserOdometry()) {
            PublishData();
            SaveTrajectory();
        }
    }

    return true;
}

bool FrontEndFlow::ReadData() {
    cloud_sub_ptr_->ParseData(cloud_data_buff_);

    static std::deque<IMUData> unsynced_imu_;
    static std::deque<VelocityData> unsynced_velocity_;
    static std::deque<GNSSData> unsynced_gnss_;

    imu_sub_ptr_->ParseData(unsynced_imu_); // 将新的数据插入到unsynced_imu_的尾部
    velocity_sub_ptr_->ParseData(unsynced_velocity_);
    gnss_sub_ptr_->ParseData(unsynced_gnss_);

    if (cloud_data_buff_.size() == 0)
        return false;

    // 根据当前点云的时间，插值获取对应的IMU，velocity，gnss数据。利用点云时间进行插值之后，存储到buff队列里面
    double cloud_time = cloud_data_buff_.front().time;
    bool valid_imu = IMUData::SyncData(unsynced_imu_, imu_data_buff_, cloud_time); 
    bool valid_velocity = VelocityData::SyncData(unsynced_velocity_, velocity_data_buff_, cloud_time);
    bool valid_gnss = GNSSData::SyncData(unsynced_gnss_, gnss_data_buff_, cloud_time);

    // 如果当前点云获取到的imu,vel,gnss中有一个没有得到时间同步的结果，则表示传感器初始化失败。这里所以需要注意就是三种数据都需要同时获取到
    static bool sensor_inited = false;
    if (!sensor_inited) {
        if (!valid_imu || !valid_velocity || !valid_gnss) {
            cloud_data_buff_.pop_front();
            ROS_WARN("sensor init failed");
            return false;
        }
        sensor_inited = true;
    }

    return true;
}

bool FrontEndFlow::InitCalibration() {
    static bool calibration_received = false;
    if (!calibration_received) {
        if (lidar_to_imu_ptr_->LookupData(lidar_to_imu_)) {
            calibration_received = true;
        }
    }

    return calibration_received;
}

// 用第一个gnss的数据进行初始化
bool FrontEndFlow::InitGNSS() {
    static bool gnss_inited = false;
    if (!gnss_inited) { 
        GNSSData gnss_data = gnss_data_buff_.front();
        gnss_data.InitOriginPosition();
        gnss_inited = true;
    }

    return gnss_inited;
}

bool FrontEndFlow::HasData() {
    if (cloud_data_buff_.size() == 0)
        return false;
    if (imu_data_buff_.size() == 0)
        return false;
    if (velocity_data_buff_.size() == 0)
        return false;
    if (gnss_data_buff_.size() == 0)
        return false;
    
    return true;
}

bool FrontEndFlow::ValidData() {
    current_cloud_data_ = cloud_data_buff_.front();
    current_imu_data_ = imu_data_buff_.front();
    current_velocity_data_ = velocity_data_buff_.front();
    current_gnss_data_ = gnss_data_buff_.front();

    double d_time = current_cloud_data_.time - current_imu_data_.time;
    if (d_time < -0.05) { // 点云数据太早了，都没有imu数据能够对应上
        cloud_data_buff_.pop_front();
        return false;
    }

    if (d_time > 0.05) { // 点云数据比较迟，所以需要将前面的imu等数据删掉
        imu_data_buff_.pop_front();
        velocity_data_buff_.pop_front();
        gnss_data_buff_.pop_front();
        return false;
    }

    cloud_data_buff_.pop_front();
    imu_data_buff_.pop_front();
    velocity_data_buff_.pop_front();
    gnss_data_buff_.pop_front();

    return true;
}

// GNSS里程计数据，平移就用GPS->XYZ值，旋转使用IMU的四元数得到的姿态值,并将其转到激光雷达坐标系下
bool FrontEndFlow::UpdateGNSSOdometry() {
    gnss_odometry_ = Eigen::Matrix4f::Identity();

    current_gnss_data_.UpdateXYZ();
    gnss_odometry_(0,3) = current_gnss_data_.local_E;
    gnss_odometry_(1,3) = current_gnss_data_.local_N;
    gnss_odometry_(2,3) = current_gnss_data_.local_U;
    gnss_odometry_.block<3,3>(0,0) = current_imu_data_.GetOrientationMatrix();
    gnss_odometry_ *= lidar_to_imu_; // T_lidar2world = T_imu2world*T_lidar2imu

    return true;
}


bool FrontEndFlow::UpdateLaserOdometry() {
    static bool front_end_pose_inited = false;
    if (!front_end_pose_inited) { // 激光雷达里程计的初始值使用组合导航得到的数据
        front_end_pose_inited = true;
        front_end_ptr_->SetInitPose(gnss_odometry_);
        return front_end_ptr_->Update(current_cloud_data_, laser_odometry_);
    }

    laser_odometry_ = Eigen::Matrix4f::Identity();
    return front_end_ptr_->Update(current_cloud_data_, laser_odometry_); // 根据当前帧点云进行关键帧判断，更新局部地图，进行点云配准，得到当前帧点云的全局位姿(laser_odomeytry_)
} 

bool FrontEndFlow::PublishData() {
    gnss_pub_ptr_->Publish(gnss_odometry_); // 发布GNSS数据
    laser_odom_pub_ptr_->Publish(laser_odometry_); // 发布激光雷达里程计

    front_end_ptr_->GetCurrentScan(current_scan_ptr_); // 发布将当前帧点云配准之后的结果点云
    cloud_pub_ptr_->Publish(current_scan_ptr_);

    if (front_end_ptr_->GetNewLocalMap(local_map_ptr_)) // 获取新的局部地图
        local_map_pub_ptr_->Publish(local_map_ptr_); // 发布局部地图

    return true;
}

bool FrontEndFlow::SaveTrajectory() {
    // 先创建文件夹和文件，再进行数据保存
    static std::ofstream ground_truth, laser_odom;
    static bool is_file_created = false;
    if (!is_file_created) {
        if (!FileManager::CreateDirectory(WORK_SPACE_PATH + "/slam_data/trajectory"))
            return false;
        if (!FileManager::CreateFile(ground_truth, WORK_SPACE_PATH + "/slam_data/trajectory/ground_truth.txt"))
            return false;
        if (!FileManager::CreateFile(laser_odom, WORK_SPACE_PATH + "/slam_data/trajectory/laser_odom.txt"))
            return false;
        is_file_created = true;
    }

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            ground_truth << gnss_odometry_(i, j); // T矩阵一行一行进行保存
            laser_odom << laser_odometry_(i, j);
            if (i == 2 && j == 3) { // T矩阵的最后一个数据保存完成后，需要进行换行
                ground_truth << std::endl;
                laser_odom << std::endl;
            } else { // 如果数据没有保存完成，则数据之间使用空格隔开
                ground_truth << " ";
                laser_odom << " ";
            }
        }
    }

    return true;
}

bool FrontEndFlow::SaveMap() {
    return front_end_ptr_->SaveMap();
}

bool FrontEndFlow::PublishGlobalMap() {
    if (front_end_ptr_->GetNewGlobalMap(global_map_ptr_)) { 
        global_map_pub_ptr_->Publish(global_map_ptr_);
        global_map_ptr_.reset(new CloudData::CLOUD());
    }
    return true;
}
}