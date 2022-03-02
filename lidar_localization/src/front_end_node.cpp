/*
 * @Description: 前端里程计的node文件
 * @Author: Ren Qian
 * @Date: 2020-02-05 02:56:27
 */
#include <ros/ros.h>
#include "glog/logging.h"

#include <lidar_localization/saveMap.h>
#include "lidar_localization/global_defination/global_defination.h"
#include "lidar_localization/front_end/front_end_flow.hpp"

using namespace lidar_localization;

std::shared_ptr<FrontEndFlow> _front_end_flow_ptr;

bool save_map_callback(saveMap::Request &request, saveMap::Response &response) {
    response.succeed = _front_end_flow_ptr->SaveMap(); // 保存地图
    _front_end_flow_ptr->PublishGlobalMap(); // 发布全局地图
    return response.succeed;
}

int main(int argc, char *argv[]) {
    // GLOG记录
    google::InitGoogleLogging(argv[0]);
    FLAGS_log_dir = WORK_SPACE_PATH + "/Log";
    FLAGS_alsologtostderr = 1;

    // ros节点定义
    ros::init(argc, argv, "front_end_node");
    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("save_map", save_map_callback); // 收到命令后发布service，
    _front_end_flow_ptr = std::make_shared<FrontEndFlow>(nh); // 前端指针

    ros::Rate rate(100);
    while (ros::ok()) {
        ros::spinOnce();

        _front_end_flow_ptr->Run();

        rate.sleep();
    }

    return 0;
}