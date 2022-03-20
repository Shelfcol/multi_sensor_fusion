// #include "lidar_localization/ceres_tutorial/powell_function.hpp"
// #include "lidar_localization/ceres_tutorial/powell_function2.hpp"
#include "lidar_localization/g2o_tutorial/curve_fitting_g2o.hpp"
// #include "lidar_localization/ceres_tutorial/pose_estimation_3d2d.hpp"
// #include "lidar_localization/ceres_tutorial/rosenbrock_function.hpp"
// // #include "lidar_localization/ceres_tutorial/slam/pose_graph_2d/slam_2d.hpp"
// #include "lidar_localization/ceres_tutorial/slam/pose_graph_3d/slam_3d.hpp"
#include <iostream>
using namespace std;
int main()
{
    // TestPowellFunc();
    // TestPowellFunc2();
    TestCurveFittingG2O();
    // TEST_PNP();
    // TestRosenbrockFunc();
    // TestSlam2d(lidar_localization::WORK_SPACE_PATH+"/slam_data/g2o_dataset/input_INTEL_g2o.g2o");
    // TestSlam3d(lidar_localization::WORK_SPACE_PATH+"/slam_data/g2o_dataset/sphere_bignoise_vertex3.g2o");
    return 0;
}
