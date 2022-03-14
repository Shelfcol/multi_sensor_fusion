#include <iostream>
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pclomp/ndt_omp.h>
#include <pclomp/gicp_omp.h>

#include <ndt_cpu/NormalDistributionsTransform.h>
#include <eigen3/Eigen/Dense>

Eigen::Translation3d init_translation(-2, -2, -2);
Eigen::AngleAxisd init_rotation(Eigen::Quaterniond(1, 0, 0, 0));
Eigen::Matrix4d init_guess = (init_translation * init_rotation) * Eigen::Matrix4d::Identity();

// align point clouds and measure processing time
pcl::PointCloud<pcl::PointXYZ>::Ptr align(pcl::Registration<pcl::PointXYZ, pcl::PointXYZ> &registration, const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud ) {
  registration.setInputTarget(target_cloud);
  registration.setInputSource(source_cloud);
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());

  auto t1 = ros::WallTime::now();
  registration.align(*aligned, init_guess.cast<float>());
  auto t2 = ros::WallTime::now();
  std::cout << "single : " << (t2 - t1).toSec() * 1000 << "[msec]" << std::endl;

  for(int i=0; i<10; i++) {
    registration.align(*aligned, init_guess.cast<float>());
  }
  auto t3 = ros::WallTime::now();
  std::cout << "10times: " << (t3 - t2).toSec() * 1000 << "[msec]" << std::endl;
  std::cout << "fitness: " << registration.getFitnessScore() << std::endl << std::endl;

  return aligned;
}


int main(int argc, char** argv) {
  if(argc != 3) {
    std::cout << "usage: align target.pcd source.pcd" << std::endl;
    return 0;
  }

  std::string target_pcd = argv[1];
  std::string source_pcd = argv[2];

  pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());

  if(pcl::io::loadPCDFile(target_pcd, *target_cloud)) {
    std::cerr << "failed to load " << target_pcd << std::endl;
    return 0;
  }
  if(pcl::io::loadPCDFile(source_pcd, *source_cloud)) {
    std::cerr << "failed to load " << source_pcd << std::endl;
    return 0;
  }

  // downsampling
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled(new pcl::PointCloud<pcl::PointXYZ>());

  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*downsampled);
  *target_cloud = *downsampled;

  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*downsampled);
  source_cloud = downsampled;

  ros::Time::init();

  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>());
  double fitness_score = 0;
  // benchmark
  std::cout << "--- pcl::NDT ---" << std::endl;
  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setResolution(1.0);
  ndt.setMaximumIterations(500);
  ndt.setStepSize(0.1);
  ndt.setTransformationEpsilon(0.01);
  ndt.setInputTarget(target_cloud);
  ndt.setInputSource(source_cloud);

  auto t11 = ros::WallTime::now();
  ndt.align(*aligned, init_guess.cast<float>());
  auto t12 = ros::WallTime::now();
  std::cout << "single : " << (t12 - t11).toSec() * 1000 << "[msec]" << std::endl;

  fitness_score = ndt.getFitnessScore();
  printf("fitness_score : %f\n", fitness_score);

  std::cout << "--- pcl::CPUNDT ---" << std::endl;
  cpu::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_cpu;
  ndt_cpu.setResolution(1.0);
  ndt_cpu.setMaximumIterations(500);
  ndt_cpu.setStepSize(0.1);
  ndt_cpu.setTransformationEpsilon(0.01);
  ndt_cpu.setInputTarget(target_cloud);
  ndt_cpu.setInputSource(source_cloud);

  auto t21 = ros::WallTime::now();
  ndt_cpu.align(*aligned, init_guess.cast<float>());
  auto t22 = ros::WallTime::now();
  std::cout << "single : " << (t22 - t21).toSec() * 1000 << "[msec]" << std::endl;

  fitness_score = ndt_cpu.getFitnessScore();
  printf("fitness_score : %f\n", fitness_score);


  std::vector<int> num_threads = {1, omp_get_max_threads()};
  std::vector<std::pair<std::string, pclomp::NeighborSearchMethod>> search_methods = {
    {"KDTREE", pclomp::KDTREE},
    {"DIRECT7", pclomp::DIRECT7},
    {"DIRECT1", pclomp::DIRECT1}
  };

  pclomp::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt_omp;
  ndt_omp.setResolution(1.0);
  ndt_omp.setMaximumIterations(500);
  ndt_omp.setStepSize(0.1);
  ndt_omp.setTransformationEpsilon(0.01);

  for(int n : num_threads) {
    for(const auto& search_method : search_methods) {
      std::cout << "--- pclomp::NDT (" << search_method.first << ", " << n << " threads) ---" << std::endl;
      ndt_omp.setNumThreads(n);
      ndt_omp.setNeighborhoodSearchMethod(search_method.second);
      ndt_omp.setInputTarget(target_cloud);
      ndt_omp.setInputSource(source_cloud);
      
      auto t1 = ros::WallTime::now();
      ndt_omp.align(*aligned, init_guess.cast<float>());
      auto t2 = ros::WallTime::now();
      std::cout << "single : " << (t2 - t1).toSec() * 1000 << "[msec]" << std::endl;
      
      fitness_score = ndt_cpu.getFitnessScore();
      printf("fitness_score : %f\n", fitness_score);
    }
  }

  // visualization
  pcl::visualization::PCLVisualizer vis("vis");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_handler(target_cloud, 255.0, 0.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_handler(source_cloud, 0.0, 255.0, 0.0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_handler(aligned, 0.0, 0.0, 255.0);
  vis.addPointCloud(target_cloud, target_handler, "target");
  vis.addPointCloud(source_cloud, source_handler, "source");
  vis.addPointCloud(aligned, aligned_handler, "aligned");
  vis.spin();

  return 0;
}
