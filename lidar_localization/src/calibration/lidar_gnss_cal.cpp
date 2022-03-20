#include "lidar_localization/calibration/lidar_gnss_cal.h"

#include <fstream>

namespace lidar_localization{

bool LidarGnssCal::Run()
{
    if(!ReadTumFormData(lidar_file_path_, lidar_pose_deq_)){
        return false;
    }
    if(!ReadTumFormData(gnss_file_path_, gnss_pose_deq_)){
        return false;
    }
    
    if(!ValidateData()){
        std::cerr<<"no data"<<std::endl;
        exit(-1);
    }
    GetHandEyeABMatrix();
    BuildAndSolveWithCeres();
    return true;
}

// 获取使用手眼标定标定模型需要的矩阵AB
bool LidarGnssCal::GetHandEyeABMatrix()
{
    for(size_t i=0;i<lidar_pose_deq_.size()-window_size_;++i)
    {
        Eigen::Matrix4d TA1 = lidar_pose_deq_.at(i).T;
        Eigen::Matrix4d TA2 = lidar_pose_deq_.at(i+window_size_).T;
        Eigen::Matrix4d TB1 = gnss_pose_deq_.at(i).T;
        Eigen::Matrix4d TB2 = gnss_pose_deq_.at(i+window_size_).T;    
        A_vec_.push_back(TA1.inverse()*TA2);
        B_vec_.push_back(TB1.inverse()*TB2);
        // std::cout<<"A ="<<A_vec_.back()<<std::endl;
        // std::cout<<"B ="<<B_vec_.back()<<std::endl;
        // static int a=5;
        // if(a--<0)  exit(0);
    }
    std::cout<<"A size = "<<A_vec_.size()<<std::endl;
}

// 利用ceres构建残差，解决问题
bool LidarGnssCal::BuildAndSolveWithCeres()
{
    ceres::Problem problem;
    ceres::LocalParameterization* local_quat = new ceres::EigenQuaternionParameterization;
    double trans[] = {0,0,0}; // 使用初始值初始化
    double quat[] = {0,0,0,1};
    problem.AddParameterBlock(trans,3);
    problem.AddParameterBlock(quat,4,local_quat);

    for(size_t i=0;i<A_vec_.size();++i)
    {
        Eigen::Matrix4d A = A_vec_[i];
        Eigen::Matrix4d B = B_vec_[i];
        ceres::CostFunction* cost_func = CalibErr::Create(A,B);
        problem.AddResidualBlock(cost_func, new ceres::HuberLoss(0.1),trans,quat);
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    Eigen::Map<Eigen::Quaterniond>  q_opt(quat);
    Eigen::Map<Eigen::Vector3d>  t_opt(trans);
    std::cout<<"t_opt " <<t_opt<<std::endl;
    return true;
}

// lidar_pose_deq_
// gnss_pose_deq_
// 将gnss_pose_deq_的数据使用插值于lidar_pose_deq_进行插值对齐，最后两个的时间戳时一致的，数量也是一致的
bool LidarGnssCal::ValidateData() 
{
    if(lidar_pose_deq_.empty() || gnss_pose_deq_.empty()) {return false;}
    double t1 = gnss_pose_deq_.front().time;
    while(!lidar_pose_deq_.empty() && lidar_pose_deq_.front().time<=gnss_pose_deq_.front().time){
        lidar_pose_deq_.pop_front();
    }

    while(!lidar_pose_deq_.empty() && lidar_pose_deq_.back().time>=gnss_pose_deq_.back().time){
        lidar_pose_deq_.pop_back();
    }
    if(lidar_pose_deq_.empty() || gnss_pose_deq_.empty()) {return false;}
    std::cout<<" lidar pose size = "<<lidar_pose_deq_.size()<<std::endl;
    std::cout<<" gnss pose size = "<<gnss_pose_deq_.size()<<std::endl;
    std::deque<PoseData>  gnss_pose_deq_tmp;
    std::deque<PoseData>  lidar_pose_deq_tmp;

    // 假设lidar_pose_dqe_的频率比
    for(size_t i=0; i<lidar_pose_deq_.size();++i){
        double sync_time = lidar_pose_deq_.at(i).time;
        for(int j=0;j<gnss_pose_deq_.size()-1;++j){
            if(gnss_pose_deq_.at(j).time>sync_time){
                break;
            }
            if(gnss_pose_deq_.at(j).time<=sync_time&&gnss_pose_deq_.at(j+1).time>=sync_time)
            {
                gnss_pose_deq_tmp.push_back(PoseInterp(gnss_pose_deq_.at(j), gnss_pose_deq_.at(j+1),sync_time));
                lidar_pose_deq_tmp.push_back(lidar_pose_deq_.at(i));
                break;
            }
        }
    }
    lidar_pose_deq_.swap(lidar_pose_deq_tmp);
    gnss_pose_deq_.swap(gnss_pose_deq_tmp);
    if(lidar_pose_deq_.empty() || gnss_pose_deq_.empty()) {return false;}
    return true;
}    


PoseData LidarGnssCal::GetT(std::string& data_file_line)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    Eigen::Quaterniond q;
    double time;
    Eigen::Vector3d trans;

    std::string data_tmp;

    std::stringstream ssr;
    ssr<<data_file_line;
    std::getline(ssr,data_tmp,' ');
    time = std::stod(data_tmp);

    std::getline(ssr,data_tmp,' ');
    trans.x() = std::stod(data_tmp);

    std::getline(ssr,data_tmp,' ');
    trans.y() = std::stod(data_tmp);

    std::getline(ssr,data_tmp,' ');
    trans.z() = std::stod(data_tmp);

    std::getline(ssr,data_tmp,' ');
    q.x() = std::stod(data_tmp);

    std::getline(ssr,data_tmp,' ');
    q.y() = std::stod(data_tmp);

    std::getline(ssr,data_tmp,' ');
    q.z() = std::stod(data_tmp);

    std::getline(ssr,data_tmp,' ');
    q.w() = std::stod(data_tmp);


    Eigen::Matrix3d T_Matrix = q.toRotationMatrix();
    T.block<3,3>(0,0) = T_Matrix;
    T.block<3,1>(0,3) = trans;
    if(0){ // test data in
        // std::cout<<"T = "<< T <<std::endl;
        std::cout<<time<<" "<<trans.x()<<" "<<trans.y()<<" "<<trans.z()<<" "<<q.x()<<" "<<q.y()<<" "<<q.z()<<" "<<q.w()<<" \n";
        static int a = 5;
        a--;
        if(a<0)
            exit(0); 
    }

    PoseData pose_data(time,T);
    return pose_data;
} 

// 读取kitti类型数据
bool LidarGnssCal::ReadTumFormData(const std::string& file_path, std::deque<PoseData>& pose_deq)
{
    std::ifstream  data_file(file_path,std::ios::in);
    if(!data_file.is_open()){
        std::cerr<<"failure to open file"<<file_path<<std::endl;
        return false;
    }
    pose_deq.clear();
    std::string data_file_line;
    std::string data_tmp;
    while(std::getline(data_file,data_file_line)){
        pose_deq.push_back(GetT(data_file_line));
    }
    return true;
};
}