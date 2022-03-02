#ifndef POSE_ESTIMATION_3DD_HPP_
#define POSE_ESTIMATION_3DD_HPP_

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void bundleAdjustment (
    const vector<Point3d> points_3d,
    const vector<Point2d> points_2d,
    const Mat& K,
    Mat& R, Mat& t
);

void bundleAdjustment2 (
    const vector<Point3d> points_3d,
    const vector<Point2d> points_2d,
    const Mat& K,
    Mat& R, Mat& t
);
// 直接使用1_depth.png的深度图作为3D位置，和2.png的像素位置求解PNP，免去了单目的初始化过程
void TEST_PNP()
{
    //-- 读取图像
    string img1_path = "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/include/lidar_localization/ceres_tutorial/1.png";
    string img2_path = "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/include/lidar_localization/ceres_tutorial/2.png";
    string img1depth_path = "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/include/lidar_localization/ceres_tutorial/1_depth.png";
    string img2depth_path = "/home/gxf/multi-sensor-fusion/chapter_1/src/chapter_1/lidar_localization/include/lidar_localization/ceres_tutorial/2_depth.png";

    
    Mat img_1 = imread ( img1_path.c_str(), IMREAD_COLOR );
    Mat img_2 = imread ( img2_path.c_str(), IMREAD_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread ( img1depth_path.c_str(), IMREAD_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3d> pts_3d;
    vector<Point2d> pts_2d;
    for ( DMatch m:matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3d ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    Mat r, t;
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cout<<"angle_axis:"<<r<<endl;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;

    bundleAdjustment2 ( pts_3d, pts_2d, K, R, t );
}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

// 旋转用轴角表示，输入的是3D点和2D点，和K
struct ReprojectionErr
{
public:
    ReprojectionErr(const cv::Point3d& p_3_, const cv::Point2d& p_2_, const cv::Mat& K_):p_3(p_3_),p_2(p_2_),K(K_){}
    template<typename T>
    bool operator()(const T* const camera, T* residuals) const
    {
        // camera的0 1 2表示旋转，345表示平移
        T point[3]={T(p_3.x),T(p_3.y),T(p_3.z)};
        T p_trans[3];
        ceres::AngleAxisRotatePoint(camera, point, p_trans); // 旋转
        p_trans[0]+=camera[3];// 平移
        p_trans[1]+=camera[4];
        p_trans[2]+=camera[5];
        double fx = K.at<double>(0,0);
        double fy = K.at<double>(1,1);
        double cx = K.at<double>(0,2);
        double cy = K.at<double>(1,2);
        residuals[0]=T(p_2.x)-fx*p_trans[0]/p_trans[2]-cx;
        residuals[1]=T(p_2.y)-fy*p_trans[1]/p_trans[2]-cy;
        return true;
    }
    //避免每次重复创建实例和析构实例，使用工厂模式
    static ceres::CostFunction* Create(const cv::Point3d& p_3_, const cv::Point2d& p_2_, const cv::Mat& K_)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionErr,2,6>(new ReprojectionErr(p_3_,p_2_,K_));
    }
private:
    cv::Mat K;
    cv::Point3d p_3;
    cv::Point2d p_2;
};



// 根据对应的3D和2D点，求解旋转平移矩阵
void bundleAdjustment (
    const vector< Point3d > points_3d,
    const vector< Point2d > points_2d,
    const Mat& K,
    Mat& R, Mat& t)
{
    // 给求解得到的Rt一个小的扰动，作为BA初值
    double angle_axis[3];
    double Rot[9]={ R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
                    R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
                    R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2)
                    };
    ceres::RotationMatrixToAngleAxis(Rot, angle_axis);
   
    double camera []= {angle_axis[0]+0.1,angle_axis[1]+0.1,angle_axis[2]+0.1,t.at<double>(0,0)+0.1,t.at<double>(1,0)+1,t.at<double>(2,0)+0.5};
    printf("true angle_axis  trans :(%f,%f,%f,%f,%f,%f)\n",angle_axis[0],angle_axis[1],angle_axis[2],t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));
    ceres::Problem problem;
    for(size_t i=0;i<points_3d.size();++i)
    {
        ceres::CostFunction* cost_func = ReprojectionErr::Create(points_3d[i],points_2d[i],K);
        problem.AddResidualBlock(cost_func,new ceres::HuberLoss(2),camera);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, & summary);

    std::cout<<summary.BriefReport()<<std::endl;
    //Ceres Solver Report: Iterations: 7, Initial cost: 2.851085e+04, Final cost: 1.255662e+02, Termination: CONVERGENCE
    printf("calc angle_axis  trans :(%f,%f,%f,%f,%f,%f)\n",camera[0],camera[1],camera[2],camera[3],camera[4],camera[5]);
}

// 旋转用轴角表示，输入的是3D点和2D点，同时优化3D点
struct Reprojection3DErr
{
public:
    Reprojection3DErr(const cv::Point2d& p_2_, const cv::Mat& K_):p_2(p_2_),K(K_){}
    template<typename T>
    bool operator()(const T* const camera, const T* const p_3, T* residuals) const
    {
        // camera的0 1 2表示旋转，345表示平移
        T p_trans[3];
        ceres::AngleAxisRotatePoint(camera, p_3, p_trans); // 旋转
        p_trans[0]+=camera[3];// 平移
        p_trans[1]+=camera[4];
        p_trans[2]+=camera[5];
        double fx = K.at<double>(0,0);
        double fy = K.at<double>(1,1);
        double cx = K.at<double>(0,2);
        double cy = K.at<double>(1,2);
        residuals[0]=T(p_2.x)-fx*p_trans[0]/p_trans[2]-cx;
        residuals[1]=T(p_2.y)-fy*p_trans[1]/p_trans[2]-cy;
        return true;
    }
    //避免每次重复创建实例和析构实例，使用工厂模式
    static ceres::CostFunction* Create(const cv::Point2d& p_2_, const cv::Mat& K_)
    {
        return new ceres::AutoDiffCostFunction<Reprojection3DErr,2,6,3>(new Reprojection3DErr(p_2_,K_));
    }
private:
    cv::Mat K;
    cv::Point2d p_2;
};


// 根据对应的3D和2D点，求解R和t，同时3D点也是优化量，最终于不优化3D点对比最终的cost
void bundleAdjustment2 (
    const vector< Point3d > points_3d,
    const vector< Point2d > points_2d,
    const Mat& K,
    Mat& R, Mat& t)
{
    printf("optimize R,t,3D points:\n");
    // 给求解得到的Rt一个小的扰动，作为BA初值
    double angle_axis[3];
    double Rot[9]={ R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),
                    R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),
                    R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2)
                    };
    ceres::RotationMatrixToAngleAxis(Rot, angle_axis);
   
    double camera []= {angle_axis[0]+0.1,angle_axis[1]+0.1,angle_axis[2]+0.1,t.at<double>(0,0)+0.1,t.at<double>(1,0)+1,t.at<double>(2,0)+0.5};
    printf("true angle_axis  trans :(%f,%f,%f,%f,%f,%f)\n",angle_axis[0],angle_axis[1],angle_axis[2],t.at<double>(0,0),t.at<double>(1,0),t.at<double>(2,0));
    ceres::Problem problem;
    for(size_t i=0;i<points_3d.size();++i)
    {
        ceres::CostFunction* cost_func = Reprojection3DErr::Create(points_2d[i],K);
        double p_3[]={points_3d[i].x,points_3d[i].y,points_3d[i].z};
        problem.AddResidualBlock(cost_func,new ceres::HuberLoss(2),camera,p_3);
    }
    printf("6\n");
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout<<summary.FullReport()<<std::endl;
    printf("calc angle_axis  trans :(%f,%f,%f,%f,%f,%f)\n",camera[0],camera[1],camera[2],camera[3],camera[4],camera[5]);
}




#endif