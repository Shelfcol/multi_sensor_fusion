#ifndef CURVE_FITTING_G2O_HPP_
#define CURVE_FITTING_G2O_HPP_

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "g2o/stuff/sampler.h"
#include "g2o/stuff/command_args.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include <vector>
using namespace std;

// y=exp(0.3x+0.1)  y=exp(para[0]x+para[1])


const int kNumObservations = 67;
// clang-format off


// 定义顶点
class CurveFittingVertex: public g2o::BaseVertex<2,Eigen::Vector2d> // 顶点维杜，顶点数据类型
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    virtual void setToOriginImpl(){ // 顶点初始值？？
        _estimate<<0,0; // _estimate表示顶点的评估值
    }
    virtual void oplusImpl(const double* update){
        _estimate += Eigen::Map<const Eigen::Vector2d>(update);
    }
    virtual bool read(istream& in){}
    virtual bool write(ostream& out) const {}
};

// 定义边
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>//观测值维度，观测值类型，顶点类型
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CurveFittingEdge(const double& x):BaseUnaryEdge(), x_(x){}
    void computeError(){
        // 还原这条边的顶点，这里根据继承的边确定
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]); //得到边的第一个顶点
        const Eigen::Vector2d ab = v->estimate(); // 顶点的估计值，类型为顶点的类型
        _error(0,0) = _measurement-std::exp(ab(0,0)*x_+ab(1,0)); //_error是D维向量 Eigen::Matrix<double, D, 1, Eigen::ColMajor>
    }
    virtual bool read(istream& in){}
    virtual bool write(ostream& out)const {}
public:
    double x_;
};

void TestCurveFittingG2O()
{
    const double data[] = {
        0.000000e+00, 1.133898e+00,
        7.500000e-02, 1.334902e+00,
        1.500000e-01, 1.213546e+00,
        2.250000e-01, 1.252016e+00,
        3.000000e-01, 1.392265e+00,
        3.750000e-01, 1.314458e+00,
        4.500000e-01, 1.472541e+00,
        5.250000e-01, 1.536218e+00,
        6.000000e-01, 1.355679e+00,
        6.750000e-01, 1.463566e+00,
        7.500000e-01, 1.490201e+00,
        8.250000e-01, 1.658699e+00,
        9.000000e-01, 1.067574e+00,
        9.750000e-01, 1.464629e+00,
        1.050000e+00, 1.402653e+00,
        1.125000e+00, 1.713141e+00,
        1.200000e+00, 1.527021e+00,
        1.275000e+00, 1.702632e+00,
        1.350000e+00, 1.423899e+00,
        1.425000e+00, 1.543078e+00,
        1.500000e+00, 1.664015e+00,
        1.575000e+00, 1.732484e+00,
        1.650000e+00, 1.543296e+00,
        1.725000e+00, 1.959523e+00,
        1.800000e+00, 1.685132e+00,
        1.875000e+00, 1.951791e+00,
        1.950000e+00, 2.095346e+00,
        2.025000e+00, 2.361460e+00,
        2.100000e+00, 2.169119e+00,
        2.175000e+00, 2.061745e+00,
        2.250000e+00, 2.178641e+00,
        2.325000e+00, 2.104346e+00,
        2.400000e+00, 2.584470e+00,
        2.475000e+00, 1.914158e+00,
        2.550000e+00, 2.368375e+00,
        2.625000e+00, 2.686125e+00,
        2.700000e+00, 2.712395e+00,
        2.775000e+00, 2.499511e+00,
        2.850000e+00, 2.558897e+00,
        2.925000e+00, 2.309154e+00,
        3.000000e+00, 2.869503e+00,
        3.075000e+00, 3.116645e+00,
        3.150000e+00, 3.094907e+00,
        3.225000e+00, 2.471759e+00,
        3.300000e+00, 3.017131e+00,
        3.375000e+00, 3.232381e+00,
        3.450000e+00, 2.944596e+00,
        3.525000e+00, 3.385343e+00,
        3.600000e+00, 3.199826e+00,
        3.675000e+00, 3.423039e+00,
        3.750000e+00, 3.621552e+00,
        3.825000e+00, 3.559255e+00,
        3.900000e+00, 3.530713e+00,
        3.975000e+00, 3.561766e+00,
        4.050000e+00, 3.544574e+00,
        4.125000e+00, 3.867945e+00,
        4.200000e+00, 4.049776e+00,
        4.275000e+00, 3.885601e+00,
        4.350000e+00, 4.110505e+00,
        4.425000e+00, 4.345320e+00,
        4.500000e+00, 4.161241e+00,
        4.575000e+00, 4.363407e+00,
        4.650000e+00, 4.161576e+00,
        4.725000e+00, 4.619728e+00,
        4.800000e+00, 4.737410e+00,
        4.875000e+00, 4.727863e+00,
        4.950000e+00, 4.669206e+00,
        };

    // 构建图优化，先设定g2o
    // 矩阵块
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<2,1>> Block; // 每个误差项优化变量维度为2，误差值维度为1

    // 创建线性求解器 linearSolver
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
    Block* solver_ptr = new Block(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    
    g2o::SparseOptimizer optimizer; // 定义优化器
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // 添加顶点
    CurveFittingVertex* v = new CurveFittingVertex();
    v->setEstimate(Eigen::Vector2d(0,0)); // 设置顶点初始值
    v->setId(0); // 设置顶点的ID号
    optimizer.addVertex(v); //! 不加入addEdge就会报异常

    // 向图中添加边
    for(int i=0;i<kNumObservations;++i)
    {
        CurveFittingEdge* edge = new CurveFittingEdge(data[2*i]);
        edge->setId(i); // 设置边的编号
        edge->setVertex(0,v); //给这条边设置顶点，这条边的顶点编号从0开始
        edge->setMeasurement(data[2*i+1]);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity());
        optimizer.addEdge(edge);
    }
    
    // 执行优化
    optimizer.initializeOptimization();
    optimizer.optimize(100); // 迭代次数
    
    // 输出优化值
    Eigen::Vector2d ab_estimate = v->estimate();
    std::cout<<"ab estimate = "<<ab_estimate;
}


#endif