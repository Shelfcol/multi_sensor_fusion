#ifndef ROSENBROCK_HPP_
#define ROSENBROCK_HPP_
#include "ceres/ceres.h"
#include "glog/logging.h"

// f(x,y) = (1-x)^2 + 100(y - x^2)^2;
// 使用ceres解决无约束优化问题
// f(x) = (1-x[0])^2+100(x[1]-x[0]^2)^2

struct RosenbrockFunc
{
    template<typename T>
    bool operator()(const T* const x, T* residuals) const{
        residuals[0]=(1.0-x[0])*(1.0-x[0])+100.0*(x[1]-x[0]*x[0])*(x[1]-x[0]*x[0]);
        return true;
    }
    static ceres::CostFunction* Create(){
        return new ceres::AutoDiffCostFunction<RosenbrockFunc,1,2>(new RosenbrockFunc);
    }
};

void TestRosenbrockFunc()
{
    double x[]={0.9,0.9}; // 必须要给一个较好的初值，否则容易陷入局部最优
    ceres::Problem problem;
    problem.AddResidualBlock(RosenbrockFunc::Create(),nullptr,x);
    ceres::Solver::Options options;
    options.max_num_iterations = 5000;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    std::cout<<summary.BriefReport()<<std::endl;
    printf("output: (%f,%f)\n",x[0],x[1]);
}


#endif