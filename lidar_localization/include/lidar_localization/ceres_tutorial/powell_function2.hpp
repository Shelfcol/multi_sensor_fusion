#ifndef POWELL_FUNCTION2_HPP_
#define POWELL_FUNCTION2_HPP_

#include "ceres/ceres.h"

// An example program that minimizes Powell's singular function.
//
//   F = 1/2 (f1^2 + f2^2 + f3^2 + f4^2)
//
//   f1 = x1 + 10*x2;
//   f2 = sqrt(5) * (x3 - x4)
//   f3 = (x2 - 2*x3)^2
//   f4 = sqrt(10) * (x1 - x4)^2
//
// The starting values are x1 = 3, x2 = -1, x3 = 0, x4 = 1.
// The minimum is 0 at (x1, x2, x3, x4) = 0.

// x = [x1,x2,x3,x4]   4x1

// 定义CostFunction模型
struct F1CostFunc
{
    template<typename T>
    bool operator()(const T* const x, T* residual) const{
        residual[0] = x[0]+T(10)*x[1];
        return true;
    }
};

struct F2CostFunc
{
    template<typename T>
    bool operator()(const T* const x, T* residual) const{
        residual[0]=sqrt(5)*(x[2]-x[3]);
        return true;
    }
};

struct F3CostFunc
{
    template<typename T>
    bool operator()(const T* const x, T* residual)const{
        residual[0] = (x[1]-T(2)*x[2])*(x[1]-T(2)*x[2]);
        return true;
    }
};

struct F4CostFunc
{
    template<typename T>
    bool operator()(const T* const x, T* residual) const{
        residual[0] = sqrt(10)*(x[0]-x[3])*(x[0]-x[3]);
        return true;
    }
};




void TestPowellFunc2()
{
    // double x1 = 3, x2 = -1, x3 = 0, x4 = 1;
    double x[]={1,1,1,1};
    // double x[]={3,-1,0,1};
    ceres::Problem problem;
    ceres::CostFunction* f1CostFunc = new ceres::AutoDiffCostFunction<F1CostFunc,1,4>(new F1CostFunc); // 损失函数模型，残差维度，各个输入参数维度
    ceres::CostFunction* f2CostFunc = new ceres::AutoDiffCostFunction<F2CostFunc,1,4>(new F2CostFunc); // 损失函数模型，残差维度，各个输入参数维度
    ceres::CostFunction* f3CostFunc = new ceres::AutoDiffCostFunction<F3CostFunc,1,4>(new F3CostFunc); // 损失函数模型，残差维度，各个输入参数维度
    ceres::CostFunction* f4CostFunc = new ceres::AutoDiffCostFunction<F4CostFunc,1,4>(new F4CostFunc); // 损失函数模型，残差维度，各个输入参数维度
    problem.AddResidualBlock(f1CostFunc,nullptr,x); // 不能使用核函数
    problem.AddResidualBlock(f2CostFunc,nullptr,x);
    problem.AddResidualBlock(f3CostFunc,nullptr,x);
    problem.AddResidualBlock(f4CostFunc,nullptr,x);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);

    std::cout<<summary.BriefReport()<<"\n";
    printf("x1 =%f, x2=%f, x3=%f, x4=%f\n",x[0],x[1],x[2],x[3]);
}

#endif
