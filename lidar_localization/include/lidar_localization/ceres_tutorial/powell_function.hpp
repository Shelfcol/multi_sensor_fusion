#ifndef POWELL_FUNCTION_HPP_
#define POWELL_FUNCTION_HPP_

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

// 定义CostFunction模型
struct F1CostFunction
{
    template<typename T>
    bool operator()(const T* const x1, const T* const x2, T* residual) const{
        residual[0] = x1[0]+T(10)*x2[0];
        return true;
    }
};

struct F2CostFunction
{
    template<typename T>
    bool operator()(const T* const x3, const T* const x4, T* residual) const{
        residual[0]=sqrt(5)*(x3[0]-x4[0]);
        return true;
    }
};

struct F3CostFunction
{
    template<typename T>
    bool operator()(const T* const x2, const T* const x3, T* residual)const{
        residual[0] = (x2[0]-T(2)*x3[0])*(x2[0]-T(2)*x3[0]);
        return true;
    }
};

struct F4CostFunction
{
    template<typename T>
    bool operator()(const T* const x1, const T* const x4, T* residual) const{
        residual[0] = sqrt(10)*(x1[0]-x4[0])*(x1[0]-x4[0]);
        return true;
    }
};




void TestPowellFunc()
{
    double x1 = 3, x2 = -1, x3 = 0, x4 = 1;
    ceres::Problem problem;
    ceres::CostFunction* f1CostFunc = new ceres::AutoDiffCostFunction<F1CostFunction,1,1,1>(new F1CostFunction); // 损失函数模型，残差维度，各个输入参数维度
    ceres::CostFunction* f2CostFunc = new ceres::AutoDiffCostFunction<F2CostFunction,1,1,1>(new F2CostFunction); // 损失函数模型，残差维度，各个输入参数维度
    ceres::CostFunction* f3CostFunc = new ceres::AutoDiffCostFunction<F3CostFunction,1,1,1>(new F3CostFunction); // 损失函数模型，残差维度，各个输入参数维度
    ceres::CostFunction* f4CostFunc = new ceres::AutoDiffCostFunction<F4CostFunction,1,1,1>(new F4CostFunction); // 损失函数模型，残差维度，各个输入参数维度
    problem.AddResidualBlock(f1CostFunc,nullptr,&x1,&x2);
    problem.AddResidualBlock(f2CostFunc,nullptr,&x3,&x4);
    problem.AddResidualBlock(f3CostFunc,nullptr,&x2,&x3);
    problem.AddResidualBlock(f4CostFunc,nullptr,&x1,&x4);

    ceres::Solver::Options options;
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);
    std::cout<<summary.BriefReport()<<"\n";
    printf("x1 =%f, x2=%f, x3=%f, x4=%f\n",x1,x2,x3,x4);
}

#endif
