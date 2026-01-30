#ifndef MATH_PRACTICE_NONLINEAR_OPTIMIZATION_HPP_
#define MATH_PRACTICE_NONLINEAR_OPTIMIZATION_HPP_
#include <Eigen/Dense>
#include <string>
#include <ceres/ceres.h>

namespace nonlinear_optimization {

// 定义代价函数结构体 fx = exp(a * x) - y
struct ExponentialResidual {
    ExponentialResidual(double x, double y) : x_(x), y_(y) {}

    // 必须实现一个模板运算符 ()
    // T 是模板类型，可以是 double，也可以是 Ceres 内部用于求导的特殊类型
    template <typename T>
    bool operator()(const T* const a, T* residual) const {
        // 残差公式: r = exp(a * x) - y
        // 注意：数学函数如 exp 必须使用 ceres::exp
        residual[0] = ceres::exp(a[0] * T(x_)) - T(y_);
        return true;
    }

 private:
    const double x_;
    const double y_;
};

void GradientDescentUsage();
void NewTonUsage();
void GaussNewtonUsage();
}  // namespace nonlinear_optimization

#endif  // MATH_PRACTICE_NONLINEAR_OPTIMIZATION_HPP_
