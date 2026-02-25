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

// 定义点到点 ICP 的代价函数结构体 (残差维度 3 = x, y, z 误差)
struct PointToPointResidual {
    PointToPointResidual(const Eigen::Vector3d& p_src, const Eigen::Vector3d& p_tgt)
        : p_src_(p_src), p_tgt_(p_tgt) {}

    // 参数：q_ptr(四元数指针), t_ptr(平移向量指针), residual(残差指针)
    template <typename T>
    bool operator()(const T* const q_ptr, const T* const t_ptr, T* residual) const {
        // 使用 Eigen::Map 将裸指针映射为 Eigen 的数据结构
        // 注意：Eigen 四元数的内存布局是 [x, y, z, w]
        Eigen::Map<const Eigen::Quaternion<T>> q(q_ptr);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(t_ptr);

        // 核心公式：将源点云 p_src 通过当前优化的位姿 (q, t) 变换到目标坐标系
        Eigen::Matrix<T, 3, 1> p_transformed = q * p_src_.cast<T>() + t;

        // 计算残差 = 变换后的点 - 目标点
        residual[0] = p_transformed[0] - T(p_tgt_[0]);
        residual[1] = p_transformed[1] - T(p_tgt_[1]);
        residual[2] = p_transformed[2] - T(p_tgt_[2]);

        return true;
    }

 private:
    const Eigen::Vector3d p_src_;
    const Eigen::Vector3d p_tgt_;
};

void GradientDescentUsage();
void NewTonUsage();
void GaussNewtonUsage();
void LevenbergMarquardtUsage();
void CeresSolverUsage();
void CeresPoseOptimizationUsage();
}  // namespace nonlinear_optimization

#endif  // MATH_PRACTICE_NONLINEAR_OPTIMIZATION_HPP_
