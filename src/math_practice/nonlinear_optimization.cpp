#include "math_practice/nonlinear_optimization.hpp"

#include <vector>
#include "math_practice/spdlog_enhance.hpp"

namespace nonlinear_optimization {
void GradientDescentUsage() {
  // 1. 生成模拟观测数据 (模型: y = exp(a * x))
  double a_true = 0.8;
  std::vector<double> x_data, y_data;

  // 填充 100 个带噪声的采样点
  for (int i = 0; i < 100; ++i) {
    double x = i / 100.0;
    x_data.push_back(x);
    // 加入微小的高斯噪声
    double noise = 0.01 * (rand() / static_cast<double>(RAND_MAX));
    y_data.push_back(std::exp(a_true * x) + noise);
  }

  // 2. 优化参数初始化
  double a_est = 0.0;    // 初始猜测值
  double alpha = 0.005;   // 学习率 (步长)
  int iterations = 100;  // 迭代次数

  LOG_INFO("开始最速下降法优化: 初始参数 a: {}, 步长 alpha: {}, 迭代总数: {}",
           a_est, alpha, iterations);

  // 3. 迭代优化
  for (int iter = 0; iter < iterations; ++iter) {
    double total_gradient = 0;
    double total_cost = 0;

    for (size_t i = 0; i < x_data.size(); ++i) {
      double x = x_data[i];
      double y = y_data[i];

      // 计算残差: e = exp(ax) - y
      double prediction = std::exp(a_est * x);
      double error = prediction - y;

      // 计算雅可比 (一阶导数): de/da = x * exp(ax)
      double jacobian = x * std::exp(a_est * x);

      // 梯度更新方向: g = J^T * e
      total_gradient += jacobian * error;
      total_cost += 0.5 * error * error;
    }

    // 执行一步“下山”：a = a - alpha * gradient
    a_est -= alpha * total_gradient;

    // 按照你要求的风格打印关键迭代日志
    if (iter % 1 == 0) {
      LOG_INFO(
          "迭代轮次: {}, 当前参数 a: {}, 梯度的模: {}, 当前总误差 Cost: {}",
          iter, a_est, std::abs(total_gradient), total_cost);
    }
  }

  LOG_INFO("优化完成! 最终估计 a: {}, 真值对比: {}, 误差偏移: {}", a_est,
           a_true, std::abs(a_est - a_true));
}
}  // namespace nonlinear_optimization