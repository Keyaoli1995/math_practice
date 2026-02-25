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
  double alpha = 0.005;  // 学习率 (步长)
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
    if (std::abs(alpha * total_gradient) < 1e-8) break;  // 收敛判定
  }

  LOG_INFO("优化完成! 最终估计 a: {}, 真值对比: {}, 误差偏移: {}", a_est,
           a_true, std::abs(a_est - a_true));
}

void NewTonUsage() {
  // 1. 数据准备 (真值 a = 0.8)
  double a_true = 0.8;
  std::vector<double> x_data, y_data;
  for (int i = 0; i < 100; ++i) {
    double x = i / 100.0;
    x_data.push_back(x);
    double noise = 0.01 * (rand() / double(RAND_MAX));
    y_data.push_back(std::exp(a_true * x) + noise);
  }

  // 2. 牛顿法参数初始化
  double a_est = 0.0;    // 初始猜测
  int iterations = 100;  // 注意：牛顿法收敛极快，通常 10 次就够了

  LOG_INFO("开始牛顿法优化: 初始参数 a: {}, 迭代总数: {}", a_est, iterations);

  for (int iter = 0; iter < iterations; ++iter) {
    double H = 0;  // Hessian 矩阵 (本例中是标量)
    double g = 0;  // 梯度 Gradient
    double total_cost = 0;

    for (size_t i = 0; i < x_data.size(); ++i) {
      double x = x_data[i];
      double y = y_data[i];

      double exp_ax = std::exp(a_est * x);
      double error = exp_ax - y;

      // 计算 Jacobian (de/da)
      double J = x * exp_ax;

      // 计算梯度 g = J * error
      g += J * error;

      // 计算 Hessian = J*J + error * (dJ/da)
      // 其中 dJ/da = x^2 * exp_ax
      H += J * J + error * (x * x * exp_ax);

      total_cost += 0.5 * error * error;
    }

    // 3. 解牛顿增量方程: H * da = -g
    double da = -g / H;

    // 4. 更新参数
    a_est += da;

    LOG_INFO("迭代轮次: {}, 当前参数 a: {}, 增量 da: {}, 当前总误差 Cost: {}",
             iter, a_est, da, total_cost);

    if (std::abs(da) < 1e-8) break;  // 收敛判定
  }

  LOG_INFO("牛顿法优化完成! 最终估计 a: {}, 真值对比: {}, 误差偏移: {}", a_est,
           a_true, std::abs(a_est - a_true));
}

void GaussNewtonUsage() {
  // 1. 生成数据 (真值 a = 0.8)
  double a_true = 0.8;
  std::vector<double> x_data, y_data;
  for (int i = 0; i < 100; ++i) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(std::exp(a_true * x) + 0.01 * rand() / double(RAND_MAX));
  }

  // 2. 高斯-牛顿参数初始化
  double a_est = 0.0;
  int iterations = 10;

  LOG_INFO("开始高斯-牛顿法优化: 初始参数 a: {}, 迭代总数: {}", a_est,
           iterations);

  for (int iter = 0; iter < iterations; ++iter) {
    // 在 GN 中，我们习惯称 J^T * J 为 H (近似 Hessian)
    double H = 0;
    double g = 0;  // 梯度 g = J^T * f
    double total_cost = 0;

    for (size_t i = 0; i < x_data.size(); ++i) {
      double x = x_data[i];
      double y = y_data[i];

      double exp_ax = std::exp(a_est * x);
      double error = exp_ax - y;

      // 计算 Jacobian (残差对 a 的导数)
      double J = x * exp_ax;

      // 核心公式：H = J * J^T, g = -J * error
      H += J * J;
      g += -J * error;

      total_cost += 0.5 * error * error;
    }

    // 3. 求解线性方程组 H * da = g (这里 H 只是个标量，直接除就行)
    // 在 SLAM 复杂问题中，这里会用到你学过的 QR 或 SVD 分解
    double da = g / H;

    // 4. 更新
    a_est += da;

    LOG_INFO("迭代轮次: {}, 当前参数 a: {}, 增量 da: {}, 当前总误差 Cost: {}",
             iter, a_est, da, total_cost);

    if (std::abs(da) < 1e-8) break;
  }

  LOG_INFO("高斯-牛顿优化完成! 最终估计 a: {}, 真值对比: {}, 误差偏移: {}",
           a_est, a_true, std::abs(a_est - a_true));
}

void LevenbergMarquardtUsage() {
  // 1. 数据准备
  double a_true = 0.8;
  std::vector<double> x_data, y_data;
  srand(42);
  for (int i = 0; i < 100; ++i) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(std::exp(a_true * x) + 0.01 * rand() / double(RAND_MAX));
  }

  // 2. LM 参数初始化
  double a_est = 0.0;
  double lambda = 1e-3;
  int max_iterations = 100;

  LOG_INFO("开始精细化 LM 算法优化: 初始 a: {}, 初始 lambda: {}", a_est,
           lambda);

  for (int iter = 0; iter < max_iterations; ++iter) {
    // --- 步骤 A: 计算当前点的 Jacobian, Hessian 和 Cost ---
    double H = 0;
    double g_neg = 0;  // J^T * (y - y_pred)
    double current_cost = 0;

    for (size_t i = 0; i < x_data.size(); ++i) {
      double exp_ax = std::exp(a_est * x_data[i]);
      double error = y_data[i] - exp_ax;
      double J = x_data[i] * exp_ax;
      H += J * J;
      g_neg += J * error;
      current_cost += 0.5 * error * error;
    }

    // --- 步骤 B: 求解 LM 方程 (H + lambda) * da = g_neg ---
    double da = g_neg / (H + lambda);

    // --- 步骤 C: 尝试更新并计算新 Cost ---
    double a_new = a_est + da;
    double new_cost = 0;
    for (size_t i = 0; i < x_data.size(); ++i) {
      double e = y_data[i] - std::exp(a_new * x_data[i]);
      new_cost += 0.5 * e * e;
    }

    // --- 步骤 D: 计算增益比例 rho ---
    double actual_reduction = current_cost - new_cost;
    double predicted_reduction = 0.5 * da * (lambda * da + g_neg);
    double rho = actual_reduction / (predicted_reduction + 1e-18);

    // --- 步骤 E: 根据 rho 判断是否接受更新并调节 lambda ---
    if (rho > 0) {
      // 成功：接受更新参数
      a_est = a_new;

      // 调节 lambda 阈值逻辑
      if (rho > 0.75) {
        lambda *= 0.33;  // 预测很准，减小阻尼
      } else if (rho < 0.25) {
        lambda *= 2.0;  // 预测一般，增大阻尼
      }
      // 在 0.25 到 0.75 之间保持 lambda 不变

      LOG_INFO("迭代 {}: [成功] a: {}, cost: {}, rho: {}, lambda: {}", iter,
               a_est, new_cost, rho, lambda);

      // 收敛判断
      if (std::abs(da) < 1e-8) {
        LOG_INFO("算法收敛。");
        break;
      }
    } else {
      // 失败：不更新 a_est，只增大 lambda
      lambda *= 10.0;
      LOG_INFO("迭代 {}: [失败] 拒绝更新, rho: {}, lambda 增大至: {}", iter,
               rho, lambda);

      if (lambda > 1e12) {
        LOG_INFO("lambda 过大，停止优化。");
        break;
      }
    }
  }

  LOG_INFO("最终估计结果 a_est: {} (真实值: 0.8)", a_est);
}

void CeresSolverUsage() {
  // 1. 模拟数据准备 (真值 a = 0.8)
  double a_true = 0.8;
  std::vector<double> x_data, y_data;
  for (int i = 0; i < 100; ++i) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(std::exp(a_true * x) + 0.01 * rand() / double(RAND_MAX));
  }

  // 2. 初始化待优化参数
  double a_est = 0.0;

  // 3. 构建 Ceres 问题
  ceres::Problem problem;
  for (size_t i = 0; i < x_data.size(); ++i) {
    // 使用自动求导的代价函数<代价函数类, 残差维度, 参数维度>
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1>(
            new ExponentialResidual(x_data[i], y_data[i]));
    // 向问题中添加残差块：代价函数, 核函数(null), 待优化参数地址
    problem.AddResidualBlock(cost_function, nullptr, &a_est);
  }

  // 4. 配置求解器选项
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;

  // 5. 求解
  ceres::Solver::Summary summary;
  LOG_INFO("Ceres 开始优化: 初始 a: {}", a_est);
  ceres::Solve(options, &problem, &summary);

  // 输出结果
  LOG_INFO(
      "Ceres 优化完成! 最终估计 a: {}, 迭代总数: {}, 初始误差: {}, 最终误差: "
      "{}",
      a_est, (int)summary.iterations.size(), summary.initial_cost,
      summary.final_cost);
  if (summary.termination_type == ceres::CONVERGENCE) {
    LOG_INFO("状态: 算法已收敛, 误差偏移: {}", std::abs(a_est - a_true));
  }
}

void CeresPoseOptimizationUsage() {
  // 1. 模拟数据准备：生成一组真实位姿 (Ground Truth)
  // 假设真实的旋转是绕 Z 轴旋转 45 度，平移是 (1.0, 2.0, 3.0)
  Eigen::AngleAxisd gt_rotation_vector(M_PI / 4.0, Eigen::Vector3d(0, 0, 1));
  Eigen::Quaterniond gt_q(gt_rotation_vector);
  Eigen::Vector3d gt_t(1.0, 2.0, 3.0);

  std::vector<Eigen::Vector3d> source_points, target_points;
  srand(42);

  // 生成 100 个随机三维点，并计算变换加噪声后的目标点
  for (int i = 0; i < 100; ++i) {
    Eigen::Vector3d p_src(10.0 * rand() / double(RAND_MAX),
                          10.0 * rand() / double(RAND_MAX),
                          10.0 * rand() / double(RAND_MAX));
    source_points.push_back(p_src);

    // 变换：p_tgt = R * p_src + t + noise
    Eigen::Vector3d noise(0.05 * rand() / double(RAND_MAX),
                          0.05 * rand() / double(RAND_MAX),
                          0.05 * rand() / double(RAND_MAX));
    Eigen::Vector3d p_tgt = gt_q * p_src + gt_t + noise;
    target_points.push_back(p_tgt);
  }
  // 打印一些初始数据对比
  LOG_INFO("source_points[0]: [{:.4f}, {:.4f}, {:.4f}] --> target_points[0]: [{:.4f}, {:.4f}, {:.4f}]",
           source_points[0].x(), source_points[0].y(), source_points[0].z(),
           target_points[0].x(), target_points[0].y(), target_points[0].z());

  // 2. 初始化待优化的参数
  // Eigen 四元数的内存布局是 [x, y, z, w]。初始化为单位四元数 (无旋转)
  double q_est[4] = {0.0, 0.0, 0.0, 1.0};
  // 初始化平移为 (0, 0, 0)
  double t_est[3] = {0.0, 0.0, 0.0};

  // 3. 构建 Ceres 问题
  ceres::Problem problem;

  // 重点：告诉 Ceres，q_est 是一个四元数，不能使用普通的加法更新
  // 它必须在流形（Manifold）上进行乘法更新，以保证更新后依然是合法的单位四元数
  // 注意：如果你使用的 Ceres 版本比较老（< 2.1），请将下面这行换成：
  // problem.AddParameterBlock(q_est, 4, new
  // ceres::EigenQuaternionParameterization());
  problem.AddParameterBlock(q_est, 4, new ceres::EigenQuaternionManifold());
  problem.AddParameterBlock(t_est,
                            3);  // 平移向量用普通加法即可，不需要 Manifold

  for (size_t i = 0; i < source_points.size(); ++i) {
    // 自动求导：<代价函数类, 残差维度(3), 旋转参数维度(4), 平移参数维度(3)>
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<PointToPointResidual, 3, 4, 3>(
            new PointToPointResidual(source_points[i], target_points[i]));

    problem.AddResidualBlock(cost_function, nullptr, q_est, t_est);
  }

  // 4. 配置求解器
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;

  // 5. 求解
  ceres::Solver::Summary summary;
  LOG_INFO("Ceres 位姿优化开始! 初始平移: [{}, {}, {}]", t_est[0], t_est[1],
           t_est[2]);
  ceres::Solve(options, &problem, &summary);

  // 6. 输出结果对比
  LOG_INFO("Ceres 位姿优化完成! 迭代总数: {}, 初始误差: {}, 最终误差: {}",
           (int)summary.iterations.size(), summary.initial_cost,
           summary.final_cost);

  if (summary.termination_type == ceres::CONVERGENCE) {
    Eigen::Map<Eigen::Quaterniond> final_q(q_est);
    LOG_INFO("状态: 算法已收敛");
    LOG_INFO(
        "真实平移 t: [1.0, 2.0, 3.0]  --> 估计平移 t: [{:.4f}, {:.4f}, {:.4f}]",
        t_est[0], t_est[1], t_est[2]);
    LOG_INFO(
        "真实旋转 q: [{:.4f}, {:.4f}, {:.4f}, {:.4f}] --> 估计旋转 q: [{:.4f}, "
        "{:.4f}, {:.4f}, {:.4f}]",
        gt_q.x(), gt_q.y(), gt_q.z(), gt_q.w(), final_q.x(), final_q.y(),
        final_q.z(), final_q.w());
  }
}

}  // namespace nonlinear_optimization