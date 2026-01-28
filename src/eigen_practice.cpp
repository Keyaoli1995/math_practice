#include "math_practice/eigen_practice.hpp"

#include <string>

#include "math_practice/spdlog_enhance.hpp"

namespace eigen_practice {
void InitVectorAndMatrix() {
  // 1. 静态矩阵与向量：编译时确定大小
  spdlog::info(
      "创建一个3维度的0向量: Eigen::Vector3d v3 = Eigen::Vector3d::Zero();");
  Eigen::Vector3d v3 = Eigen::Vector3d::Zero();  // 创建一个3维零向量
  spdlog::info(
      "创建一个3x3维度的单位阵: Eigen::Matrix3d m33 = "
      "Eigen::Matrix3d::Identity();");
  Eigen::Matrix3d m33 = Eigen::Matrix3d::Identity();  // 3x3 单位阵
  // 2. 动态矩阵与向量
  spdlog::info("创建一个动态维度的矩阵");
  spdlog::info("执行--Eigen::MatrixXd m_dynamic(10, 10); ");
  Eigen::MatrixXd m_dynamic(10, 10);  // 这时候矩阵的各个元素是没有值的
  spdlog::info("m_dynamic的维度: row = {}, col = {}", m_dynamic.rows(),
               m_dynamic.cols());
  spdlog::info("现在m_dynamic中的元素时什么值呢?");
  spdlog::info("打印m_dynamic的前3个元素");
  for (int i = 0; i < 3; i++) {
    spdlog::info("第{}个元素 = {}", i, m_dynamic(i));
  }
  spdlog::info("执行--m_dynamic = Eigen::MatrixXd::Ones(5, 5); ");
  m_dynamic = Eigen::MatrixXd::Ones(5, 5);
  spdlog::info("m_dynamic的维度: row = {}, col = {}", m_dynamic.rows(),
               m_dynamic.cols());
  for (int i = 0; i < 3; i++) {
    spdlog::info("第{}个元素 = {}", i, m_dynamic(i));
  }
}

void VisitVectorAndMatrix() {
  // 方式 A：通过构造函数直接赋值（适用于小尺寸向量）
  Eigen::Vector3d v1(1.0, 2.0, 3.0);
  // 方式 B：先声明，后赋值
  Eigen::Vector3d v2;
  v2 << 1.0, 2.0, 3.0;  // 使用逗号初始化器（Comma Initializer）
  // 方式 C：创建全为某个常数的向量
  Eigen::Vector3d v3 = Eigen::Vector3d::Constant(1.5);  // [1.5, 1.5, 1.5]
  LOG_INFO("访问Vector中的元素: v1 = [{}, {}, {}]", v1(0), v1(1), v1(2));
  // 方式 A：逗号初始化（最直观）
  Eigen::Matrix3d m1;
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  // 方式 B：创建全为某个常数的矩阵
  Eigen::Matrix3d m2 = Eigen::Matrix3d::Constant(0.5);
  // 方式 C：创建随机矩阵（在测试算法鲁棒性时常用）
  Eigen::Matrix3d m3 = Eigen::Matrix3d::Random();
  spdlog::info("m1 row1: {}, {}, {}", m1(0, 0), m1(0, 1), m1(0, 2));
  spdlog::info("m1 row2: {}, {}, {}", m1(1, 0), m1(1, 1), m1(1, 2));
  spdlog::info("m1 row3: {}, {}, {}", m1(2, 0), m1(2, 1), m1(2, 2));

  // 示例:创建一个3*3的双精度矩阵m(非单位阵),然后提取它的第2列(索引为1)赋值给一个Vector3d变量v
  Eigen::Matrix3d m;
  m << 10, 11, 12, 13, 14, 15, 16, 17, 18;
  Eigen::Vector3d v = m.col(1);
  LOG_INFO(
      "创建一个3*3的双精度矩阵m(非单位阵),然后提取它的第2列(索引为1)"
      "赋值给一个Vector3d变量v");
  LOG_INFO("Eigen::Matrix3d m =");
  LOG_INFO("[{}, {}, {}]", m(0, 0), m(0, 1), m(0, 2));
  LOG_INFO("[{}, {}, {}]", m(1, 0), m(1, 1), m(1, 2));
  LOG_INFO("[{}, {}, {}]", m(2, 0), m(2, 1), m(2, 2));
  LOG_INFO("Eigen::Vector3d v = [{}, {}, {}]", v(0), v(1), v(2));
  m.col(0) = v;
  LOG_INFO("把提取出的向量放在原矩阵的第一列");
  LOG_INFO("Eigen::Matrix3d m =");
  LOG_INFO("[{}, {}, {}]", m(0, 0), m(0, 1), m(0, 2));
  LOG_INFO("[{}, {}, {}]", m(1, 0), m(1, 1), m(1, 2));
  LOG_INFO("[{}, {}, {}]", m(2, 0), m(2, 1), m(2, 2));
  /**
   * 示例2：块操作（Block Operations）
   * 在 SLAM 中，我们不仅会提取一列，有时还需要提取一个“局部”。
   * 比如从一个 $4 \times 4$ 的变换矩阵中提取左上角的 $3 \times 3$ 旋转矩阵。
   */
  Eigen::MatrixXd T = Eigen::MatrixXd::Identity(4, 4);
  LOG_INFO("T是一个4*4单位阵");
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  LOG_INFO("从T中提取旋转部分R");
  LOG_INFO("R = [{}, {}, {}]", R(0, 0), R(0, 1), R(0, 2));
  LOG_INFO("    [{}, {}, {}]", R(1, 0), R(1, 1), R(1, 2));
  LOG_INFO("    [{}, {}, {}]", R(2, 0), R(2, 1), R(2, 2));
}

void ConvertPosAtt() {
  // 点 $\mathbf{P}_A = [1.0, 0.5, 0.0]^T$
  // $A$ 系相对于 $W$ 系旋转了 $45^\circ$（绕 $Z$ 轴），并平移了 $[2.0, 1.0,
  // 0.0]^T$ 求点A在世界坐标系下的位置

  // 1. 定义点 P_A
  Eigen::Vector3d p_a(1.0, 0.5, 0.0);

  // 2. 定义旋转矩阵 R (绕Z轴旋转45度)
  double angle = M_PI / 4.0;  // 45度
  Eigen::Matrix3d R;
  R << cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0, 0, 0, 1;
  // 3. 定义平移向量 t
  Eigen::Vector3d t(2.0, 1.0, 0.0);
  // 5. 方案二：使用 4x4 变换矩阵 T (SLAM 标准做法)
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;       // 填充旋转部分
  T.topRightCorner<3, 1>() = t;  // 填充平移部分
                                 // 将 p_a 变为齐次坐标 [x, y, z, 1]
  Eigen::Vector4d p_a_homo(1.0, 0.5, 0.0, 1.0);
  Eigen::Vector4d p_w_homo = T * p_a_homo;
  LOG_INFO("p_w = [{}, {}, {}]", p_w_homo(0), p_w_homo(1), p_w_homo(2));
}

void AngleAxisUsage() {
  // C_b^n b n 开始时重合 b执行旋转
  Eigen::AngleAxis<double> rotation_vector(M_PI / 2.0,
                                           Eigen::Vector3d::UnitZ());
  Eigen::Vector3d point(1, 0, 0);
  LOG_INFO("旋转前的点坐标: [{}, {}, {}]", point(0), point(1), point(2));
  Eigen::Vector3d rotated_point = rotation_vector * point;
  LOG_INFO("旋转后的点坐标: [{}, {}, {}]", rotated_point(0), rotated_point(1),
           rotated_point(2));
  // 构造方式2
  Eigen::Vector3d vec(0, 0, 5);
  Eigen::AngleAxisd rotation_vector_1(M_PI_2, vec.normalized());
  Eigen::Vector3d rotated_point_1 = rotation_vector_1 * point;
  LOG_INFO("旋转后的点坐标: [{}, {}, {}]", rotated_point_1(0),
           rotated_point_1(1), rotated_point_1(2));
}

void QuaternionUsage() {
  // 通过矩阵构造四元数
  Eigen::Matrix3d m1 = Eigen::Matrix3d::Identity();
  Eigen::Quaterniond q1(m1);
  LOG_INFO("通过3维单位阵构造的四元数q1 = [{}, {}, {}, {}]", q1.coeffs()(0),
           q1.coeffs()(1), q1.coeffs()(2), q1.coeffs()(3));
  // 通过构造函数直接将w x y z赋值给四元数
  Eigen::Quaternion<double> q2(1, 0, 0, 0);
  // 通过旋转矢量构造四元数
  // 1. 定义旋转轴 (必须是单位向量) 和 旋转角度 (弧度)
  Eigen::Vector3d axis(0, 0, 1);  // 绕 Z 轴
  double angle = M_PI_2;          // 旋转 90 度
  // 2. 构造 AngleAxisUsage 对象
  Eigen::AngleAxisd rotation_vector(angle, axis);
  // 3. 转换为四元数
  Eigen::Quaterniond q3(rotation_vector);
  Eigen::Vector3d point(1, 0, 0);
  LOG_INFO("旋转前的点坐标: [{}, {}, {}]", point(0), point(1), point(2));
  Eigen::Vector3d rotated_point = q3 * point;
  LOG_INFO("旋转后的点坐标: [{}, {}, {}]", rotated_point(0), rotated_point(1),
           rotated_point(2));
}

void IsometryUsage() {
  // 1. 初始化：创建一个单位变换矩阵 (Identity)
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

  // 2. 旋转部分：可以使用旋转向量、四元数或旋转矩阵
  Eigen::AngleAxisd rv(M_PI_2, Eigen::Vector3d::UnitZ());
  T.rotate(rv);  // 相当于在 T 内部设置了旋转部分

  // 3. 平移部分：直接传入一个 Vector3d
  T.pretranslate(Eigen::Vector3d(1.0, 2.0, 3.0));
  // T.translate(Eigen::Vector3d(1.0, 2.0, 3.0));
  // 注意：pretranslate 是在左侧乘，translate 是在右侧乘，SLAM 中常用
  // pretranslate

  // 4. 访问提取
  Eigen::Matrix3d R = T.rotation();     // 提取旋转
  Eigen::Vector3d t = T.translation();  // 提取平移
  LOG_INFO("T的旋转部分 R = [{}, {}, {}]", R(0, 0), R(0, 1), R(0, 2));
  LOG_INFO("               [{}, {}, {}]", R(1, 0), R(1, 1), R(1, 2));
  LOG_INFO("               [{}, {}, {}]", R(2, 0), R(2, 1), R(2, 2));
  LOG_INFO("T的平移部分 t = [{}, {}, {}]", t(0), t(1), t(2));

  // 5. 坐标变换：直接乘向量，不需要手动转齐次坐标！
  Eigen::Vector3d p_a(1, 0, 0);
  LOG_INFO("旋转前的点坐标: [{}, {}, {}]", p_a(0), p_a(1), p_a(2));
  Eigen::Vector3d p_w = T * p_a;  // Eigen 底层自动帮你处理了 Rp + t
  LOG_INFO("旋转后的点坐标: [{}, {}, {}]", p_w(0), p_w(1), p_w(2));
}

void LuSolver(const Eigen::Matrix3d& A, const Eigen::Vector3d& b,
              const std::string& desc);

void LuDecompositionUsage() {
  // --- 情况 1: 满秩 (唯一解) ---
  // 三个平面相交于一点
  Eigen::Matrix3d A_unique;
  A_unique << 1, 2, 3, 0, 1, 4, 5, 6, 0;
  Eigen::Vector3d b_unique(14, 13, 22);  // 预期解 [8, -3, 4]
  LuSolver(A_unique, b_unique, "满秩唯一解");

  // --- 情况 2: 亏秩且相容 (无穷多解) ---
  // 第三行是第一行和第二行的和，三个平面交于一条线
  Eigen::Matrix3d A_inf;
  A_inf << 1, 1, 1, 0, 1, 2, 1, 2, 3;  // Row3 = Row1 + Row2
  Eigen::Vector3d b_inf(6, 5, 11);     // b 也满足这种关系，所以有解
  LuSolver(A_inf, b_inf, "亏秩无穷多解");

  // --- 情况 3: 亏秩且矛盾 (无解) ---
  // 矩阵部分和上面一样，但 b 不满足比例关系
  Eigen::Matrix3d A_none;
  A_none << 1, 1, 1, 0, 1, 2, 1, 2, 3;
  Eigen::Vector3d b_none(6, 5, 100);  // 显然 Row1+Row2 != 100，逻辑矛盾
  LuSolver(A_none, b_none, "亏秩无解(矛盾)");
}

/**
 * @brief 使用 PartialPivLU 尝试求解 3x3 方程组
 */
void LuSolver(const Eigen::Matrix3d& A, const Eigen::Vector3d& b,
              const std::string& desc) {
  LOG_INFO("--- 案例: {} ---", desc);

  // 1. 进行分解
  Eigen::PartialPivLU<Eigen::Matrix3d> lu(A);

  // 2. 检查奇异性 (LU 无法处理亏秩矩阵)
  // 对于 3x3 矩阵，行列式是一个非常直观的判别式
  double det = A.determinant();
  if (std::abs(det) < 1e-9) {
    LOG_WARN("矩阵 A 亏秩 (det={:.6f}),LU 无法保证唯一解", det);
  }

  // 3. 执行求解
  Eigen::Vector3d x = lu.solve(b);

  // 4. 通过残差进行严格报错判断
  double residual = (A * x - b).norm();

  if (residual > 1e-6) {
    LOG_ERROR("求解报错: 残差过大 ({:.6f})。该系统可能矛盾(无解)或严重亏秩。",
              residual);
  } else {
    // 如果 det 接近 0 但残差很小，说明是无穷多解情况下的一个特解
    if (std::abs(det) < 1e-9) {
      LOG_WARN("检测到无穷多解,LU 返回了一个特解 x: [{:.2f}, {:.2f}, {:.2f}]",
               x(0), x(1), x(2));
    } else {
      LOG_INFO("求解成功! 唯一解 x: [{:.2f}, {:.2f}, {:.2f}]", x(0), x(1),
               x(2));
    }
  }
}

void QRSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
              const std::string& case_name);

void QRDecompositionUsage() {
  // 情况 1: 长方形矩阵 5x3，无解 (超定矛盾)
  Eigen::MatrixXd A1(5, 3);
  A1 << 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1;
  Eigen::VectorXd b1(5);
  b1 << 1, 1, 1, 3, 100;  // 最后一个 100 制造了巨大的矛盾 (z既等于1又等于100)
  QRSolver(A1, b1, "长方形超定无解");

  // 情况 2: 方阵 3x3，亏秩且相容 (无穷多解)
  Eigen::Matrix3d A2;
  A2 << 1, 1, 1, 2, 2, 2, 3, 3, 3;  // 三行全线性相关
  Eigen::Vector3d b2(3, 6, 9);      // b 满足比例，自洽
  QRSolver(A2, b2, "3x3 亏秩无穷多解");

  // 定义一个 5x3 矩阵
  Eigen::MatrixXd A3(5, 3);
  A3 << 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1;

  // 情况 3: 长方形矩阵，但 b 是根据 A * [1, 2, 3] 严格计算出来的
  // 这代表了多个观测值之间没有任何冲突
  Eigen::Vector3d true_x(1, 2, 3);
  Eigen::VectorXd b_exact = A3 * true_x;

  QRSolver(A3, b_exact, "长方形超定精确解");
}

void QRSolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
              const std::string& case_name) {
  LOG_INFO("--- 案例: {} ---", case_name);

  // 1. 构造增广矩阵 [A | b]
  Eigen::MatrixXd augmented(A.rows(), A.cols() + 1);
  augmented << A, b;

  // 构建求解器
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(A);
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_augment(augmented);

  // 2. 计算秩
  int rank_A = qr.rank();
  int rank_Aug = qr_augment.rank();
  int n = A.cols();

  LOG_INFO("矩阵 A 的秩: {}, 增广阵 [A|b] 的秩: {}, 未知数 n: {}", rank_A,
           rank_Aug, n);

  // 3. 逻辑判定
  if (rank_A == rank_Aug && rank_A == n) {
    LOG_INFO("判定结果: rank(A) == rank(A|b) == n -> 系统有[唯一精确解]");
  } else if (rank_A < rank_Aug) {
    LOG_WARN("判定结果: rank(A) < rank(A|b) -> 系统[无解]");
  } else if (rank_A == rank_Aug && rank_A < n) {
    LOG_WARN("判定结果: rank(A) == rank(A|b) < n -> 系统有[无穷多解]");
  }

  // 4. 求解
  Eigen::VectorXd x = qr.solve(b);

  LOG_INFO("求解结果 x: [{:.2f}, {:.2f}, {:.2f}]", x(0), x(1), x(2));

  // 5. 残差验证
  double residual = (A * x - b).norm();
  LOG_INFO("残差范数: {:.10f}", residual);
  if (residual < 1e-10) {
    LOG_INFO("验证成功: 残差接近 0,这是一个完美的精确解!");
  }
}

void SVDsolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
               const std::string& case_name);

void SVDDecompositionUsage() {
  // 情况 1: 严重亏秩的 4x2 矩阵 (无穷多解)
  // 所有的行都是 [1, 1] 的倍数，实际上只有一个有效方程 x + y = 2
  Eigen::MatrixXd A_inf(4, 2);
  A_inf << 1, 1, 2, 2, 3, 3, 4, 4;
  Eigen::VectorXd b_inf(4);
  b_inf << 2, 4, 6, 8;  // 完全自洽
  SVDsolver(A_inf, b_inf, "4x2 严重亏秩-无穷多解");

  // 情况 2: 极度病态/矛盾的矩阵
  // 几乎平行的两条线，LU 和 QR 可能会在这里因为数值精度抖动
  Eigen::MatrixXd A_bad(2, 2);
  A_bad << 1.000, 1.000, 1.000, 1.001;
  Eigen::VectorXd b_bad(2);
  b_bad << 2.000, 2.001;
  SVDsolver(A_bad, b_bad, "2x2 极度病态矩阵");

  // 构造一个 4x2 的超定矩阵 (4个方程，2个未知数)
  Eigen::MatrixXd A(4, 2);
  A << 1, 1,  // x + y = 2
      1, -1,  // x - y = 0
      1, 0,   // x = 1.1 (与前两个矛盾)
      0, 1;   // y = 0.9 (与前两个矛盾)

  Eigen::VectorXd b(4);
  b << 2.0, 0.0, 1.1, 0.9;

  // 如果没有矛盾，解应该是 [1, 1]。现在的 b 向量制造了轻微的矛盾。
  SVDsolver(A, b, "4x2 超定矩阵");
}

/**
 * @brief 使用 SVD 分解处理最具挑战性的矩阵
 */
void SVDsolver(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
               const std::string& case_name) {
  LOG_INFO("--- 案例: {} ---", case_name);

  // 1. 构造增广矩阵并计算秩
  Eigen::MatrixXd augmented(A.rows(), A.cols() + 1);
  augmented << A, b;

  // SVD 本身就可以非常精确地测量秩
  // ComputeThinU | ComputeThinV 对长方形矩阵更高效
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);

  // 设置一个阈值来判断奇异值是否为 0
  svd.setThreshold(1e-9);
  int rank_A = svd.rank();

  // 对增广阵也做一次 SVD 判定秩
  int rank_Aug = augmented.fullPivHouseholderQr().rank();
  int n = A.cols();

  LOG_INFO("矩阵 A 的秩: {}, 增广阵 [A|b] 的秩: {}, 未知数 n: {}", rank_A,
           rank_Aug, n);

  // 2. 打印奇异值 (Singular Values)
  // 奇异值的大小直接反映了矩阵在各个维度上的“强度”
  LOG_INFO("奇异值分布: [{}]",
           fmt::join(svd.singularValues().data(),
                     svd.singularValues().data() + svd.singularValues().size(),
                     ", "));

  // 3. 求解
  Eigen::VectorXd x = svd.solve(b);

  if (x.size() >= 2) {
    LOG_INFO("SVD 求解结果 x: [{:.2f}, {:.2f}]", x(0), x(1));
  }

  // 4. 残差与解的范数
  double residual = (A * x - b).norm();
  double x_norm = x.norm();
  LOG_INFO("残差范数: {:.6f}, 解向量的范数 ||x||: {:.6f}", residual, x_norm);

  // 5. 逻辑解释
  if (rank_A < n && rank_A == rank_Aug) {
    LOG_INFO(
        "判定结果: 无穷多解。SVD 返回的是所有解中 ||x|| "
        "最小的那个（最小范数解）");
  } else if (rank_A < rank_Aug) {
    LOG_WARN("判定结果: 无解。SVD 返回的是最小二乘解");
  }
}

}  // namespace eigen_practice