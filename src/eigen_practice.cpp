#include "math_practice/eigen_practice.hpp"

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

void AngleAxis() {
  Eigen::AngleAxis<double> rotation_vector(M_PI / 4.0, Eigen::Vector3d::UnitZ());
}

}  // namespace eigen_practice