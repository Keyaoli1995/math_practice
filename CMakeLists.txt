cmake_minimum_required(VERSION 3.10)
project(math_practice)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 查找依赖
find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(Ceres REQUIRED)

# 2. 定义头文件目录
# 让本项目和依赖本项目的其他包都能找到头文件
include_directories(include)

# 3. 编译库 (Library)
add_library(math_practice_lib 
src/math_practice/eigen_practice.cpp
src/math_practice/nonlinear_optimization.cpp
)
target_link_libraries(math_practice_lib PUBLIC Eigen3::Eigen spdlog::spdlog Ceres::ceres)

# 4. 编译可执行文件 (Executable)
add_executable(eigen_practice_node src/main_eigen.cpp)
target_link_libraries(eigen_practice_node math_practice_lib)

add_executable(nonlinear_optimization_node src/main_nonlinear_optimization.cpp)
target_link_libraries(nonlinear_optimization_node math_practice_lib)
# --- 5. Install 逻辑开始 ---

# 安装可执行文件到 lib/math_practice 目录下
install(TARGETS
  eigen_practice_node
  nonlinear_optimization_node
  DESTINATION lib/${PROJECT_NAME}
)

# 安装静态库（可选，如果你以后想给别的包用）
install(TARGETS
  math_practice_lib
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# 安装头文件目录到 install 空间
install(DIRECTORY include/
  DESTINATION include/
)

# --- Install 逻辑结束 ---

# 6. ament 打包 (生成环境配置)
ament_package()