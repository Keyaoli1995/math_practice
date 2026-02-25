#include <memory>

#include "math_practice/nonlinear_optimization.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"  // 控制台（带颜色）
#include "spdlog/spdlog.h"
int main(int argc, char const* argv[]) {
  // 1. 创建控制台 Sink
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  console_sink->set_level(spdlog::level::info);  // 控制台只打印 info 及以上

  // 3. 将两个 Sink 组合进一个 Logger
  // 参数："global_logger" 是名字，{...} 是 Sink 列表
  auto logger = std::make_shared<spdlog::logger>(
      "global_logger", spdlog::sinks_init_list{console_sink});

  // 4. 设置为默认 Logger
  spdlog::set_default_logger(logger);

  // 5. 设置全局级别（Logger 的级别会过滤掉 Sink 的级别）
  spdlog::set_level(spdlog::level::debug);
  spdlog::info("Initialize logger system success.");

  nonlinear_optimization::GradientDescentUsage();
  nonlinear_optimization::NewTonUsage();
  nonlinear_optimization::GaussNewtonUsage();
  nonlinear_optimization::LevenbergMarquardtUsage();
  nonlinear_optimization::CeresSolverUsage();
  nonlinear_optimization::CeresPoseOptimizationUsage();

  // 程序结束前（可选但推荐）
  spdlog::shutdown();
  return 0;
}
