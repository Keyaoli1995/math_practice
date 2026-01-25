// spdlog_enhance.hpp
#ifndef MATH_PRACTICE_SPDLOG_ENHANCE_HPP  // 添加头文件保护
#define MATH_PRACTICE_SPDLOG_ENHANCE_HPP

#include "spdlog/spdlog.h"
#include <string>

// 方法1：简单的函数名宏
#define LOG_INFO(...) \
    spdlog::info("[{}] {}", __FUNCTION__, fmt::format(__VA_ARGS__))

#define LOG_WARN(...) \
    spdlog::warn("[{}] {}", __FUNCTION__, fmt::format(__VA_ARGS__))

#define LOG_ERROR(...) \
    spdlog::error("[{}] {}", __FUNCTION__, fmt::format(__VA_ARGS__))

#define LOG_DEBUG(...) \
    spdlog::debug("[{}] {}", __FUNCTION__, fmt::format(__VA_ARGS__))

#endif  // MATH_PRACTICE_SPDLOG_ENHANCE_HPP
