#pragma once

#include "CUDA/Base.h"

#include "CUDA/Vec2.h"
#include <SFML/Graphics.hpp>

template <typename T>
__device__ __host__ inline utils::math::vec2<T> to_utils_vec2(const sf::Vector2<T> sf_vec2) { return {sf_vec2.x, sf_vec2.y}; }

template <typename T>
__device__ __host__ inline sf::Vector2<T> to_sf_vec2(const utils::math::vec2<T> utils_vec2) { return {utils_vec2.x, utils_vec2.y}; }
