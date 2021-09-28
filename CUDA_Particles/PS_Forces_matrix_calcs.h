#pragma once

#include <SFML/Graphics.hpp>
#include "CUDA/vec2.h"
#include "CUDA/Base.h"

namespace Particle_system { extern float mass; }

__host__ __device__
inline void clamp_vertex(sf::Vertex& vertex, float width, float height)
	{
	// Bounce
	if (vertex.position.x >= width) { vertex.position.x -= (vertex.position.x - width); }
	else if (vertex.position.x < 0.f) { vertex.position.x = -vertex.position.x; }
	if (vertex.position.y >= height) { vertex.position.y -= (vertex.position.y - height); }
	else if (vertex.position.y < 0.f) { vertex.position.y = -vertex.position.y; }

	// Clamp if after bounces the particle is still outside
	if (vertex.position.x >= width) { vertex.position.x = width - 1; }
	else if (vertex.position.x < 0.f) { vertex.position.x = 0; }
	if (vertex.position.y >= height) { vertex.position.y = height - 1; }
	else if (vertex.position.y < 0.f) { vertex.position.y = 0; }
	}

__host__ __device__
inline utils::CUDA::math::vec2f calc_force(const utils::CUDA::math::vec2f& particle_position, const utils::CUDA::math::vec2f& target_position, float mass_a, float mass_b) noexcept
	{
	namespace utm = utils::CUDA::math;
	utm::vec2f this_to_current_vector{target_position - particle_position};
	float dist = this_to_current_vector.magnitude();
	float strength = dist < std::numeric_limits<float>::epsilon() ? 0.f : /*0.000000000066742f * */ mass_a * mass_b / dist;
	return this_to_current_vector.normal() * strength;
	}

__host__ __device__
inline utils::CUDA::math::vec2f calc_force_at(const sf::Vertex& this_vertex, size_t x, size_t y, float mass)
	{
	namespace utm = utils::CUDA::math;

	utm::vec2f other{static_cast<float>(x), static_cast<float>(y)};
	utm::vec2f self{std::floorf(this_vertex.position.x), std::floorf(this_vertex.position.y)};

	return calc_force(self, other, mass, mass);
	}