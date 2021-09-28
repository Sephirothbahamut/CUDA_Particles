#pragma once

#include <algorithm>
#include <execution>
#include <functional>
#include <ranges>

#include <utils/containers/matrix.h>

#include "PS_Base.h"

#include "Vector_extension.h"
#include "Color_manip.h"

#include "Index_range.h"

namespace Particle_system
	{

	template <CPU_Type type = CPU_Type::Sequential>
	class CPU_Forces_matrix : public base<std::allocator<sf::Vertex>>
		{
		public:
			CPU_Forces_matrix(const init& init) noexcept : base{init, type == CPU_Type::Sequential ? sf::Color::Yellow : sf::Color{255, 127, 0}},
				velocities{init.width, init.height}
				{}

			virtual void update_impl() noexcept override final
				{
				zero_velocities();
				set_velocities();
				apply_velocities();
				}

		protected:
			const char* get_name() const noexcept override final
				{
				if constexpr (type == CPU_Type::Sequential) { return  "CPU Forces matrix  Seq "; }
				if constexpr (type == CPU_Type::Parallel)   { return  "CPU Forces matrix  Par "; }
				}

		private:
			utils::matrix_dyn<utils::CUDA::math::vec2f> velocities;

			void zero_velocities() noexcept { for (auto& velocity : velocities) { velocity.x = 0; velocity.y = 0; } }
			void set_velocities() noexcept
				{
				namespace utm = utils::CUDA::math;

				if constexpr (type == CPU_Type::Sequential)
					{
					for (const auto& vertex : vertices)
						{
						size_t discrete_x = static_cast<size_t>(vertex.position.x);
						size_t discrete_y = static_cast<size_t>(vertex.position.y);

						for (size_t y = 0; y < velocities.height(); y++)
							{
							for (size_t x = 0; x < velocities.width(); x++)
								{
								auto& current_vector = velocities[{x, y}];
								current_vector += calc_force_at(vertex, x, y, Particle_system::mass);
								}
							}
						}
					}
				if constexpr (type == CPU_Type::Parallel)
					{
					utils::Index_range indices{velocities.size()};

					std::for_each(std::execution::par, indices.begin(), indices.end(), [&velocities = this->velocities, &vertices = this->vertices] (size_t index)
						{
						utils::CUDA::math::vec2f velocity{0, 0};
						size_t x{velocities.get_x(index)};
						size_t y{velocities.get_y(index)};

						for (const auto& vertex : vertices) { velocity += calc_force_at(vertex, x, y, Particle_system::mass); }
						velocities[index] = velocity;
						});
					}
				}
			void apply_velocities() noexcept
				{
				namespace utm = utils::CUDA::math;

				if constexpr (type == CPU_Type::Sequential)
					{
					for (auto& vertex : vertices) { apply_force(vertex); }
					}
				if constexpr (type == CPU_Type::Parallel)
					{
					std::for_each(std::execution::par, vertices.begin(), vertices.end(), [&a = *this](sf::Vertex& vertex) { a.apply_force(vertex); });
					}
				}

			void apply_force(sf::Vertex& vertex) noexcept
				{
				namespace utm = utils::CUDA::math;
				size_t discrete_x = static_cast<size_t>(vertex.position.x);
				size_t discrete_y = static_cast<size_t>(vertex.position.y);

				auto& velocity = velocities[{discrete_x, discrete_y}];
				vertex.position += to_sf_vec2(velocity);

				clamp_vertex(vertex, width, height);
				}
		};
	}