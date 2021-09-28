#pragma once

#include <algorithm>
#include <execution>
#include <functional>

#include <utils/containers/matrix.h>

#include "PS_Base.h"

#include "Vector_extension.h"
#include "Color_manip.h"
#include "Index_range.h"


namespace Particle_system
	{
	template <CPU_Type type>
	class CPU_Nested_foreach : public base<std::allocator<sf::Vertex>>
		{
		public:
			CPU_Nested_foreach(const init& init) noexcept : base{init, type == CPU_Type::Sequential ? sf::Color::Cyan : sf::Color::Magenta},
				velocities(init.amount)
				{ zero_velocities(); }

		protected:
			const char* get_name() const noexcept override final
				{
				if constexpr (type == CPU_Type::Sequential) { return  "CPU Nested foreach Seq "; }
				if constexpr (type == CPU_Type::Parallel)   { return  "CPU Nested foreach Par "; }
				}

		private:
			std::vector<utils::CUDA::math::vec2f> velocities;

			void zero_velocities() noexcept
				{
				if constexpr (type == CPU_Type::Sequential) { for (auto& velocity : velocities) { velocity = {0.f, 0.f}; } }
				if constexpr (type == CPU_Type::Parallel)
					{
					utils::Index_range indices{velocities.size()};
					std::for_each(std::execution::par, indices.begin(), indices.end(), [&] (size_t index) { velocities[index] = {0.f, 0.f}; });
					}
				}

			virtual void update_impl() noexcept override final
				{
				if constexpr (type == CPU_Type::Sequential)
					{
					for (size_t i = 0; i < vertices.size(); i++) { update_velocity(i); }
					for (size_t i = 0; i < vertices.size(); i++) { apply_velocity (i); }
					}
				if constexpr (type == CPU_Type::Parallel)
					{
					utils::Index_range indices{velocities.size()};

					std::for_each(std::execution::par, indices.begin(), indices.end(), [&] (size_t index) { update_velocity(index); });
					std::for_each(std::execution::par, indices.begin(), indices.end(), [&] (size_t index) { apply_velocity(index); });
					}
				}

			void update_velocity(size_t index) noexcept
				{
				utils::CUDA::math::vec2f velocity = velocities[index];
				utils::CUDA::math::vec2f this_thread_vertex_position{to_utils_vec2(vertices[index].position)};

				for (const auto& other_vertex : vertices) 
					{
					velocity += calc_force(this_thread_vertex_position, to_utils_vec2(other_vertex.position), mass, mass);
					}
				velocities[index] = velocity;
				};

			void apply_velocity(size_t index) noexcept
				{
				auto& vertex  {vertices  [index]};
				auto& velocity{velocities[index]};

				vertex.position += to_sf_vec2(velocity);
				set_vertex_color(vertex, velocity);

				vertices[index] = vertex;
				};

			void set_vertex_color(sf::Vertex& vertex, const utils::CUDA::math::vec2f& velocity) const noexcept
				{
				float hue = velocity.magnitude2();

				hue = std::clamp(hue, 0.f, 240.f);
				hue = 240.f - hue;

				vertex.color = mysf::HSVtoRGB(hue, 1.f, 1.f);
				}
		};
	}