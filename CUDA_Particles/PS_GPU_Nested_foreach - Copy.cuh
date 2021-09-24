#pragma once

#include <algorithm>
#include <execution>

#include "Vector_extension.h"

#include "CUDA/Matrix.h"
#include "CUDA/Vector.h"
#include "PS_Base.h"


namespace Particle_system
	{
	class CUDA_Nested_foreach: public base<utils::CUDA::pinned_allocator<sf::Vertex>>
		{
		public:
			using forces_t = utils::CUDA::vector<utils::math::vec2f>;
			using device_forces_t = utils::CUDA::device_vector<utils::math::vec2f>;
			using device_vertices_t = utils::CUDA::device_vector<sf::Vertex>;

			CUDA_Nested_foreach(const init& init) noexcept : base{init, sf::Color::Green},
				forces{init.amount}
				{}

			virtual void update() noexcept override final;

		protected:
			forces_t forces;

			const char* get_name() const noexcept override final { return  "GPU Nested foreach Par"; }
		};
	}