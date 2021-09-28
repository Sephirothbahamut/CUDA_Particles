#pragma once

#include <algorithm>
#include <execution>

#include "Vector_extension.h"

#include "CUDA/Matrix.h"
#include "CUDA/Vector.h"
#include "PS_Base.h"


namespace Particle_system
	{
	class CUDA_Forces_matrix : public base<utils::CUDA::pinned_allocator<sf::Vertex>>
		{
		public:
			using forces_t          = utils::CUDA::matrix       <utils::CUDA::math::vec2f>;
			using device_forces_t   = utils::CUDA::device_matrix<utils::CUDA::math::vec2f>;
			using device_vertices_t = utils::CUDA::device_vector<sf::Vertex>;

			CUDA_Forces_matrix(const init& init) noexcept : base{init, sf::Color::Green},
				velocities{init.width, init.height}
				{}

		protected:
			forces_t velocities;

			const char* get_name() const noexcept override final { return "GPU Forces matrix      "; }

			virtual void update_impl() noexcept override final;
		};
	}