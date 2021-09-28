#pragma once

#include <algorithm>
#include <execution>

#include "Vector_extension.h"

#include "CUDA/Base.h"
#include "CUDA/Matrix.h"
#include "CUDA/Vector.h"
#include "PS_Base.h"


namespace Particle_system
	{
	class CUDA_Nested_foreach : public base<utils::CUDA::pinned_allocator<sf::Vertex>>
		{
		public:
			using forces_t          = utils::CUDA::vector       <utils::CUDA::math::vec2f>;
			using device_forces_t   = utils::CUDA::device_vector<utils::CUDA::math::vec2f>;
			using device_vertices_t = utils::CUDA::device_vector<sf::Vertex>;

			CUDA_Nested_foreach(const init& init) noexcept : base{init, sf::Color::Green},
				velocities{init.amount}, particles_per_block{256}
				{ zero_velocities(); }

				virtual void update_impl() noexcept override final;

				void update_auto_size() noexcept;
				virtual void update_inner() noexcept;

		protected:
			forces_t velocities;
			size_t particles_per_block;

			virtual void zero_velocities() noexcept;
			const char* get_name() const noexcept override       { return  "GPU Nested foreach     "; }
		};

	class CUDA_Nested_foreach_smem : public CUDA_Nested_foreach
		{
		public:
			using CUDA_Nested_foreach::CUDA_Nested_foreach;
			virtual void update_inner() noexcept final override;
			const char* get_name() const noexcept override final { return  "GPU Nested foreach smem"; }
		};
	}