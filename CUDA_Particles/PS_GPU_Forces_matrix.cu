#include "PS_GPU_Forces_matrix.cuh"

#include "CUDA/Launcher.h"

namespace Particle_system { extern float mass; }

namespace Particle_system
	{
	__global__
	void _apply_velocities(const CUDA_Forces_matrix::device_forces_t velocities, CUDA_Forces_matrix::device_vertices_t vertices)
		{
		size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};
		if (index < vertices.size())
			{
			auto vertex{vertices[index]};

			size_t discrete_x{static_cast<size_t>(vertex.position.x)};
			size_t discrete_y{static_cast<size_t>(vertex.position.y)};

			utils::CUDA::math::vec2f velocity{velocities[velocities.get_index(discrete_x, discrete_y)]};

			vertex.position.x -= velocity.x;
			vertex.position.y -= velocity.y;

			clamp_vertex(vertex, velocities.width(), velocities.height());

			vertices[index] = vertex;
			}
		}

	__global__
	void _set_velocities(CUDA_Forces_matrix::device_forces_t velocities, const CUDA_Forces_matrix::device_vertices_t vertices, float mass)
		{
		size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};

		if (index < velocities.size())
			{
			//Copy value
			utils::CUDA::math::vec2f velocity{0.f, 0.f};
			auto coords = velocities.get_coords(index);

			for (const auto& vertex : vertices) { velocity += calc_force_at(vertex, coords.x, coords.y, mass); }
			velocities[index] = velocity;
			}
		}

	void CUDA_Forces_matrix::update_impl() noexcept
		{
		size_t threads{velocities.size() < 1024 ? velocities.size() : 1024};
		size_t blocks {velocities.size() < 1024 ? 1 : velocities.size() / 1024};

		utils::CUDA::Launcher<_set_velocities>  {blocks, threads}(velocities.get_device_matrix(), utils::CUDA::device_vector{vertices}, mass);
		utils::CUDA::Launcher<_apply_velocities>{blocks, threads}(velocities.get_device_matrix(), utils::CUDA::device_vector{vertices});
		cudaDeviceSynchronize();
		}
	}