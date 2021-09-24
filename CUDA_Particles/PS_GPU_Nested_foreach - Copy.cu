#include "PS_GPU_Nested_foreach.cuh"


namespace Particle_system
	{
	__global__
	void _set_forces(CUDA_Nested_foreach::device_forces_t& forces, CUDA_Nested_foreach::device_vertices_t vertices, size_t block_size)
		{
		size_t index{(blockIdx.x * block_size) + threadIdx.x};

		if (index < vertices.size())
			{
			utils::math::vec2f this_pos{to_utils_vec2(vertices[index].position)};
			utils::math::vec2f force{0.f, 0.f};

#pragma unroll 128
			for (const auto& other_vertex : vertices)
				{
				utils::math::vec2f other_pos{to_utils_vec2(other_vertex.position)};
				force -= calc_force(this_pos, other_pos);
				}

			forces[index] = force;
			}
		}
	__global__
	void _apply_forces(CUDA_Nested_foreach::device_forces_t& forces, CUDA_Nested_foreach::device_vertices_t vertices, size_t block_size)
		{
		size_t index{(blockIdx.x * block_size) + threadIdx.x};

		if (index < vertices.size())
			{
			auto& vertex{vertices[index]};
			const auto& force{forces[index]};
			vertex.position.x += force.x;
			vertex.position.y += force.y;
			}
		}

	void CUDA_Nested_foreach::update() noexcept
		{
		size_t threads_n{vertices.size() < 1024 ? vertices.size() : 1024};
		size_t blocks_n {vertices.size() < 1024 ? 1 : vertices.size() / 1024};

		auto& device_forces   = forces.get_device_ref();
		utils::CUDA::device_vector device_vertices{vertices};

		_set_forces  <<<blocks_n, threads_n>>>(device_forces, device_vertices, threads_n);
		_apply_forces<<<blocks_n, threads_n>>>(forces.get_device_ref(), utils::CUDA::device_vector{vertices}, threads_n);
		cudaDeviceSynchronize();
		}
	}