#define UTILS_IMPL
#include "PS_GPU_Nested_foreach.cuh"

#include "CUDA/Launcher.h"

#include "CUDA/Color_manip.h"

namespace Particle_system
	{
	extern float mass;
	
	__global__
	void _update_velocities(CUDA_Nested_foreach::device_forces_t velocities, CUDA_Nested_foreach::device_vertices_t vertices, float mass)
		{
		size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};
		if (index < vertices.size())
			{
			utils::CUDA::math::vec2f this_pos{to_utils_vec2(vertices[index].position)};

			utils::CUDA::math::vec2f velocity{velocities[index]};
			
			for (const auto& other_vertex : vertices)
				{
				utils::CUDA::math::vec2f other_pos{to_utils_vec2(other_vertex.position)};
				velocity += calc_force(this_pos, other_pos, mass, mass);
				}
			velocities[index] = velocity;
			}
		}

	__global__
	void _update_velocities_smem(CUDA_Nested_foreach::device_forces_t velocities, CUDA_Nested_foreach::device_vertices_t vertices, float mass)
		{
		size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};

		bool vertex_exists{index < vertices.size()};

		utils::CUDA::math::vec2f this_pos;
		if (vertex_exists) { this_pos = to_utils_vec2(vertices[index].position); }

		utils::CUDA::math::vec2f velocity;
		if (vertex_exists) { velocity = velocities[index]; }

		extern __shared__ sf::Vertex current_other_vertices_data[];
		utils::CUDA::shared_vector smem_vertices{current_other_vertices_data, vertices, blockDim.x};

		smem_vertices.load(threadIdx.x, [&] ()
			{
			for (const auto& other_vertex : smem_vertices)
				{
				utils::CUDA::math::vec2f other_pos{to_utils_vec2(other_vertex.position)};
				velocity += calc_force(this_pos, other_pos, mass, mass);
				}
			});

		if (vertex_exists) { velocities[index] = velocity; }
		}

	__device__
	void _set_vertex_color(sf::Vertex& vertex, const utils::CUDA::math::vec2f& velocity)
		{
		float hue{velocity.magnitude2()};
		clamp(hue, 0.f, 240.f);
		hue = 240.f - hue;

		auto color = utils::CUDA::mysf::HSVtoRGB(hue, 1.f, 1.f);
		vertex.color.r = color.r;
		vertex.color.g = color.g;
		vertex.color.b = color.b;
		}

	__global__
	void _apply_velocities(CUDA_Nested_foreach::device_forces_t velocities, CUDA_Nested_foreach::device_vertices_t vertices)
		{
		size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};

		if (index < vertices.size())
			{
			      auto& vertex   {vertices   [index]};
			const auto& velocity {velocities [index]};
			vertex.position.x += velocity.x;
			vertex.position.y += velocity.y;

			_set_vertex_color(vertex, velocity);
			}
		}

	__global__
	void _zero_velocities(CUDA_Nested_foreach::device_forces_t velocities)
		{
		size_t index{(blockIdx.x * blockDim.x) + threadIdx.x};

		if (index < velocities.size()) { velocities[index] = {0.f, 0.f}; }
		}

	void CUDA_Nested_foreach::zero_velocities() noexcept
		{
		size_t threads_n{velocities.size() < 1024 ? velocities.size() : 1024};

		size_t blocks_n {velocities.size() / 1024 + ((velocities.size() % 1024) ? 1 : 0)};

		utils::CUDA::Launcher<_zero_velocities>{blocks_n, threads_n} (velocities.get_device_vector());
		cudaDeviceSynchronize();
		}

	void CUDA_Nested_foreach::update_impl() noexcept { update_inner(); }

	void CUDA_Nested_foreach::update_inner() noexcept
		{
		auto device_velocities = velocities.get_device_vector();
		utils::CUDA::device_vector device_vertices{vertices};

		size_t threads_n{vertices.size() < particles_per_block ? vertices.size() : particles_per_block};

		size_t blocks_n{vertices.size() / particles_per_block + ((vertices.size() % particles_per_block) ? 1 : 0)};
		size_t smem_size{threads_n * sizeof(sf::Vertex)};

		utils::CUDA::Launcher<_update_velocities>  {blocks_n, threads_n, smem_size} (device_velocities, device_vertices, Particle_system::mass);
		utils::CUDA::Launcher<_apply_velocities>{blocks_n, threads_n, smem_size} (device_velocities, device_vertices);
		cudaDeviceSynchronize();
		}

	void CUDA_Nested_foreach_smem::update_inner() noexcept
		{
		auto device_velocities = velocities.get_device_vector();
		utils::CUDA::device_vector device_vertices{vertices};

		size_t threads_n{vertices.size() < particles_per_block ? vertices.size() : particles_per_block};

		size_t blocks_n {vertices.size() / particles_per_block + ((vertices.size() % particles_per_block) ? 1 : 0)};
		size_t smem_size{threads_n * sizeof(sf::Vertex)};

		utils::CUDA::Launcher<_update_velocities_smem>{blocks_n, threads_n, smem_size} (device_velocities, device_vertices, Particle_system::mass);
		utils::CUDA::Launcher<_apply_velocities>   {blocks_n, threads_n, smem_size} (device_velocities, device_vertices);
		cudaDeviceSynchronize();
		}

	/// <summary>
	/// Attempts to find the thread count which leads to the best timings.
	/// Note: unreliable, the optimal value on the first iteration might be subjected to external factors and ultimately not be optimal overall.
	/// </summary>
	void CUDA_Nested_foreach::update_auto_size() noexcept
		{
		static std::chrono::nanoseconds best_time{-1};
		static size_t best_particles_per_block   {32};
		static bool found_best                   {false};
		particles_per_block = 32;

		std::chrono::steady_clock::time_point tp;
		if (!found_best) { tp = std::chrono::steady_clock::now(); }

		update_inner();

		if (!found_best)
			{
			auto tt = std::chrono::steady_clock::now() - tp;

			if (best_time < std::chrono::nanoseconds{0} || tt < best_time)
				{
				best_time = tt;
				best_particles_per_block = particles_per_block;
				}

			if (particles_per_block == 512)
				{
				found_best = true;
				particles_per_block = best_particles_per_block;
				std::cout << "Best time with particles count = " << particles_per_block << std::endl;
				reset_average_time();
				}
			else { particles_per_block += 32; }
			}
		}
	}