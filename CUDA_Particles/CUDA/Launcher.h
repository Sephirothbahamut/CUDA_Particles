#pragma once
#include "Base.h"

namespace utils::CUDA
	{
	/// <summary>
	/// Wrapper for CUDA's kernel calls. Allows intellisense to report eventual errors in the kernel call parameters before compile time.
	/// </summary>
	template <auto device_function>
	struct Launcher
		{
		Launcher() = default;

		/// <summary>
		/// Prepare to launch the templated
		/// </summary>
		/// <param name="blocks_n">Amount of blocks</param>
		/// <param name="threads_n">Amount of threads</param>
		Launcher(size_t blocks_n, size_t threads_n)
			: Launcher{blocks_n, threads_n, 0}
			{}

		/// <summary>
		/// Prepare to launch the templated
		/// </summary>
		/// <param name="blocks_n">Amount of blocks</param>
		/// <param name="threads_n">Amount of threads</param>
		/// <param name="smem_size">Shared memory size</param>
		Launcher(size_t blocks_n, size_t threads_n, size_t smem_size)
			: blocks_n{blocks_n}, threads_n{threads_n}, smem_size{smem_size}
			{}

		size_t blocks_n{1};
		size_t threads_n{1024};
		size_t smem_size{0};

		/// <summary>
		/// Launch the kernel with the given parameters.
		/// </summary>
		template <typename ...Args>
		void operator()(Args&&... args) const noexcept
			{
			device_function <<<blocks_n, threads_n, smem_size>>> (std::forward<Args>(args)...);
			}
		};
	}