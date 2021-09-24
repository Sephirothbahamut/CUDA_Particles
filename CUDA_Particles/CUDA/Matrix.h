#pragma once
#include <stdexcept>

#include <utils/containers/matrix.h>
#include <utils/compilation/debug.h>

#include "Base.h"

namespace utils::CUDA
	{
	template <typename T, utils::matrix_memory MEMORY>
	class matrix;

	/// <summary>
	/// Device side matrix abstraction.
	/// Can be constructed from pinned memory (TODO) or returned from a global memory matrix manager (see utils::CUDA::matrix)
	/// Note: does NOT have ownership of any resource. Ownership is managed exclusively by the host side.
	/// </summary>
	/// <typeparam name="T">The type stored in the vector.</typeparam>
	template <typename T, utils::matrix_memory MEMORY = utils::matrix_memory::width_first >
	class device_matrix
		{
		template <typename T, utils::matrix_memory MEMORY>
		friend class matrix;
		public:
			struct coords_t { size_t x{0}; size_t y{0}; };

			__device__ size_t width()  const noexcept { return _width; }
			__device__ size_t height() const noexcept { return _height; }
			__device__ size_t size()   const noexcept { return _width * _height; }

			__device__ T* data() { return _arr; }

			__device__ size_t get_index(size_t x, size_t y) const noexcept
				{
				if constexpr (MEMORY == utils::matrix_memory::width_first) { return x + (y * _width); }
				else { return y + (x * _height); }
				}
			__device__ size_t   get_x(size_t index) const noexcept { if constexpr (MEMORY == utils::matrix_memory::width_first) { return index % width(); } else { return index / height(); } } //TODO test
			__device__ size_t   get_y(size_t index) const noexcept { if constexpr (MEMORY == utils::matrix_memory::width_first) { return index / width(); } else { return index % height(); } } //TODO test
			__device__ coords_t get_coords(size_t index) const noexcept { return {get_x(index), get_y(index)}; }
			
			__device__ const T& operator[](size_t i)        const noexcept { return _arr[i]; }
			__device__       T& operator[](size_t i)              noexcept { return _arr[i]; }
			__device__ const T& operator[](coords_t coords) const noexcept { return _arr[get_index(coords.x, coords.y)]; }
			__device__       T& operator[](coords_t coords)       noexcept { return _arr[get_index(coords.x, coords.y)]; }

			__device__ const auto begin()   const noexcept { return _arr; }
			__device__       auto begin()         noexcept { return _arr; }
			__device__ const auto end()     const noexcept { return _arr + size(); }
			__device__       auto end()           noexcept { return _arr + size(); }

		private:
			__host__ device_matrix(size_t width, size_t height, T* arr) : _width(width), _height(height), _arr(arr)
				{}

			size_t _width;
			size_t _height;
			T* _arr;
		};

	/// <summary>
	/// Host side RAII manager for a matrix in device global memory.
	/// </summary>
	/// <typeparam name="T">The type stored in the vector.</typeparam>
	template <typename T, utils::matrix_memory MEMORY = utils::matrix_memory::width_first >
	class matrix
		{
		using device_matrix_t = device_matrix<T, MEMORY>;
		using Host_matrix_t   = utils::matrix_dyn<T, MEMORY>;
		public:
			// Allocates a vector in cuda memory
			matrix(size_t width, size_t height) : _width{width}, _height{height}
				{
				cuda_check(cudaMalloc((void**)&arr_ptr, width * height * sizeof(T)));

				device_matrix_t tmp{width, height, arr_ptr};
				}

			/// <summary>
			/// Allocates a matrix that only exists on the device's global memory and initializes it with the values of the given host-side matrix.
			/// </summary>
			/// <param name="vec">The host side vector to take data from.</param>
			matrix(const Host_matrix_t& mat) : matrix{mat.width(), mat.height()} { from(mat); }

			/// <summary>
			/// Retrieve a device_matrix to use in kernels which wraps the globally allocated memory.
			/// </summary>
			/// <returns></returns>
			device_matrix_t get_device_matrix() noexcept { return {width(), height(), arr_ptr}; }

			/// <summary>
			/// Copies data from host to device. Assumes both have the same size (checked in debug mode, unchecked in release)
			/// </summary>
			void from(const Host_matrix_t& mat) utils_if_debug(noexcept)
				{
#ifdef utils_is_debug
				if (mat.width() != width() || mat.height() != height()) { throw std::out_of_range{"Trying to copy matrix from CPU to GPU, but sizes don't match."}; }
#endif
				cuda_check(cudaMemcpy(arr_ptr, mat.data(), mat.size() * sizeof(T), cudaMemcpyHostToDevice));
				}

			/// <summary>
			/// Copies data from device to host. Assumes both have the same size (checked in debug mode, unchecked in release)
			/// </summary>
			void to(Host_matrix_t& mat) const utils_if_debug(noexcept)
				{
#ifdef utils_is_debug
				if (mat.width() != width() || mat.height() != height()) { throw std::out_of_range{"Trying to copy matrix from CPU to GPU, but sizes don't match."}; }
#endif
				cuda_check(cudaMemcpy(mat.data(), arr_ptr, mat.size() * sizeof(T), cudaMemcpyDeviceToHost));
				}

			size_t size()   const noexcept { return _width * _height; }
			size_t width()  const noexcept { return _width;  }
			size_t height() const noexcept { return _height; }

			~matrix() { if (arr_ptr) { cuda_check(cudaFree(arr_ptr)); } }

		private:
			T* arr_ptr{nullptr};

			const size_t _width {0};
			const size_t _height{0};
		};
	}