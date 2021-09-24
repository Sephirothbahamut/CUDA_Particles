#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cstdlib>

namespace utils::CUDA
	{
	/// <summary>
	/// Prints eventual CUDA errors before terminating the program
	/// </summary>
	inline void cuda_check(cudaError_t err)
		{
		if (err != cudaSuccess)
			{
			fprintf(stderr, "Cuda error: %s\n", cudaGetErrorString(err));
			exit(0);
			}
		}
	}

// Makes sure intellisense recognizes __syncthreads as a defined function
#ifdef __INTELLISENSE__
void __syncthreads() {};
#endif