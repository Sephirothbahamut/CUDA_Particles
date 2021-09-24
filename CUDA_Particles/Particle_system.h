#pragma once

#include "PS_Base.h"
#include "PS_CPU_Nested_foreach.h"
#include "PS_CPU_Forces_matrix.h"
#include "PS_GPU_Nested_foreach.cuh"
#include "PS_GPU_Forces_matrix.cuh"

namespace Particle_system
	{
	inline float mass{0.05};

	using CPU_Forces_matrix_Seq  = CPU_Forces_matrix <CPU_Type::Sequential>;
	using CPU_Forces_matrix_Par  = CPU_Forces_matrix <CPU_Type::Parallel>;
	using CPU_Nested_foreach_Seq = CPU_Nested_foreach<CPU_Type::Sequential>;
	using CPU_Nested_foreach_Par = CPU_Nested_foreach<CPU_Type::Parallel>;
	}