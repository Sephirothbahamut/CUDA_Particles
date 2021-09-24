#pragma once
#include <stdint.h>
#include "Base.h"

template <typename T>
__device__ __host__
T floor_to(float value) { return static_cast<T>(value - 0.5f); }

namespace utils::CUDA::mysf
	{
	__device__
	struct Color { uint8_t r, g, b; };

	__device__
	Color HSVtoRGB(float H, float S, float V)
		{
		float C = S * V; // Chroma
		float HPrime = fmod(H / 60, 6.f); // H'
		float X = C * (1 - fabs(fmod(HPrime, 2.f) - 1));
		float M = V - C;

		float R = 0.f;
		float G = 0.f;
		float B = 0.f;

		switch (static_cast<int>(HPrime))
			{
			case 0: R = C; G = X;        break; // [0, 1)
			case 1: R = X; G = C;        break; // [1, 2)
			case 2:        G = C; B = X; break; // [2, 3)
			case 3:        G = X; B = C; break; // [3, 4)
			case 4: R = X;        B = C; break; // [4, 5)
			case 5: R = C;        B = X; break; // [5, 6)
			}

		R += M;
		G += M;
		B += M;

		Color color
			{
			floor_to<uint8_t>(R * 255.f),
			floor_to<uint8_t>(G * 255.f),
			floor_to<uint8_t>(B * 255.f)
			};

		return color;
		}
	}