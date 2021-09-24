#pragma once
#include <ostream>
#include <cmath>
#include <numbers>

#include "../CUDA/Base.h"

namespace utils::math
	{
	template <typename T> class vec2;
	}

namespace utils::math
	{
	//fast typenames
	using vec2i   = vec2<int>;
	using vec2i8  = vec2<int8_t>;
	using vec2i16 = vec2<int16_t>;
	using vec2i32 = vec2<int32_t>;
	using vec2i64 = vec2<int64_t>;

	using vec2u   = vec2<unsigned>;
	using vec2u8  = vec2<uint8_t>;
	using vec2u16 = vec2<uint16_t>;
	using vec2u32 = vec2<uint32_t>;
	using vec2u46 = vec2<uint64_t>;

	using vec2s = vec2<size_t>;
	using vec2f = vec2<float>;
	using vec2d = vec2<double>;

	template <typename T>
	class vec2
		{
		public:
			__host__ __device__ vec2<T>()         noexcept = default;
			__host__ __device__ vec2<T>(T x, T y) noexcept : x(x), y(y) {};
			__host__ __device__ vec2<T>(T xy)     noexcept : x(xy), y(xy) {}
			__host__ __device__ static vec2<T> rr()    noexcept { return {T{ 1}, T{ 0}}; }
			__host__ __device__ static vec2<T> ll()    noexcept { return {T{-1}, T{ 0}}; }
			__host__ __device__ static vec2<T> up()    noexcept { return {T{ 0}, T{-1}}; }
			__host__ __device__ static vec2<T> dw()    noexcept { return {T{ 0}, T{ 1}}; }
			__host__ __device__ static vec2<T> right() noexcept { return rr(); }
			__host__ __device__ static vec2<T> left()  noexcept { return ll(); }
			__host__ __device__ static vec2<T> down()  noexcept { return dw(); }
			__host__ __device__ static vec2<T> zero()  noexcept { return {}; }

			T x = 0, y = 0;

			__host__ __device__ T magnitude2()    const noexcept { return x * x + y * y; }
			__host__ __device__ T magnitude()     const noexcept { return std::sqrt(magnitude2()); }
			__host__ __device__ vec2<T>  normal() const noexcept { return magnitude() ? *this / magnitude() : *this; }
			__host__ __device__ vec2<T>& normalize()    noexcept { return *this = normal(); }

			__host__ __device__ float to_angle() const noexcept
				{
				return (std::atan2f(x, y) * 180.f / static_cast<float>(std::acos(-1)/*numbers::pi*/)) + 180.f;
				}

			// OPERATORS
			__host__ __device__ vec2<T>  operator-() const noexcept { return {-x, -y}; }

			// VEC & SCALAR OPERATORS
			__host__ __device__ vec2<T>  operator++() noexcept { return *this += T(1); }
			__host__ __device__ vec2<T>  operator--() noexcept { return *this -= T(1); }
			
			__host__ __device__ vec2<T>  operator+ (const T n) const noexcept { return {normal() * (magnitude() + n)}; }
			__host__ __device__ vec2<T>  operator- (const T n) const noexcept { return {normal() * (magnitude() - n)}; }
			__host__ __device__ vec2<T>  operator* (const T n) const noexcept { return {x * n, y * n}; }
			__host__ __device__ vec2<T>  operator/ (const T n) const noexcept { return {x / n, y / n}; }
			
			__host__ __device__ vec2<T>& operator+=(const T n)       noexcept { return *this = *this + n; }
			__host__ __device__ vec2<T>& operator-=(const T n)       noexcept { return *this = *this - n; }
			__host__ __device__ vec2<T>& operator*=(const T n)       noexcept { return *this = *this * n; }
			__host__ __device__ vec2<T>& operator/=(const T n)       noexcept { return *this = *this / n; }
			__host__ __device__ vec2<T>& operator= (const T n)       noexcept { return normalize() *= n; }

			// VEC & VEC OPERATORS
			__host__ __device__ vec2<T>  operator+ (const vec2<T>& oth) const noexcept { return {x + oth.x, y + oth.y}; }
			__host__ __device__ vec2<T>  operator- (const vec2<T>& oth) const noexcept { return {x - oth.x, y - oth.y}; }
			__host__ __device__ vec2<T>  operator* (const vec2<T>& oth) const noexcept { return {x * oth.x, y * oth.y}; }
			__host__ __device__ vec2<T>  operator/ (const vec2<T>& oth) const noexcept { return {x / oth.x, y / oth.y}; }
			
			__host__ __device__ vec2<T>& operator+=(const vec2<T>& oth)       noexcept { return *this = *this + oth; }
			__host__ __device__ vec2<T>& operator-=(const vec2<T>& oth)       noexcept { return *this = *this - oth; }
			__host__ __device__ vec2<T>& operator*=(const vec2<T>& oth)       noexcept { return *this = *this * oth; }
			__host__ __device__ vec2<T>& operator/=(const vec2<T>& oth)       noexcept { return *this = *this / oth; }
			
			__host__ __device__ bool     operator==(const vec2<T>& oth) const noexcept { return x == oth.x && y == oth.y; }
			__host__ __device__ bool     operator!=(const vec2<T>& oth) const noexcept { return !(*this == oth); }
			
			__host__ __device__ static T dot     (const vec2<T>& a, const vec2<T>& b) noexcept { return a.x * b.x + a.y * b.y; }
			__host__ __device__ static T distance(const vec2<T>& a, const vec2<T>& b) noexcept
				{
				T dx = a.x - b.x;
				T dy = a.y - b.y;
				return std::sqrt(dx * dx + dy * dy);
				}
		};
	}
