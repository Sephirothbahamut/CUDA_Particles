#pragma once

#include <vector>
#include <random>
#include <algorithm>

#include "PS_Forces_matrix_calcs.h"

#include <SFML/Graphics.hpp>

#include "Elements.h"


template<class T>
__host__ __device__ inline constexpr const T& clamp(T& v, const T& lo, const T& hi)
	{
	return v = (v < lo) ? lo : ((v >= hi) ? hi : v);
	}

namespace Particle_system
	{
	enum class CPU_Type { Sequential, Parallel };

	struct init
		{
		size_t amount{1};
		size_t width;
		size_t height;
		std::seed_seq random_seed;
		};

	template <typename Allocator = std::allocator<sf::Vertex>>
	class base : public Updatable
		{
		public:
			using vertices_t = std::vector<sf::Vertex, Allocator>;

			base(const init& init, sf::Color c = sf::Color::White) noexcept :
				width{static_cast<float>(init.width)}, height{static_cast<float>(init.height)}
				{
				std::default_random_engine random_engine{init.random_seed};
				std::uniform_real_distribution<float> x_distribution{0.f, static_cast<float>(init.width)};
				std::uniform_real_distribution<float> y_distribution{0.f, static_cast<float>(init.height)};

				vertices.reserve(init.amount);
				for (size_t i = 0; i < init.amount; i++)
					{
					float x = x_distribution(random_engine);
					float y = y_distribution(random_engine);

					vertices.emplace_back(sf::Vector2f{x, y}, c);
					}
				}

		protected:
			vertices_t vertices;
			float width, height;



		private:
			void draw(sf::RenderTarget& target, sf::RenderStates states) const final override { target.draw(vertices.data(), vertices.size(), sf::Points, states); }

		};
	}