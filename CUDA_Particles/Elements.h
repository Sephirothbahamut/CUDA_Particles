#pragma once
#include <vector>
#include <memory>
#include <chrono>
#include <iostream>

#include <utils/math/average.h>

#include <SFML/Graphics.hpp>

template <typename T, typename Parent, typename... Args>
T& vector_ptr_emplace(std::vector<std::unique_ptr<Parent>>& vec, Args&&... args)
	{
	auto ptr = std::make_unique<T>(std::forward<Args>(args)...);
	T& element = *ptr;
	vec.push_back(std::move(ptr));
	return element;
	}

struct Updatable : public sf::Drawable
	{
	public:
		bool paused {true};
		bool visible{true};

		void update_time() noexcept
			{
			if (!paused)
				{
				auto tp = std::chrono::steady_clock::now();
				update_impl();
				auto tt = std::chrono::steady_clock::now() - tp;
				average_time += tt;
				//std::cout << get_name() << " time taken: " << tt.count() << std::endl;
				}
			};
		std::chrono::nanoseconds get_average_time() const noexcept { return average_time; }
		void reset_average_time() noexcept { average_time.reset(); }

		virtual void update() noexcept { if (!paused) { update_impl(); } }

		virtual const char* get_name() const noexcept = 0;

		void draw(sf::RenderTarget& rt) const noexcept { if (visible) { rt.draw(*this); } }

	protected:
		virtual void update_impl() noexcept = 0;

	private:
		utils::cumulative_average<std::chrono::nanoseconds> average_time;
	};
