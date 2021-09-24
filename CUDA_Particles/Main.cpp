#include <iostream>
#include "Window.h"
#include "Particle_system.h"

inline sf::Font font;

int main()
	{
	font.loadFromFile("consola.ttf");

	namespace PS = Particle_system;
	auto _60fps_delay = std::chrono::nanoseconds{16666666};

	Window wnd("CUDA Particles", 2000, 1000, _60fps_delay);

	PS::init particle_system_initializer
		{
		//.amount{50176},
		.amount{50000},
		//.amount{512},
		.width {wnd.get_width()},
		.height{wnd.get_height()},
		.random_seed{1}
		};

	vector_ptr_emplace<PS::CPU_Nested_foreach_Seq  >(wnd.updatables, particle_system_initializer);
	vector_ptr_emplace<PS::CPU_Nested_foreach_Par  >(wnd.updatables, particle_system_initializer);
	vector_ptr_emplace<PS::CPU_Forces_matrix_Seq   >(wnd.updatables, particle_system_initializer);
	vector_ptr_emplace<PS::CPU_Forces_matrix_Par   >(wnd.updatables, particle_system_initializer);
	vector_ptr_emplace<PS::CUDA_Forces_matrix      >(wnd.updatables, particle_system_initializer);
	vector_ptr_emplace<PS::CUDA_Nested_foreach     >(wnd.updatables, particle_system_initializer);
	vector_ptr_emplace<PS::CUDA_Nested_foreach_smem>(wnd.updatables, particle_system_initializer);

	wnd.run();
	}
