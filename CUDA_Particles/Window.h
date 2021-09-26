#pragma once
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <limits>
#include <sstream>

#include <SFML/Graphics.hpp>

#include "Elements.h"

namespace Particle_system { extern float mass; }

extern sf::Font font;

class Window
	{
	public:
		Window(const std::string& title, unsigned width, unsigned height, std::chrono::nanoseconds delay, size_t max_step = std::numeric_limits<size_t>::max()) :
			window{{width, height}, title, sf::Style::Default, sf::ContextSettings{0U, 0U, 8U}},
			delay{delay},
			max_step{max_step}
			{
			mass_text.setColor({126, 180, 255});
			mass_text.setPosition({16, 16});
			mass_text.setCharacterSize(16);

			toggle_text.setColor({126, 180, 255});
			toggle_text.setPosition({16, 48});
			toggle_text.setCharacterSize(16);

			updatables_states_text.setColor({126, 180, 255});
			updatables_states_text.setPosition({16, 86});
			updatables_states_text.setCharacterSize(16);

			ui_back.setPosition(0, 0);
			ui_back.setFillColor({0, 0, 0, 180});
			ui_back.setOutlineColor({127, 127, 127, 255});
			ui_back.setOutlineThickness(2);
			}

		void run() 
			{
			update_updatables_states_text();
			update_mass_text();
			update_toggle_text();

			float width = std::max
				(
				updatables_states_text.getGlobalBounds().left + updatables_states_text.getGlobalBounds().width, std::max(
				mass_text             .getGlobalBounds().left + mass_text             .getGlobalBounds().width,
				toggle_text           .getGlobalBounds().left + toggle_text           .getGlobalBounds().width)
				);
			float height = updatables_states_text.getGlobalBounds().top + updatables_states_text.getGlobalBounds().height;

			ui_back.setSize({width + 16.f, height + 16.f});

			main_loop(); 
			}

		std::vector<std::unique_ptr<Updatable>> updatables;

		size_t get_width()  { return window.getSize().x; }
		size_t get_height() { return window.getSize().y; }

		void print_times()
			{
			std::stringstream sstr;
			sstr << "Average times:\n";
			for (const auto& updatable : updatables) { sstr << updatable->get_name() << " - " << std::setw(15) << updatable->get_average_time() << "\n"; }
			sstr << "_______________________________________________";
			std::cout << sstr.str() << std::endl;
			}

	private:
		sf::RenderWindow window;
		std::chrono::nanoseconds delay;
		size_t step{0};
		size_t max_step{std::numeric_limits<size_t>::max()};

		sf::Text mass_text             {"", font};
		sf::Text updatables_states_text{"", font};
		sf::Text toggle_text           {"", font};
		sf::RectangleShape ui_back;

		//Dragging
		bool is_dragging = false;
		sf::Vector2f mouse_prev = {0, 0};

		//Toggles
		enum class toggle_t { state, visibility } toggle{toggle_t::state};

		void main_loop()
			{
			while (/*window.isOpen() &&*/ step < max_step)
				{
				auto tp{std::chrono::steady_clock::now()};
				poll_events();
				draw();

				auto dt{std::chrono::steady_clock::now() - tp};
				std::this_thread::sleep_for(delay - dt);

				step++;
				}
			}

		void poll_events()
			{
			for (auto event = sf::Event{}; window.pollEvent(event);)
				{
				switch (event.type)
					{
					case sf::Event::Closed: window.close(); break;
					case sf::Event::KeyPressed:          handle_key_pressed   (event.key);         break;
					case sf::Event::Resized:             handle_resize        (event.size);        break;
					case sf::Event::MouseButtonPressed:  handle_mouse_pressed (event.mouseButton); break;
					case sf::Event::MouseButtonReleased: handle_mouse_released(event.mouseButton); break;
					case sf::Event::MouseMoved:          handle_mouse_moved   (event.mouseMove);   break;
					default:
						break;
					}
				}
			for (auto& updatable : updatables) { updatable->update_time(); }
			}
		void draw()
			{
			window.clear(sf::Color::Black);
			for (const auto& updatable : updatables) { updatable->draw(window); }

			window.draw(ui_back);
			window.draw(mass_text);
			window.draw(toggle_text);
			window.draw(updatables_states_text);
			window.display();
			}

		void update_mass_text() noexcept { mass_text.setString("Particles mass * mass: " + std::to_string(Particle_system::mass)); }
		void update_updatables_states_text() noexcept
			{
			std::stringstream sstr;

			sstr << "Particle systems:\n";
			for (size_t i = 0; i < updatables.size(); i++)
				{
				const auto& updatable = *updatables[i];
				sstr << "[" << (i + 1) << "] " << updatable.get_name() << " ";
				sstr << (updatable.paused  ? "paused   " : "running  ") << " ";
				sstr << (updatable.visible ? "visible  " : "invisible") << "\n";
				}

			updatables_states_text.setString(sstr.str());
			}
		void update_toggle_text() { toggle_text.setString(std::string{"[tab] toggle "} + (toggle == toggle_t::state ? "state" : "visibility")); }

		void toggle_updatable(size_t index)
			{
			if (index < updatables.size())
				{
				     if (toggle == toggle_t::state)      { updatables[index]->paused  = !updatables[index]->paused;  }
				else if (toggle == toggle_t::visibility) { updatables[index]->visible = !updatables[index]->visible; }
				update_updatables_states_text();
				}
			}

		void handle_key_pressed(sf::Event::KeyEvent e)
			{
			switch (e.code)
				{
				case sf::Keyboard::Q:        print_times(); break;
				case sf::Keyboard::Add:      Particle_system::mass += 0.1; update_mass_text(); break;
				case sf::Keyboard::Subtract: Particle_system::mass -= 0.1; update_mass_text(); break;
				case sf::Keyboard::Num1:
				case sf::Keyboard::Num2:
				case sf::Keyboard::Num3:
				case sf::Keyboard::Num4:
				case sf::Keyboard::Num5:
				case sf::Keyboard::Num6:
				case sf::Keyboard::Num7:
				case sf::Keyboard::Num8:
				case sf::Keyboard::Num9: toggle_updatable(static_cast<size_t>(e.code) - sf::Keyboard::Num1); break;
				case sf::Keyboard::Num0: toggle_updatable(10); break;
				case sf::Keyboard::Tab:  toggle = toggle == toggle_t::state ? toggle_t::visibility : toggle_t::state; update_toggle_text(); break;
				}
			}

		void handle_resize(sf::Event::SizeEvent e) noexcept
			{
			auto view = window.getView();
			view.setSize(e.width, e.height);
			window.setView(view);
			}
		void handle_mouse_pressed(sf::Event::MouseButtonEvent e) noexcept 
			{
			switch (e.button)
				{
				case sf::Mouse::Button::Left:
				case sf::Mouse::Button::Middle:
				case sf::Mouse::Button::Right:
					is_dragging = true;

					mouse_prev = window.mapPixelToCoords({e.x, e.y});
					break;
				}
			}
		void handle_mouse_released(sf::Event::MouseButtonEvent e)
			{
			switch (e.button)
				{
				case sf::Mouse::Button::Left:
				case sf::Mouse::Button::Middle:
				case sf::Mouse::Button::Right:
					is_dragging = false;
					break;
				}
			}
		void handle_mouse_moved(sf::Event::MouseMoveEvent e)
			{
			if (is_dragging)
				{
				auto view = window.getView();
				auto pos = view.getCenter();

				auto mouse_new = window.mapPixelToCoords({e.x, e.y});

				auto delta = mouse_prev - mouse_new;

				view.setCenter(pos + delta);
				window.setView(view);
				}
			}
	};