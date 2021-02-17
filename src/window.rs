use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{Window, WindowBuilder};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const PAINT_FPS_COUNTER: bool = false;
const ABORT_ON_FRAME_DRAW_FAILURE: bool = true;

trait EdgeTrigger {
    fn pressed(&self);
    fn released(&self);
}

#[derive(Copy, Clone)]
struct EdgeTriggerRef {
    id: u64,
    key: VirtualKeyCode,
}

struct KeypressSentinel {
    pressed: RefCell<bool>,
}

impl KeypressSentinel {
    fn new() -> Self {
	Self{
	    pressed: RefCell::new(false),
	}
    }

    fn reset(&self) {
	*self.pressed.borrow_mut() = false;
    }

    fn was_pressed(&self) -> bool {
	*self.pressed.borrow()
    }
}

impl EdgeTrigger for KeypressSentinel {
    fn pressed(&self) {
	*self.pressed.borrow_mut() = true;
    }

    fn released(&self) {
    }
}

struct KeyboardModifiers {
    shift: bool,
    ctrl: bool,
    alt: bool,
    logo: bool,
}

struct KeyboardState {
    edge_triggers: HashMap<VirtualKeyCode, Vec<(EdgeTriggerRef, Rc<dyn EdgeTrigger>)>>,
    keys_down: HashSet<VirtualKeyCode>,
    next_edge_trigger_id: u64,
    modifiers: KeyboardModifiers,
}

impl KeyboardState {
    fn new() -> Self {
	Self{
	    edge_triggers: HashMap::new(),
	    keys_down: HashSet::new(),
	    next_edge_trigger_id: 0,
	    modifiers: KeyboardModifiers{
		shift: false,
		ctrl: false,
		alt: false,
		logo: false,
	    },
	}
    }

    fn register_edge_trigger(
	&mut self,
	key: VirtualKeyCode,
	trigger: Rc<dyn EdgeTrigger>,
    ) -> EdgeTriggerRef {
	let trigger_ref = EdgeTriggerRef{
	    id: self.next_edge_trigger_id,
	    key,
	};
	// TODO: handle wrap-around
	self.next_edge_trigger_id += 1;
	if let Some(triggers) = self.edge_triggers.get_mut(&key) {
	    triggers.push((trigger_ref, trigger));
	} else {
	    let mut triggers = Vec::new();
	    triggers.push((trigger_ref, trigger));
	    self.edge_triggers.insert(key, triggers);
	}

	trigger_ref
    }

    #[allow(unused)]
    fn unregister_edge_trigger(
	&mut self,
	trigger_ref: EdgeTriggerRef,
    ) {
	if let Some(triggers) = self.edge_triggers.get_mut(&trigger_ref.key) {
	    for i in 0..triggers.len() {
		if triggers[i].0.id == trigger_ref.id {
		    triggers.remove(i);
		    return;
		}
	    }
	}
    }

    #[allow(unused)]
    fn key_is_down(&self, key: VirtualKeyCode) -> bool {
	self.keys_down.contains(&key)
    }

    fn get_down_keys(&self) -> &HashSet<VirtualKeyCode> {
	&self.keys_down
    }

    fn set_key_pressed(&mut self, key: VirtualKeyCode) {
	use VirtualKeyCode::*;
	match key {
	    // We use the modifiers field to track these.
	    // We should be able to just use the "modifiers changed" event, but that
	    // seems to be annoyingly laggy.
	    LShift | RShift => {
		self.modifiers.shift = true;
	    },
	    LControl | RControl => {
		self.modifiers.ctrl = true;
	    },
	    LAlt | RAlt => {
		self.modifiers.alt = true;
	    },
	    LWin | RWin => {
		self.modifiers.logo = true;
	    },
	    _ => {
		self.keys_down.insert(key);
	    },
	}
	if let Some(triggers) = self.edge_triggers.get(&key) {
	    for (_, trigger) in triggers {
		trigger.pressed();
	    }
	}
    }

    fn set_key_released(&mut self, key: VirtualKeyCode) {
	self.keys_down.remove(&key);
	if let Some(triggers) = self.edge_triggers.get(&key) {
	    for (_, trigger) in triggers {
		trigger.released();
	    }
	}
    }
}

pub fn init_window(
    event_loop: &EventLoop<()>,
    title: &str,
    width: u32,
    height: u32
) -> Window {
    WindowBuilder::new()
        .with_title(title)
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .build(event_loop)
        .expect("Failed to create window")
}

pub trait VulkanApp {
    fn draw_frame(&mut self, raw_input: egui::RawInput) -> anyhow::Result<()>;
    fn get_fps(&self) -> u32;
    fn wait_device_idle(&self) -> anyhow::Result<()>;
    fn resize_framebuffer(&mut self);
    fn window_ref(&self) -> &Window;
    fn set_yaw_speed(&mut self, speed: f32);
    fn set_pitch_speed(&mut self, speed: f32);
    fn set_roll_speed(&mut self, speed: f32);
    fn set_x_speed(&mut self, speed: f32);
    fn set_y_speed(&mut self, speed: f32);
    fn set_z_speed(&mut self, speed: f32);
    fn toggle_uniform_twiddler(&mut self) -> bool;
    fn get_window_size(&self) -> (usize, usize);
    fn get_window_scale(&self) -> f32;
}

fn mutually_exclusive(a: &mut bool, b: &mut bool) {
    if *a && *b {
	*a = false;
	*b = false;
    }
}

struct CameraMotionController {
    speed_boost: bool,
    yaw_left: bool,
    yaw_right: bool,
    pitch_up: bool,
    pitch_down: bool,
    roll_left: bool,
    roll_right: bool,
    move_forward: bool,
    move_backward: bool,
    move_left: bool,
    move_right: bool,
    move_up: bool,
    move_down: bool,
}

impl CameraMotionController {
    fn new() -> Self {
	Self{
	    speed_boost: false,
	    yaw_left: false,
	    yaw_right: false,
	    pitch_up: false,
	    pitch_down: false,
	    roll_left: false,
	    roll_right: false,
	    move_forward: false,
	    move_backward: false,
	    move_left: false,
	    move_right: false,
	    move_up: false,
	    move_down: false,
	}
    }

    #[allow(unused)]
    fn reset(&mut self) {
	self.speed_boost = false;
	self.yaw_left = false;
	self.yaw_right = false;
	self.pitch_up = false;
	self.pitch_down = false;
	self.roll_left = false;
	self.roll_right = false;
	self.move_forward = false;
	self.move_backward = false;
	self.move_left = false;
	self.move_right = false;
	self.move_up = false;
	self.move_down = false;
    }

    fn process_keys(&mut self, keyboard_state: &KeyboardState) {
	let keys = keyboard_state.get_down_keys();
	if keyboard_state.modifiers.shift {
	    self.speed_boost = true;
	}

	if keys.contains(&VirtualKeyCode::Left) {
	    self.yaw_left = true;
	}
	if keys.contains(&VirtualKeyCode::Right) {
	    self.yaw_right = true;
	}

	if keys.contains(&VirtualKeyCode::Down) {
	    self.pitch_up = true;
	}
	if keys.contains(&VirtualKeyCode::Up) {
	    self.pitch_down = true;
	}

	if keys.contains(&VirtualKeyCode::Q) {
	    self.roll_left = true;
	}
	if keys.contains(&VirtualKeyCode::E) {
	    self.roll_right = true;
	}

	if keys.contains(&VirtualKeyCode::W) {
	    self.move_forward = true;
	}
	if keys.contains(&VirtualKeyCode::S) {
	    self.move_backward = true;
	}

	if keys.contains(&VirtualKeyCode::A) {
	    self.move_left = true;
	}
	if keys.contains(&VirtualKeyCode::D) {
	    self.move_right = true;
	}

	if keys.contains(&VirtualKeyCode::Space) {
	    self.move_up = true;
	}
	if keys.contains(&VirtualKeyCode::C) {
	    self.move_down = true;
	}

	self.fix_input();
    }

    fn fix_input(&mut self) {
	mutually_exclusive(&mut self.yaw_left, &mut self.yaw_right);
	mutually_exclusive(&mut self.pitch_up, &mut self.pitch_down);
	mutually_exclusive(&mut self.roll_left, &mut self.roll_right);

	mutually_exclusive(&mut self.move_forward, &mut self.move_backward);
	mutually_exclusive(&mut self.move_left, &mut self.move_right);
	mutually_exclusive(&mut self.move_up, &mut self.move_down);
    }
}

pub fn main_loop<A: 'static + VulkanApp>(event_loop: EventLoop<()>, mut vulkan_app: A) {
    let mut keyboard_state = KeyboardState::new();

    let quit_sentinel = Rc::new(KeypressSentinel::new());
    keyboard_state.register_edge_trigger(
	VirtualKeyCode::Escape,
	{
	    // The Rust compiler is amazingly bad at inferring types in certain situations.
	    let temp = Rc::clone(&quit_sentinel);
	    temp
	},
    );

    let twiddler_sentinel = Rc::new(KeypressSentinel::new());
    keyboard_state.register_edge_trigger(
	VirtualKeyCode::F3,
	{
	    // The Rust compiler is amazingly bad at inferring types in certain situations.
	    let temp = Rc::clone(&twiddler_sentinel);
	    temp
	},
    );

    let mut camera_controller = CameraMotionController::new();
    let mut raw_input: egui::RawInput = Default::default();
    event_loop.run(move |event, _, control_flow| {
	let (window_width, window_height) = vulkan_app.get_window_size();
	raw_input.screen_rect = Some(egui::math::Rect{
	    min: egui::math::Pos2{
		x: 0.0,
		y: 0.0,
	    },
	    max: egui::math::Pos2{
		x: window_width as f32,
		y: window_height as f32,
	    },
	});

        match event {
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::CloseRequested => {
                        vulkan_app.wait_device_idle().unwrap();
                        *control_flow = ControlFlow::Exit;
                    },
		    WindowEvent::CursorMoved { position, .. } => {
			let scale = vulkan_app.get_window_scale();
			raw_input.mouse_pos = Some(egui::math::Pos2{
			    x: position.x as f32 / scale,
			    y: position.y as f32 / scale,
			});
		    },
		    WindowEvent::MouseInput { state, button, .. } => {
			if button == winit::event::MouseButton::Left {
			    raw_input.mouse_down = state == ElementState::Pressed;
			}
		    },
		    WindowEvent::ModifiersChanged(state) => {
			keyboard_state.modifiers.shift = state.shift();
			keyboard_state.modifiers.ctrl = state.ctrl();
			keyboard_state.modifiers.alt = state.alt();
			keyboard_state.modifiers.logo = state.logo();
		    },
                    WindowEvent::KeyboardInput { input, .. } => {
                        match input {
                            KeyboardInput { virtual_keycode, state, .. } => {
				if let Some(key) = virtual_keycode {
				    if let Some(egui_key) = convert_key_to_egui(key) {
					raw_input.events.push(egui::Event::Key{
					    key: egui_key,
					    pressed: state == ElementState::Pressed,
					    modifiers: egui::Modifiers{
						alt: keyboard_state.modifiers.alt,
						ctrl: keyboard_state.modifiers.ctrl,
						shift: keyboard_state.modifiers.shift,
						mac_cmd: is_mac_cmd_pressed(
						    keyboard_state.modifiers.logo,
						),
						command: is_command_pressed(
						    keyboard_state.modifiers.ctrl,
						    keyboard_state.modifiers.logo,
						),
					    },
					});
				    }
				    match state {
					ElementState::Pressed => 
					    keyboard_state.set_key_pressed(key),
					ElementState::Released => 
					    keyboard_state.set_key_released(key),
				    }
				}
                                if quit_sentinel.was_pressed() {
                                    vulkan_app.wait_device_idle().unwrap();
                                    *control_flow = ControlFlow::Exit;
                                }
                                if twiddler_sentinel.was_pressed() {
                                    if vulkan_app.toggle_uniform_twiddler() {
                                        println!("Activated uniform twiddler app");
                                    } else {
                                        println!("Deactivated uniform twiddler app");
                                    };
				    twiddler_sentinel.reset();
                                }

				camera_controller.reset();
				camera_controller.process_keys(&keyboard_state);

                                let rotation_speed = if camera_controller.speed_boost {
                                    100.0
                                } else {
                                    20.0
                                };

                                if camera_controller.yaw_left {
                                    vulkan_app.set_yaw_speed(rotation_speed);
                                } else if camera_controller.yaw_right {
                                    vulkan_app.set_yaw_speed(-rotation_speed);
                                } else {
                                    vulkan_app.set_yaw_speed(0.0);
                                }

                                if camera_controller.pitch_down {
                                    vulkan_app.set_pitch_speed(-rotation_speed);
                                } else if camera_controller.pitch_up {
                                    vulkan_app.set_pitch_speed(rotation_speed);
                                } else {
                                    vulkan_app.set_pitch_speed(0.0);
                                }

                                if camera_controller.roll_left {
                                    vulkan_app.set_roll_speed(-rotation_speed);
                                } else if camera_controller.roll_right {
                                    vulkan_app.set_roll_speed(rotation_speed);
                                } else {
                                    vulkan_app.set_roll_speed(0.0);
                                }

                                let translation_speed = if camera_controller.speed_boost {
                                    2.5
                                } else {
                                    0.5
                                };

                                if camera_controller.move_up {
                                    vulkan_app.set_z_speed(translation_speed);
                                } else if camera_controller.move_down {
                                    vulkan_app.set_z_speed(-translation_speed);
                                } else {
                                    vulkan_app.set_z_speed(0.0);
                                }

                                if camera_controller.move_forward {
                                    vulkan_app.set_y_speed(translation_speed);
                                } else if camera_controller.move_backward {
                                    vulkan_app.set_y_speed(-translation_speed);
                                } else {
                                    vulkan_app.set_y_speed(0.0);
                                }

                                if camera_controller.move_left {
                                    vulkan_app.set_x_speed(translation_speed);
                                } else if camera_controller.move_right {
                                    vulkan_app.set_x_speed(-translation_speed);
                                } else {
                                    vulkan_app.set_x_speed(0.0);
                                }
                            }
                        }
                    },
                    WindowEvent::Resized(_new_size) => {
                        vulkan_app.wait_device_idle().unwrap();
                        vulkan_app.resize_framebuffer();
                    },
                    _ => {},
                }
            },
            Event::MainEventsCleared => {
                vulkan_app.window_ref().request_redraw();
            },
            Event::RedrawRequested(_window_id) => {
                if let Err(e) = vulkan_app.draw_frame(raw_input.clone()) {
		    println!("Failed to draw frame: {:?}", e);
		    if ABORT_ON_FRAME_DRAW_FAILURE {
			std::process::abort();
		    }
		}
		raw_input.events.clear();
                if PAINT_FPS_COUNTER {
                    println!("FPS: {}", vulkan_app.get_fps());
                }
            },
            Event::LoopDestroyed => {
                vulkan_app.wait_device_idle().unwrap();
            },
            _ => (),
        }
    })
}

fn convert_key_to_egui(key: VirtualKeyCode) -> Option<egui::Key> {
    use egui::Key::*;
    Some(match key {
	VirtualKeyCode::Down => ArrowDown,
	VirtualKeyCode::Left => ArrowLeft,
	VirtualKeyCode::Right => ArrowRight,
	VirtualKeyCode::Up => ArrowUp,
	VirtualKeyCode::Escape => Escape,
	VirtualKeyCode::Tab => Tab,
	VirtualKeyCode::Back => Backspace,
	VirtualKeyCode::Return => Enter,
	VirtualKeyCode::Space => Space,
	VirtualKeyCode::Insert => Insert,
	VirtualKeyCode::Delete => Delete,
	VirtualKeyCode::Home => Home,
	VirtualKeyCode::End => End,
	VirtualKeyCode::PageUp => PageUp,
	VirtualKeyCode::PageDown => PageDown,
	VirtualKeyCode::Key0 | VirtualKeyCode::Numpad0 => Num0,
	VirtualKeyCode::Key1 | VirtualKeyCode::Numpad1 => Num1,
	VirtualKeyCode::Key2 | VirtualKeyCode::Numpad2 => Num2,
	VirtualKeyCode::Key3 | VirtualKeyCode::Numpad3 => Num3,
	VirtualKeyCode::Key4 | VirtualKeyCode::Numpad4 => Num4,
	VirtualKeyCode::Key5 | VirtualKeyCode::Numpad5 => Num5,
	VirtualKeyCode::Key6 | VirtualKeyCode::Numpad6 => Num6,
	VirtualKeyCode::Key7 | VirtualKeyCode::Numpad7 => Num7,
	VirtualKeyCode::Key8 | VirtualKeyCode::Numpad8 => Num8,
	VirtualKeyCode::Key9 | VirtualKeyCode::Numpad9 => Num9,
	VirtualKeyCode::A => A,
	VirtualKeyCode::B => B,
	VirtualKeyCode::C => C,
	VirtualKeyCode::D => D,
	VirtualKeyCode::E => E,
	VirtualKeyCode::F => F,
	VirtualKeyCode::G => G,
	VirtualKeyCode::H => H,
	VirtualKeyCode::I => I,
	VirtualKeyCode::J => J,
	VirtualKeyCode::K => K,
	VirtualKeyCode::L => L,
	VirtualKeyCode::M => M,
	VirtualKeyCode::N => N,
	VirtualKeyCode::O => O,
	VirtualKeyCode::P => P,
	VirtualKeyCode::Q => Q,
	VirtualKeyCode::R => R,
	VirtualKeyCode::S => S,
	VirtualKeyCode::T => T,
	VirtualKeyCode::U => U,
	VirtualKeyCode::V => V,
	VirtualKeyCode::W => W,
	VirtualKeyCode::X => X,
	VirtualKeyCode::Y => Y,
	VirtualKeyCode::Z => Z,
	_ => return None,
    })
}

#[cfg(target_os = "macos")]
fn is_mac_cmd_pressed(pressed: bool) -> bool {
    pressed
}

#[cfg(not(target_os = "macos"))]
fn is_mac_cmd_pressed(_pressed: bool) -> bool {
    false
}

#[cfg(target_os = "macos")]
fn is_command_pressed(_ctrl: bool, logo: bool) -> bool {
    logo
}

#[cfg(not(target_os = "macos"))]
fn is_command_pressed(ctrl: bool, _logo: bool) -> bool {
    ctrl
}
