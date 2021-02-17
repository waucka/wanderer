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

struct KeyboardState {
    edge_triggers: HashMap<VirtualKeyCode, Vec<(EdgeTriggerRef, Rc<dyn EdgeTrigger>)>>,
    keys_down: HashSet<VirtualKeyCode>,
    next_edge_trigger_id: u64,
}

impl KeyboardState {
    fn new() -> Self {
	Self{
	    edge_triggers: HashMap::new(),
	    keys_down: HashSet::new(),
	    next_edge_trigger_id: 0,
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
	self.keys_down.insert(key);
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

    fn process_keys(&mut self, keys: &HashSet<VirtualKeyCode>) {
	if keys.contains(&VirtualKeyCode::LShift) {
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
                    WindowEvent::KeyboardInput { input, .. } => {
                        match input {
                            KeyboardInput { virtual_keycode, state, .. } => {
				if let Some(key) = virtual_keycode {
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
				camera_controller.process_keys(keyboard_state.get_down_keys());

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
