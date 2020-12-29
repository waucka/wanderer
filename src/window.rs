use winit::event::{Event, VirtualKeyCode, ElementState, KeyboardInput, WindowEvent};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{Window, WindowBuilder};

const PAINT_FPS_COUNTER: bool = false;

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
    fn draw_frame(&mut self) -> anyhow::Result<()>;
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
    fn toggle_diffuse(&mut self) -> bool;
    fn toggle_specular(&mut self) -> bool;
}

pub fn main_loop<A: 'static + VulkanApp>(event_loop: EventLoop<()>, mut vulkan_app: A) {
    let mut w_down = false;
    let mut a_down = false;
    let mut s_down = false;
    let mut d_down = false;
    let mut q_down = false;
    let mut e_down = false;
    let mut space_down = false;
    let mut c_down = false;
    let mut up_arrow_down = false;
    let mut left_arrow_down = false;
    let mut down_arrow_down = false;
    let mut right_arrow_down = false;
    let mut shift_down = false;
    event_loop.run(move |event, _, control_flow| {
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
                                match (virtual_keycode, state) {
                                    (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
                                        vulkan_app.wait_device_idle().unwrap();
                                        *control_flow = ControlFlow::Exit;
                                    },
                                    (Some(VirtualKeyCode::W), state) => {
                                        w_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::A), state) => {
                                        a_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::S), state) => {
                                        s_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::D), state) => {
                                        d_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::Q), state) => {
                                        q_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::E), state) => {
                                        e_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::Space), state) => {
                                        space_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::C), state) => {
                                        c_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::Up), state) => {
                                        up_arrow_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::Left), state) => {
                                        left_arrow_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::Down), state) => {
                                        down_arrow_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::Right), state) => {
                                        right_arrow_down = state == ElementState::Pressed;
                                    },
                                    (Some(VirtualKeyCode::LShift), ElementState::Pressed) => {
                                        shift_down = true;
                                    },
                                    (Some(VirtualKeyCode::LShift), ElementState::Released) => {
                                        shift_down = false;
                                    },
                                    (Some(VirtualKeyCode::T), ElementState::Released) => {
                                        if vulkan_app.toggle_diffuse() {
                                            println!("Toggled diffuse lighting on");
                                        } else {
                                            println!("Toggled diffuse lighting off");
                                        };
                                    }
                                    (Some(VirtualKeyCode::Y), ElementState::Released) => {
                                        if vulkan_app.toggle_specular() {
                                            println!("Toggled specular lighting on");
                                        } else {
                                            println!("Toggled specular lighting off");
                                        };
                                    }
                                    _ => {},
                                }

                                let rotation_speed = if shift_down {
                                    100.0
                                } else {
                                    20.0
                                };

                                if left_arrow_down && !right_arrow_down {
                                    vulkan_app.set_yaw_speed(rotation_speed);
                                } else if right_arrow_down && !left_arrow_down {
                                    vulkan_app.set_yaw_speed(-rotation_speed);
                                } else {
                                    vulkan_app.set_yaw_speed(0.0);
                                }

                                if up_arrow_down && !down_arrow_down {
                                    vulkan_app.set_pitch_speed(-rotation_speed);
                                } else if down_arrow_down && !up_arrow_down {
                                    vulkan_app.set_pitch_speed(rotation_speed);
                                } else {
                                    vulkan_app.set_pitch_speed(0.0);
                                }

                                if q_down && !e_down {
                                    vulkan_app.set_roll_speed(-rotation_speed);
                                } else if e_down && !q_down {
                                    vulkan_app.set_roll_speed(rotation_speed);
                                } else {
                                    vulkan_app.set_roll_speed(0.0);
                                }

                                let translation_speed = if shift_down {
                                    2.5
                                } else {
                                    0.5
                                };

                                if space_down && !c_down {
                                    vulkan_app.set_z_speed(translation_speed);
                                } else if c_down && !space_down {
                                    vulkan_app.set_z_speed(-translation_speed);
                                } else {
                                    vulkan_app.set_z_speed(0.0);
                                }

                                if w_down && !s_down {
                                    vulkan_app.set_y_speed(translation_speed);
                                } else if s_down && !w_down {
                                    vulkan_app.set_y_speed(-translation_speed);
                                } else {
                                    vulkan_app.set_y_speed(0.0);
                                }

                                if d_down && !a_down {
                                    vulkan_app.set_x_speed(-translation_speed);
                                } else if a_down && !d_down {
                                    vulkan_app.set_x_speed(translation_speed);
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
                if let Err(e) = vulkan_app.draw_frame() {
		    println!("Failed to draw frame: {:?}", e);
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
