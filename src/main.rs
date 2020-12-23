use winit::event_loop::EventLoop;

use ash::vk;

use cgmath::{Deg, Matrix4, Point3, Vector3, Vector4, InnerSpace};
use std::path::Path;
use std::rc::Rc;

mod platforms;
mod window;
mod debug;
mod utils;
mod support;
mod models;

use window::VulkanApp;
use models::{Model, Vertex};
use support::{Device, DeviceBuilder};
use support::renderer::{Presenter, GlobalUniform, ObjectType, ObjectTypeRenderer, RenderObject};
use support::shader::{VertexShader, FragmentShader};
use support::texture::{Texture, Sampler};
use support::buffer::{VertexBuffer, IndexBuffer};
use support::command_buffer::CommandBuffer;

const WINDOW_TITLE: &'static str = "Wanderer";
const WINDOW_WIDTH: usize = 1024;
const WINDOW_HEIGHT: usize = 768;
const MODEL_PATH: &'static str = "viking_room.obj";
const TEXTURE_PATH: &'static str = "viking_room.png";

#[repr(C)]
#[derive(Clone)]
struct UniformBufferObject {
    #[allow(unused)]
    model: Matrix4<f32>,
    #[allow(unused)]
    view: Matrix4<f32>,
    #[allow(unused)]
    proj: Matrix4<f32>,
    #[allow(unused)]
    view_pos: Vector4<f32>,
    #[allow(unused)]
    use_diffuse: u32,
    #[allow(unused)]
    use_specular: u32,
}

struct StaticGeometryUBO {
    #[allow(unused)]
    tint: Vector4<f32>,
}

struct StaticInstanceUBO {
    #[allow(unused)]
    tint: Vector4<f32>,
}

struct VulkanApp21 {
    device: Device,
    presenter: Presenter,
    global_uniform: GlobalUniform<UniformBufferObject>,
    _static_geometry_type: Rc<ObjectType<StaticGeometryUBO, StaticInstanceUBO>>,
    viking_room: RenderObject<UniformBufferObject, StaticGeometryUBO, StaticInstanceUBO>,
    static_geometry_renderer: ObjectTypeRenderer<Vertex,
                                                 UniformBufferObject,
                                                 StaticGeometryUBO,
                                                 StaticInstanceUBO>,
    global_descriptor_sets: Vec<vk::DescriptorSet>,
    type_descriptor_sets: Vec<vk::DescriptorSet>,
    instance_descriptor_sets: Vec<vk::DescriptorSet>,
    _model: Model,
    vertex_buffer: VertexBuffer<Vertex>,
    index_buffer: IndexBuffer,
    command_buffers: Vec<CommandBuffer>,

    is_framebuffer_resized: bool,
    rotation_speed: f32,
    tilt_speed: f32,
    camera_speed: Vector3<f32>,
    view_dir: Vector4<f32>,
    view_tilt_axis: Vector4<f32>,
}

impl VulkanApp21 {
    fn create_command_buffers(&mut self) -> anyhow::Result<()> {
	//println!("Creating command buffers...");
        let mut command_buffers = vec![];
        let max_frames_in_flight = self.presenter.get_num_images();
        for i in 0..max_frames_in_flight {
            let descriptor_sets = [
                self.global_descriptor_sets[i],
                self.type_descriptor_sets[i],
                self.instance_descriptor_sets[i],
            ];
            let buf = CommandBuffer::new(
		&self.device,
		vk::CommandBufferLevel::PRIMARY,
		self.device.get_default_graphics_queue(),
	    )?;
	    {
		let writer = buf.record(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE)?;
		let render_pass_writer = writer.begin_render_pass(
                    &self.presenter,
                    &[
			vk::ClearValue{
                            color: vk::ClearColorValue{
				float32: [0.0, 0.0, 0.0, 1.0],
                            }
			},
			vk::ClearValue{
                            depth_stencil: vk::ClearDepthStencilValue{
				depth: 1.0,
				stencil: 0,
                            }
			},
                    ],
                    i,
		);
		self.viking_room.write_draw_command(
	            render_pass_writer,
	            &self.static_geometry_renderer,
	            &self.vertex_buffer,
	            &self.index_buffer,
	            &descriptor_sets,
		);
	    }
            command_buffers.push(buf);
        }
	self.command_buffers = command_buffers;
	Ok(())
    }

    pub fn new(event_loop: &EventLoop<()>) -> anyhow::Result<Self> {
	//println!("Creating a device...");
        let device = Device::new(
	    event_loop,
            DeviceBuilder::new()
                .with_window_title(WINDOW_TITLE)
                .with_application_version(0, 1, 0)
                .with_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
                .with_validation(true)
		.with_default_extensions()
        )?;
        let msaa_samples = device.get_max_usable_sample_count();
        let presenter = Presenter::new(
            &device,
	    msaa_samples,
            60,
        )?;
        let max_frames_in_flight = presenter.get_num_images();

        let vert_shader: VertexShader<Vertex> =
            VertexShader::from_spv_file(
                &device,
                Path::new("./lighting.vert.spv"),
            )?;
        let frag_shader = FragmentShader::from_spv_file(
            &device,
            Path::new("./lighting.frag.spv"),
        )?;

        let (width, height) = presenter.get_dimensions();

	//println!("Creating global uniform...");
        let global_uniform = GlobalUniform::new(
	    &device,
	    UniformBufferObject{
                model: Matrix4::from_angle_z(Deg(0.0)),
                view: Matrix4::look_at_dir(
                    Point3::new(0.0, 2.0, 0.0),
                    Vector3::new(0.0, -2.0, 0.0).normalize(),
                    Vector3::new(0.0, 0.0, 1.0),
                ),
                proj: {
                    let mut proj = cgmath::perspective(
                        Deg(45.0),
                        width as f32
                            / height as f32,
                        0.1,
                        10.0,
                    );
                    proj[1][1] = proj[1][1] * -1.0;
                    proj
                },
                view_pos: Vector4::new(0.0, 2.0, 0.0, 0.0),
                use_diffuse: 0xffffffff,
                use_specular: 0xffffffff,
            },
	    max_frames_in_flight,
        )?;

        device.check_mipmap_support(vk::Format::R8G8B8A8_UNORM)?;
        let texture = Texture::from_file(&device, &Path::new(TEXTURE_PATH))?;
        let sampler = Rc::new(Sampler::new(
	    &device,
	    texture.get_mip_levels(),
	    vk::Filter::LINEAR,
	    vk::Filter::LINEAR,
	    vk::SamplerMipmapMode::LINEAR,
	    vk::SamplerAddressMode::REPEAT,
        )?);
        let textures = vec![(sampler, Rc::new(texture))];
	let num_textures = textures.len();

        let static_geometry_type = Rc::new(ObjectType::new(
	    &device,
	    StaticGeometryUBO{
                tint: Vector4::new(1.0, 1.0, 1.0, 1.0),
            },
	    textures,
	    max_frames_in_flight,
        )?);

	//println!("Loading model...");
        let model = models::Model::load(Path::new(MODEL_PATH))?;
        let vertex_buffer = VertexBuffer::new(&device, model.get_vertices())?;
        let index_buffer = IndexBuffer::new(&device, model.get_indices())?;
        let viking_room = RenderObject::new(
	    &device,
	    static_geometry_type.clone(),
	    StaticInstanceUBO{
                tint: Vector4::new(1.0, 1.0, 1.0, 1.0),
            },
        )?;

	//println!("Creating static geometry renderer...");
        let static_geometry_renderer = ObjectTypeRenderer::new(
	    &device,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
	    &presenter,
	    vert_shader,
	    frag_shader,
	    msaa_samples,
	    num_textures,
        )?;

	//println!("Creating descriptor sets...");
        let global_descriptor_sets = global_uniform.create_descriptor_sets(&device)?;
        let type_descriptor_sets = static_geometry_type.create_descriptor_sets(&device)?;
        let instance_descriptor_sets = viking_room.create_descriptor_sets(&device)?;

        let mut vk_app = VulkanApp21{
            device,
            presenter,
            global_uniform,
            _static_geometry_type: static_geometry_type,
            viking_room: viking_room,
            static_geometry_renderer,
	    global_descriptor_sets,
	    type_descriptor_sets,
	    instance_descriptor_sets,
            _model: model,
            vertex_buffer: vertex_buffer,
            index_buffer: index_buffer,
            command_buffers: vec![],

            is_framebuffer_resized: false,
            rotation_speed: 0.0,
            tilt_speed: 0.0,
            camera_speed: [0.0, 0.0, 0.0].into(),
            view_dir: Vector4::new(0.0, -2.0, 0.0, 0.0).normalize(),
            view_tilt_axis: Vector4::new(1.0, 0.0, 0.0, 0.0).normalize(),
        };

	vk_app.create_command_buffers()?;

	Ok(vk_app)
    }

    fn update_uniform_buffer(&mut self, current_image: usize, delta_time: f32) {
        self.view_dir =
            Matrix4::from_axis_angle(
                self.view_tilt_axis.truncate(),
                Deg(self.tilt_speed * delta_time),
            ) * self.view_dir;
        self.view_dir =
            Matrix4::from_axis_angle(
                Vector3::new(0.0, 0.0, 1.0),
                Deg(self.rotation_speed * delta_time),
            ) * self.view_dir;
        self.view_tilt_axis =
            Matrix4::from_axis_angle(
                Vector3::new(0.0, 0.0, 1.0),
                Deg(self.rotation_speed * delta_time),
            ) * self.view_tilt_axis;

        let up_vector = self.view_dir.truncate().cross(self.view_tilt_axis.truncate()).extend(0.0);
	let camera_speed = self.camera_speed;
	let view_dir = self.view_dir;
	let view_tilt_axis = self.view_tilt_axis;
        self.global_uniform.get_uniform_set().update_and_upload(
	    current_image,
	    |uniform_transform: &mut UniformBufferObject| -> anyhow::Result<()> {
		uniform_transform.view_pos =
                    uniform_transform.view_pos
                    + (up_vector * camera_speed.z)
                    + (view_dir * camera_speed.y)
                    + (view_tilt_axis * -camera_speed.x);

		uniform_transform.view =
                    Matrix4::look_at_dir(
			Point3::new(
                            uniform_transform.view_pos.x,
                            uniform_transform.view_pos.y,
                            uniform_transform.view_pos.z,
			),
			view_dir.truncate(),
			Vector3::new(0.0, 0.0, 1.0),
                    );
		Ok(())
            }).unwrap();
    }
}

impl VulkanApp for VulkanApp21 {
    fn draw_frame(&mut self) -> anyhow::Result<()>{
	// Hopefully, this will give me the precision I need for the calculation but the
	// compactness and speed I want for the result.
	let delta_time = ((self.presenter.ns_since_last_frame() as f64) / 1_000_000_000_f64) as f32;
        self.presenter.wait_for_next_frame()?;
	let mut maybe_new_dimensions = None;
        let image_index = self.presenter.acquire_next_image(
	    &mut |width: usize, height: usize| -> anyhow::Result<()> {
		maybe_new_dimensions = Some((width, height));
		Ok(())
	    },
	)?;

	if self.is_framebuffer_resized {
	    maybe_new_dimensions = Some(self.presenter.get_dimensions());
	    self.is_framebuffer_resized = false;
	}

	if let Some((width, height)) = maybe_new_dimensions {
	    self.static_geometry_renderer.update_viewport(
		width,
		height,
		&self.presenter,
	    )?;
	    self.create_command_buffers()?;
	    maybe_new_dimensions = None;
	}

        self.update_uniform_buffer(image_index as usize, delta_time);

        self.presenter.submit_graphics_command_buffer(&self.command_buffers[image_index as usize])?;
	self.presenter.present_frame(
	    image_index,
	    &mut |width: usize, height: usize| -> anyhow::Result<()> {
		maybe_new_dimensions = Some((width, height));
		Ok(())
	    },
	)?;
	if let Some((width, height)) = maybe_new_dimensions {
	    self.static_geometry_renderer.update_viewport(
		width,
		height,
		&self.presenter,
	    )?;
	    self.create_command_buffers()?;
	}
        Ok(())
    }

    fn get_fps(&self) -> u32 {
	self.presenter.get_current_fps()
    }

    fn wait_device_idle(&self) -> anyhow::Result<()> {
        self.device.wait_for_idle()
    }

    fn resize_framebuffer(&mut self) {
        self.is_framebuffer_resized = true;
    }

    fn window_ref(&self) -> &winit::window::Window {
        &self.device.window_ref()
    }

    fn set_rotation_speed(&mut self, speed: f32) {
        self.rotation_speed = speed;
    }

    fn set_tilt_speed(&mut self, speed: f32) {
        self.tilt_speed = speed;
    }

    fn set_x_speed(&mut self, speed: f32) {
        self.camera_speed.x = speed;
    }

    fn set_y_speed(&mut self, speed: f32) {
        self.camera_speed.y = speed;
    }

    fn set_z_speed(&mut self, speed: f32) {
        self.camera_speed.z = speed;
    }

    fn toggle_diffuse(&mut self) -> bool {
        self.global_uniform.get_uniform_set().update(
	    |uniform_transform: &mut UniformBufferObject| -> anyhow::Result<bool> {
		uniform_transform.use_diffuse = if uniform_transform.use_diffuse > 0 {
                    0
		} else {
                    0xffffffff
		};
		dbg!(uniform_transform.use_diffuse);
		Ok(uniform_transform.use_diffuse != 0)
            }).unwrap()
    }

    fn toggle_specular(&mut self) -> bool {
        self.global_uniform.get_uniform_set().update(
	    |uniform_transform: &mut UniformBufferObject| -> anyhow::Result<bool> {
		uniform_transform.use_specular = if uniform_transform.use_specular > 0 {
                    0
		} else {
                    0xffffffff
		};
		dbg!(uniform_transform.use_specular);
		Ok(uniform_transform.use_specular != 0)
            }).unwrap()
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let vulkan_app = match VulkanApp21::new(&event_loop) {
	Ok(v) => v,
	Err(e) => panic!("Failed to create app: {:?}", e),
    };
    window::main_loop(event_loop, vulkan_app);
}
