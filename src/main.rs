use winit::event_loop::EventLoop;
use ash::vk;
use cgmath::{Deg, Matrix4, Point3, Vector3, Vector4, InnerSpace};

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::ptr;
use std::rc::Rc;

mod platforms;
mod window;
mod debug;
mod utils;
mod support;
mod scene;
mod objects;
mod models;

use window::VulkanApp;
use models::{Model, Vertex};
use support::{Device, DeviceBuilder};
use support::descriptor::{
    DescriptorPool,
    DescriptorSetLayout,
    DescriptorRef,
    UniformBufferRef,
};
use support::renderer::{Presenter, Pipeline};
use support::shader::{VertexShader, FragmentShader};
use support::texture::Material;
use support::buffer::{VertexBuffer, IndexBuffer, UniformBufferSet};
use objects::{StaticGeometrySet, StaticGeometrySetRenderer};
use scene::{Scene, Renderable};

const WINDOW_TITLE: &'static str = "Wanderer";
const WINDOW_WIDTH: usize = 1024;
const WINDOW_HEIGHT: usize = 768;
const MODEL_PATH: &'static str = "viking_room.obj";
//const TEXTURE_PATH: &'static str = "viking_room.png";

#[repr(C)]
#[derive(Clone)]
struct UniformBufferObject {
    #[allow(unused)]
    view: Matrix4<f32>,
    #[allow(unused)]
    proj: Matrix4<f32>,
    #[allow(unused)]
    view_pos: Vector4<f32>,
    #[allow(unused)]
    view_dir: Vector4<f32>,
    #[allow(unused)]
    use_diffuse: u32,
    #[allow(unused)]
    use_specular: u32,
}

struct VulkanApp21 {
    device: Device,
    presenter: Presenter,
    #[allow(unused)]
    global_pool: DescriptorPool,
    #[allow(unused)]
    global_descriptor_set_layout: DescriptorSetLayout,
    global_uniform_buffer_set: UniformBufferSet<UniformBufferObject>,
    static_geometry_pipelines: Vec<Rc<RefCell<Pipeline<Vertex>>>>,
    #[allow(unused)]
    viking_room_set: Rc<StaticGeometrySetRenderer<Vertex>>,
    #[allow(unused)]
    global_descriptor_sets: Vec<vk::DescriptorSet>,
    scene: Scene,
    materials: Vec<Rc<Material>>,
    _model: Model,

    is_framebuffer_resized: bool,
    yaw_speed: f32,
    pitch_speed: f32,
    roll_speed: f32,
    camera_speed: Vector3<f32>,
    view_dir: Vector4<f32>,
    view_up: Vector4<f32>,
    view_right: Vector4<f32>,
}

impl Drop for VulkanApp21 {
    fn drop(&mut self) {
	self.static_geometry_pipelines.clear();
	self.materials.clear();
    }
}

impl VulkanApp21 {
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
	dbg!(max_frames_in_flight);
        device.check_mipmap_support(vk::Format::R8G8B8A8_UNORM)?;

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

	let mut global_pool = {
	    let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
	    pool_sizes.insert(
		vk::DescriptorType::UNIFORM_BUFFER,
		// I think I only need 2, but let's play it safe.
		10,
	    );
	    DescriptorPool::new(
		&device,
		pool_sizes,
		// I think I only need 2, but let's play it safe.
		10,
	    )?
	};

	// TODO: I feel likw the descriptor layout, uniform buffers, and descriptor sets
	//       could be created together in some convenient way.
	let global_descriptor_set_layout = DescriptorSetLayout::new(
	    &device,
	    vec![vk::DescriptorSetLayoutBinding{
		binding: 0,
		descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		descriptor_count: 1,
		stage_flags: vk::ShaderStageFlags::ALL,
		p_immutable_samplers: ptr::null(),
	    }],
	)?;

	let global_uniform_buffer_set = UniformBufferSet::new(
	    &device,
	    UniformBufferObject{
                view: Matrix4::look_at_dir(
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::new(0.0, 1.0, 0.0).normalize(),
                    Vector3::new(0.0, 0.0, 1.0),
                ),
                proj: {
                    let mut proj = cgmath::perspective(
                        Deg(45.0),
                        width as f32
                            / height as f32,
                        0.1,
                        100.0,
                    );
                    proj[1][1] = proj[1][1] * -1.0;
                    proj
                },
		view_pos: Vector4::new(0.0, 0.0, 0.0, 1.0),
		view_dir: Vector4::new(0.0, 1.0, 0.0, 1.0),
		use_diffuse: 0xffffffff,
		use_specular: 0xffffffff,
	    },
	    max_frames_in_flight,
	)?;

	let mut global_descriptor_sets = vec![];
	for frame_idx in 0..max_frames_in_flight {
	    let items: Vec<Box<dyn DescriptorRef>> = vec![
		Box::new(UniformBufferRef::new(vec![
		    global_uniform_buffer_set.get_buffer(frame_idx)?,
		])),
	    ];
	    let sets = global_pool.create_descriptor_sets(
		1,
		&global_descriptor_set_layout,
		&items,
	    )?;
	    global_descriptor_sets.push(sets[0]);
	}

	let material = Material::from_files(
	    &device,
	    &Path::new("./assets/textures/MetalPlates001/MetalPlates001_4K_Color.jpg"),
	    &Path::new("./assets/textures/MetalPlates001/MetalPlates001_4K_Normal.jpg"),
	    &Path::new("./assets/textures/MetalPlates001/MetalPlates001_4K_Material.jpg"),
	)?;
	let material2 = Material::from_files(
	    &device,
	    &Path::new("./assets/textures/MetalPlates001/MetalPlates001_4K_Color.jpg"),
	    &Path::new("./assets/textures/MetalPlates001/MetalPlates001_4K_Normal.jpg"),
	    &Path::new("./assets/textures/MetalPlates001/MetalPlates001_4K_Material.jpg"),
	)?;
        let materials = vec![Rc::new(material), Rc::new(material2)];

	//println!("Loading model...");
        let model = models::Model::load(Path::new(MODEL_PATH))?;
        let vertex_buffer = VertexBuffer::new(&device, model.get_vertices())?;
        let index_buffer = IndexBuffer::new(&device, model.get_indices())?;


	let mut viking_room_geometry_set = StaticGeometrySet::new(
	    &device,
	    global_descriptor_sets.clone(),
	    Rc::new(vertex_buffer),
	    Rc::new(index_buffer),
	    &materials,
	    max_frames_in_flight,
	)?;
	viking_room_geometry_set.add(
	    &device,
	    Matrix4::from_scale(1.0),
	)?;

	let set_layouts = [
	    &global_descriptor_set_layout,
	    viking_room_geometry_set.get_type_layout(),
	    viking_room_geometry_set.get_instance_layout(),
	];

	let static_geometry_pipeline = Rc::new(RefCell::new(Pipeline::new(
	    &device,
	    WINDOW_WIDTH,
	    WINDOW_HEIGHT,
	    &presenter,
	    vert_shader,
	    frag_shader,
	    msaa_samples,
	    &set_layouts,
	)?));

	let viking_room_set = Rc::new(StaticGeometrySetRenderer::new(
	    static_geometry_pipeline.clone(),
	    viking_room_geometry_set,
	));

	let renderables: Vec<Rc<dyn Renderable>> = vec![
	    viking_room_set.clone(),
	];

	let scene = Scene::new(
	    &device,
	    &presenter,
	    renderables,
	    device.get_default_graphics_queue(),
	    max_frames_in_flight,
	)?;

        Ok(VulkanApp21{
	    device,
	    presenter,
	    global_pool,
	    global_descriptor_set_layout,
	    global_uniform_buffer_set,
	    static_geometry_pipelines: vec![static_geometry_pipeline],
	    viking_room_set,
	    global_descriptor_sets,
	    scene,
	    materials,
            _model: model,

            is_framebuffer_resized: false,
            yaw_speed: 0.0,
            pitch_speed: 0.0,
            roll_speed: 0.0,
            camera_speed: [0.0, 0.0, 0.0].into(),
            view_dir: Vector4::new(0.0, 1.0, 0.0, 1.0).normalize(),
	    view_up: Vector4::new(0.0, 0.0, 1.0, 1.0).normalize(),
	    view_right: Vector4::new(1.0, 0.0, 0.0, 1.0).normalize(),
        })
    }

    fn update_uniform_buffer(&mut self, current_image: usize, delta_time: f32) {
	let pitch_transform = Matrix4::from_axis_angle(
            self.view_right.truncate(),
            Deg(self.pitch_speed * delta_time),
        );

	self.view_dir = pitch_transform * self.view_dir;
	self.view_up = pitch_transform * self.view_up;
	self.view_right = pitch_transform * self.view_right;

	let yaw_transform = Matrix4::from_axis_angle(
            self.view_up.truncate(),
            Deg(self.yaw_speed * delta_time),
        );

	self.view_dir = yaw_transform * self.view_dir;
	self.view_up = yaw_transform * self.view_up;
	self.view_right = yaw_transform * self.view_right;

	let roll_transform = Matrix4::from_axis_angle(
            self.view_dir.truncate(),
            Deg(self.roll_speed * delta_time),
        );

	self.view_dir = roll_transform * self.view_dir;
	self.view_up = roll_transform * self.view_up;
	self.view_right = roll_transform * self.view_right;

	let camera_speed = self.camera_speed;
	let view_dir = self.view_dir;
	let view_up = self.view_up;
	let view_right = self.view_right;
        self.global_uniform_buffer_set.update_and_upload(
	    current_image,
	    |uniform_transform: &mut UniformBufferObject| -> anyhow::Result<()> {
		uniform_transform.view_pos =
                    uniform_transform.view_pos
                    + (view_up * camera_speed.z * delta_time)
                    + (view_dir * camera_speed.y * delta_time)
                    + (view_right * -camera_speed.x * delta_time);

		uniform_transform.view =
                    Matrix4::look_at_dir(
			Point3::new(
                            uniform_transform.view_pos.x,
                            uniform_transform.view_pos.y,
                            uniform_transform.view_pos.z,
			),
			view_dir.truncate(),
			view_up.truncate(),
                    );
		Ok(())
            }).unwrap();
    }
}

impl VulkanApp for VulkanApp21 {
    fn draw_frame(&mut self) -> anyhow::Result<()>{
	// Hopefully, this will give me the precision I need for the calculation but the
	// compactness and speed I want for the result.
        let since_last_frame = self.presenter.wait_for_next_frame()?;
	let delta_time = ((since_last_frame.as_nanos() as f64) / 1_000_000_000_f64) as f32;
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
	    {
		let mut pipeline = self.static_geometry_pipelines[0].borrow_mut();
		pipeline.update_viewport(
		    width,
		    height,
		    &self.presenter,
		)?;
	    }
	    self.scene.rebuild_command_buffers(&self.device, &self.presenter)?;
	    maybe_new_dimensions = None;
	}

        self.update_uniform_buffer(image_index as usize, delta_time);

        //self.presenter.submit_graphics_command_buffer(&self.command_buffers[image_index as usize])?;
	let (image_available_semaphore, render_finished_semaphore, inflight_fence) =
	    self.presenter.get_sync_objects();
	self.scene.submit_command_buffer(
	    image_index as usize,
	    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
	    image_available_semaphore,
	    render_finished_semaphore,
	    inflight_fence,
	)?;
	self.presenter.present_frame(
	    image_index,
	    &mut |width: usize, height: usize| -> anyhow::Result<()> {
		maybe_new_dimensions = Some((width, height));
		Ok(())
	    },
	)?;
	if let Some((width, height)) = maybe_new_dimensions {
	    {
		let mut pipeline = self.static_geometry_pipelines[0].borrow_mut();
		pipeline.update_viewport(
		    width,
		    height,
		    &self.presenter,
		)?;
	    }
	    self.scene.rebuild_command_buffers(&self.device, &self.presenter)?;
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

    fn set_yaw_speed(&mut self, speed: f32) {
        self.yaw_speed = speed;
    }

    fn set_pitch_speed(&mut self, speed: f32) {
        self.pitch_speed = speed;
    }

    fn set_roll_speed(&mut self, speed: f32) {
        self.roll_speed = speed;
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
        self.global_uniform_buffer_set.update(
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
        self.global_uniform_buffer_set.update(
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
    /*let timer = timer::Timer::new();
    let guard = timer.schedule_with_delay(chrono::Duration::seconds(5), move || {
	println!("BAILING OUT");
	std::process::abort();
    });
    guard.ignore();*/
    let event_loop = EventLoop::new();
    let vulkan_app = match VulkanApp21::new(&event_loop) {
	Ok(v) => v,
	Err(e) => panic!("Failed to create app: {:?}", e),
    };
    window::main_loop(event_loop, vulkan_app);
}
