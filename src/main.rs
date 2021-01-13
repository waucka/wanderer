use winit::event_loop::EventLoop;
use ash::vk;
use cgmath::{Deg, Matrix4, Point3, Vector3, Vector4, InnerSpace};
use glsl_layout::{AsStd140};
use anyhow::anyhow;

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
use models::{Model, ModelNonIndexed, Vertex};
use support::{Device, DeviceBuilder};
use support::command_buffer::CommandBuffer;
use support::descriptor::{
    DescriptorPool,
    DescriptorSetLayout,
    DescriptorRef,
    InputAttachmentRef,
    UniformBufferRef,
};
use support::renderer::{
    Presenter,
    Pipeline,
    PipelineParameters,
    RenderPass,
    RenderPassBuilder,
    AttachmentDescription,
    AttachmentSet,
    AttachmentRef,
    Subpass,
};
use support::shader::{VertexShader, FragmentShader};
use support::texture::{Material, Texture};
use support::buffer::{VertexBuffer, IndexBuffer, UniformBufferSet};
use support::image::ImageBuilder;
use objects::{StaticGeometrySet, StaticGeometrySetRenderer, PostProcessingStep};
use scene::{Scene, Renderable};
use utils::{NullVertex, Matrix4f, Vector4f};

const WINDOW_TITLE: &'static str = "Wanderer";
const WINDOW_WIDTH: usize = 1024;
const WINDOW_HEIGHT: usize = 768;
const MODEL_PATH: &'static str = "viking_room.obj";
//const TEXTURE_PATH: &'static str = "viking_room.png";

#[derive(Debug, Default, Clone, Copy, AsStd140)]
struct UniformBufferObject {
    #[allow(unused)]
    view: Matrix4f,
    #[allow(unused)]
    proj: Matrix4f,
    #[allow(unused)]
    view_pos: Vector4f,
    #[allow(unused)]
    view_dir: Vector4f,
    #[allow(unused)]
    light_positions: [Vector4f; 4],
    #[allow(unused)]
    light_colors: [Vector4f; 4],
    #[allow(unused)]
    use_parallax: glsl_layout::boolean,
    #[allow(unused)]
    use_ao: glsl_layout::boolean,
}

#[derive(Debug, Default, Clone, Copy, AsStd140)]
struct HdrControlUniform {
    #[allow(unused)]
    exposure: f32,
    #[allow(unused)]
    gamma: f32,
    #[allow(unused)]
    algo: u32,
}

struct VulkanApp21 {
    device: Device,
    presenter: Presenter,
    render_pass: RenderPass,
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
    attachment_set: AttachmentSet,
    render_target_color: AttachmentRef,
    postprocessing_pipelines: Vec<Rc<RefCell<Pipeline<NullVertex>>>>,
    hdr: PostProcessingStep,
    hdr_uniform_buffer_set: UniformBufferSet<HdrControlUniform>,
    hdr_descriptor_set_layout: DescriptorSetLayout,
    command_buffers: Vec<CommandBuffer>,
    materials: Vec<Rc<Material>>,
    _model: ModelNonIndexed,
    msaa_samples: vk::SampleCountFlags,

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
	// TODO: figure out how to support MSAA for this engine or use FXAA or something
        let msaa_samples = vk::SampleCountFlags::TYPE_1;//device.get_max_usable_sample_count();
	let (surface_format, depth_format) = Presenter::get_swapchain_image_formats(&device);
	let mut render_pass_builder = RenderPassBuilder::new(
	    AttachmentDescription::standard_color_final(
		surface_format,
		false,
		vk::ImageUsageFlags::COLOR_ATTACHMENT,
	    ),
	);
	let render_target_color = render_pass_builder.add_attachment(AttachmentDescription::new(
	    true,
	    vk::ImageUsageFlags::COLOR_ATTACHMENT |
	    vk::ImageUsageFlags::INPUT_ATTACHMENT |
	    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
	    vk::Format::R16G16B16_SFLOAT,
	    vk::ImageAspectFlags::COLOR,
	    vk::AttachmentLoadOp::CLEAR,
	    vk::AttachmentStoreOp::STORE,
	    vk::AttachmentLoadOp::DONT_CARE,
	    vk::AttachmentStoreOp::DONT_CARE,
	    vk::ImageLayout::UNDEFINED,
	    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
	));
	let render_target_depth = render_pass_builder.add_attachment(AttachmentDescription::standard_depth(
	    depth_format,
	    true,
	    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT |
	    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
	));
	let presentation_target = render_pass_builder.get_swapchain_attachment();
	// Rendering subpass
	render_pass_builder.add_subpass({
	    let mut subpass = Subpass::new(vk::PipelineBindPoint::GRAPHICS);
	    subpass.add_color_attachment(
		render_target_color.as_vk(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
		None,
	    );
	    subpass.set_depth_attachment(
		render_target_depth.as_vk(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
	    );
	    subpass
	});
	// Postprocessing subpass
	render_pass_builder.add_subpass({
	    let mut subpass = Subpass::new(vk::PipelineBindPoint::GRAPHICS);
	    subpass.add_color_attachment(
		presentation_target.as_vk(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
		None,
	    );
	    subpass.add_input_attachment(
		render_target_color.as_vk(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
	    );
	    subpass
	});
	render_pass_builder.add_standard_entry_dep();
	render_pass_builder.add_dep(vk::SubpassDependency{
	    src_subpass: 0,
	    dst_subpass: 1,
	    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
	    dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
	    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
	    dst_access_mask: vk::AccessFlags::SHADER_READ,
	    dependency_flags: vk::DependencyFlags::empty(),
	});
        let render_pass = RenderPass::new(
            &device,
            msaa_samples,
	    render_pass_builder,
        )?;
        let presenter = Presenter::new(
            &device,
	    &render_pass,
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
		// This is going to be shared between rendering and HDR tonemapping.
		20,
	    );
	    pool_sizes.insert(
		vk::DescriptorType::INPUT_ATTACHMENT,
		// This is going to be shared between rendering and HDR tonemapping.
		20,
	    );
	    DescriptorPool::new(
		&device,
		pool_sizes,
		// This is going to be shared between rendering and HDR tonemapping.
		20,
	    )?
	};

	// TODO: I feel like the descriptor layout, uniform buffers, and descriptor sets
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
                view: {
		    let view_matrix = Matrix4::look_at_dir(
			Point3::new(0.0, -2.0, 0.0),
			Vector3::new(0.0, 1.0, 0.0).normalize(),
			Vector3::new(0.0, 0.0, 1.0),
                    );
		    view_matrix.into()
		},
                proj: {
                    let mut proj = cgmath::perspective(
                        Deg(45.0),
                        width as f32
                            / height as f32,
                        0.1,
                        100.0,
                    );
                    proj[1][1] = proj[1][1] * -1.0;
		    proj.into()
                },
		view_pos: [0.0, -2.0, 0.0, 1.0].into(),
		view_dir: [0.0, 1.0, 0.0, 1.0].into(),
		light_positions: [
		    [0.0, 0.0, 10.0, 1.0].into(),
		    [0.0, 0.0, 0.0, 1.0].into(),
		    [0.0, 0.0, 0.0, 1.0].into(),
		    [0.0, 0.0, 0.0, 1.0].into(),
		],
		light_colors: [
		    [500.0, 500.0, 500.0, 500.0].into(),
		    [0.0, 0.0, 0.0, 0.0].into(),
		    [0.0, 0.0, 0.0, 0.0].into(),
		    [0.0, 0.0, 0.0, 0.0].into(),
		],
		use_parallax: true.into(),
		use_ao: true.into(),
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
	    &Path::new("./assets/textures/Bricks034/Bricks034_4K_Color.jpg"),
	    &Path::new("./assets/textures/Bricks034/Bricks034_4K_Normal.jpg"),
	    &Path::new("./assets/textures/Bricks034/Bricks034_4K_Properties.png"),
	)?;
	let material3 = Material::from_files(
	    &device,
	    &Path::new("./assets/textures/bricks_simple/bricks2.jpg"),
	    &Path::new("./assets/textures/bricks_simple/bricks2_normal.jpg"),
	    &Path::new("./assets/textures/bricks_simple/bricks2_disp.jpg"),
	)?;
	let material4 = Material::from_files(
	    &device,
	    &Path::new("./assets/textures/WoodenTable_01/WoodenTable_01_8-bit_Diffuse.png"),
	    &Path::new("./assets/textures/WoodenTable_01/WoodenTable_01_8-bit_Normal.png"),
	    &Path::new("./assets/textures/WoodenTable_01/WoodenTable_01_8-bit_Properties.png"),
	)?;
        let materials = vec![Rc::new(material), Rc::new(material2), Rc::new(material3), Rc::new(material4)];
        // let materials = vec![Rc::new(material)];

	//println!("Loading model...");
        //let model = models::Model::load(Path::new(MODEL_PATH))?;
	//let model = models::ModelNonIndexed::load(Path::new(MODEL_PATH))?;
	let model = models::ModelNonIndexed::load(Path::new("./assets/models/WoodenTable_01.obj"))?;
	//let model = models::ModelNonIndexed::load(Path::new("cube.obj"))?;
        let vertex_buffer = VertexBuffer::new(&device, model.get_vertices())?;
        //let index_buffer = IndexBuffer::new(&device, model.get_indices())?;


	let mut viking_room_geometry_set = StaticGeometrySet::new(
	    &device,
	    global_descriptor_sets.clone(),
	    Rc::new(vertex_buffer),
	    None,
	    //Some(Rc::new(index_buffer)),
	    &materials,
	    max_frames_in_flight,
	)?;
	viking_room_geometry_set.add(
	    &device,
	    Matrix4::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), Deg(90.0)) * Matrix4::from_scale(1.0),
	)?;

	/*viking_room_geometry_set.add(
	    &device,
	    Matrix4::from_translation(Vector3::new(0.0, 0.0, 5.0)) * Matrix4::from_scale(0.5),
	)?;*/

	let set_layouts = [
	    &global_descriptor_set_layout,
	    viking_room_geometry_set.get_type_layout(),
	    viking_room_geometry_set.get_instance_layout(),
	];

	let static_geometry_pipeline = Rc::new(RefCell::new(Pipeline::new(
	    &device,
	    WINDOW_WIDTH,
	    WINDOW_HEIGHT,
	    &render_pass,
	    vert_shader,
	    frag_shader,
	    &set_layouts,
	    PipelineParameters::new()
		.with_msaa_samples(msaa_samples)
		.with_cull_mode(vk::CullModeFlags::NONE)
		.with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
		.with_depth_test()
		.with_depth_write()
		.with_depth_compare_op(vk::CompareOp::LESS)
		.with_subpass(0),
	)?));

	let viking_room_set = Rc::new(StaticGeometrySetRenderer::new(
	    static_geometry_pipeline.clone(),
	    viking_room_geometry_set,
	));

	let renderables: Vec<Rc<dyn Renderable>> = vec![
	    viking_room_set.clone(),
	];

	let scene = Scene::new(
	    renderables,
	);

	let attachment_set = AttachmentSet::for_renderpass(
	    &render_pass,
	    WINDOW_WIDTH,
	    WINDOW_HEIGHT,
	    msaa_samples,
	)?;

	// Begin HDR setup
        let vert_shader_hdr: VertexShader<NullVertex> =
            VertexShader::from_spv_file(
                &device,
                Path::new("./hdr.vert.spv"),
            )?;
        let frag_shader_hdr = FragmentShader::from_spv_file(
            &device,
            Path::new("./hdr.frag.spv"),
        )?;

	let hdr_descriptor_set_layout = DescriptorSetLayout::new(
	    &device,
	    vec![
		vk::DescriptorSetLayoutBinding{
		    binding: 0,
		    descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
		    descriptor_count: 1,
		    stage_flags: vk::ShaderStageFlags::FRAGMENT,
		    p_immutable_samplers: ptr::null(),
		},
		vk::DescriptorSetLayoutBinding{
		    binding: 1,
		    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		    descriptor_count: 1,
		    stage_flags: vk::ShaderStageFlags::FRAGMENT,
		    p_immutable_samplers: ptr::null(),
		},
	    ],
	)?;

	let hdr_uniform_buffer_set = UniformBufferSet::new(
	    &device,
	    HdrControlUniform{
		exposure: 1.0,
		gamma: 2.2,
		algo: 2,
	    },
	    max_frames_in_flight,
	)?;

	let mut hdr_descriptor_sets = vec![];
	for frame_idx in 0..max_frames_in_flight {
	    let items: Vec<Box<dyn DescriptorRef>> = vec![
		// Texture 0 is the texture we wrote in HDR format.
		Box::new(InputAttachmentRef::new(
		    attachment_set.get(&render_target_color),
		    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
		)),
		Box::new(UniformBufferRef::new(vec![
		    hdr_uniform_buffer_set.get_buffer(frame_idx)?,
		])),
	    ];
	    let sets = global_pool.create_descriptor_sets(
		1,
		&hdr_descriptor_set_layout,
		&items,
	    )?;
	    hdr_descriptor_sets.push(sets[0]);
	}

	let set_layouts_hdr = [&hdr_descriptor_set_layout];

	let hdr_resolve_pipeline = Rc::new(RefCell::new(Pipeline::new(
	    &device,
	    WINDOW_WIDTH,
	    WINDOW_HEIGHT,
	    &render_pass,
	    vert_shader_hdr,
	    frag_shader_hdr,
	    &set_layouts_hdr,
	    PipelineParameters::new()
		.with_cull_mode(vk::CullModeFlags::FRONT)
		.with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
		.with_subpass(1),
	)?));

	let hdr = PostProcessingStep::new(
	    hdr_descriptor_sets,
	    hdr_resolve_pipeline.clone(),
	);

	// End HDR setup

        Ok({
	    let mut this = VulkanApp21{
		device,
		presenter,
		render_pass,
		global_pool,
		global_descriptor_set_layout,
		global_uniform_buffer_set,
		static_geometry_pipelines: vec![static_geometry_pipeline],
		viking_room_set,
		global_descriptor_sets,
		scene,
		attachment_set,
		render_target_color,
		postprocessing_pipelines: vec![hdr_resolve_pipeline],
		hdr,
		hdr_uniform_buffer_set,
		hdr_descriptor_set_layout,
		command_buffers: Vec::new(),
		materials,
		_model: model,
		msaa_samples,

		is_framebuffer_resized: false,
		yaw_speed: 0.0,
		pitch_speed: 0.0,
		roll_speed: 0.0,
		camera_speed: [0.0, 0.0, 0.0].into(),
		view_dir: Vector4::new(0.0, 1.0, 0.0, 1.0).normalize(),
		view_up: Vector4::new(0.0, 0.0, 1.0, 1.0).normalize(),
		view_right: Vector4::new(1.0, 0.0, 0.0, 1.0).normalize(),
            };
	    this.rebuild_command_buffers()?;
	    this
	})
    }

    fn rebuild_command_buffers(&mut self) -> anyhow::Result<()> {
        let max_frames_in_flight = self.presenter.get_num_images();
	self.command_buffers.clear();

	let clear_values = [
            vk::ClearValue{
                color: vk::ClearColorValue{
                    float32: [0.0, 0.0, 0.0, 1.0],
                }
            },
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
        ];

	for frame_idx in 0..max_frames_in_flight {
	    let command_buffer = CommandBuffer::new(
		&self.device,
		vk::CommandBufferLevel::PRIMARY,
		self.device.get_default_graphics_queue(),
	    )?;
	    {
		let buffer_writer = command_buffer.record(
		    vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
		)?;
		{
		    let render_pass_writer = buffer_writer.begin_render_pass(
			&self.presenter,
			&self.render_pass,
			&clear_values,
			&self.attachment_set,
			frame_idx,
		    );
		    self.scene.write_command_buffer(frame_idx, &render_pass_writer)?;
		    render_pass_writer.next_subpass();
		    self.hdr.write_draw_command(frame_idx, &render_pass_writer)?;
		}
	    }
	    self.command_buffers.push(command_buffer);
	}
	Ok(())
    }

    fn submit_command_buffer(&self, idx: usize) -> anyhow::Result<()> {
	if idx > self.command_buffers.len() {
	    return Err(anyhow!(
		"Tried to submit command buffer #{} of {}",
		idx,
		self.command_buffers.len(),
	    ));
	}
	self.presenter.submit_command_buffer(
	    // TODO: idx is out of sync with current_frame!
	    &self.command_buffers[idx],
	    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
	)
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
		uniform_transform.view_pos = {
		    let mut view_pos: Vector4<f32> = uniform_transform.view_pos.into();
                    view_pos += view_up * camera_speed.z * delta_time;
		    view_pos += view_dir * camera_speed.y * delta_time;
		    view_pos += view_right * -camera_speed.x * delta_time;
		    view_pos.w = 1.0;
		    view_pos.into()
		};

		uniform_transform.view = {
		    let view_pos: Vector4<f32> = uniform_transform.view_pos.into();
                    let view_matrix: Matrix4<f32> = Matrix4::look_at_dir(
			Point3::new(
                            view_pos.x,
                            view_pos.y,
                            view_pos.z,
			),
			view_dir.truncate(),
			view_up.truncate(),
                    );
		    view_matrix.into()
		};
		Ok(())
            }).unwrap();
    }

    fn resize(&mut self, width: usize, height: usize) -> anyhow::Result<()> {
	println!("Resizing...");
	for pipeline_cell in self.static_geometry_pipelines.iter_mut() {
	    let mut pipeline = pipeline_cell.borrow_mut();
	    pipeline.update_viewport(
		width,
		height,
		&self.render_pass,
	    )?;
	}
	for pipeline_cell in self.postprocessing_pipelines.iter_mut() {
	    let mut pipeline = pipeline_cell.borrow_mut();
	    pipeline.update_viewport(
		width,
		height,
		&self.render_pass,
	    )?;
	}
	self.attachment_set.resize(&self.render_pass, width, height, self.msaa_samples)?;
	let mut hdr_descriptor_sets = vec![];
	for frame_idx in 0..self.presenter.get_num_images() {
	    let items: Vec<Box<dyn DescriptorRef>> = vec![
		// Texture 0 is the texture we wrote in HDR format.
		Box::new(InputAttachmentRef::new(
		    self.attachment_set.get(&self.render_target_color),
		    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
		)),
		Box::new(UniformBufferRef::new(vec![
		    self.hdr_uniform_buffer_set.get_buffer(frame_idx)?,
		])),
	    ];
	    let sets = self.global_pool.create_descriptor_sets(
		1,
		&self.hdr_descriptor_set_layout,
		&items,
	    )?;
	    hdr_descriptor_sets.push(sets[0]);
	}
	self.hdr.replace_descriptor_sets(hdr_descriptor_sets);
	// TODO: I don't like having this in here, but any resize operation requires
	//       the command buffers to be re-created, right?
	self.rebuild_command_buffers()?;
	Ok(())
    }
}

impl VulkanApp for VulkanApp21 {
    // TODO: lots of repetition here.  I need to find a solution for that.
    fn draw_frame(&mut self) -> anyhow::Result<()>{
	if self.is_framebuffer_resized {
	    self.presenter.fit_to_window(&self.render_pass)?;
	    let (width, height) = self.presenter.get_dimensions();
	    self.is_framebuffer_resized = false;
	    println!("Resizing before doing anything else");
	    self.resize(width, height)?;
	}
	// Hopefully, this will give me the precision I need for the calculation but the
	// compactness and speed I want for the result.
	let mut maybe_new_dimensions = None;
        let since_last_frame = self.presenter.wait_for_next_frame()?;
	let delta_time = ((since_last_frame.as_nanos() as f64) / 1_000_000_000_f64) as f32;
        let image_index = self.presenter.acquire_next_image(
	    &self.render_pass,
	    &mut |width: usize, height: usize| -> anyhow::Result<()> {
		maybe_new_dimensions = Some((width, height));
		Ok(())
	    },
	)?;

	if let Some((width, height)) = maybe_new_dimensions {
	    println!("Resizing after acquiring an image from the swapchain");
	    self.resize(width, height)?;
	}

        self.update_uniform_buffer(image_index as usize, delta_time);

        //self.presenter.submit_graphics_command_buffer(&self.command_buffers[image_index as usize])?;
	self.submit_command_buffer(image_index as usize)?;
	self.presenter.present_frame(
	    image_index,
	    &self.render_pass,
	    &mut |width: usize, height: usize| -> anyhow::Result<()> {
		maybe_new_dimensions = Some((width, height));
		Ok(())
	    },
	)?;
	if let Some((width, height)) = maybe_new_dimensions {
	    println!("Resizing after rendering");
	    self.resize(width, height)?;
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

    fn toggle_parallax(&mut self) -> bool {
        self.global_uniform_buffer_set.update(
	    |uniform_transform: &mut UniformBufferObject| -> anyhow::Result<bool> {
		let val: bool = uniform_transform.use_parallax.into();
		uniform_transform.use_parallax = (!val).into();
		Ok(uniform_transform.use_parallax.into())
            }).unwrap()
    }

    fn toggle_ao(&mut self) -> bool {
        self.global_uniform_buffer_set.update(
	    |uniform_transform: &mut UniformBufferObject| -> anyhow::Result<bool> {
		let val: bool = uniform_transform.use_ao.into();
		uniform_transform.use_ao = (!val).into();
		Ok(uniform_transform.use_ao.into())
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
