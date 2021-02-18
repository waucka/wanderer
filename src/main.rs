use winit::event_loop::EventLoop;
use ash::vk;
use cgmath::{Deg, Matrix4, Point3, Vector3, Vector4, InnerSpace};
use glsl_layout::{AsStd140};

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

mod platforms;
mod window;
mod debug;
mod utils;
mod support;
mod scene;
mod objects;
mod models;
mod ui_app;

use window::VulkanApp;
use models::{/*Model,*/ ModelNonIndexed};
use support::{Device, DeviceBuilder, PerFrameSet, FrameId};
use support::command_buffer::{CommandBuffer, SecondaryCommandBuffer};
use support::descriptor::{
    DescriptorBindings,
    DescriptorPool,
    DescriptorSet,
    DescriptorSetLayout,
    DescriptorRef,
    InputAttachmentRef,
    UniformBufferRef,
};
use support::renderer::{
    Presenter,
    RenderPass,
    RenderPassBuilder,
    AttachmentDescription,
    AttachmentSet,
    AttachmentRef,
    Subpass,
    SubpassRef,
};
use support::texture::{Material};
use support::buffer::{VertexBuffer/*, IndexBuffer*/, UniformBuffer};
use objects::{StaticGeometryRenderer, PostProcessingStep};
use scene::{Scene, Renderable};
use ui_app::{UIApp, UIAppRenderer};
use utils::{Matrix4f, Vector4f};

const WINDOW_TITLE: &'static str = "Wanderer";
const WINDOW_WIDTH: usize = 1024;
const WINDOW_HEIGHT: usize = 768;
//const MODEL_PATH: &'static str = "viking_room.obj";
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

impl UniformBufferObject {
    fn get_twiddler_data(&self) -> Rc<ui_app::UniformData> {
	let mut items: HashMap<String, Box<dyn ui_app::UniformDataItem>> = HashMap::new();
	items.insert(
	    "use_parallax".to_owned(),
	    Box::new(ui_app::UniformDataItemBool::new(self.use_parallax.into())),
	);
	items.insert(
	    "use_ao".to_owned(),
	    Box::new(ui_app::UniformDataItemBool::new(self.use_ao.into())),
	);
	Rc::new(ui_app::UniformData::new(items))
    }

    fn set_data(&mut self, twiddler: Rc<ui_app::UniformTwiddler>) {
	let uniform_data = twiddler.get_uniform_data();
	let items = uniform_data.get_items();

	if let Some(item) = items.get("use_parallax") {
	    if let ui_app::UniformDataVar::Bool(use_parallax) = item.get_value() {
		self.use_parallax = use_parallax.into();
	    }
	}

	if let Some(item) = items.get("use_ao") {
	    if let ui_app::UniformDataVar::Bool(use_ao) = item.get_value() {
		self.use_ao = use_ao.into();
	    }
	}
    }
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

impl HdrControlUniform {
    fn get_twiddler_data(&self) -> Rc<ui_app::UniformData> {
	let mut items: HashMap<String, Box<dyn ui_app::UniformDataItem>> = HashMap::new();
	items.insert(
	    "exposure".to_owned(),
	    Box::new(ui_app::UniformDataItemSliderSFloat::new(self.exposure, 0.0..=10.0)),
	);
	items.insert(
	    "gamma".to_owned(),
	    Box::new(ui_app::UniformDataItemSliderSFloat::new(self.gamma, 1.0..=5.0)),
	);
	items.insert(
	    "algo".to_owned(),
	    Box::new(ui_app::UniformDataItemRadio::new(
		self.algo,
		vec![
		    ("No-op".to_owned(), 0),
		    ("Linear".to_owned(), 1),
		    ("Reinhard simple".to_owned(), 2),
		    ("Invalid".to_owned(), 9001),
		],
	    )),
	);
	Rc::new(ui_app::UniformData::new(items))
    }

    fn set_data(&mut self, twiddler: Rc<ui_app::UniformTwiddler>) {
	let uniform_data = twiddler.get_uniform_data();
	let items = uniform_data.get_items();

	if let Some(item) = items.get("exposure") {
	    if let ui_app::UniformDataVar::SFloat(exposure) = item.get_value() {
		self.exposure = exposure;
	    }
	}

	if let Some(item) = items.get("gamma") {
	    if let ui_app::UniformDataVar::SFloat(gamma) = item.get_value() {
		self.gamma = gamma;
	    }
	}

	if let Some(item) = items.get("algo") {
	    if let ui_app::UniformDataVar::UInt(algo) = item.get_value() {
		self.algo = algo;
	    }
	}
    }
}

struct SecondaryBufferSet {
    scene: Vec<Rc<SecondaryCommandBuffer>>,
    hdr: Rc<SecondaryCommandBuffer>,
    ui: Rc<SecondaryCommandBuffer>,
}

impl Clone for SecondaryBufferSet {
    fn clone(&self) -> Self {
	Self{
	    scene: self.scene.clone(),
	    hdr: Rc::clone(&self.hdr),
	    ui: Rc::clone(&self.ui),
	}
    }
}


pub struct UIManager {
    renderer: RefCell<UIAppRenderer>,
    apps: Vec<Rc<dyn UIApp>>,
    app_context: ui_app::AppContext,
    egui_ctx: egui::CtxRef,
}

impl UIManager {
    pub fn new(
	device: &Device,
	window_width: usize,
	window_height: usize,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<Self> {
	let egui_ctx = egui::CtxRef::default();
	let mut style: egui::style::Style = egui::style::Style::clone(&egui_ctx.style());
	style.visuals.widgets.noninteractive.bg_fill = egui::paint::color::Color32::from_rgba_unmultiplied(
	    255, 128, 128, 192,
	);
	egui_ctx.set_style(style);
	Ok(Self{
	    renderer: RefCell::new(UIAppRenderer::new(
		device,
		window_width,
		window_height,
		render_pass,
		subpass,
		device.get_default_graphics_queue(),
	    )?),
	    apps: Vec::new(),
	    app_context: ui_app::AppContext::new(),
	    egui_ctx,
	})
    }

    pub fn get_window_scale(&self) -> f32 {
	self.egui_ctx.pixels_per_point()
    }

    pub fn update_app_data(
	&mut self,
	device: &Device,
	frame: FrameId,
	raw_input: egui::RawInput,
    ) -> anyhow::Result<()> {
	self.egui_ctx.begin_frame(raw_input);
	for app in self.apps.iter() {
	    app.update(&self.egui_ctx, &mut self.app_context);
	}
	let (_egui_output, paint_commands) = self.egui_ctx.end_frame();
	let egui_texture = self.egui_ctx.texture();
	let paint_jobs = self.egui_ctx.tessellate(paint_commands);
	self.renderer.borrow_mut().add_frame_data(
	    device,
	    frame,
	    &paint_jobs,
	    &egui_texture,
	)?;
	Ok(())
    }

    fn get_command_buffer(
	&self,
	device: &Device,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<Rc<SecondaryCommandBuffer>> {
	self.renderer.borrow_mut().create_command_buffer(
	    device,
	    render_pass,
	    subpass,
	)
    }

    fn update_pipeline_viewport(
	&self,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: &RenderPass,
    ) -> anyhow::Result<()> {
	self.renderer.borrow_mut().update_pipeline_viewport(
	    viewport_width,
	    viewport_height,
	    render_pass,
	)
    }

    fn add_app(
	&mut self,
	app: Rc<dyn UIApp>,
    ) {
	self.apps.push(app);
    }

    fn clear_apps(&mut self) {
	self.apps.clear();
    }

    fn has_apps(&self) -> bool {
	!self.apps.is_empty()
    }
}

struct GlobalFrameData {
    descriptor_set: Rc<DescriptorSet>,
    uniform_buffer: Rc<UniformBuffer<UniformBufferObject>>,
}

struct HdrFrameData {
    descriptor_set: Rc<DescriptorSet>,
    uniform_buffer: Rc<UniformBuffer<HdrControlUniform>>,
}

struct VulkanApp21 {
    device: Device,
    presenter: Presenter,
    render_pass: RenderPass,
    #[allow(unused)]
    global_pool: DescriptorPool,
    #[allow(unused)]
    global_descriptor_set_layout: DescriptorSetLayout,
    global_frame_data: PerFrameSet<GlobalFrameData>,
    global_uniform: UniformBufferObject,
    hdr_control_uniform: HdrControlUniform,
    #[allow(unused)]
    scene: Scene,
    attachment_set: AttachmentSet,
    render_target_color: AttachmentRef,
    hdr: PostProcessingStep,
    hdr_frame_data: PerFrameSet<HdrFrameData>,
    hdr_descriptor_set_layout: DescriptorSetLayout,
    ui_manager: UIManager,
    materials: Vec<Rc<Material>>,
    _model: ModelNonIndexed,
    msaa_samples: vk::SampleCountFlags,

    rendering_subpass: SubpassRef,
    postprocessing_subpass: SubpassRef,
    ui_subpass: SubpassRef,

    uniform_twiddler_app: Rc<ui_app::UniformTwiddler>,
    hdr_twiddler_app: Rc<ui_app::UniformTwiddler>,

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
	let rendering_subpass = render_pass_builder.add_subpass({
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
	let postprocessing_subpass = render_pass_builder.add_subpass({
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
	// UI subpass
	let ui_subpass = render_pass_builder.add_subpass({
	    let mut subpass = Subpass::new(vk::PipelineBindPoint::GRAPHICS);
	    subpass.add_color_attachment(
		presentation_target.as_vk(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL),
		None,
	    );
	    subpass
	});
	render_pass_builder.add_standard_entry_dep();
	render_pass_builder.add_dep(vk::SubpassDependency{
	    src_subpass: rendering_subpass.into(),
	    dst_subpass: postprocessing_subpass.into(),
	    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
	    dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
	    src_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
	    dst_access_mask: vk::AccessFlags::SHADER_READ,
	    dependency_flags: vk::DependencyFlags::empty(),
	});
	render_pass_builder.add_dep(vk::SubpassDependency{
	    src_subpass: postprocessing_subpass.into(),
	    dst_subpass: ui_subpass.into(),
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
            60,
        )?;
        let max_frames_in_flight = support::MAX_FRAMES_IN_FLIGHT;
	dbg!(max_frames_in_flight);
        device.check_mipmap_support(vk::Format::R8G8B8A8_UNORM)?;

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
	let global_descriptor_bindings = DescriptorBindings::new()
	    .with_binding(
		vk::DescriptorType::UNIFORM_BUFFER,
		1,
		vk::ShaderStageFlags::ALL,
		false,
	    );
	let global_descriptor_set_layout = DescriptorSetLayout::new(
	    &device,
	    global_descriptor_bindings,
	)?;

	let global_uniform = UniformBufferObject{
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
	};

	let uniform_twiddler_app = Rc::new(
	    ui_app::UniformTwiddler::new(
		"Global Uniform",
		global_uniform.get_twiddler_data(),
	    )
	);

	let global_frame_data = PerFrameSet::new(
	    |_| {
		let uniform_buffer = Rc::new(UniformBuffer::new(
		    &device,
		    Some(&global_uniform),
		)?);
		let items: Vec<Rc<dyn DescriptorRef>> = vec![
		    Rc::new(UniformBufferRef::new(vec![Rc::clone(&uniform_buffer)])),
		];

		let sets = global_pool.create_descriptor_sets(
		    1,
		    &global_descriptor_set_layout,
		    &items,
		)?;
		Ok(GlobalFrameData{
		    descriptor_set: Rc::clone(&sets[0]),
		    uniform_buffer 
		})
	    },
	)?;

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

	let mut viking_room_geometry = StaticGeometryRenderer::new(
	    &device,
	    &global_descriptor_set_layout,
	    global_frame_data.extract(
		|frame_data| {
		    Ok(Rc::clone(&frame_data.descriptor_set))
		}
	    )?,
	    Rc::new(vertex_buffer),
	    None,
	    //Some(Rc::new(index_buffer)),
	    &materials,
	    WINDOW_WIDTH,
	    WINDOW_HEIGHT,
	    &render_pass,
	    rendering_subpass.into(),
	    msaa_samples,
	)?;
	viking_room_geometry.add(
	    &device,
	    Matrix4::from_axis_angle(Vector3::new(1.0, 0.0, 0.0), Deg(90.0)) * Matrix4::from_scale(1.0),
	)?;

	/*viking_room_geometry_set.add(
	    &device,
	    Matrix4::from_translation(Vector3::new(0.0, 0.0, 5.0)) * Matrix4::from_scale(0.5),
	)?;*/

	let renderables: Vec<Rc<dyn Renderable>> = vec![
	    Rc::new(viking_room_geometry),
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

	let hdr_descriptor_bindings = DescriptorBindings::new()
	    .with_binding(
		vk::DescriptorType::INPUT_ATTACHMENT,
		1,
		vk::ShaderStageFlags::FRAGMENT,
		false,
	    )
	    .with_binding(
		vk::DescriptorType::UNIFORM_BUFFER,
		1,
		vk::ShaderStageFlags::FRAGMENT,
		false,
	    );
	let hdr_descriptor_set_layout = DescriptorSetLayout::new(
	    &device,
	    hdr_descriptor_bindings,
	)?;

	let hdr_control_uniform = HdrControlUniform{
	    exposure: 1.0,
	    gamma: 2.2,
	    algo: 2,
	};

	let hdr_twiddler_app = Rc::new(
	    ui_app::UniformTwiddler::new(
		"HDR Lighting",
		hdr_control_uniform.get_twiddler_data(),
	    )
	);

	let hdr_frame_data = PerFrameSet::new(
	    |_| {
		let uniform_buffer = Rc::new(UniformBuffer::new(
		    &device,
		    Some(&hdr_control_uniform),
		)?);

		let items: Vec<Rc<dyn DescriptorRef>> = vec![
		    Rc::new(InputAttachmentRef::new(
			attachment_set.get(&render_target_color),
			vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
		    )),
		    Rc::new(UniformBufferRef::new(vec![Rc::clone(&uniform_buffer)])),
		];
		let sets = global_pool.create_descriptor_sets(
		    1,
		    &hdr_descriptor_set_layout,
		    &items,
		)?;

		Ok(HdrFrameData{
		    descriptor_set: Rc::clone(&sets[0]),
		    uniform_buffer,
		})
	    },
	)?;

	let hdr = PostProcessingStep::new(
	    &device,
	    &hdr_descriptor_set_layout,
	    hdr_frame_data.extract(
		|frame_data| {
		    Ok(Rc::clone(&frame_data.descriptor_set))
		}
	    )?,
	    WINDOW_WIDTH,
	    WINDOW_HEIGHT,
	    &render_pass,
	    postprocessing_subpass.into(),
	)?;

	// End HDR setup

	let ui_manager = UIManager::new(
	    &device,
	    WINDOW_WIDTH,
	    WINDOW_HEIGHT,
	    &render_pass,
	    ui_subpass.into(),
	)?;

        Ok({
	    let mut this = VulkanApp21{
		device,
		presenter,
		render_pass,
		global_pool,
		global_descriptor_set_layout,
		global_frame_data,
		global_uniform,
		hdr_control_uniform,
		scene,
		attachment_set,
		render_target_color,
		hdr,
		hdr_frame_data,
		hdr_descriptor_set_layout,
		ui_manager,
		materials,
		_model: model,
		msaa_samples,

		rendering_subpass,
		postprocessing_subpass,
		ui_subpass,

		uniform_twiddler_app,
		hdr_twiddler_app,

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
	self.scene.rebuild_command_buffers(
	    &self.device,
	    &self.render_pass,
	    self.rendering_subpass.into(),
	)?;
	self.hdr.rebuild_command_buffers(
	    &self.device,
	    &self.render_pass,
	    self.postprocessing_subpass.into(),
	)?;
	Ok(())
    }

    fn get_secondary_buffers(
	&self,
	frame: FrameId,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<SecondaryBufferSet> {
	Ok(SecondaryBufferSet {
	    scene: self.scene.get_command_buffers(frame)?,
	    hdr: self.hdr.get_command_buffer(frame)?,
	    ui: self.ui_manager.get_command_buffer(
		&self.device,
		render_pass,
		subpass,
	    )?,
	})
    }

    fn update_uniform_buffer(&mut self, delta_time: f32) -> anyhow::Result<()> {
	let frame = self.presenter.get_current_frame();
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

	let uniform_transform = &mut self.global_uniform;
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

	uniform_transform.set_data(Rc::clone(&self.uniform_twiddler_app));
	self.hdr_control_uniform.set_data(Rc::clone(&self.hdr_twiddler_app));

	self.global_frame_data.get(frame).uniform_buffer.update(uniform_transform)?;
	self.hdr_frame_data.get(frame).uniform_buffer.update(&self.hdr_control_uniform)?;
	Ok(())
    }

    fn resize(&mut self, width: usize, height: usize) -> anyhow::Result<()> {
	println!("Resizing...");
	self.scene.update_pipeline_viewports(
	    width,
	    height,
	    &self.render_pass,
	)?;
	self.hdr.update_pipeline_viewport(
	    width,
	    height,
	    &self.render_pass,
	)?;
	self.ui_manager.update_pipeline_viewport(
	    width,
	    height,
	    &self.render_pass,
	)?;

	self.attachment_set.resize(&self.render_pass, width, height, self.msaa_samples)?;
	let hdr = &mut self.hdr;
	let global_pool = &mut self.global_pool;
	let hdr_descriptor_set_layout = &self.hdr_descriptor_set_layout;
	let render_target_color = &self.attachment_set.get(&self.render_target_color);
	let hdr_frame_data = &self.hdr_frame_data;
	hdr.replace_descriptor_sets(
	    |frame, _| {
		let items: Vec<Rc<dyn DescriptorRef>> = vec![
		    Rc::new(InputAttachmentRef::new(
			Rc::clone(render_target_color),
			vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
		    )),
		    Rc::new(UniformBufferRef::new(vec![
			Rc::clone(&hdr_frame_data.get(frame).uniform_buffer),
		    ])),
		];
		let sets = global_pool.create_descriptor_sets(
		    1,
		    hdr_descriptor_set_layout,
		    &items,
		)?;
		Ok(Rc::clone(&sets[0]))
	    }
	)?;
	// TODO: I don't like having this in here, but any resize operation requires
	//       the command buffers to be re-created, right?
	self.rebuild_command_buffers()?;
	Ok(())
    }
}

impl VulkanApp for VulkanApp21 {
    // TODO: move sync stuff into Presenter
    fn draw_frame(&mut self, raw_input: egui::RawInput) -> anyhow::Result<()>{
	if self.is_framebuffer_resized {
	    self.presenter.fit_to_window(&self.render_pass)?;
	    let (width, height) = self.presenter.get_dimensions();
	    self.is_framebuffer_resized = false;
	    println!("Resizing before doing anything else");
	    self.resize(width, height)?;
	}

	let mut maybe_new_dimensions = None;
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

	let frame = self.presenter.get_current_frame();

	self.ui_manager.update_app_data(
	    &self.device,
	    frame.next(),
	    raw_input,
	)?;

	// Fetch the command buffers for this frame
	let SecondaryBufferSet{
	    scene: scene_buffers,
	    hdr: hdr_buffer,
	    ui: ui_buffer,
	} = self.get_secondary_buffers(
	    frame,
	    &self.render_pass,
	    self.ui_subpass.into(),
	)?;

	// Create the current frame's command buffer
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

	let mut command_buffer = CommandBuffer::new(
	    &self.device,
	    self.device.get_default_graphics_pool(),
	)?;
	command_buffer.record(
	    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
	    |primary_writer| {
		primary_writer.begin_render_pass(
		    &self.presenter,
		    &self.render_pass,
		    &clear_values,
		    &self.attachment_set,
		    image_index as usize,
		    true,
		    |primary_rp_writer| {
			primary_rp_writer.execute_commands(&scene_buffers);
			primary_rp_writer.next_subpass(true);
			primary_rp_writer.execute_commands(&[Rc::clone(&hdr_buffer)]);
			primary_rp_writer.next_subpass(true);
			primary_rp_writer.execute_commands(&[Rc::clone(&ui_buffer)]);
			Ok(())
		    }
		)?;
		Ok(())
	    }
	)?;
	// Hopefully, this will give me the precision I need for the calculation but the
	// compactness and speed I want for the result.
        let since_last_frame = self.presenter.wait_for_next_frame()?;
	let delta_time = ((since_last_frame.as_nanos() as f64) / 1_000_000_000_f64) as f32;

        self.update_uniform_buffer(delta_time)?;
	self.presenter.submit_command_buffer(
	    &command_buffer,
	    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
	)?;


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

    fn toggle_uniform_twiddler(&mut self) -> bool {
	if self.ui_manager.has_apps() {
	    self.ui_manager.clear_apps();
	    false
	} else {
	    let global_uniform = Rc::clone(&self.uniform_twiddler_app);
	    let hdr_uniform = Rc::clone(&self.hdr_twiddler_app);
	    self.ui_manager.add_app(global_uniform);
	    self.ui_manager.add_app(hdr_uniform);
	    true
	}
    }

    fn get_window_size(&self) -> (usize, usize) {
	self.device.get_window_size()
    }

    fn get_window_scale(&self) -> f32 {
	self.ui_manager.get_window_scale()
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
