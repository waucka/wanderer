use ash::vk;
use memoffset::offset_of;
use glsl_layout::AsStd140;

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use super::support::{Device, PerFrameSet, FrameId, Queue};
use super::support::buffer::{VertexBuffer, IndexBuffer, UniformBuffer};
use super::support::command_buffer::{SecondaryCommandBuffer, CommandPool};
use super::support::descriptor::{
    DescriptorBindings,
    DescriptorPool,
    DescriptorSetLayout,
    DescriptorSet,
    DescriptorRef,
    UniformBufferRef,
    CombinedRef,
};
use super::support::renderer::{Pipeline, PipelineParameters, RenderPass};
use super::support::shader::{VertexShader, FragmentShader};
use super::support::texture::{Texture, Sampler};
use super::utils::Vector2f;

const DEBUG_DESCRIPTOR_SETS: bool = false;

#[derive(Debug, Default, Clone, Copy, AsStd140)]
pub struct UIData {
    #[allow(unused)]
    window_size: Vector2f,
}

pub trait UIApp {
    fn name(&self) -> &str;
    fn update(&self, ctx: &egui::CtxRef, app_ctx: &mut AppContext);
}

pub struct AppContext {
    should_quit: bool,
}

impl AppContext {
    pub fn new() -> Self {
	Self{
	    should_quit: false,
	}
    }

    pub fn signal_quit(&mut self) {
	self.should_quit = true;
    }

    pub fn quit_signaled(&self) -> bool {
	self.should_quit
    }
}

pub struct UniformTwiddler {
    label: RefCell<String>,
    value: RefCell<f32>,
    painting: RefCell<Painting>,
}

impl Default for UniformTwiddler {
    fn default() -> Self {
	Self{
	    label: RefCell::new("Hello World!".to_owned()),
	    value: RefCell::new(2.7),
	    painting: RefCell::new(Default::default()),
	}
    }
}

impl UIApp for UniformTwiddler {
    fn name(&self) -> &str {
	"Uniform Twiddler"
    }

    fn update(&self, ctx: &egui::CtxRef, app_ctx: &mut AppContext) {
	// TODO: find a better way to accomplish this.
	let mut label = self.label.borrow_mut();
	let mut value = self.value.borrow_mut();
	let mut painting = self.painting.borrow_mut();

	egui::SidePanel::left("side_panel", 200.0).show(ctx, |ui| {
	    ui.heading("Side Panel");

	    ui.horizontal(|ui| {
		ui.label("Write something: ");
		ui.text_edit_singleline(&mut label);
	    });

	    ui.add(egui::Slider::f32(&mut value, 0.0..=10.0).text("value"));
	    if ui.button("Increment").clicked {
		*value += 1.0;
	    }

	    ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
		ui.add(
		    egui::Hyperlink::new("https://github.com/emilk/egui").text("powered by egui"),
		);
	    });
	});

	egui::TopPanel::top("top_panel").show(ctx, |ui| {
	    egui::menu::bar(ui, |ui| {
		egui::menu::menu(ui, "File", |ui| {
		    if ui.button("Quit").clicked {
			app_ctx.signal_quit();
		    }
		});
	    });
	});

	egui::CentralPanel::default().show(ctx, |ui| {
	    ui.heading("Uniform Twiddler");
	    egui::warn_if_debug_build(ui);

	    ui.separator();

	    ui.heading("Central Panel");
	    ui.label("The central panel is the region left after adding TopPanels and SidePanels");
	    ui.label("It is often a great place for big things, like drawings:");

	    ui.heading("Draw with your mouse:");
	    painting.ui_control(ui);
	    egui::Frame::dark_canvas(ui.style()).show(ui, |ui| {
		painting.ui_content(ui);
	    });
	});
    }
}

struct Painting {
    lines: Vec<Vec<egui::Vec2>>,
    stroke: egui::Stroke,
}

impl Default for Painting {
    fn default() -> Self {
	Self{
	    lines: Default::default(),
	    stroke: egui::Stroke::new(1.0, egui::Color32::LIGHT_BLUE),
	}
    }
}

impl Painting {
    pub fn ui_control(&mut self, ui: &mut egui::Ui) -> egui::Response {
	ui.horizontal(|ui| {
	    self.stroke.ui(ui, "Stroke");
	    ui.separator();
	    if ui.button("Clear Painting").clicked {
		self.lines.clear()
	    }
	}).1
    }

    pub fn ui_content(&mut self, ui: &mut egui::Ui) -> egui::Response {
	let (response, painter) = ui.allocate_painter(ui.available_size_before_wrap_finite(), egui::Sense::drag());
	let rect = response.rect;

	if self.lines.is_empty() {
	    self.lines.push(vec![]);
	}

	let current_line = self.lines.last_mut().unwrap();

	if response.active {
	    if let Some(mouse_pos) = ui.input().mouse.pos {
		let canvas_pos = mouse_pos - rect.min;
		if current_line.last() != Some(&canvas_pos) {
		    current_line.push(canvas_pos);
		}
	    }
	} else if !current_line.is_empty() {
	    self.lines.push(vec![]);
	}

	for line in &self.lines {
	    if line.len() >= 2 {
		let points: Vec<egui::Pos2> = line.iter().map(|p| rect.min + *p).collect();
		painter.add(egui::PaintCmd::line(points, self.stroke));
	    }
	}

	response
    }
}

struct RenderingSet {
    vertex_buffer: Rc<VertexBuffer<egui::paint::tessellator::Vertex>>,
    index_buffer: Rc<IndexBuffer>,
    descriptor_set: Rc<DescriptorSet>,
}

struct FrameData {
    rendering_sets: Vec<RenderingSet>,
    sampler: Rc<Sampler>,
    texture: Rc<Texture>,
    texture_version: u64,
}

impl FrameData {
    fn create_command_buffer(
	&self,
	device: &Device,
	pool: Rc<CommandPool>,
	render_pass: &RenderPass,
	subpass: u32,
	pipeline: &Rc<Pipeline<egui::paint::tessellator::Vertex>>,
    ) -> anyhow::Result<Rc<SecondaryCommandBuffer>> {
	let command_buffer = SecondaryCommandBuffer::new(
	    device,
	    Rc::clone(&pool),
	)?;

	command_buffer.record(
	    vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE |
	    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
	    render_pass,
	    subpass,
	    |writer| {
		writer.join_render_pass(
		    |rp_writer| {
			rp_writer.bind_pipeline(Rc::clone(pipeline));

			for rs in &self.rendering_sets {
			    let descriptor_sets = [
				Rc::clone(&rs.descriptor_set),
			    ];

			    if DEBUG_DESCRIPTOR_SETS {
				println!("Binding descriptor sets...");
				println!("\tSet 0: {:?}", descriptor_sets[0]);
			    }

			    rp_writer.bind_descriptor_sets(pipeline.get_layout(), &descriptor_sets);
			    rp_writer.draw_indexed(
				Rc::clone(&rs.vertex_buffer),
				Rc::clone(&rs.index_buffer),
			    );
			}

			Ok(())
		    },
		)
	    },
	)?;
	Ok(command_buffer)
    }
}

pub struct UIAppRenderer {
    frame_data: VecDeque<FrameData>,
    descriptor_pools: PerFrameSet<DescriptorPool>,
    descriptor_set_layout: DescriptorSetLayout,
    uniform: UIData,
    uniform_buffers: PerFrameSet<Rc<UniformBuffer<UIData>>>,
    command_pool: Rc<CommandPool>,
    pipeline: Rc<Pipeline<egui::paint::tessellator::Vertex>>,
}

impl UIAppRenderer {
    const POOL_SIZE: u32 = super::support::MAX_FRAMES_IN_FLIGHT as u32 * 8;

    pub fn new(
	device: &Device,
	window_width: usize,
	window_height: usize,
	render_pass: &RenderPass,
	subpass: u32,
	graphics_queue: Rc<Queue>,
    ) -> anyhow::Result<Self> {
	let descriptor_pools = {
	    let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
	    pool_sizes.insert(
		vk::DescriptorType::UNIFORM_BUFFER,
		Self::POOL_SIZE / 2,
	    );
	    pool_sizes.insert(
		vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		Self::POOL_SIZE / 2,
	    );
	    PerFrameSet::new(|_| {
		DescriptorPool::new(
		    device,
		    pool_sizes.clone(),
		    Self::POOL_SIZE,
		)
	    })?
	};

	let descriptor_bindings = DescriptorBindings::new()
	    .with_binding(
		vk::DescriptorType::UNIFORM_BUFFER,
		1,
		vk::ShaderStageFlags::ALL,
		false,
	    )
	    .with_binding(
		vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		1,
		vk::ShaderStageFlags::ALL,
		false,
	    );

	let descriptor_set_layout = DescriptorSetLayout::new(
	    device,
	    descriptor_bindings,
	)?;

	let set_layouts = [&descriptor_set_layout];

        let vert_shader: VertexShader<egui::paint::tessellator::Vertex> =
            VertexShader::from_spv_file(
                device,
                Path::new("./ui.vert.spv"),
            )?;
        let frag_shader = FragmentShader::from_spv_file(
            device,
            Path::new("./ui.frag.spv"),
        )?;

	let pipeline = Rc::new(Pipeline::new(
	    &device,
	    window_width,
	    window_height,
	    render_pass,
	    vert_shader,
	    frag_shader,
	    &set_layouts,
	    PipelineParameters::new()
		.with_cull_mode(vk::CullModeFlags::NONE)
		.with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
		.with_subpass(subpass),
	)?);

	let command_pool = CommandPool::new(
	    device,
	    graphics_queue,
	    false,
	    true,
	)?;

	let frame_data = VecDeque::new();

	let uniform = UIData{
	    window_size: Vector2f::new(window_width as f32, window_height as f32),
	};
	let uniform_buffers = PerFrameSet::new(|_| {
	    Ok(Rc::new(UniformBuffer::new(
		device,
		Some(&uniform),
	    )?))
	})?;

	Ok(Self{
	    frame_data,
	    descriptor_pools,
	    descriptor_set_layout,
	    uniform,
	    uniform_buffers,
	    command_pool,
	    pipeline,
	})
    }

    // TODO: texture management needs refinement.  For example, Triangles has a texture_id member.
    pub fn add_frame_data(
	&mut self,
	device: &Device,
	frame: FrameId,
	jobs: &egui::paint::tessellator::PaintJobs,
	egui_texture: &Arc<egui::paint::Texture>,
    ) -> anyhow::Result<()> {
	self.descriptor_pools.get_mut(frame).reset()?;
	let uniform_buffer = self.uniform_buffers.get(frame);
	uniform_buffer.update(&self.uniform)?;

	let sampler = Rc::new(Sampler::new(
	    device,
	    1,
	    vk::Filter::LINEAR,
	    vk::Filter::LINEAR,
	    vk::SamplerMipmapMode::NEAREST,
	    vk::SamplerAddressMode::REPEAT,
	)?);
	let texture = Rc::new(Texture::from_egui(
	    device,
	    egui_texture,
	)?);

	let mut rendering_sets = vec![];
	for (_, triangles) in jobs.iter() {
	    let vertex_buffer = Rc::new(VertexBuffer::new(device, &triangles.vertices)?);
	    let index_buffer = Rc::new(IndexBuffer::new(device, &triangles.indices)?);
	    let items: Vec<Rc<dyn DescriptorRef>> = vec![
		Rc::new(UniformBufferRef::new(
		    vec![Rc::clone(uniform_buffer)],
		)),
		Rc::new(CombinedRef::new(
		    Rc::clone(&sampler),
		    vec![Rc::clone(&texture)],
		)),
	    ];

	    if DEBUG_DESCRIPTOR_SETS {
		println!("Creating app descriptor set with {} items...", items.len());
	    }
	    let sets = self.descriptor_pools.get_mut(frame).create_descriptor_sets(
		1,
		&self.descriptor_set_layout,
		&items,
	    )?;
	    rendering_sets.push(RenderingSet{
		vertex_buffer,
		index_buffer,
		descriptor_set: Rc::clone(&sets[0]),
	    });
	}

	self.frame_data.push_back(FrameData{
	    rendering_sets,
	    sampler,
	    texture,
	    texture_version: egui_texture.version,
	});

	Ok(())
    }

    pub fn clear_frame_data(&mut self) {
	self.frame_data.clear();
    }

    fn create_blank_command_buffer(
	&self,
	device: &Device,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<Rc<SecondaryCommandBuffer>> {
	let command_buffer = SecondaryCommandBuffer::new(
	    device,
	    Rc::clone(&self.command_pool),
	)?;

	command_buffer.record(
	    vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE |
	    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
	    render_pass,
	    subpass,
	    |writer| {
		writer.join_render_pass(
		    |_rp_writer| {
			Ok(())
		    }
		)
	    }
	)?;
	Ok(command_buffer)
    }

    pub fn create_command_buffer(
	&mut self,
	device: &Device,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<Rc<SecondaryCommandBuffer>> {
	if self.frame_data.len() < 2 {
	    // Don't use data that might not have been uploaded yet.
	    // This should only happen on the first frame after starting.
	    return self.create_blank_command_buffer(
		device,
		render_pass,
		subpass,
	    );
	}

	// By popping the frame data, we should ensure that all frame dependencies get
	// dropped when the command buffer goes away.
	// Everything that needs to stick around *should* be listed as a dependency of
	// the command buffer, so it won't be dropped until the command buffer is.
	match self.frame_data.pop_front() {
	    Some(command_buffer) => command_buffer.create_command_buffer(
		device,
		Rc::clone(&self.command_pool),
		render_pass,
		subpass,
		&self.pipeline,
	    ),
	    None => {
		self.create_blank_command_buffer(
		    device,
		    render_pass,
		    subpass,
		)
	    },
	}
    }

    pub fn update_pipeline_viewport(
	&mut self,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: &RenderPass,
    ) -> anyhow::Result<()> {
	self.uniform.window_size = Vector2f::new(viewport_width as f32, viewport_height as f32);
	self.pipeline.update_viewport(
	    viewport_width,
	    viewport_height,
	    render_pass,
	)
    }
}

impl super::support::shader::Vertex for egui::paint::tessellator::Vertex {
    fn get_binding_description() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription{
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
	vec![
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 1,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, uv) as u32,
            },
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 2,
                format: vk::Format::R8G8B8A8_UINT,
                offset: offset_of!(Self, color) as u32,
            },
        ]
    }
}
