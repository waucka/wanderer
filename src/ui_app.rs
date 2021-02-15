use ash::vk;
use memoffset::offset_of;
use glsl_layout::AsStd140;

use std::cell::RefCell;
use std::collections::HashMap;
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
use super::utils::{Vector4f};

const DEBUG_DESCRIPTOR_SETS: bool = false;

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

#[derive(Debug, Default, Clone, Copy, AsStd140)]
pub struct UIUniform {
    tint: Vector4f,
}

struct RenderingSet {
    vertex_buffer: Rc<VertexBuffer<egui::paint::tessellator::Vertex>>,
    index_buffer: Rc<IndexBuffer>,
    descriptor_set: Rc<DescriptorSet>,
}

struct UIRenderingData {
    rendering_sets: Vec<RenderingSet>,
    texture: Rc<Texture>,
    texture_version: u64,
}

struct FrameData {
    rendering_data: Option<UIRenderingData>,
    uniform_buffer: Rc<UniformBuffer<UIUniform>>,
    command_buffer: RefCell<Rc<SecondaryCommandBuffer>>,
}

impl FrameData {
    fn write_command_buffer(
	&self,
	render_pass: &RenderPass,
	subpass: u32,
	pipeline: &Rc<Pipeline<egui::paint::tessellator::Vertex>>,
    ) -> anyhow::Result<()> {
	// reset() waits for the buffer to leave the Pending state.
	self.command_buffer.borrow().reset()?;
	// TODO: does this actually have to be a RefCell?
	// TODO: should these be one-time submit?
	//       One-time submit would make a lot of sense if we were
	//       rendering the UI to a texture and compositing it.
	//       In that case, we would only execute a command buffer
	//       if the texture needed to be updated.  Each command buffer
	//       would only be run once.
	self.command_buffer.borrow().record(
	    vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
	    render_pass,
	    subpass,
	    |writer| {
		writer.join_render_pass(
		    |rp_writer| {
			match &self.rendering_data {
			    Some(ref rendering_data) => {
				rp_writer.bind_pipeline(Rc::clone(pipeline));

				for rs in &rendering_data.rendering_sets {
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
			    },
			    None => (),
			};

			Ok(())
		    },
		)
	    },
	)
    }
}

pub struct UIAppRenderer {
    frame_data: PerFrameSet<FrameData>,
    descriptor_pools: PerFrameSet<DescriptorPool>,
    descriptor_set_layout: DescriptorSetLayout,
    command_pool: Rc<CommandPool>,
    sampler: Rc<Sampler>,
    pipeline: Rc<Pipeline<egui::paint::tessellator::Vertex>>,
    uniform: UIUniform,
}

impl UIAppRenderer {
    const POOL_SIZE: u32 = super::support::MAX_FRAMES_IN_FLIGHT as u32 * 4;

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

	//TODO: need to set this up for "render stuff uploaded during previous frame" workflow!
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
		.with_cull_mode(vk::CullModeFlags::FRONT)
		.with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
		.with_subpass(subpass),
	)?);

	let command_pool = CommandPool::new(
	    device,
	    graphics_queue,
	    true,
	    true,
	)?;

	let sampler = Rc::new(Sampler::new(
	    device,
	    1,
	    vk::Filter::LINEAR,
	    vk::Filter::LINEAR,
	    vk::SamplerMipmapMode::NEAREST,
	    vk::SamplerAddressMode::REPEAT,
	)?);

	let frame_data = PerFrameSet::new(
	    |_frame| {
		let command_buffer = SecondaryCommandBuffer::new(
		    device,
		    Rc::clone(&command_pool),
		)?;
		let frame_data = FrameData{
		    rendering_data: None,
		    uniform_buffer: Rc::new(UniformBuffer::new(device, None)?),
		    command_buffer: RefCell::new(command_buffer),
		};
		frame_data.write_command_buffer(
		    render_pass,
		    subpass,
		    &pipeline,
		)?;
		Ok(frame_data)
	    },
	)?;

	Ok(Self{
	    frame_data,
	    descriptor_pools,
	    descriptor_set_layout,
	    command_pool,
	    sampler,
	    pipeline,
	    uniform: UIUniform {
		tint: Vector4f::from([1.0, 1.0, 1.0, 1.0]),
	    },
	})
    }

    // TODO: texture management needs refinement.  For example, Triangles has a texture_id member.
    fn set_app_data(
	&mut self,
	device: &Device,
	frame: FrameId,
	jobs: &egui::paint::tessellator::PaintJobs,
	egui_texture: &Arc<egui::paint::Texture>
    ) -> anyhow::Result<()> {
	let frame_data = self.frame_data.get_mut(frame);

	let texture = Rc::new(Texture::from_egui(
	    device,
	    egui_texture,
	)?);
	let mut rendering_sets = vec![];
	for (_, triangles) in jobs.iter() {
	    let vertex_buffer = Rc::new(VertexBuffer::new(device, &triangles.vertices)?);
	    let index_buffer = Rc::new(IndexBuffer::new(device, &triangles.indices)?);
	    let items: Vec<Box<dyn DescriptorRef>> = vec![
		Box::new(UniformBufferRef::new(vec![Rc::clone(&frame_data.uniform_buffer)])),
		Box::new(CombinedRef::new(
		    Rc::clone(&self.sampler),
		    vec![Rc::clone(&texture)],
		)),
	    ];

	    if DEBUG_DESCRIPTOR_SETS {
		println!("Creating type descriptor set with {} items...", items.len());
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

	frame_data.rendering_data = Some(UIRenderingData{
	    rendering_sets,
	    texture,
	    texture_version: egui_texture.version,
	});

	Ok(())
    }

    pub fn update_app_data(
	&mut self,
	device: &Device,
	frame: FrameId,
	jobs: &egui::paint::tessellator::PaintJobs,
	egui_texture: &Arc<egui::paint::Texture>,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<()> {
	let uniform_buffer = Rc::clone(&self.frame_data.get_mut(frame).uniform_buffer);
	match &mut self.frame_data.get_mut(frame).rendering_data {
	    None => {
		self.set_app_data(
		    device,
		    frame,
		    jobs,
		    egui_texture,
		)?;
		Ok(())
	    },
	    Some(ref mut rendering_data) => {
		if rendering_data.texture_version != egui_texture.version {
		    rendering_data.texture = Rc::new(Texture::from_egui(
			device,
			egui_texture,
		    )?);
		}

		// TODO: reuse the buffers if possible
		let mut rendering_sets = vec![];
		for (_, triangles) in jobs.iter() {
		    let vertex_buffer = Rc::new(VertexBuffer::new(device, &triangles.vertices)?);
		    let index_buffer = Rc::new(IndexBuffer::new(device, &triangles.indices)?);

		    let items: Vec<Box<dyn DescriptorRef>> = vec![
			Box::new(UniformBufferRef::new(vec![
			    Rc::clone(&uniform_buffer),
			])),
			Box::new(CombinedRef::new(
			    Rc::clone(&self.sampler),
			    vec![Rc::clone(&rendering_data.texture)],
			)),
		    ];

		    if DEBUG_DESCRIPTOR_SETS {
			println!("Creating type descriptor sets with {} items...", items.len());
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
		rendering_data.rendering_sets = rendering_sets;
		Ok(())
	    },
	}
    }

    // TODO: should this also target the next frame?
    pub fn remove_app_data(&mut self, frame: FrameId) {
	self.frame_data.get_mut(frame).rendering_data = None;
    }

    pub fn get_command_buffer(
	&self,
	frame: FrameId,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<Rc<SecondaryCommandBuffer>> {
	let pipeline = &self.pipeline;
	self.frame_data.get(frame).write_command_buffer(
	    render_pass,
	    subpass,
	    pipeline,
	)?;
	Ok(Rc::clone(&self.frame_data.get(frame).command_buffer.borrow()))
    }

    pub fn sync_uniform_buffers(&self, frame: FrameId) -> anyhow::Result<()> {
	self.frame_data.get(frame).uniform_buffer.update(&self.uniform)
    }

    pub fn update_pipeline_viewport(
	&self,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: &RenderPass,
    ) -> anyhow::Result<()> {
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
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
        ]
    }
}
