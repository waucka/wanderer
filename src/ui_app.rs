use ash::vk;
use memoffset::offset_of;
use glsl_layout::Uniform;

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
use super::support::renderer::{Pipeline, PipelineParameters, RenderPass, SubpassRef};
use super::support::shader::{VertexShader, FragmentShader};
use super::support::texture::{Texture, Sampler};
use super::utils::{Vector2f, Vector4f};

const DEBUG_DESCRIPTOR_SETS: bool = false;

#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct UIData {
    #[allow(unused)]
    window_size: Vector2f,
}

pub trait UIApp {
    fn name(&self) -> &str;
    fn update(&self, ctx: &egui::CtxRef, app_ctx: &mut AppContext);
}

pub struct AppContext {
    #[allow(unused)]
    should_quit: bool,
}

impl AppContext {
    pub fn new() -> Self {
        Self{
            should_quit: false,
        }
    }

    #[allow(unused)]
    pub fn signal_quit(&mut self) {
        self.should_quit = true;
    }

    #[allow(unused)]
    pub fn quit_signaled(&self) -> bool {
        self.should_quit
    }
}

pub enum UniformDataVar {
    #[allow(unused)]
    Bool(bool),
    #[allow(unused)]
    UInt(u32),
    #[allow(unused)]
    SFloat(f32),
    #[allow(unused)]
    Vec2(Vector2f),
    #[allow(unused)]
    Vec4(Vector4f),
}

pub trait UniformDataItem {
    fn ui(&mut self, ui: &mut egui::Ui);
    fn get_value(&self) -> UniformDataVar;
}

pub struct UniformDataItemBool {
    value: bool,
}

impl UniformDataItemBool {
    pub fn new(value: bool) -> Self {
        Self{
            value,
        }
    }
}

impl UniformDataItem for UniformDataItemBool {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.checkbox(&mut self.value, "");
    }

    fn get_value(&self) -> UniformDataVar {
        UniformDataVar::Bool(self.value)
    }
}

pub struct UniformDataItemSliderUInt {
    value: u32,
    range: core::ops::RangeInclusive<u32>,
    logarithmic: bool,
}

impl UniformDataItemSliderUInt {
    #[allow(unused)]
    pub fn new(value: u32, range: core::ops::RangeInclusive<u32>, logarithmic: bool) -> Self {
        Self{
            value,
            range,
            logarithmic,
        }
    }
}

impl UniformDataItem for UniformDataItemSliderUInt {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::u32(&mut self.value, self.range.clone())
                .logarithmic(self.logarithmic)
                // This is apparently the only way to get a numeric entry box
                // to the right of the slider.  I have no idea why.
                .text("")
        );
    }

    fn get_value(&self) -> UniformDataVar {
        UniformDataVar::UInt(self.value)
    }
}

pub struct UniformDataItemSliderSFloat {
    value: f32,
    range: core::ops::RangeInclusive<f32>,
    logarithmic: bool,
}

impl UniformDataItemSliderSFloat {
    pub fn new(value: f32, range: core::ops::RangeInclusive<f32>, logarithmic: bool) -> Self {
        Self{
            value,
            range,
            logarithmic,
        }
    }
}

impl UniformDataItem for UniformDataItemSliderSFloat {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::f32(&mut self.value, self.range.clone())
                .logarithmic(self.logarithmic)
                // This is apparently the only way to get a numeric entry box
                // to the right of the slider.  I have no idea why.
                .text("")
        );
    }

    fn get_value(&self) -> UniformDataVar {
        UniformDataVar::SFloat(self.value)
    }
}

pub struct UniformDataItemRadio {
    value: u32,
    options: Vec<(String, u32)>,
}

impl UniformDataItemRadio {
    pub fn new(value: u32, options: Vec<(String, u32)>) -> Self {
        Self{
            value,
            options,
        }
    }
}

impl UniformDataItem for UniformDataItemRadio {
    fn ui(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            for (opt_name, opt_value) in self.options.iter() {
                ui.radio_value(&mut self.value, *opt_value, opt_name);
            }
        });
    }

    fn get_value(&self) -> UniformDataVar {
        UniformDataVar::UInt(self.value)
    }
}

pub struct UniformData {
    items: RefCell<HashMap<String, (String, Box<dyn UniformDataItem>)>>,
    order: RefCell<Vec<String>>,
}

impl UniformData {
    pub fn new() -> Self {
        Self{
            items: RefCell::new(HashMap::new()),
            order: RefCell::new(Vec::new()),
        }
    }

    pub fn add<T: UniformDataItem + 'static>(&mut self, name: &str, description: &str, item: T) {
        self.items.borrow_mut().insert(name.to_owned(), (description.to_owned(), Box::new(item)));
        self.order.borrow_mut().push(name.to_owned());
    }

    pub fn get_order(&self) -> std::cell::Ref<Vec<String>> {
        self.order.borrow()
    }

    fn get_items_mut(&self) -> std::cell::RefMut<HashMap<String, (String, Box<dyn UniformDataItem>)>> {
        self.items.borrow_mut()
    }

    pub fn get_value(&self, name: &str) -> Option<UniformDataVar> {
        match self.items.borrow().get(name) {
            Some((_, item)) => Some(item.get_value()),
            None => None,
        }
    }
}

pub struct UniformTwiddler {
    title: String,
    uniform_data: Rc<UniformData>,
}

impl UniformTwiddler {
    pub fn new(title: &str, uniform_data: Rc<UniformData>) -> Self {
        Self{
            title: title.to_owned(),
            uniform_data,
        }
    }

    pub fn get_uniform_data(&self) -> Rc<UniformData> {
        Rc::clone(&self.uniform_data)
    }
}

impl UIApp for UniformTwiddler {
    fn name(&self) -> &str {
        "Uniform Twiddler"
    }

    fn update(&self, ctx: &egui::CtxRef, _app_ctx: &mut AppContext) {
        let mut items = self.uniform_data.get_items_mut();
        let order = self.uniform_data.get_order();
        egui::Window::new(&self.title)
            .scroll(true)
            .title_bar(true)
            .show(ctx, |ui| {
                egui::Grid::new(format!("{}-grid", &self.title))
                    .striped(true)
                    .max_col_width(ui.available_width())
                    .show(ui, |ui| {
                        for name in order.iter() {
                            match items.get_mut(name) {
                                Some((desc, value)) => {
                                    ui.label(desc.clone());
                                    value.ui(ui);
                                    ui.end_row();
                                },
                                None => (),
                            }
                        }
                    });
            });
    }
}

struct RenderingSet {
    vertex_buffer: Rc<VertexBuffer<egui::paint::Vertex>>,
    index_buffer: Rc<IndexBuffer>,
    descriptor_set: Rc<DescriptorSet>,
}

struct FrameData {
    rendering_sets: Vec<RenderingSet>,
    #[allow(unused)]
    sampler: Rc<Sampler>,
    #[allow(unused)]
    texture: Rc<Texture>,
    #[allow(unused)]
    texture_version: u64,
}

impl FrameData {
    fn create_command_buffer(
        &self,
        device: &Device,
        pool: Rc<CommandPool>,
        render_pass: &RenderPass,
        subpass: SubpassRef,
        pipeline: &Rc<Pipeline<egui::paint::Vertex>>,
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
    pipeline: Rc<Pipeline<egui::paint::Vertex>>,
}

impl UIAppRenderer {
    const POOL_SIZE: u32 = super::support::MAX_FRAMES_IN_FLIGHT as u32 * 8;

    pub fn new(
        device: &Device,
        window_width: usize,
        window_height: usize,
        render_pass: &RenderPass,
        subpass: SubpassRef,
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
                    "UI",
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

        let vert_shader: VertexShader<egui::paint::Vertex> =
            VertexShader::from_spv_file(
                device,
                Path::new("./assets/shaders/ui.vert.spv"),
            )?;
        let frag_shader = FragmentShader::from_spv_file(
            device,
            Path::new("./assets/shaders/ui.frag.spv"),
        )?;

        let pipeline = Rc::new(Pipeline::new(
            &device,
            window_width,
            window_height,
            render_pass,
            vert_shader,
            frag_shader,
            &set_layouts,
            PipelineParameters::new(subpass)
                .with_cull_mode(vk::CullModeFlags::NONE)
                .with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
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
        meshes: &Vec<egui::ClippedMesh>,
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
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            16,
        )?);
        let texture = Rc::new(Texture::from_egui(
            device,
            egui_texture,
        )?);

        let mut rendering_sets = vec![];
        for clipped_mesh in meshes.iter() {
            let mesh = &clipped_mesh.1;
            if mesh.vertices.len() == 0 {
                // I have no idea why egui sometimes gives me meshes with no vertices.
                continue;
            }
            let vertex_buffer = Rc::new(VertexBuffer::new(device, &mesh.vertices)?);
            let index_buffer = Rc::new(IndexBuffer::new(device, &mesh.indices)?);
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

    #[allow(unused)]
    pub fn clear_frame_data(&mut self) {
        self.frame_data.clear();
    }

    fn create_blank_command_buffer(
        &self,
        device: &Device,
        render_pass: &RenderPass,
        subpass: SubpassRef,
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
        subpass: SubpassRef,
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

impl super::support::shader::Vertex for egui::paint::Vertex {
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
