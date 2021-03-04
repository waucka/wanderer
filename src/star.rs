use ash::vk;
use cgmath::{Vector3, Matrix4};
use glsl_layout::Uniform;

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use super::support::{Device, PerFrameSet, FrameId};
use super::support::buffer::{VertexBuffer, IndexBuffer, UniformBuffer};
use super::support::command_buffer::{SecondaryCommandBuffer};
use super::support::descriptor::{
    DescriptorBindings,
    DescriptorPool,
    DescriptorSet,
    DescriptorSetLayout,
    CombinedRef,
    DescriptorRef,
    UniformBufferRef,
};
use super::support::renderer::{Pipeline, PipelineParameters, RenderPass};
use super::support::shader::{SimpleVertex, VertexShader, FragmentShader};
use super::support::texture::{Texture, Sampler};
use super::scene::Renderable;
use super::utils::{Vector4f, Matrix4f};

const DEBUG_DESCRIPTOR_SETS: bool = false;

#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct StarInfo {
    transform: Matrix4f,
    center: Vector4f,
    radius: f32,
    luminosity: f32,
    temperature: u32,
    heat_color_lower_bound: u32,
    heat_color_upper_bound: u32,
}

impl StarInfo {
    pub fn new(center: Vector3<f32>, radius: f32, temperature: u32, luminosity: f32) -> Self {
        Self{
            transform: {
                (Matrix4::from_translation(center) *
                 Matrix4::from_scale(radius))
                    .into()
            },
            center: center.extend(1.0).into(),
            radius,
            luminosity,
            temperature,
            heat_color_lower_bound: 0,
            heat_color_upper_bound: 40000,
        }
    }
}

struct StarRenderingData {
    instance_descriptor_set: Rc<DescriptorSet>,
    uniform_buffer: Rc<UniformBuffer<StarInfo>>,
}

pub struct StarRenderer {
    global_descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
    instance_descriptor_set_layout: DescriptorSetLayout,
    vertex_buffer: Rc<VertexBuffer<SimpleVertex>>,
    index_buffer: Rc<IndexBuffer>,
    stars: Vec<StarRenderingData>,
    texture_ref: Rc<dyn DescriptorRef>,
    instance_pool: DescriptorPool,
    pipeline: Rc<Pipeline<SimpleVertex>>,
    // TODO: ARGH!  I don't like using a RefCell here!
    command_buffers: RefCell<PerFrameSet<Rc<SecondaryCommandBuffer>>>,
}

pub type StarId = usize;

impl StarRenderer {
    const POOL_SIZE: u32 = 8;

    pub fn new(
        device: &Device,
        global_descriptor_set_layout: &DescriptorSetLayout,
        global_descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
        heat_texture: Rc<Texture>,
        window_width: usize,
        window_height: usize,
        render_pass: &RenderPass,
        subpass: u32,
        msaa_samples: vk::SampleCountFlags,
    ) -> anyhow::Result<Self> {
        // Load shaders first; the files not being present is the most likely cause of failure in this function.
        let vert_shader: VertexShader<SimpleVertex> =
            VertexShader::from_spv_file(
                device,
                Path::new("./assets/shaders/star.vert.spv"),
            )?;
        let frag_shader = FragmentShader::from_spv_file(
            device,
            Path::new("./assets/shaders/star.frag.spv"),
        )?;

        let (vertex_buffer, index_buffer) = {
            // These vertices and indices come from a cube I created in Blender and exported to .obj format.
            let vertices = vec![
                SimpleVertex::new( 1.0,  1.0, -1.0, 1.0),
                SimpleVertex::new( 1.0, -1.0, -1.0, 1.0),
                SimpleVertex::new( 1.0,  1.0,  1.0, 1.0),
                SimpleVertex::new( 1.0, -1.0,  1.0, 1.0),
                SimpleVertex::new(-1.0,  1.0, -1.0, 1.0),
                SimpleVertex::new(-1.0, -1.0, -1.0, 1.0),
                SimpleVertex::new(-1.0,  1.0,  1.0, 1.0),
                SimpleVertex::new(-1.0, -1.0,  1.0, 1.0),
            ];
            let indices = vec![
                0, 4, 6,
                0, 6, 2,
                3, 2, 6,
                3, 6, 7,
                7, 6, 4,
                7, 4, 5,
                5, 1, 3,
                5, 3, 7,
                1, 0, 2,
                1, 2, 3,
                5, 4, 0,
                5, 0, 1,
            ];
            (
                Rc::new(VertexBuffer::new(device, &vertices)?),
                Rc::new(IndexBuffer::new(device, &indices)?),
            )
        };

        let instance_descriptor_bindings = DescriptorBindings::new()
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
                true
            );

        let instance_descriptor_set_layout = DescriptorSetLayout::new(
            device,
            instance_descriptor_bindings,
        )?;

        let sampler = Sampler::new(
            device,
            1,
            vk::Filter::LINEAR,
            vk::Filter::LINEAR,
            vk::SamplerMipmapMode::NEAREST,
            vk::SamplerAddressMode::CLAMP_TO_EDGE,
            0,
        )?;

        let texture_ref = Rc::new(CombinedRef::new(
            Rc::new(sampler),
            vec![heat_texture],
        ));

        let instance_pool = {
            let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
            pool_sizes.insert(
                vk::DescriptorType::UNIFORM_BUFFER,
                Self::POOL_SIZE / 2,
            );
            pool_sizes.insert(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                Self::POOL_SIZE / 2,
            );
            DescriptorPool::new(
                device,
                pool_sizes,
                Self::POOL_SIZE,
            )?
        };

        let set_layouts = [
            global_descriptor_set_layout,
            &instance_descriptor_set_layout,
        ];

        let pipeline = Rc::new(Pipeline::new(
            &device,
            window_width,
            window_height,
            render_pass,
            vert_shader,
            frag_shader,
            &set_layouts,
            PipelineParameters::new()
                .with_msaa_samples(msaa_samples)
                .with_cull_mode(vk::CullModeFlags::BACK)
                .with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .with_depth_test()
                .with_depth_write()
                .with_depth_compare_op(vk::CompareOp::LESS)
                .with_subpass(subpass),
        )?);

        let objects = Vec::new();
        let command_buffers = RefCell::new(PerFrameSet::new(
            |frame| {
                let command_buffer = SecondaryCommandBuffer::new(
                    device,
                    device.get_default_graphics_pool(),
                )?;
                StarRenderer::write_command_buffer(
                    &command_buffer,
                    frame,
                    render_pass,
                    subpass,
                    &global_descriptor_sets,
                    &vertex_buffer,
                    &index_buffer,
                    &objects,
                    &pipeline,
                )?;
                Ok(command_buffer)
            },
        )?);

        Ok(Self{
            global_descriptor_sets,
            instance_descriptor_set_layout,
            vertex_buffer,
            index_buffer,
            instance_pool,
            stars: Vec::new(),
            texture_ref,
            pipeline,
            command_buffers,
        })
    }

    pub fn add(
        &mut self,
        device: &Device,
        star_info: &StarInfo,
    ) -> anyhow::Result<StarId> {
        let uniform_buffer = Rc::new(UniformBuffer::new(
            device,
            Some(star_info),
        )?);

        let items: Vec<Rc<dyn DescriptorRef>> = vec![
            Rc::new(UniformBufferRef::new(vec![Rc::clone(&uniform_buffer)])),
            Rc::clone(&self.texture_ref),
        ];

        if DEBUG_DESCRIPTOR_SETS {
            println!("Creating instance descriptor sets with {} items...", items.len());
        }
        let sets = self.instance_pool.create_descriptor_sets(
            1,
            &self.instance_descriptor_set_layout,
            &items,
        )?;
        let instance_descriptor_set = Rc::clone(&sets[0]);

        let star_id = self.stars.len();
        self.stars.push(StarRenderingData{
            instance_descriptor_set,
            uniform_buffer,
        });
        Ok(star_id)
    }

    #[allow(unused)]
    pub fn clear(&mut self) {
        self.stars.clear();
    }

    pub fn get_instance_layout(&self) -> &DescriptorSetLayout {
        &self.instance_descriptor_set_layout
    }

    fn write_command_buffer(
        command_buffer: &Rc<SecondaryCommandBuffer>,
        frame: FrameId,
        render_pass: &RenderPass,
        subpass: u32,
        global_descriptor_sets: &PerFrameSet<Rc<DescriptorSet>>,
        vertex_buffer: &Rc<VertexBuffer<SimpleVertex>>,
        index_buffer: &Rc<IndexBuffer>,
        stars: &[StarRenderingData],
        pipeline: &Rc<Pipeline<SimpleVertex>>,
    ) -> anyhow::Result<()> {
        command_buffer.record(
            vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            render_pass,
            subpass,
            |writer| {
                writer.join_render_pass(
                    |rp_writer| {
                        rp_writer.bind_pipeline(Rc::clone(pipeline));
                        for star in stars.iter() {
                            let descriptor_sets = [
                                Rc::clone(global_descriptor_sets.get(frame)),
                                Rc::clone(&star.instance_descriptor_set),
                            ];
                            if DEBUG_DESCRIPTOR_SETS {
                                println!("Binding descriptor sets...");
                                println!("\tSet 0: {:?}", descriptor_sets[0]);
                                println!("\tSet 1: {:?}", descriptor_sets[1]);
                            }
                            rp_writer.bind_descriptor_sets(pipeline.get_layout(), &descriptor_sets);
                            rp_writer.draw_indexed(
                                Rc::clone(&vertex_buffer),
                                Rc::clone(index_buffer),
                            );
                        }
                        Ok(())
                    },
                )
            },
        )
    }
}

impl Renderable for StarRenderer {
    fn get_command_buffer(&self, frame: FrameId) -> anyhow::Result<Rc<SecondaryCommandBuffer>> {
        Ok(Rc::clone(self.command_buffers.borrow().get(frame)))
    }

    fn rebuild_command_buffers(
        &self,
        device: &Device,
        render_pass: &RenderPass,
        subpass: u32,
    ) -> anyhow::Result<()> {
        let global_descriptor_sets = &self.global_descriptor_sets;
        let vertex_buffer = &self.vertex_buffer;
        let index_buffer = &self.index_buffer;
        let stars = &self.stars;
        let pipeline = &self.pipeline;
        self.command_buffers.borrow_mut().replace(
            |frame, _| {
                let command_buffer = SecondaryCommandBuffer::new(
                    device,
                    device.get_default_graphics_pool(),
                )?;
                StarRenderer::write_command_buffer(
                    &command_buffer,
                    frame,
                    render_pass,
                    subpass,
                    global_descriptor_sets,
                    vertex_buffer,
                    index_buffer,
                    stars,
                    pipeline,
                )?;
                Ok(command_buffer)
            },
        )
    }

    fn sync_uniform_buffers(&self, _frame: FrameId) -> anyhow::Result<()> {
        Ok(())
    }

    fn update_pipeline_viewport(
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
