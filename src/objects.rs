use ash::vk;
use anyhow::anyhow;
use cgmath::Matrix4;
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
use super::support::shader::{Vertex, VertexShader, FragmentShader};
use super::support::texture::Material;
use super::scene::Renderable;
use super::utils::{NullVertex, Vector4f, Matrix4f};

const DEBUG_DESCRIPTOR_SETS: bool = false;

// TODO: get rid of static geometry type UBO after validating that
//       this approach works.

#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct StaticGeometryTypeUBO {
    tint: Vector4f,
}

pub struct StaticGeometryRenderer<V: Vertex> {
    global_descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
    type_descriptor_set_layout: DescriptorSetLayout,
    type_descriptor_set: Rc<DescriptorSet>,
    instance_descriptor_set_layout: DescriptorSetLayout,
    vertex_buffer: Rc<VertexBuffer<V>>,
    index_buffer: Option<Rc<IndexBuffer>>,
    uniform_buffer: Rc<UniformBuffer<StaticGeometryTypeUBO>>,
    type_pool: DescriptorPool,
    instance_pool: DescriptorPool,
    objects: Vec<StaticGeometry>,
    pipeline: Rc<Pipeline<V>>,
    // TODO: ARGH!  I don't like using a RefCell here!
    command_buffers: RefCell<PerFrameSet<Rc<SecondaryCommandBuffer>>>,
}

pub type StaticGeometryId = usize;

impl<V: Vertex + 'static> StaticGeometryRenderer<V> {
    const NUM_UNIFORM_BUFFERS_PER_INSTANCE: u32 = 1;
    const MAX_TEXTURES: u32 = 256;
    const NUM_IMAGES_PER_TEXTURE: u32 = 3;
    const POOL_SIZE: u32 = 1024;

    pub fn new(
        device: &Device,
        global_descriptor_set_layout: &DescriptorSetLayout,
        global_descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
        vertex_buffer: Rc<VertexBuffer<V>>,
        index_buffer: Option<Rc<IndexBuffer>>,
        materials: &[Rc<Material>],
        window_width: usize,
        window_height: usize,
        render_pass: &RenderPass,
        subpass: u32,
        msaa_samples: vk::SampleCountFlags,
    ) -> anyhow::Result<Self> {
        // Load shaders first; the files not being present is the most likely cause of failure in this function.
        let vert_shader: VertexShader<V> =
            VertexShader::from_spv_file(
                device,
                Path::new("./assets/shaders/lighting.vert.spv"),
            )?;
        let frag_shader = FragmentShader::from_spv_file(
            device,
            Path::new("./assets/shaders/lighting.frag.spv"),
        )?;

        let uniform_buffer = Rc::new(UniformBuffer::new(
            device,
            Some(&StaticGeometryTypeUBO{
                tint: [1.0, 1.0, 1.0, 1.0].into(),
            }),
        )?);

        let type_descriptor_bindings = DescriptorBindings::new()
            .with_binding(
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::ALL,
                false,
            )
            .with_binding(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                Self::MAX_TEXTURES,
                vk::ShaderStageFlags::ALL,
                true
            )
            .with_binding(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                Self::MAX_TEXTURES,
                vk::ShaderStageFlags::ALL,
                true
            )
            .with_binding(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                Self::MAX_TEXTURES,
                vk::ShaderStageFlags::ALL,
                true
            );

        let type_descriptor_set_layout = DescriptorSetLayout::new(
            device,
            type_descriptor_bindings,
        )?;

        let instance_descriptor_bindings = DescriptorBindings::new()
            .with_binding(
                vk::DescriptorType::UNIFORM_BUFFER,
                Self::NUM_UNIFORM_BUFFERS_PER_INSTANCE,
                vk::ShaderStageFlags::ALL,
                false,
            );

        let instance_descriptor_set_layout = DescriptorSetLayout::new(
            device,
            instance_descriptor_bindings,
        )?;

        let mut type_pool = {
            let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
            pool_sizes.insert(
                vk::DescriptorType::UNIFORM_BUFFER,
                Self::POOL_SIZE * Self::NUM_UNIFORM_BUFFERS_PER_INSTANCE,
            );
            pool_sizes.insert(
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                Self::MAX_TEXTURES * Self::NUM_IMAGES_PER_TEXTURE,
            );
            DescriptorPool::new(
                "Static Geometry Type",
                device,
                pool_sizes,
                Self::POOL_SIZE,
            )?
        };

        let (samplers, textures_color, textures_normal, textures_props) = {
            let mut samplers = Vec::new();
            let mut textures_color = Vec::new();
            let mut textures_normal = Vec::new();
            let mut textures_props = Vec::new();
            for material in materials.iter() {
                let sampler = material.get_sampler();
                let color = material.get_color_texture();
                let normal = material.get_normal_texture();
                let props = material.get_properties_texture();
                samplers.push(sampler);
                textures_color.push(color);
                textures_normal.push(normal);
                textures_props.push(props);
            }
            for _ in materials.len()..(Self::MAX_TEXTURES as usize) {
                let material = &materials[0];
                let sampler = material.get_sampler();
                let color = material.get_color_texture();
                let normal = material.get_normal_texture();
                let props = material.get_properties_texture();
                samplers.push(sampler);
                textures_color.push(color);
                textures_normal.push(normal);
                textures_props.push(props);
            }
            if samplers.len() > (Self::MAX_TEXTURES as usize) {
                panic!("{} samplers for {} textures!", samplers.len(), Self::MAX_TEXTURES);
            }
            (samplers, textures_color, textures_normal, textures_props)
        };

        let mut items: Vec<Rc<dyn DescriptorRef>> = vec![
            Rc::new(UniformBufferRef::new(vec![Rc::clone(&uniform_buffer)])),
        ];
        items.push(Rc::new(CombinedRef::new_per(
            samplers.clone(),
            textures_color.clone(),
        )?));
        items.push(Rc::new(CombinedRef::new_per(
            samplers.clone(),
            textures_normal.clone(),
        )?));
        items.push(Rc::new(CombinedRef::new_per(
            samplers.clone(),
            textures_props.clone(),
        )?));

        if DEBUG_DESCRIPTOR_SETS {
            println!("Creating type descriptor sets with {} items...", items.len());
        }
        let sets = type_pool.create_descriptor_sets(
            1,
            &type_descriptor_set_layout,
            &items,
        )?;
        let type_descriptor_set = Rc::clone(&sets[0]);

        let instance_pool = {
            let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
            pool_sizes.insert(
                vk::DescriptorType::UNIFORM_BUFFER,
                Self::POOL_SIZE * Self::NUM_UNIFORM_BUFFERS_PER_INSTANCE,
            );
            DescriptorPool::new(
                "Static Geometry Instance",
                device,
                pool_sizes,
                Self::POOL_SIZE,
            )?
        };

        let set_layouts = [
            global_descriptor_set_layout,
            &type_descriptor_set_layout,
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
                StaticGeometryRenderer::write_command_buffer(
                    &command_buffer,
                    frame,
                    render_pass,
                    subpass,
                    &global_descriptor_sets,
                    &type_descriptor_set,
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
            type_descriptor_set_layout,
            type_descriptor_set,
            instance_descriptor_set_layout,
            vertex_buffer,
            index_buffer,
            uniform_buffer,
            type_pool,
            instance_pool,
            objects,
            pipeline,
            command_buffers,
        })
    }

    pub fn add(
        &mut self,
        device: &Device,
        model_matrix: Matrix4<f32>,
    ) -> anyhow::Result<StaticGeometryId> {
        let uniform_buffer = Rc::new(UniformBuffer::new(
            device,
            Some(&StaticGeometryInstanceUBO{
                model: model_matrix.into(),
            }),
        )?);

        let items: Vec<Rc<dyn DescriptorRef>> = vec![
            Rc::new(UniformBufferRef::new(vec![Rc::clone(&uniform_buffer)])),
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

        let object_id = self.objects.len();
        self.objects.push(StaticGeometry{
            instance_descriptor_set,
            uniform_buffer,
        });
        Ok(object_id)
    }

    #[allow(unused)]
    pub fn clear(&mut self) {
        self.objects.clear();
    }

    #[allow(unused)]
    pub fn get_mut(&mut self, object_id: StaticGeometryId) -> anyhow::Result<&mut StaticGeometry> {
        if object_id >= self.objects.len() {
            Err(anyhow!("Tried to modify invalid static geometry object {}", object_id))
        } else {
            Ok(&mut self.objects[object_id])
        }
    }

    pub fn get_type_layout(&self) -> &DescriptorSetLayout {
        &self.type_descriptor_set_layout
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
        type_descriptor_set: &Rc<DescriptorSet>,
        vertex_buffer: &Rc<VertexBuffer<V>>,
        index_buffer: &Option<Rc<IndexBuffer>>,
        objects: &[StaticGeometry],
        pipeline: &Rc<Pipeline<V>>,
    ) -> anyhow::Result<()> {
        command_buffer.record(
            vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            render_pass,
            subpass,
            |writer| {
                writer.join_render_pass(
                    |rp_writer| {
                        rp_writer.bind_pipeline(Rc::clone(pipeline));
                        for object in objects.iter() {
                            let descriptor_sets = [
                                Rc::clone(global_descriptor_sets.get(frame)),
                                Rc::clone(type_descriptor_set),
                                Rc::clone(&object.instance_descriptor_set),
                            ];
                            if DEBUG_DESCRIPTOR_SETS {
                                println!("Binding descriptor sets...");
                                println!("\tSet 0: {:?}", descriptor_sets[0]);
                                println!("\tSet 1: {:?}", descriptor_sets[1]);
                                println!("\tSet 2: {:?}", descriptor_sets[2]);
                            }
                            rp_writer.bind_descriptor_sets(pipeline.get_layout(), &descriptor_sets);
                            match &index_buffer {
                                Some(idx_buf) => rp_writer.draw_indexed(
                                    Rc::clone(&vertex_buffer),
                                    Rc::clone(idx_buf),
                                ),
                                None => rp_writer.draw(
                                    Rc::clone(&vertex_buffer),
                                ),
                            };
                        }
                        Ok(())
                    },
                )
            },
        )
    }
}

impl<V: Vertex + 'static> Renderable for StaticGeometryRenderer<V> {
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
        let type_descriptor_set = &self.type_descriptor_set;
        let vertex_buffer = &self.vertex_buffer;
        let index_buffer = &self.index_buffer;
        let objects = &self.objects;
        let pipeline = &self.pipeline;
        self.command_buffers.borrow_mut().replace(
            |frame, _| {
                let command_buffer = SecondaryCommandBuffer::new(
                    device,
                    device.get_default_graphics_pool(),
                )?;
                StaticGeometryRenderer::write_command_buffer(
                    &command_buffer,
                    frame,
                    render_pass,
                    subpass,
                    global_descriptor_sets,
                    type_descriptor_set,
                    vertex_buffer,
                    index_buffer,
                    objects,
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

#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct StaticGeometryInstanceUBO {
    #[allow(unused)]
    model: Matrix4f,
}

pub struct StaticGeometry {
    instance_descriptor_set: Rc<DescriptorSet>,
    uniform_buffer: Rc<UniformBuffer<StaticGeometryInstanceUBO>>,
}

//TODO: This should probably be moved to a new module called "postprocessing" or something.

pub struct PostProcessingStep {
    global_descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
    pipeline: Rc<Pipeline<NullVertex>>,
    command_buffers: RefCell<PerFrameSet<Rc<SecondaryCommandBuffer>>>,
}

impl PostProcessingStep {
    pub fn new(
        device: &Device,
        global_descriptor_set_layout: &DescriptorSetLayout,
        global_descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
        window_width: usize,
        window_height: usize,
        render_pass: &RenderPass,
        subpass: u32,
    ) -> anyhow::Result<Self> {
        let vert_shader: VertexShader<NullVertex> =
            VertexShader::from_spv_file(
                device,
                Path::new("./assets/shaders/hdr.vert.spv"),
            )?;
        let frag_shader = FragmentShader::from_spv_file(
            device,
            Path::new("./assets/shaders/hdr.frag.spv"),
        )?;

        let set_layouts = [global_descriptor_set_layout];

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
                .with_subpass(1),
        )?);

        let command_buffers = RefCell::new(PerFrameSet::new(
            |frame| {
                let command_buffer = SecondaryCommandBuffer::new(
                    device,
                    device.get_default_graphics_pool(),
                )?;
                PostProcessingStep::write_command_buffer(
                    &command_buffer,
                    frame,
                    render_pass,
                    subpass,
                    &global_descriptor_sets,
                    &pipeline,
                )?;
                Ok(command_buffer)
            },
        )?);

        Ok(Self{
            global_descriptor_sets,
            pipeline,
            command_buffers,
        })
    }

    pub fn replace_descriptor_sets<F>(&mut self, replacer: F) -> anyhow::Result<()>
    where
        F: FnMut(FrameId, &Rc<DescriptorSet>) -> anyhow::Result<Rc<DescriptorSet>>
    {
        self.global_descriptor_sets.replace(replacer)
    }

    fn write_command_buffer(
        command_buffer: &Rc<SecondaryCommandBuffer>,
        frame: FrameId,
        render_pass: &RenderPass,
        subpass: u32,
        global_descriptor_sets: &PerFrameSet<Rc<DescriptorSet>>,
        pipeline: &Rc<Pipeline<NullVertex>>,
    ) -> anyhow::Result<()> {
        command_buffer.record(
            vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
            render_pass,
            subpass,
            |writer| {
                writer.join_render_pass(
                    |rp_writer| {
                        rp_writer.bind_pipeline(Rc::clone(pipeline));
                        let descriptor_sets = [
                            Rc::clone(&global_descriptor_sets.get(frame)),
                        ];
                        if DEBUG_DESCRIPTOR_SETS {
                            println!("Binding descriptor sets...");
                            println!("\tSet 0: {:?}", descriptor_sets[0]);
                        }
                        rp_writer.bind_descriptor_sets(pipeline.get_layout(), &descriptor_sets);
                        rp_writer.draw_no_vbo(3, 1);
                        Ok(())
                    },
                )
            },
        )
    }
}

impl Renderable for PostProcessingStep {
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
        let pipeline = &self.pipeline;
        self.command_buffers.borrow_mut().replace(
            |frame, _| {
                let command_buffer = SecondaryCommandBuffer::new(
                    device,
                    device.get_default_graphics_pool(),
                )?;
                PostProcessingStep::write_command_buffer(
                    &command_buffer,
                    frame,
                    render_pass,
                    subpass,
                    global_descriptor_sets,
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
