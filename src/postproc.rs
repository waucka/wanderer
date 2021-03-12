use ash::vk;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::support::{Device, PerFrameSet, FrameId, Queue};
use super::support::command_buffer::{SecondaryCommandBuffer, RenderPassWriter, CommandPool};
use super::support::descriptor::{
    DescriptorBindings,
    DescriptorPool,
    DescriptorSet,
    DescriptorSetLayout,
    DescriptorRef,
    InputAttachmentRef,
};
use super::support::renderer::{PipelineParameters, RenderPass, SubpassRef};
use super::support::texture::Texture;

pub struct PostProcResources {
    _command_pool: Rc<CommandPool>,
    pools: [DescriptorPool; 2],
    current_pool: usize,
    descriptor_set_layout: DescriptorSetLayout,
    descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
    command_buffers: RefCell<PerFrameSet<Rc<SecondaryCommandBuffer>>>,
}

impl PostProcResources {
    pub fn new(
        device: &Device,
        pixel_sources: &PerFrameSet<Rc<Texture>>,
        queue: Rc<Queue>,
    ) -> anyhow::Result<Self> {
        let descriptor_bindings = DescriptorBindings::new()
            .with_binding(
                vk::DescriptorType::INPUT_ATTACHMENT,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                false,
            );
        let descriptor_set_layout = DescriptorSetLayout::new(
            device,
            descriptor_bindings,
        )?;

        let command_pool = CommandPool::new(
            &device,
            queue,
            true,
            false,
        )?;

        let mut pools = [
            {
                let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
                pool_sizes.insert(
                    vk::DescriptorType::INPUT_ATTACHMENT,
                    2,
                );
                DescriptorPool::new(
                    "Post-processing",
                    device,
                    pool_sizes,
                    2,
                )?
            },
            {
                let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
                pool_sizes.insert(
                    vk::DescriptorType::INPUT_ATTACHMENT,
                    2,
                );
                DescriptorPool::new(
                    "Post-processing",
                    device,
                    pool_sizes,
                    2,
                )?
            }
        ];

        let descriptor_sets = Self::build_descriptor_sets(
            &mut pools[0],
            &descriptor_set_layout,
            pixel_sources,
        )?;

        let command_buffers = RefCell::new(PerFrameSet::new(
            |_| {
                SecondaryCommandBuffer::new(device, Rc::clone(&command_pool))
            })?);

        Ok(Self{
            _command_pool: command_pool,
            pools,
            current_pool: 0,
            descriptor_set_layout,
            descriptor_sets,
            command_buffers,
        })
    }

    fn build_descriptor_sets(
        pool: &mut DescriptorPool,
        descriptor_set_layout: &DescriptorSetLayout,
        pixel_sources: &PerFrameSet<Rc<Texture>>,
    ) -> anyhow::Result<PerFrameSet<Rc<DescriptorSet>>> {
        let descriptor_sets = PerFrameSet::new(
            |frame| {
                let items: Vec<Rc<dyn DescriptorRef>> = vec![
                    Rc::new(InputAttachmentRef::new(
                        Rc::clone(pixel_sources.get(frame)),
                        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    )),
                ];
                let sets = pool.create_descriptor_sets(
                    1,
                    descriptor_set_layout,
                    &items,
                )?;

                Ok(Rc::clone(&sets[0]))
            }
        )?;
        Ok(descriptor_sets)
    }

    pub fn update_pixel_sources(
        &mut self,
        pixel_sources: &PerFrameSet<Rc<Texture>>,
    ) -> anyhow::Result<()> {
        self.current_pool = (self.current_pool + 1) % 2;
        self.pools[self.current_pool].reset()?;
        self.descriptor_sets = Self::build_descriptor_sets(
            &mut self.pools[self.current_pool],
            &self.descriptor_set_layout,
            pixel_sources,
        )?;
        Ok(())
    }

    pub fn get_descriptor_set(&self, frame: FrameId) -> Rc<DescriptorSet> {
        Rc::clone(self.descriptor_sets.get(frame))
    }

    pub fn get_descriptor_set_layout(&self) -> &DescriptorSetLayout {
        &self.descriptor_set_layout
    }

    pub fn rebuild_command_buffers(
        &self,
        postproc: &dyn PostProcStep,
        render_pass: &RenderPass,
        subpass: SubpassRef,
    ) -> anyhow::Result<()> {
        self.command_buffers.borrow_mut().foreach(
            |frame, command_buffer| {
                command_buffer.reset()?;
                command_buffer.record(
                    vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE,
                    render_pass,
                    subpass,
                    |writer| {
                        // TODO: obviate the need for this by having an alternative to
                        //       CommandBuffer::record that does a continue.
                        writer.join_render_pass(
                            |rp_writer| {
                                postproc.bind_resources(
                                    frame,
                                    self.get_descriptor_set(frame),
                                    rp_writer,
                                )?;
                                Self::write_draw_command(rp_writer);
                                Ok(())
                            },
                        )
                    },
                )
            }
        )?;
        Ok(())
    }

    pub fn get_command_buffer(
        &self,
        frame: FrameId,
    ) -> Rc<SecondaryCommandBuffer> {
        Rc::clone(self.command_buffers.borrow().get(frame))
    }

    pub fn write_draw_command(
        rp_writer: &mut RenderPassWriter,
    ) {
        rp_writer.draw_no_vbo(3, 1);
    }

    pub fn get_pipeline_parameters(subpass: SubpassRef) -> PipelineParameters {
        PipelineParameters::new(subpass)
            .with_cull_mode(vk::CullModeFlags::FRONT)
            .with_front_face(vk::FrontFace::COUNTER_CLOCKWISE)
    }
}

pub trait PostProcStep {
    fn bind_resources(
        &self,
        frame: FrameId,
        rt_descriptor_set: Rc<DescriptorSet>,
        writer: &mut RenderPassWriter,
    ) -> anyhow::Result<()>;
    fn update_viewport(
        &mut self,
        pixel_sources: &PerFrameSet<Rc<Texture>>,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: &RenderPass,
        subpass: SubpassRef,
    ) -> anyhow::Result<()>;
    fn get_command_buffer(
        &self,
        frame: FrameId,
    ) -> Rc<SecondaryCommandBuffer>;
}
