use ash::vk;
use anyhow::anyhow;

use std::rc::Rc;

use super::support::{Device, Queue};
use super::support::command_buffer::{CommandBuffer, RenderPassWriter};
use super::support::renderer::Presenter;

pub trait Renderable {
    fn write_draw_command(&self, idx: usize, writer: &RenderPassWriter) -> anyhow::Result<()>;
    fn sync_uniform_buffers(&self, idx: usize) -> anyhow::Result<()>;
}

pub struct Scene {
    renderables: Vec<Rc<dyn Renderable>>,
    queue: Rc<Queue>,
    num_command_buffers: usize,
    command_buffers: Vec<CommandBuffer>,
}

impl Scene {
    pub fn new(
	device: &Device,
	presenter: &Presenter,
	renderables: Vec<Rc<dyn Renderable>>,
	queue: Rc<Queue>,
	num_command_buffers: usize,
    ) -> anyhow::Result<Self> {
	let mut this = Self{
	    renderables,
	    queue,
	    num_command_buffers,
	    command_buffers: Vec::new(),
	};

	this.rebuild_command_buffers(device, presenter)?;

	Ok(this)
    }

    // This should mainly be useful for handling pipeline updates.
    pub fn clear_renderables(&mut self) {
	self.renderables.clear();
    }

    pub fn set_renderables(
	&mut self,
	device: &Device,
	presenter: &Presenter,
	renderables: Vec<Rc<dyn Renderable>>,
    ) -> anyhow::Result<()> {
	self.renderables = renderables;
	self.rebuild_command_buffers(device, presenter)
    }

    pub fn rebuild_command_buffers(
	&mut self,
	device: &Device,
	presenter: &Presenter,
    ) -> anyhow::Result<()> {
	self.command_buffers.clear();

	// TODO: this will probably need to be configurable eventually.
	let clear_values = [
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

	for framebuffer_index in 0..self.num_command_buffers {
	    let command_buffer = CommandBuffer::new(
		device,
		vk::CommandBufferLevel::PRIMARY,
		self.queue.clone(),
	    )?;
	    {
		let buffer_writer = command_buffer.record(
		    vk::CommandBufferUsageFlags::SIMULTANEOUS_USE,
		)?;
		{
		    let render_pass_writer = buffer_writer.begin_render_pass(
			presenter,
			&clear_values,
			framebuffer_index,
		    );
		    for renderable in self.renderables.iter() {
			println!("Writing a draw command to command buffer {}...", framebuffer_index);
			renderable.write_draw_command(framebuffer_index, &render_pass_writer)?;
		    }
		}
	    }
	    self.command_buffers.push(command_buffer);
	}
	println!("{} command buffers present", self.command_buffers.len());
	Ok(())
    }

    pub fn submit_command_buffer(
	&self,
	idx: usize,
	wait_stage: vk::PipelineStageFlags,
	image_available_semaphore: vk::Semaphore,
	render_finished_semaphore: vk::Semaphore,
	inflight_fence: vk::Fence,
    ) -> anyhow::Result<()> {
	if idx > self.command_buffers.len() {
	    return Err(anyhow!(
		"Tried to submit command buffer #{} of {}",
		idx,
		self.command_buffers.len(),
	    ));
	}

	self.command_buffers[idx].submit_synced(
	    wait_stage,
	    image_available_semaphore,
	    render_finished_semaphore,
	    inflight_fence,
	)
    }
}
