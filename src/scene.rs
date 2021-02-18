use std::rc::Rc;

use super::support::{Device, FrameId};
use super::support::command_buffer::SecondaryCommandBuffer;
use super::support::renderer::RenderPass;

pub trait Renderable {
    fn get_command_buffer(&self, frame: FrameId) -> anyhow::Result<Rc<SecondaryCommandBuffer>>;
    fn rebuild_command_buffers(
	&self,
	device: &Device,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<()>;
    // TODO: figure out upload/use synchronization that doesn't rely on sitting around waiting for
    //       the buffer to be uploaded.
    fn sync_uniform_buffers(&self, frame: FrameId) -> anyhow::Result<()>;
    fn update_pipeline_viewport(
	&self,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: &RenderPass,
    ) -> anyhow::Result<()>;
}

// TODO: make this object useful again
pub struct Scene {
    renderables: Vec<Rc<dyn Renderable>>,
}

impl Scene {
    pub fn new(
	renderables: Vec<Rc<dyn Renderable>>,
    ) -> Self {
	Self{
	    renderables,
	}
    }

    // This should mainly be useful for handling pipeline updates.
    #[allow(unused)]
    pub fn clear_renderables(&mut self) {
	self.renderables.clear();
    }

    #[allow(unused)]
    pub fn set_renderables(
	&mut self,
	renderables: Vec<Rc<dyn Renderable>>,
    ) {
	self.renderables = renderables;
    }

    pub fn get_command_buffers(&self, frame: FrameId) -> anyhow::Result<Vec<Rc<SecondaryCommandBuffer>>> {
	let mut command_buffers = Vec::new();
	for renderable in self.renderables.iter() {
	    command_buffers.push(renderable.get_command_buffer(frame)?);
	}
	Ok(command_buffers)
    }

    pub fn rebuild_command_buffers(
	&self,
	device: &Device,
	render_pass: &RenderPass,
	subpass: u32,
    ) -> anyhow::Result<()> {
	for renderable in self.renderables.iter() {
	    renderable.rebuild_command_buffers(device, render_pass, subpass)?;
	}
	Ok(())
    }

    pub fn sync_uniform_buffers(&self, frame: FrameId) -> anyhow::Result<()> {
	for renderable in self.renderables.iter() {
	    renderable.sync_uniform_buffers(frame)?;
	}
	Ok(())
    }

    pub fn update_pipeline_viewports(
	&self,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: &RenderPass,
    ) -> anyhow::Result<()> {
	for renderable in self.renderables.iter() {
	    renderable.update_pipeline_viewport(
		viewport_width,
		viewport_height,
		render_pass,
	    )?;
	}
	Ok(())
    }
}
