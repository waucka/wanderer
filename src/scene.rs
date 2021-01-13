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
    pub fn clear_renderables(&mut self) {
	self.renderables.clear();
    }

    pub fn set_renderables(
	&mut self,
	renderables: Vec<Rc<dyn Renderable>>,
    ) {
	self.renderables = renderables;
    }

    pub fn write_command_buffer(
	&mut self,
	framebuffer_index: usize,
	writer: &RenderPassWriter,
    ) -> anyhow::Result<()> {
	for renderable in self.renderables.iter() {
	    renderable.write_draw_command(framebuffer_index, writer)?;
	}
	Ok(())
    }
}
