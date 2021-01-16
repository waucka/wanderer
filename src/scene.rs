use std::rc::Rc;

use super::support::command_buffer::RenderPassWriter;

pub trait Renderable {
    fn write_draw_command(&self, idx: usize, writer: &mut RenderPassWriter) -> anyhow::Result<()>;
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

    pub fn write_command_buffer(
	&self,
	framebuffer_index: usize,
	writer: &mut RenderPassWriter,
    ) -> anyhow::Result<()> {
	for renderable in self.renderables.iter() {
	    renderable.write_draw_command(framebuffer_index, writer)?;
	}
	Ok(())
    }
}
