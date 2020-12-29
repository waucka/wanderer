use ash::version::DeviceV1_0;
use ash::vk;

use std::cell::RefCell;
use std::rc::Rc;
use std::ptr;

use super::{Device, InnerDevice, Queue};
use super::renderer::{Presenter, Pipeline};
use super::buffer::{VertexBuffer, IndexBuffer, UploadSourceBuffer, HasBuffer};
use super::image::Image;
use super::shader::Vertex;

pub struct CommandBuffer {
    device: Rc<InnerDevice>,
    queue: Rc<Queue>,
    buf: vk::CommandBuffer,
}

impl CommandBuffer {
    pub fn new(
	device: &Device,
	level: vk::CommandBufferLevel,
	queue: Rc<Queue>,
    ) -> anyhow::Result<Self> {
	CommandBuffer::from_inner_device(device.inner.clone(), level, queue)
    }

    fn from_inner_device(
	device: Rc<InnerDevice>,
	level: vk::CommandBufferLevel,
	queue: Rc<Queue>,
    ) -> anyhow::Result<Self> {
        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo{
            s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
            p_next: ptr::null(),
            command_buffer_count: 1,
            command_pool: queue.command_pool,
            level,
        };

        let command_buffer = unsafe {
            device.device
                .allocate_command_buffers(&command_buffer_allocate_info)?
        }[0];

	Ok(Self{
	    device,
	    buf: command_buffer,
	    queue,
	})
    }

    pub fn record(&self, usage_flags: vk::CommandBufferUsageFlags) -> anyhow::Result<BufferWriter> {
        let command_buffer_begin_info = vk::CommandBufferBeginInfo{
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            p_next: ptr::null(),
            p_inheritance_info: ptr::null(),
            flags: usage_flags,
        };

        unsafe {
            self.device.device
                .begin_command_buffer(self.buf, &command_buffer_begin_info)?;
        }

        Ok(BufferWriter{
	    device: self.device.clone(),
            command_buffer: self,
        })
    }

    pub fn submit_synced(
	&self,
	wait_stage: vk::PipelineStageFlags,
	image_available_semaphore: vk::Semaphore,
	render_finished_semaphore: vk::Semaphore,
	inflight_fence: vk::Fence,
    ) -> anyhow::Result<()> {
	let wait_semaphores = [image_available_semaphore];
	let signal_semaphores = [render_finished_semaphore];
	let wait_stages = [wait_stage];
	let wait_fences = [inflight_fence];

        let submit_infos = [vk::SubmitInfo{
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &self.buf,
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
        }];

        unsafe {
            self.device.device
                .reset_fences(&wait_fences)?;
            self.device.device
                .queue_submit(
                    self.queue.get(),
                    &submit_infos,
                    inflight_fence,
                )?;
        }
	Ok(())
    }

    pub fn submit_unsynced(
        &self,
    ) -> anyhow::Result<()> {
        let buffers_to_submit = [self.buf];

        let submit_infos = [vk::SubmitInfo{
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: 0,
            p_wait_semaphores: ptr::null(),
            p_wait_dst_stage_mask: ptr::null(),
            command_buffer_count: 1,
            p_command_buffers: buffers_to_submit.as_ptr(),
            signal_semaphore_count: 0,
            p_signal_semaphores: ptr::null(),
        }];

        unsafe {
            self.device.device
                .queue_submit(self.queue.get(), &submit_infos, vk::Fence::null())?;
	    // TODO: do I actually want to do this wait here?
            self.device.device
                .queue_wait_idle(self.queue.get())?;
        }

        Ok(())
    }

    #[allow(unused)]
    pub fn run_oneshot<T>(
	device: &Device,
	queue: Rc<Queue>,
	cmd_fn: T,
    ) -> anyhow::Result<()>
    where
        T: FnMut(&BufferWriter) -> anyhow::Result<()>
    {
	CommandBuffer::run_oneshot_internal(device.inner.clone(), queue, cmd_fn)
    }

    pub (in super) fn run_oneshot_internal<T>(
	device: Rc<InnerDevice>,
	queue: Rc<Queue>,
	mut cmd_fn: T,
    ) -> anyhow::Result<()>
    where
        T: FnMut(&BufferWriter) -> anyhow::Result<()>
    {
        let cmd_buf = CommandBuffer::from_inner_device(
	    device,
	    vk::CommandBufferLevel::PRIMARY,
	    queue,
	)?;
	{
	    let writer = cmd_buf.record(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)?;
            cmd_fn(&writer)?;
	}
        cmd_buf.submit_unsynced()?;
        Ok(())
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
	let buffers = [self.buf];
	unsafe {
	    self.device.device.free_command_buffers(self.queue.command_pool, &buffers);
	}
    }
}

pub struct BufferWriter<'a> {
    device: Rc<InnerDevice>,
    command_buffer: &'a CommandBuffer,
}

impl<'a> BufferWriter<'a> {
    pub fn begin_render_pass(
	&self,
	presenter: &Presenter,
	clear_values: &[vk::ClearValue],
	framebuffer_index: usize,
    ) -> RenderPassWriter {
	let render_area = vk::Rect2D{
	    offset: vk::Offset2D{
		x: 0,
		y: 0,
	    },
	    extent: presenter.get_render_extent()
	};
	let render_pass_begin_info = vk::RenderPassBeginInfo{
	    s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
	    p_next: ptr::null(),
	    render_pass: presenter.render_pass.render_pass,
	    framebuffer: presenter.get_framebuffer(framebuffer_index),
	    render_area,
	    clear_value_count: clear_values.len() as u32,
	    p_clear_values: clear_values.as_ptr(),
	};

	unsafe {
	    self.device.device.cmd_begin_render_pass(
		self.command_buffer.buf,
		&render_pass_begin_info,
		vk::SubpassContents::INLINE,
	    );
	}

	RenderPassWriter{
	    device: self.device.clone(),
	    command_buffer: self.command_buffer,
	}
    }

    pub fn pipeline_barrier(
	&self,
	src_stage_mask: vk::PipelineStageFlags,
	dst_stage_mask: vk::PipelineStageFlags,
	deps: vk::DependencyFlags,
	memory_barriers: &[vk::MemoryBarrier],
	buffer_memory_barriers: &[vk::BufferMemoryBarrier],
	image_memory_barriers: &[vk::ImageMemoryBarrier],
    ) {
	unsafe {
	    self.device.device.cmd_pipeline_barrier(
		self.command_buffer.buf,
		src_stage_mask,
		dst_stage_mask,
		deps,
		memory_barriers,
		buffer_memory_barriers,
		image_memory_barriers,
	    );
	}
    }

    pub fn copy_buffer<S: HasBuffer, D: HasBuffer>(
	&self,
        src_buffer: &S,
        dst_buffer: &D,
	copy_regions: &[vk::BufferCopy],
    ) {
	unsafe {
	    let src_buf = src_buffer.get_buffer();
	    let dst_buf = dst_buffer.get_buffer();
	    self.device.device.cmd_copy_buffer(
		self.command_buffer.buf,
		src_buf,
		dst_buf,
		&copy_regions,
	    );
	}
    }

    pub fn copy_buffer_to_image(
	&self,
	src_buffer: &UploadSourceBuffer,
	image: &Image,
    ) {
        let buffer_image_regions = [vk::BufferImageCopy{
            image_subresource: vk::ImageSubresourceLayers{
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: image.extent,
            buffer_offset: 0,
            buffer_image_height: 0,
            buffer_row_length: 0,
            image_offset: vk::Offset3D{ x: 0, y: 0, z: 0 },
        }];

	unsafe {
            self.device.device.cmd_copy_buffer_to_image(
		self.command_buffer.buf,
		src_buffer.get_buffer(),
		image.img,
		vk::ImageLayout::TRANSFER_DST_OPTIMAL,
		&buffer_image_regions,
            );
	}
    }

    pub fn blit_image(
	&self,
	img_src: &Image,
	layout_src: vk::ImageLayout,
	img_dst: &Image,
	layout_dst: vk::ImageLayout,
	regions: &[vk::ImageBlit],
	filter: vk::Filter,
    ) {
	unsafe {
            self.device.device.cmd_blit_image(
		self.command_buffer.buf,
		img_src.img,
		layout_src,
		img_dst.img,
		layout_dst,
		regions,
		filter,
            );
	}
    }
}

impl<'a> Drop for BufferWriter<'a> {
    fn drop(&mut self) {
	unsafe {
	    if let Err(e) = self.device.device.end_command_buffer(self.command_buffer.buf) {
		println!("Failed to end command buffer: {:?}", e);
	    }
	}
    }
}

pub struct RenderPassWriter<'a> {
    device: Rc<InnerDevice>,
    command_buffer: &'a CommandBuffer,
}

impl<'a> RenderPassWriter<'a> {
    pub fn bind_pipeline<V: Vertex>(
	&self,
	pipeline: Rc<RefCell<Pipeline<V>>>,
    ) {
	unsafe {
	    self.device.device.cmd_bind_pipeline(
		self.command_buffer.buf,
		vk::PipelineBindPoint::GRAPHICS,
		pipeline.borrow().pipeline,
	    );
	}
    }

    pub fn bind_descriptor_sets(
	&self,
	pipeline_layout: vk::PipelineLayout,
	descriptor_sets: &[vk::DescriptorSet],
    ) {
	unsafe {
	    self.device.device.cmd_bind_descriptor_sets(
		self.command_buffer.buf,
		vk::PipelineBindPoint::GRAPHICS,
		pipeline_layout,
		0,
		&descriptor_sets,
		&[],
	    );
	}
    }

    pub fn draw_indexed<T>(
	&self,
	vertex_buffer: &VertexBuffer<T>,
	index_buffer: &IndexBuffer,
    ) {
	let vertex_buffers = [vertex_buffer.get_buffer()];
	let offsets = [0_u64];

	unsafe {
	    self.device.device.cmd_bind_vertex_buffers(
		self.command_buffer.buf,
		0,
		&vertex_buffers,
		&offsets,
	    );
	    self.device.device.cmd_bind_index_buffer(
		self.command_buffer.buf,
		index_buffer.get_buffer(),
		0,
		vk::IndexType::UINT32,
	    );
	    self.device.device.cmd_draw_indexed(
		self.command_buffer.buf,
		index_buffer.len() as u32,
		1, 0, 0, 0,
	    );
	}
    }
}

impl<'a> Drop for RenderPassWriter<'a> {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.cmd_end_render_pass(self.command_buffer.buf);
	}
    }
}
