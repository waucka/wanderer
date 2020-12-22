use ash::version::DeviceV1_0;
use ash::vk;
use anyhow::anyhow;

use std::rc::Rc;
use std::ptr;

use super::{Device, InnerDevice, Queue};
use super::command_buffer::CommandBuffer;

pub trait HasBuffer {
    fn get_buffer(&self) -> vk::Buffer;
}

struct MemoryMapping<T> {
    device: Rc<InnerDevice>,
    mem: vk::DeviceMemory,
    data_ptr: *mut T,
}

impl<T> MemoryMapping<T> {
    fn new(buf: &Buffer) -> anyhow::Result<Self> {
        let req = unsafe {
	    buf.device.device.get_buffer_memory_requirements(buf.buf)
	};
        Ok(Self {
            device: buf.device.clone(),
            data_ptr: unsafe {
		buf.device.device
		    .map_memory(
			buf.mem,
			0,
			req.size,
			vk::MemoryMapFlags::empty(),
		    )? as *mut T
	    },
	    mem: buf.mem,
        })
    }

    fn copy_slice(&self, src: &[T]) {
	unsafe {
            self.data_ptr.copy_from_nonoverlapping(src.as_ptr(), src.len());
	}
    }

    fn copy_item(&self, src: &T) {
	unsafe {
            self.data_ptr.copy_from_nonoverlapping(src, 1);
	}
    }
}

impl<T> Drop for MemoryMapping<T> {
    fn drop(&mut self) {
	unsafe {
            self.device.device.unmap_memory(self.mem);
	}
    }
}

struct Buffer {
    device: Rc<InnerDevice>,
    pub (in super) buf: vk::Buffer,
    mem: vk::DeviceMemory,
}

impl Buffer {
    fn new(
        device: &Device,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required_memory_flags: vk::MemoryPropertyFlags,
        sharing_mode: vk::SharingMode,
    ) -> anyhow::Result<Self> {
        let buffer_create_info = vk::BufferCreateInfo{
            s_type: vk::StructureType::BUFFER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::BufferCreateFlags::empty(),
            size: size,
            usage: usage,
            sharing_mode: sharing_mode,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
        };

        let buffer = unsafe {
            device.inner.device
                .create_buffer(&buffer_create_info, None)?
        };

        let mem_requirements = unsafe { device.inner.device.get_buffer_memory_requirements(buffer) };
        let memory_type = super::utils::find_memory_type(
            mem_requirements.memory_type_bits,
            required_memory_flags,
            &device.inner.memory_properties,
        );

        let allocate_info = vk::MemoryAllocateInfo{
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: mem_requirements.size,
            memory_type_index: memory_type,
        };

        let buffer_memory = unsafe {
            device.inner.device
                .allocate_memory(&allocate_info, None)?
        };

        unsafe {
            device.inner.device
                .bind_buffer_memory(buffer, buffer_memory, 0)?
        }

        Ok(Buffer{
            device: device.inner.clone(),
            buf: buffer,
            mem: buffer_memory,
        })
    }

    pub fn copy(
        &self,
        dst_buffer: &Buffer,
        size: vk::DeviceSize,
	queue: Rc<Queue>,
    ) -> anyhow::Result<()> {
        CommandBuffer::run_oneshot_internal(
	    self.device.clone(),
	    queue,
	    |writer| {
		let copy_regions = [vk::BufferCopy{
		    src_offset: 0,
		    dst_offset: 0,
		    size,
		}];

		writer.copy_buffer(self, dst_buffer, &copy_regions);
		Ok(())
	    }
	)
    }

    pub fn with_memory_mapping<T, F>(&self, mmap_fn: F) -> anyhow::Result<()>
    where
        F: Fn(&MemoryMapping<T>) -> anyhow::Result<()> {
        let mmap = MemoryMapping::new(self)?;
        mmap_fn(&mmap)
    }
}

impl HasBuffer for Buffer {
    fn get_buffer(&self) -> vk::Buffer {
	self.buf
    }
}


impl Drop for Buffer {
    fn drop(&mut self) {
	unsafe {
            self.device.device.destroy_buffer(self.buf, None);
            self.device.device.free_memory(self.mem, None);
	}
    }
}

pub struct UploadSourceBuffer {
    buf: Buffer,
}

impl UploadSourceBuffer {
    pub fn new(
        device: &Device,
        size: vk::DeviceSize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            buf: Buffer::new(
		device,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::SharingMode::EXCLUSIVE,
            )?,
        })
    }

    pub fn copy_data<T>(&self, data: &[T]) -> anyhow::Result<()> {
        self.buf.with_memory_mapping(|mmap| {
            mmap.copy_slice(data);
            Ok(())
	})
    }
}

impl HasBuffer for UploadSourceBuffer {
    fn get_buffer(&self) -> vk::Buffer {
	self.buf.buf
    }
}

pub struct VertexBuffer<T> {
    buf: Buffer,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> VertexBuffer<T> {
    pub fn new(
	device: &Device,
	data: &[T],
    ) -> anyhow::Result<Self> {
	let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
	let upload_buffer = UploadSourceBuffer::new(device, buffer_size)?;
	upload_buffer.copy_data(data)?;
	let vertex_buffer = Buffer::new(
	    device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
	    // TODO: is this really what we want?
	    vk::SharingMode::EXCLUSIVE,
	)?;

	upload_buffer.buf.copy(
	    &vertex_buffer,
	    buffer_size,
	    device.inner.default_transfer_queue.clone(),
	)?;

	Ok(Self{
	    buf: vertex_buffer,
	    _phantom: std::marker::PhantomData,
	})
    }
}

impl<V> HasBuffer for VertexBuffer<V> {
    fn get_buffer(&self) -> vk::Buffer {
	self.buf.buf
    }
}

pub struct IndexBuffer {
    buf: Buffer,
    len: usize,
}

impl IndexBuffer {
    pub fn new(
	device: &Device,
	data: &[u32],
    ) -> anyhow::Result<Self> {
	let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
	let upload_buffer = UploadSourceBuffer::new(device, buffer_size)?;
	upload_buffer.copy_data(data)?;
	let index_buffer = Buffer::new(
	    device,
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
	    // TODO: Is this actually what we want in multiqueue scenarios?
	    vk::SharingMode::EXCLUSIVE,
	)?;

	upload_buffer.buf.copy(
	    &index_buffer,
	    buffer_size,
	    device.inner.default_transfer_queue.clone(),
	)?;

	Ok(Self{
	    buf: index_buffer,
	    len: data.len(),
	})
    }

    pub fn len(&self) -> usize {
	self.len
    }
}

impl HasBuffer for IndexBuffer {
    fn get_buffer(&self) -> vk::Buffer {
	self.buf.buf
    }
}

pub struct UniformBuffer<T> {
    buf: Buffer,
    size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T> UniformBuffer<T> {
    pub fn new(
	device: &Device,
	initial_value: Option<&T>,
    ) -> anyhow::Result<Self> {
	let size = std::mem::size_of::<T>();
	let buffer = Buffer::new(
	    device,
            size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
	    vk::SharingMode::EXCLUSIVE,
	)?;

	let mut this = Self{
	    buf: buffer,
	    size,
	    _phantom: std::marker::PhantomData,
	};

	if let Some(initval) = initial_value {
	    this.update(initval)?;
	}

	Ok(this)
    }

    pub fn update(
	&mut self,
	new_value: &T,
    ) -> anyhow::Result<()> {
	let new_value_size = std::mem::size_of_val::<T>(new_value);
        if self.size > new_value_size {
	    Err(anyhow!("Tried to write {} bytes to a {}-byte uniform buffer!", new_value_size, self.size))
	} else {
            self.buf.with_memory_mapping(|mmap| {
		mmap.copy_item(new_value);
		Ok(())
	    })
	}
    }

    pub fn len(&self) -> vk::DeviceSize {
	self.size as vk::DeviceSize
    }
}

impl<T> HasBuffer for UniformBuffer<T> {
    fn get_buffer(&self) -> vk::Buffer {
	self.buf.buf
    }
}

/*impl<T> HasUniformDescriptor for UniformBuffer<T> {
    fn create_write_descriptor_set(
	&self,
	dst_set: vk::DescriptorSet,
	dst_binding: u32,
	dst_array_element: u32,
    ) -> vk::WriteDescriptorSet {
	let descriptor_buffer_info = [vk::DescriptorBufferInfo{
            buffer: self.buf.buf,
            offset: 0,
            range: self.size as u64,
        }];

	vk::WriteDescriptorSet{
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set,
            dst_binding,
            dst_array_element,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_image_info: ptr::null(),
            p_buffer_info: descriptor_buffer_info.as_ptr(),
            p_texel_buffer_view: ptr::null(),
        }
    }
}*/
