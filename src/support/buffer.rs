use ash::vk;
use anyhow::anyhow;
use glsl_layout::Uniform;

use std::rc::Rc;
use std::ptr;

use super::{Device, InnerDevice, MemoryUsage};
use super::command_buffer::{CommandBuffer, CommandPool};

pub trait HasBuffer {
    fn get_buffer(&self) -> vk::Buffer;
}

pub struct MemoryMapping<T> {
    device: Rc<InnerDevice>,
    allocation: vk_mem::Allocation,
    data_ptr: *mut T,
}

impl<T> MemoryMapping<T> {
    fn new(buf: &Buffer) -> anyhow::Result<Self> {
        Ok(Self {
            device: buf.device.clone(),
            data_ptr: buf.device.map_memory(
                &buf.allocation,
            )? as *mut T,
            allocation: buf.allocation,
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
        match self.device.unmap_memory(&self.allocation) {
            Ok(_) => (),
            Err(e) => println!("Failed to unmap memory: {}", e),
        }
    }
}

pub struct Buffer {
    device: Rc<InnerDevice>,
    pub (in super) buf: vk::Buffer,
    allocation: vk_mem::Allocation,
    _allocation_info: vk_mem::AllocationInfo,
    size: vk::DeviceSize,
}

impl Buffer {
    fn new(
        device: Rc<InnerDevice>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryUsage,
        sharing_mode: vk::SharingMode,
    ) -> anyhow::Result<Self> {
        if size == 0 {
            return Err(anyhow!("Buffer size must be greater than zero."));
        }
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

        let (buffer, allocation, allocation_info) = device.create_buffer(
            memory_usage,
            &buffer_create_info,
        )?;

        Ok(Buffer{
            device: device.clone(),
            buf: buffer,
            allocation,
            _allocation_info: allocation_info,
            size,
        })
    }

    pub fn copy(
        src_buffer: Rc<Buffer>,
        dst_buffer: Rc<Buffer>,
        pool: Rc<CommandPool>,
    ) -> anyhow::Result<()> {
        if src_buffer.size > dst_buffer.size {
            return Err(anyhow!(
                "Tried to copy a {} byte buffer into a {} byte buffer!",
                src_buffer.size,
                dst_buffer.size,
            ));
        }
        CommandBuffer::run_oneshot_internal(
           src_buffer.device.clone(),
            pool,
            |writer| {
                let copy_regions = [vk::BufferCopy{
                    src_offset: 0,
                    dst_offset: 0,
                    size: src_buffer.size,
                }];

                writer.copy_buffer(
                    Rc::clone(&src_buffer),
                    Rc::clone(&dst_buffer),
                    &copy_regions,
                );
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
        match self.device.destroy_buffer(self.buf, &self.allocation) {
            Ok(_) => (),
            Err(e) => println!("Failed to destroy buffer: {}", e),
        }
    }
}

pub struct UploadSourceBuffer {
    buf: Rc<Buffer>,
}

impl UploadSourceBuffer {
    pub fn new(
        device: &Device,
        size: vk::DeviceSize,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            buf: Rc::new(Buffer::new(
                Rc::clone(&device.inner),
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryUsage::CpuOnly,
                vk::SharingMode::EXCLUSIVE,
            )?),
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

impl Drop for UploadSourceBuffer {
    fn drop(&mut self) {
        // Drop has been implemented solely so that UploadSourceBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}

pub struct VertexBuffer<T> {
    buf: Rc<Buffer>,
    len: usize,
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
        let vertex_buffer = Rc::new(Buffer::new(
            Rc::clone(&device.inner),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryUsage::GpuOnly,
            // TODO: is this really what we want?
            vk::SharingMode::EXCLUSIVE,
        )?);

        Buffer::copy(
            Rc::clone(&upload_buffer.buf),
            Rc::clone(&vertex_buffer),
            device.inner.get_default_transfer_pool(),
        )?;

        Ok(Self{
            buf: vertex_buffer,
            len: data.len(),
            _phantom: std::marker::PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl<V> HasBuffer for VertexBuffer<V> {
    fn get_buffer(&self) -> vk::Buffer {
        self.buf.buf
    }
}

impl<V> Drop for VertexBuffer<V> {
    fn drop(&mut self) {
        // Drop has been implemented solely so that VertexBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}

pub struct IndexBuffer {
    buf: Rc<Buffer>,
    len: usize,
}

impl IndexBuffer {
    #[allow(unused)]
    pub fn new(
        device: &Device,
        data: &[u32],
    ) -> anyhow::Result<Self> {
        let buffer_size = std::mem::size_of_val(data) as vk::DeviceSize;
        let upload_buffer = UploadSourceBuffer::new(device, buffer_size)?;
        upload_buffer.copy_data(data)?;
        let index_buffer = Rc::new(Buffer::new(
            Rc::clone(&device.inner),
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryUsage::GpuOnly,
            // TODO: Is this actually what we want in multiqueue scenarios?
            vk::SharingMode::EXCLUSIVE,
        )?);

        Buffer::copy(
            Rc::clone(&upload_buffer.buf),
            Rc::clone(&index_buffer),
            device.inner.get_default_transfer_pool(),
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

impl Drop for IndexBuffer {
    fn drop(&mut self) {
        // Drop has been implemented solely so that IndexBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}

pub struct UniformBuffer<T: Uniform>
where <T as Uniform>::Std140: Sized
{
    buf: Rc<Buffer>,
    size: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Uniform> UniformBuffer<T>
where <T as Uniform>::Std140: Sized
{
    pub fn new(
        device: &Device,
        initial_value: Option<&T>,
    ) -> anyhow::Result<Self> {
        let size = std::mem::size_of::<T::Std140>();
        let buffer = Rc::new(Buffer::new(
            Rc::clone(&device.inner),
            size as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            // TODO: allow GpuOnly for infrequently-updated buffers
            MemoryUsage::CpuToGpu,
            vk::SharingMode::EXCLUSIVE,
        )?);

        let this = Self{
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
        &self,
        new_value: &T,
    ) -> anyhow::Result<()> {
        let std140_val = new_value.std140();
        let std140_value_size = std::mem::size_of_val::<T::Std140>(&std140_val);
        if self.size < std140_value_size {
            Err(anyhow!("Tried to write {} bytes to a {}-byte uniform buffer!", std140_value_size, self.size))
        } else {
            self.buf.with_memory_mapping(|mmap| {
                mmap.copy_item(&std140_val);
                Ok(())
            })
        }
    }

    pub fn len(&self) -> vk::DeviceSize {
        self.size as vk::DeviceSize
    }
}

impl<T: Uniform> HasBuffer for UniformBuffer<T>
where <T as Uniform>::Std140: Sized
{
    fn get_buffer(&self) -> vk::Buffer {
        self.buf.buf
    }
}

impl<T: Uniform> Drop for UniformBuffer<T>
where <T as Uniform>::Std140: Sized
{
    fn drop(&mut self) {
        // Drop has been implemented solely so that UniformBuffers can be recorded as
        // dependencies for CommandBuffers.
    }
}
