use ash::version::DeviceV1_0;
use ash::vk;

use std::rc::Rc;
use std::ptr;

use super::{Device, InnerDevice};
use super::utils::{Defaulted, find_memory_type};
use super::command_buffer::CommandBuffer;
use super::buffer::UploadSourceBuffer;

pub struct ImageBuilder {
    size: Defaulted<Vec<u32>>,
    image_type: Defaulted<vk::ImageType>,
    mip_levels: Defaulted<u32>,
    num_samples: Defaulted<vk::SampleCountFlags>,
    format: Defaulted<vk::Format>,
    tiling: Defaulted<vk::ImageTiling>,
    usage: Defaulted<vk::ImageUsageFlags>,
    required_memory_properties: Defaulted<vk::MemoryPropertyFlags>,
    sharing_mode: Defaulted<vk::SharingMode>,
}

impl ImageBuilder {
    #[allow(unused)]
    pub fn new1d(length: u32) -> Self {
        ImageBuilder::new(vec![length], vk::ImageType::TYPE_1D)
    }

    #[allow(unused)]
    pub fn new2d(width: u32, height: u32) -> Self {
        ImageBuilder::new(vec![width, height], vk::ImageType::TYPE_2D)
    }

    #[allow(unused)]
    pub fn new3d(width: u32, height: u32, depth: u32) -> Self {
        ImageBuilder::new(vec![width, height, depth], vk::ImageType::TYPE_3D)
    }

    fn new(size: Vec<u32>, image_type: vk::ImageType) -> Self {
        Self{
            size: Defaulted::new(size),
            image_type: Defaulted::new(image_type),
            mip_levels: Defaulted::new(1),
            num_samples: Defaulted::new(vk::SampleCountFlags::TYPE_1),
            format: Defaulted::new(vk::Format::R8G8B8A8_UNORM),
            tiling: Defaulted::new(vk::ImageTiling::OPTIMAL),
            usage: Defaulted::new(vk::ImageUsageFlags::TRANSFER_DST),
            required_memory_properties: Defaulted::new(vk::MemoryPropertyFlags::DEVICE_LOCAL),
            sharing_mode: Defaulted::new(vk::SharingMode::EXCLUSIVE),
        }
    }

    impl_defaulted_setter!(with_mip_levels, mip_levels, u32);
    impl_defaulted_setter!(with_num_samples, num_samples, vk::SampleCountFlags);
    impl_defaulted_setter!(with_format, format, vk::Format);
    impl_defaulted_setter!(with_tiling, tiling, vk::ImageTiling);
    impl_defaulted_setter!(with_usage, usage, vk::ImageUsageFlags);
    impl_defaulted_setter!(with_required_memory_properties, required_memory_properties, vk::MemoryPropertyFlags);
    impl_defaulted_setter!(with_sharing_mode, sharing_mode, vk::SharingMode);
}

pub struct Image {
    device: Rc<InnerDevice>,
    pub (in super) img: vk::Image,
    mem: Option<vk::DeviceMemory>,
    pub (in super) extent: vk::Extent3D,
    format: vk::Format,
    image_type: vk::ImageType,
}

impl Image {
    pub fn new(device: &Device, builder: ImageBuilder) -> anyhow::Result<Self> {
	Image::new_internal(device.inner.clone(), builder)
    }

    pub (in super) fn from_vk_image(
	device: Rc<InnerDevice>,
	image: vk::Image,
	extent: vk::Extent3D,
	format: vk::Format,
	image_type: vk::ImageType,
    ) -> Self {
	Self {
	    device,
	    img: image,
	    mem: None,
	    extent,
	    format,
	    image_type,
	}
    }

    pub (in super) fn new_internal(
	device: Rc<InnerDevice>,
	builder: ImageBuilder,
    ) -> anyhow::Result<Self> {
        let image_type = *builder.image_type.get_value();
        let format = *builder.format.get_value();
        let dimensions = builder.size.get_value();
        let extent = match image_type {
            vk::ImageType::TYPE_1D => vk::Extent3D{
                width: dimensions[0],
                height: 1,
                depth: 1,
            },
            vk::ImageType::TYPE_2D => vk::Extent3D{
                width: dimensions[0],
                height: dimensions[1],
                depth: 1,
            },
            vk::ImageType::TYPE_3D => vk::Extent3D{
                width: dimensions[0],
                height: dimensions[1],
                depth: dimensions[2],
            },
	    _ => panic!("Invalid image type (what did you do?)"),
        };
        let image_create_info = vk::ImageCreateInfo{
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageCreateFlags::empty(),
            image_type,
            format,
            extent,
            mip_levels: *builder.mip_levels.get_value(),
            array_layers: 1,
            samples: *builder.num_samples.get_value(),
            tiling: *builder.tiling.get_value(),
            usage: *builder.usage.get_value(),
            sharing_mode: *builder.sharing_mode.get_value(),
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            initial_layout: vk::ImageLayout::UNDEFINED,
        };

        let texture_image = unsafe {
            device.device
                .create_image(&image_create_info, None)?
        };

        let image_memory_requirement = unsafe {
            device.device.get_image_memory_requirements(texture_image)
        };
        let memory_allocate_info = vk::MemoryAllocateInfo{
            s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
            p_next: ptr::null(),
            allocation_size: image_memory_requirement.size,
            memory_type_index: find_memory_type(
                image_memory_requirement.memory_type_bits,
                *builder.required_memory_properties.get_value(),
                &device.memory_properties,
            ),
        };

        let texture_image_memory = unsafe {
            device.device
                .allocate_memory(&memory_allocate_info, None)?
        };

        unsafe {
            device.device
                .bind_image_memory(texture_image, texture_image_memory, 0)?
        }

        Ok(Image{
            device: device,
            img: texture_image,
            mem: Some(texture_image_memory),
            extent,
            format,
            image_type,
        })
    }

    pub fn transition_layout(
	&self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) -> anyhow::Result<()> {
        CommandBuffer::run_oneshot_internal(
	    self.device.clone(),
	    self.device.default_graphics_queue.clone(),
	    |writer| {
		let src_access_mask;
		let dst_access_mask;
		let source_stage;
		let destination_stage;

		if old_layout == vk::ImageLayout::UNDEFINED
                    && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
		{
                    src_access_mask = vk::AccessFlags::empty();
                    dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                    source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                    destination_stage = vk::PipelineStageFlags::TRANSFER;
		} else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
                    && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
		{
                    src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                    dst_access_mask = vk::AccessFlags::SHADER_READ;
                    source_stage = vk::PipelineStageFlags::TRANSFER;
                    destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
		} else {
                    return Err(anyhow::anyhow!("Unsupported layout transition"));
		}

		let image_barriers = [vk::ImageMemoryBarrier{
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask,
                    dst_access_mask,
                    old_layout,
                    new_layout,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: self.img,
                    subresource_range: vk::ImageSubresourceRange{
			aspect_mask: vk::ImageAspectFlags::COLOR,
			base_mip_level: 0,
			level_count: mip_levels,
			base_array_layer: 0,
			layer_count: 1,
                    },
		}];

                writer.pipeline_barrier(
                    source_stage,
                    destination_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &image_barriers,
                );
		Ok(())
            })
    }

    pub fn copy_buffer(
	&self,
        buffer: &UploadSourceBuffer,
    ) -> anyhow::Result<()> {
	CommandBuffer::run_oneshot_internal(
	    self.device.clone(),
	    self.device.default_transfer_queue.clone(),
	    |writer| {
		writer.copy_buffer_to_image(
		    buffer,
		    self,
		);
		Ok(())
            })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
	unsafe {
            self.device.device.destroy_image(self.img, None);
	    if let Some(mem) = self.mem {
		self.device.device.free_memory(mem, None);
	    }
	}
    }
}

pub struct ImageView {
    device: Rc<InnerDevice>,
    pub (in super) view: vk::ImageView,
}

impl ImageView {
    pub (in super) fn from_image(
        image: &Image,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> anyhow::Result<ImageView> {
        let imageview_create_info = vk::ImageViewCreateInfo{
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ImageViewCreateFlags::empty(),
            view_type: match image.image_type {
                vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
                vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
                vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
		_ => panic!("Invalid image type (what did you do?)"),
            },
            format: image.format,
            components: vk::ComponentMapping{
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            },
            subresource_range: vk::ImageSubresourceRange{
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            image: image.img,
        };

        Ok(Self{
            device: image.device.clone(),
            view: unsafe {
                image.device.device
                    .create_image_view(&imageview_create_info, None)?
            },
        })
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
	unsafe {
            self.device.device.destroy_image_view(self.view, None);
	}
    }
}
