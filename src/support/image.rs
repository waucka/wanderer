use ash::version::DeviceV1_0;
use ash::vk;

use std::rc::Rc;
use std::ptr;
use std::os::raw::c_void;

use super::{Device, InnerDevice};
use super::utils::{Defaulted, find_memory_type};
use super::command_buffer::CommandBuffer;
use super::buffer::UploadSourceBuffer;

#[derive(Clone)]
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
    pub fn new1d(length: usize) -> Self {
        ImageBuilder::new(vec![length as u32], vk::ImageType::TYPE_1D)
    }

    #[allow(unused)]
    pub fn new2d(width: usize, height: usize) -> Self {
        ImageBuilder::new(vec![width as u32, height as u32], vk::ImageType::TYPE_2D)
    }

    #[allow(unused)]
    pub fn new3d(width: usize, height: usize, depth: usize) -> Self {
        ImageBuilder::new(vec![width as u32, height as u32, depth as u32], vk::ImageType::TYPE_3D)
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
    pub (in super) format: vk::Format,
    image_type: vk::ImageType,
    pub (in super) layout: vk::ImageLayout,
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
            layout: vk::ImageLayout::UNDEFINED,
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

        let format_list_bs = vk::ImageFormatListCreateInfo{
            s_type: vk::StructureType::IMAGE_FORMAT_LIST_CREATE_INFO,
            p_next: ptr::null(),
            view_format_count: 1,
            p_view_formats: &format,
        };

        let image_create_info = vk::ImageCreateInfo{
            s_type: vk::StructureType::IMAGE_CREATE_INFO,
            p_next: (&format_list_bs as *const _) as *const c_void,
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
            layout: vk::ImageLayout::UNDEFINED,
        })
    }

    pub fn transition_layout(
        &mut self,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) -> anyhow::Result<()> {
        CommandBuffer::run_oneshot_internal(
            self.device.clone(),
            self.device.get_default_graphics_pool(),
            |writer| {
                let src_access_mask;
                let dst_access_mask;
                let source_stage;
                let destination_stage;
                let mut aspect_mask = vk::ImageAspectFlags::COLOR;

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
                } else if old_layout == vk::ImageLayout::UNDEFINED
                    && new_layout == vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
                {
                    src_access_mask = vk::AccessFlags::empty();
                    dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
                    source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                    destination_stage = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
                } else if old_layout == vk::ImageLayout::UNDEFINED
                    && new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                {
                    src_access_mask = vk::AccessFlags::empty();
                    dst_access_mask = vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
                    source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                    destination_stage = vk::PipelineStageFlags::ALL_GRAPHICS;
                    aspect_mask = vk::ImageAspectFlags::DEPTH;
                } else if old_layout == vk::ImageLayout::UNDEFINED
                    && new_layout == vk::ImageLayout::GENERAL
                {
                    src_access_mask = vk::AccessFlags::empty();
                    dst_access_mask = vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::INPUT_ATTACHMENT_READ;
                    source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
                    destination_stage = vk::PipelineStageFlags::ALL_GRAPHICS;
                } else if old_layout == new_layout {
                    return Ok(());
                } else {
                    return Err(anyhow::anyhow!(
                        "Unsupported layout transition {:?} -> {:?}",
                        old_layout,
                        new_layout,
                    ));
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
                        aspect_mask,
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
            })?;
        self.layout = new_layout;
        Ok(())
    }

    pub fn generate_mipmaps(
        &mut self,
        mip_levels: u32,
    ) -> anyhow::Result<()>{
        use std::cmp::max;
        super::command_buffer::CommandBuffer::run_oneshot_internal(
            Rc::clone(&self.device),
            self.device.get_default_graphics_pool(),
            |writer| {
                let mut image_barrier = vk::ImageMemoryBarrier{
                    s_type: vk::StructureType::IMAGE_MEMORY_BARRIER,
                    p_next: ptr::null(),
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::empty(),
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::UNDEFINED,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image: self.img,
                    subresource_range: vk::ImageSubresourceRange{
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                };

                let mut mip_width = self.extent.width as i32;
                let mut mip_height = self.extent.height as i32;

                for i in 1..mip_levels {
                    image_barrier.subresource_range.base_mip_level = i - 1;
                    image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                    image_barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                    image_barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

                    writer.pipeline_barrier(
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[image_barrier.clone()],
                    );

                    let blits = [vk::ImageBlit{
                        src_subresource: vk::ImageSubresourceLayers{
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: i - 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_offsets: [
                            vk::Offset3D{ x: 0, y: 0, z: 0 },
                            vk::Offset3D{
                                x: mip_width,
                                y: mip_height,
                                z: 1,
                            },
                        ],
                        dst_subresource: vk::ImageSubresourceLayers{
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: i,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        dst_offsets: [
                            vk::Offset3D{ x: 0, y: 0, z: 0 },
                            vk::Offset3D{
                                x: max(mip_width / 2, 1),
                                y: max(mip_height / 2, 1),
                                z: 1,
                            },
                        ],
                    }];

                    unsafe {
                        writer.blit_image_no_deps(
                            self,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            self,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &blits,
                            vk::Filter::LINEAR,
                        );
                    }

                    image_barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                    image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                    image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                    image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                    writer.pipeline_barrier(
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[image_barrier.clone()],
                    );

                    mip_width = max(mip_width / 2, 1);
                    mip_height = max(mip_height / 2, 1);
                }

                image_barrier.subresource_range.base_mip_level = mip_levels - 1;
                image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                image_barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                image_barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                writer.pipeline_barrier(
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[image_barrier.clone()],
                );
                Ok(())
            })?;

        self.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

        Ok(())
    }

    #[allow(unused)]
    pub fn copy_buffer(
        buffer: Rc<UploadSourceBuffer>,
        dst_img: Rc<Image>,
    ) -> anyhow::Result<()> {
        CommandBuffer::run_oneshot_internal(
            Rc::clone(&dst_img.device),
            dst_img.device.get_default_transfer_pool(),
            |writer| {
                writer.copy_buffer_to_image(
                    Rc::clone(&buffer),
                    Rc::clone(&dst_img),
                );
                Ok(())
            })
    }

    pub (in super) unsafe fn copy_buffer_no_deps(
        buffer: &UploadSourceBuffer,
        dst_img: &Image,
    ) -> anyhow::Result<()> {
        CommandBuffer::run_oneshot_internal(
            Rc::clone(&dst_img.device),
            dst_img.device.get_default_transfer_pool(),
            |writer| {
                writer.copy_buffer_to_image_no_deps(
                    buffer,
                    dst_img,
                );
                Ok(())
            })
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        unsafe {
            //println!("Dropping image {:?}", self.img);
            if let Some(mem) = self.mem {
                // If we don't have the memory for it, then we aren't
                // responsible for destroying it.
                self.device.device.destroy_image(self.img, None);
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
    pub fn from_image(
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
        //println!("Dropping image view {:?}", self.view);
        unsafe {
            self.device.device.destroy_image_view(self.view, None);
        }
    }
}
