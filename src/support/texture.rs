use ash::version::DeviceV1_0;
use ash::vk;
use image::GenericImageView;

use std::rc::Rc;
use std::ptr;
use std::path::Path;
use std::cmp::max;

use super::{Device, InnerDevice};
use super::image::{Image, ImageView, ImageBuilder};
use super::buffer::UploadSourceBuffer;

pub struct Texture {
    device: Rc<InnerDevice>,
    image: Image,
    pub (in super) image_view: ImageView,
    mip_levels: u32,
}

impl Texture {
    pub fn from_file(device: &Device, image_path: &Path) -> anyhow::Result<Self> {
	let mut image_object = image::open(image_path).unwrap();
	image_object = image_object.flipv();
	let (image_width, image_height) = (image_object.width(), image_object.height());
	let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
	let image_data = match &image_object {
            image::DynamicImage::ImageLuma8(_) |
            image::DynamicImage::ImageBgr8(_) |
            image::DynamicImage::ImageRgb8(_) |
            image::DynamicImage::ImageLumaA8(_) |
            image::DynamicImage::ImageBgra8(_) |
            image::DynamicImage::ImageRgba8(_) => image_object.to_rgba8().into_raw(),
            _ => panic!("Unsupported image type (probably 16 bits per channel)"),
	};
	let mip_levels = ((max(image_width, image_height) as f32)
			  .log2()
			  .floor() as u32) + 1;

	if image_size <= 0 {
            panic!("Failed to load texture image");
	}

	let upload_buffer = UploadSourceBuffer::new(device, image_size)?;
	upload_buffer.copy_data(&image_data)?;

	let image = Image::new(
	    device,
	    ImageBuilder::new2d(image_width, image_height)
		.with_mip_levels(mip_levels)
		.with_num_samples(vk::SampleCountFlags::TYPE_1)
		.with_format(vk::Format::R8G8B8A8_UNORM)
		.with_tiling(vk::ImageTiling::OPTIMAL)
		.with_usage(
		    vk::ImageUsageFlags::TRANSFER_SRC |
		    vk::ImageUsageFlags::TRANSFER_DST |
		    vk::ImageUsageFlags::SAMPLED,
		)
		.with_required_memory_properties(vk::MemoryPropertyFlags::DEVICE_LOCAL)
		.with_sharing_mode(vk::SharingMode::EXCLUSIVE)
	)?;

	image.transition_layout(
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
	)?;

	image.copy_buffer(&upload_buffer)?;

        let image_view = ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?;

        let tex = Self{
            device: device.inner.clone(),
            image,
            image_view,
	    mip_levels,
        };

	tex.generate_mipmaps()?;

        Ok(tex)
    }

    fn generate_mipmaps(
	&self,
    ) -> anyhow::Result<()>{
        super::command_buffer::CommandBuffer::run_oneshot_internal(
	    self.device.clone(),
	    self.device.get_default_transfer_queue(),
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
                image: self.image.img,
                subresource_range: vk::ImageSubresourceRange{
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };

            let mut mip_width = self.image.extent.width as i32;
            let mut mip_height = self.image.extent.height as i32;

            for i in 1..self.mip_levels {
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

                writer.blit_image(
                    &self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    &self.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                );

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

            image_barrier.subresource_range.base_mip_level = self.mip_levels - 1;
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
        })
    }

    pub fn get_mip_levels(&self) -> u32 {
	self.mip_levels
    }
}

pub struct Sampler {
    device: Rc<InnerDevice>,
    pub (in super) sampler: vk::Sampler,
}

impl Sampler {
    pub fn new(
	device: &Device,
	mip_levels: u32,
	min_filter: vk::Filter,
	mag_filter: vk::Filter,
	mipmap_mode: vk::SamplerMipmapMode,
	address_mode: vk::SamplerAddressMode,
    ) -> anyhow::Result<Self> {
	let sampler_create_info = vk::SamplerCreateInfo{
            s_type: vk::StructureType::SAMPLER_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SamplerCreateFlags::empty(),
            min_filter: min_filter,
            mag_filter: mag_filter,
            mipmap_mode: mipmap_mode,
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mip_lod_bias: 0.0,
            anisotropy_enable: vk::TRUE,
            max_anisotropy: 16.0,
            compare_enable: vk::FALSE,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0,
            max_lod: mip_levels as f32,
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: vk::FALSE,
	};

	unsafe {
	    Ok(Self {
		device: device.inner.clone(),
		sampler: device.inner.device.create_sampler(&sampler_create_info, None)?,
	    })
	}
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
	unsafe {
            self.device.device.destroy_sampler(self.sampler, None);
	}
    }
}

/*pub struct TextureSampler<'a> {
    texture: &'a Texture,
    sampler: &'a Sampler,
}

impl<'a> TextureSampler<'a> {
    pub fn new(texture: &'a Texture, sampler: &'a Sampler) -> Self {
	Self {
	    texture,
	    sampler,
	}
    }
}*/

/*impl<'a> HasUniformDescriptor for TextureSampler<'a> {
    fn create_write_descriptor_set(
	&self,
	dst_set: vk::DescriptorSet,
	dst_binding: u32,
	dst_array_element: u32,
    ) -> vk::WriteDescriptorSet {
	let descriptor_image_info = [vk::DescriptorImageInfo{
            sampler: self.sampler.sampler,
            image_view: self.texture.image_view.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];

	vk::WriteDescriptorSet{
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            p_next: ptr::null(),
            dst_set,
            dst_binding,
            dst_array_element,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: descriptor_image_info.as_ptr(),
            p_buffer_info: ptr::null(),
            p_texel_buffer_view: ptr::null(),
        }
    }
}*/
