use ash::version::DeviceV1_0;
use ash::vk;
use anyhow::anyhow;
use image::{GenericImageView, DynamicImage};

use std::sync::Arc;
use std::rc::Rc;
use std::ptr;
use std::path::Path;
use std::cmp::max;

use super::{Device, InnerDevice};
use super::image::{Image, ImageView, ImageBuilder};
use super::buffer::UploadSourceBuffer;

pub struct Texture {
    pub (in super) image: Rc<Image>,
    pub (in super) image_view: Rc<ImageView>,
    mip_levels: u32,
}

impl Texture {
    pub fn from_float_tex(
        device: &Device,
        path: &Path,
    ) -> anyhow::Result<Self> {
        let data_bytes = std::fs::read(path)?;
        if data_bytes.len() % 4 != 0 {
            return Err(anyhow!("Float texture is {} bytes long; which is not a multiple of 4.", data_bytes.len()));
        }
        let num_values = data_bytes.len() / 4;
        let mip_levels = 1;
        /*let data_float = Vec::with_capacity(num_values);
        for i in 0..num_values {
            data_float.push(f32::from_le_bytes([
                data_bytes[i + 0],
                data_bytes[i + 1],
                data_bytes[i + 2],
                data_bytes[i + 3],
            ]));
        }*/
        let image_size =
            (std::mem::size_of::<u8>() * data_bytes.len()) as vk::DeviceSize;
        let upload_buffer = UploadSourceBuffer::new(device, image_size)?;
        upload_buffer.copy_data(&data_bytes)?;

        let mut image = Image::new(
            device,
            ImageBuilder::new1d(num_values as usize)
                .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(vk::Format::R32G32B32_SFLOAT)
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

        unsafe {
            Image::copy_buffer_no_deps(&upload_buffer, &image)?;
        }

        image.transition_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mip_levels,
        )?;

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
        })
    }

    pub fn from_exr(
        device: &Device,
        path: &Path,
    ) -> anyhow::Result<Self> {
        let image = {
            use exr::prelude::*;
            read()
                .no_deep_data()
                .largest_resolution_level()
                .specific_channels()
                .required("R")
                .required("G")
                .required("B")
                .collect_pixels(
                    |resolution, _| {
                        let num_values = resolution.width() * resolution.height() * 3;
                        let empty_image = vec![0.0; num_values];
                        empty_image
                    },
                    |pixel_vector, position, (r, g, b): (f32, f32, f32)| {
                        println!(
                            "[{}, {}] = ({}, {}, {})",
                            position.x(), position.y(),
                            r, g, b,
                        );
                        let y = position.y() + 1;
                        pixel_vector[y * position.x() * 3 + 0] = r;
                        pixel_vector[y * position.x() * 3 + 1] = g;
                        pixel_vector[y * position.x() * 3 + 2] = b;
                    },
                )
                .all_layers()
                .all_attributes()
                .from_file(path)?
        };

        let first_layer = match image.layer_data.first() {
            Some(layer) => layer,
            None => return Err(anyhow!("OpenEXR image {} contains no layers", path.display())),
        };
        let pixels = &first_layer.channel_data.pixels;
        dbg!(pixels[pixels.len()-3]);
        dbg!(pixels[pixels.len()-2]);
        dbg!(pixels[pixels.len()-1]);

        let image_size =
            (std::mem::size_of::<f32>() * pixels.len()) as vk::DeviceSize;
        let upload_buffer = UploadSourceBuffer::new(device, image_size)?;
        upload_buffer.copy_data(pixels)?;

        let mip_levels = 1;
        let mut image = Image::new(
            device,
            ImageBuilder::new1d(pixels.len() / 3)
                .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(vk::Format::R32G32B32_SFLOAT)
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

        unsafe {
            Image::copy_buffer_no_deps(&upload_buffer, &image)?;
        }

        image.transition_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mip_levels,
        )?;

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
        })
    }

    pub fn from_egui(
        device: &Device,
        egui_texture: &Arc<egui::paint::Texture>,
    ) -> anyhow::Result<Self> {
        let (image_width, image_height) = (egui_texture.width as u32, egui_texture.height as u32);
        let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
        let mip_levels = 1;

        let upload_buffer = UploadSourceBuffer::new(device, image_size)?;
        let srgba_pixels: Vec<egui::paint::Color32> = egui_texture.srgba_pixels().collect();
        upload_buffer.copy_data(&srgba_pixels)?;

        let mut image = Image::new(
            device,
            ImageBuilder::new2d(image_width as usize, image_height as usize)
                .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(vk::Format::R8G8B8A8_SRGB)
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

        unsafe {
            Image::copy_buffer_no_deps(&upload_buffer, &image)?;
        }

        image.transition_layout(
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            mip_levels,
        )?;

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
        })
    }

    pub fn from_image_builder(
        device: &Device,
        aspect: vk::ImageAspectFlags,
        mip_levels: u32,
        desired_layout: vk::ImageLayout,
        builder: ImageBuilder,
    ) -> anyhow::Result<Self> {
        Self::from_image_builder_internal(
            device.inner.clone(),
            aspect,
            mip_levels,
            desired_layout,
            builder,
        )
    }

    pub (in super) fn from_image_builder_internal(
        device: Rc<InnerDevice>,
        aspect: vk::ImageAspectFlags,
        mip_levels: u32,
        desired_layout: vk::ImageLayout,
        builder: ImageBuilder,
    ) -> anyhow::Result<Self> {
        let mut image = Image::new_internal(device.clone(), builder)?;
        image.transition_layout(vk::ImageLayout::UNDEFINED, desired_layout, mip_levels)?;
        let image_view = Rc::new(ImageView::from_image(
            &image,
            aspect,
            mip_levels,
        )?);
        let image = Rc::new(image);
        Ok(Self{
            image,
            image_view,
            mip_levels,
        })
    }

    #[allow(unused)]
    pub fn get_image_debug_str(&self) -> String {
        format!("{:?}", self.image.img)
    }

    pub fn from_file(device: &Device, image_path: &Path, srgb: bool, mipmapped: bool) -> anyhow::Result<Self> {
        let start = std::time::Instant::now();
        let image_object = match image::open(image_path) {
            Ok(v) => v,
            Err(e) => return Err(anyhow!("{}: {}", image_path.display(), e)),
        };
        println!("Loaded {} in {}ms", image_path.display(), start.elapsed().as_millis());
        Self::from_image(device, image_object, srgb, mipmapped)
    }

    pub fn from_image(device: &Device, image_object: DynamicImage, srgb: bool, mipmapped: bool) -> anyhow::Result<Self> {
        let (image_width, image_height) = (image_object.width(), image_object.height());
        let image_size =
            (std::mem::size_of::<u8>() as u32 * image_width * image_height * 4) as vk::DeviceSize;
        let image_data = match &image_object {
            DynamicImage::ImageLuma8(_) |
            DynamicImage::ImageBgr8(_) |
            DynamicImage::ImageRgb8(_) |
            DynamicImage::ImageLumaA8(_) |
            DynamicImage::ImageBgra8(_) |
            DynamicImage::ImageRgba8(_) => image_object.to_rgba8().into_raw(),
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

        let mut image = Image::new(
            device,
            ImageBuilder::new2d(image_width as usize, image_height as usize)
                .with_mip_levels(mip_levels)
                .with_num_samples(vk::SampleCountFlags::TYPE_1)
                .with_format(if srgb {
                    vk::Format::R8G8B8A8_SRGB
                } else {
                    vk::Format::R8G8B8A8_UNORM
                })
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

        unsafe {
            Image::copy_buffer_no_deps(&upload_buffer, &image)?;
        }

        if mipmapped {
            let start = std::time::Instant::now();
            image.generate_mipmaps(
                mip_levels,
            )?;
            println!("Generated mipmaps in {}ms", start.elapsed().as_millis());
        } else {
            image.transition_layout(
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mip_levels,
            )?;
        }

        let image_view = Rc::new(ImageView::from_image(
            &image,
            vk::ImageAspectFlags::COLOR,
            mip_levels,
        )?);

        let image = Rc::new(image);

        Ok(Self{
            image,
            image_view,
            mip_levels,
        })
    }

    pub fn get_mip_levels(&self) -> u32 {
        self.mip_levels
    }

    #[allow(unused)]
    pub fn get_descriptor_info(&self, sampler: &Sampler) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo{
            sampler: sampler.sampler,
            image_view: self.image_view.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
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
            // TODO: make this configurable
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

#[allow(unused)]
pub struct CombinedTexture {
    sampler: Sampler,
    texture: Texture,
}

impl CombinedTexture {
    #[allow(unused)]
    pub fn new(
        sampler: Sampler,
        texture: Texture,
    ) -> Self {
        Self{
            sampler,
            texture,
        }
    }

    #[allow(unused)]
    pub fn get_descriptor_info(&self) -> vk::DescriptorImageInfo {
        vk::DescriptorImageInfo{
            sampler: self.sampler.sampler,
            image_view: self.texture.image_view.view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }
}

pub struct Material {
    color: Rc<Texture>,
    normal: Rc<Texture>,
    // r = displacement, g = roughness, b = metalness, a = ambient occlusion
    material: Rc<Texture>,
    sampler: Rc<Sampler>,
}

impl Material {
    pub fn from_files(
        device: &Device,
        color_path: &Path,
        normal_path: &Path,
        material_path: &Path,
    ) -> anyhow::Result<Self> {
        let color_texture = Texture::from_file(device, color_path, true, true)?;
        let mip_levels = color_texture.get_mip_levels();
        Ok(Self {
            color: Rc::new(color_texture),
            normal: Rc::new(Texture::from_file(device, normal_path, false, true)?),
            material: Rc::new(Texture::from_file(device, material_path, false, true)?),
            sampler: Rc::new(Sampler::new(
                &device,
                mip_levels,
                vk::Filter::LINEAR,
                vk::Filter::LINEAR,
                vk::SamplerMipmapMode::LINEAR,
                vk::SamplerAddressMode::REPEAT,
            )?),
        })
    }

    pub fn get_sampler(&self) -> Rc<Sampler> {
        self.sampler.clone()
    }

    pub fn get_color_texture(&self) -> Rc<Texture> {
        self.color.clone()
    }

    pub fn get_normal_texture(&self) -> Rc<Texture> {
        self.normal.clone()
    }

    pub fn get_properties_texture(&self) -> Rc<Texture> {
        self.material.clone()
    }

    // TODO: is this legitimately unused?
    #[allow(unused)]
    pub fn get_color_descriptor_info(&self) -> vk::DescriptorImageInfo {
        self.color.get_descriptor_info(&self.sampler)
    }

    #[allow(unused)]
    pub fn get_normal_descriptor_info(&self) -> vk::DescriptorImageInfo {
        self.normal.get_descriptor_info(&self.sampler)
    }

    #[allow(unused)]
    pub fn get_material_descriptor_info(&self) -> vk::DescriptorImageInfo {
        self.material.get_descriptor_info(&self.sampler)
    }
}
