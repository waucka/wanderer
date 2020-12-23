use ash::version::DeviceV1_0;
use ash::vk;
use anyhow::anyhow;

use std::ffi::CString;
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::ptr;
use std::marker::PhantomData;

use super::{Device, InnerDevice, Queue};
use super::command_buffer::CommandBuffer;
use super::image::{Image, ImageView, ImageBuilder};
use super::texture::Texture;
use super::buffer::{VertexBuffer, IndexBuffer, UniformBuffer, HasBuffer};
use super::texture::Sampler;
use super::shader::{VertexShader, FragmentShader, Vertex, GenericShader};

pub struct Presenter {
    device: Rc<InnerDevice>,
    swapchain: Swapchain,
    pub (in super) render_pass: RenderPass,
    msaa_samples: vk::SampleCountFlags,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    inflight_fences: Vec<vk::Fence>,
    max_frames_in_flight: usize,
    last_frame: Instant,
    last_frame_duration: Duration,
    current_frame: usize,
    desired_fps: u32,

    graphics_queue: Rc<Queue>,
    present_queue: Rc<Queue>,
}

impl Presenter {
    pub fn new(
	device: &Device,
	msaa_samples: vk::SampleCountFlags,
	desired_fps: u32,
    ) -> anyhow::Result<Self> {
	let swapchain_support = super::query_swapchain_support(
	    device.inner.physical_device,
	    &device.inner.surface_loader,
	    device.inner.surface,
	);
	let surface_format = choose_swapchain_format(&swapchain_support.formats);
	let depth_format = super::utils::find_depth_format(
	    &device.inner.instance,
	    device.inner.physical_device,
	);

        let render_pass = RenderPass::new(
            &device,
            surface_format.format,
	    depth_format,
            msaa_samples,
        )?;

	let swapchain = Swapchain::new(device.inner.clone(), msaa_samples, &render_pass)?;

        let mut image_available_semaphores = vec![];
        let mut render_finished_semaphores = vec![];
        let mut inflight_fences = vec![];

	let semaphore_create_info = vk::SemaphoreCreateInfo{
            s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::SemaphoreCreateFlags::empty(),
	};

	let fence_create_info = vk::FenceCreateInfo{
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::FenceCreateFlags::SIGNALED,
	};

	// TODO: decide if I want to couple the in-flight image limit
	// to the swapchain size or not.
	let max_frames_in_flight = swapchain.get_num_images();

	for _ in 0..max_frames_in_flight {
            unsafe {
		let image_available_semaphore = device.inner.device
                    .create_semaphore(&semaphore_create_info, None)?;
		let render_finished_semaphore = device.inner.device
                    .create_semaphore(&semaphore_create_info, None)?;
		let inflight_fence = device.inner.device
                    .create_fence(&fence_create_info, None)?;

                image_available_semaphores
                    .push(image_available_semaphore);
                render_finished_semaphores
                    .push(render_finished_semaphore);
		inflight_fences.push(inflight_fence);
            }
	}

	Ok(Self{
	    device: device.inner.clone(),
	    swapchain,
	    render_pass,
	    msaa_samples,

	    image_available_semaphores,
	    render_finished_semaphores,
	    inflight_fences,
	    max_frames_in_flight,
	    current_frame: 0,
	    last_frame: Instant::now(),
	    last_frame_duration: Duration::new(0, 0),
	    desired_fps,

	    graphics_queue: device.inner.get_default_graphics_queue(),
	    present_queue: device.inner.get_default_present_queue(),
	})
    }

    // TODO: decide if I want this or the below swapchain image count function!
    pub fn get_num_images(&self) -> usize {
	self.max_frames_in_flight
    }

    pub fn get_dimensions(&self) -> (usize, usize) {
	self.swapchain.get_dimensions()
    }

    pub fn get_render_extent(&self) -> vk::Extent2D {
	self.swapchain.swapchain_extent
    }

    pub (in super) fn get_framebuffer(&self, framebuffer_index: usize) -> vk::Framebuffer {
	self.swapchain.frames[framebuffer_index].framebuffer.framebuffer
    }

    #[allow(unused)]
    pub fn set_desired_fps(&mut self, desired_fps: u32) {
	self.desired_fps = desired_fps;
    }

    pub fn get_current_fps(&self) -> u32 {
	let ns = self.last_frame_duration.as_nanos();
	if ns == 0 {
	    0
	} else {
	    let fps = 1_000_000_000_f64 / (ns as f64);
	    fps as u32
	}
    }

    pub fn wait_for_next_frame(&self) -> anyhow::Result<Duration> {
        let millis_since_last_frame = self.last_frame.elapsed().as_millis() as i64;
        let millis_until_next_frame = (
	    ((1_f32 / self.desired_fps as f32) * 1000_f32) as i64
	) - millis_since_last_frame;
        if millis_until_next_frame > 2 {
            //println!("Sleeping {}ms", millis_until_next_frame);
            std::thread::sleep(Duration::from_millis(millis_until_next_frame as u64));
        }

        let wait_fences = [self.inflight_fences[self.current_frame]];
        unsafe {
            self.device.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)?;
        }
	Ok(self.last_frame.elapsed())
    }

    pub fn acquire_next_image<F>(&mut self, viewport_update: &mut F) -> anyhow::Result<u32>
    where
        F: FnMut(usize, usize) -> anyhow::Result<()>
    {
        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.swapchain.swapchain_loader
                .acquire_next_image(
                    self.swapchain.swapchain,
                    std::u64::MAX,
                    self.image_available_semaphores[self.current_frame],
                    vk::Fence::null(),
                );
            match result {
                Ok(res) => res,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.fit_to_window()?;
			let (width, height) = self.get_dimensions();
			viewport_update(width, height)?;
			// TODO: I hope I don't regret doing this.
			return self.acquire_next_image(viewport_update);
                    },
                    _ => return Err(anyhow!("Failed to acquire swap chain image")),
                },
            }
        };

	Ok(image_index)
    }

    // This is allegedly not even close to complete.  I disagree.
    // The Vulkan tutorial says we need to re-create the following:
    // - swapchain
    // - image views
    // - render pass
    // - graphics pipeline
    // - framebuffers
    // - command buffers
    //
    // Of those, this re-creates the swapchain, image views, and framebuffers.
    // Discussion on Reddit indicates that the render pass only needs to be
    // re-created if the new swapchain images have a different format.
    // WHY THE FUCK WOULD THAT EVER HAPPEN?  Changing the color depth?
    // What is this, 1998?  I had to look up how to do that on a modern OS.
    // I'm not going to all that trouble to support people doing weird shit.
    // Fuck that.  Apparently, it's the same deal for the pipeline.
    // So, fuck that, too.  That leaves us with the command buffers, which
    // are currently the user's responsibility.  I hope to sidestep that in some way.
    pub fn fit_to_window(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.device.device
                .device_wait_idle()?;
        }
	//println!("Replacing swapchain!");

	//self.swapchain.replace(Swapchain::new(
	// TODO: I hope this doesn't cause any problems with order of creation/destruction.
	self.swapchain = Swapchain::new(
	    self.device.clone(),
	    self.msaa_samples,
	    &self.render_pass,
	)?;

	Ok(())
    }

    pub fn get_swapchain_image_count(&self) -> usize {
	self.swapchain.frames.len()
    }

    pub fn submit_graphics_command_buffer(
	&self,
	command_buffer: &CommandBuffer,
    ) -> anyhow::Result<()> {
	command_buffer.submit_synced(
	    self.graphics_queue.clone(),
	    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
	    self.image_available_semaphores[self.current_frame],
	    self.render_finished_semaphores[self.current_frame],
	    self.inflight_fences[self.current_frame],
	)
    }

    pub fn present_frame<F>(&mut self, image_index: u32, viewport_update: &mut F) -> anyhow::Result<()>
    where
        F: FnMut(usize, usize) -> anyhow::Result<()>
    {
        let swapchains = [self.swapchain.swapchain];
	let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
        let present_info = vk::PresentInfoKHR{
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
        };

        let result = unsafe {
            self.swapchain.swapchain_loader
                .queue_present(self.present_queue.get(), &present_info)
        };

	let is_resized = match result {
            Ok(_) => false,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => return Err(anyhow!("Failed to submit swap to presentation queue")),
            },
        };

	if is_resized {
	    self.fit_to_window()?;
	    let (width, height) = self.get_dimensions();
	    viewport_update(width, height)?;
	}

        self.current_frame = (self.current_frame + 1) % self.get_swapchain_image_count();
	self.last_frame_duration = self.last_frame.elapsed();
        self.last_frame = Instant::now();
	Ok(())
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
	//println!("Dropping a Presenter...");
	unsafe {
	    for sem in self.image_available_semaphores.iter() {
		self.device.device.destroy_semaphore(*sem, None);
	    }
	    for sem in self.render_finished_semaphores.iter() {
		self.device.device.destroy_semaphore(*sem, None);
	    }
	    for fence in self.inflight_fences.iter() {
		self.device.device.destroy_fence(*fence, None);
	    }
	}
    }
}

pub fn choose_swapchain_format(
    available_formats: &Vec<vk::SurfaceFormatKHR>,
) -> vk::SurfaceFormatKHR {
    for &available_format in available_formats.iter() {
        if available_format.format == vk::Format::B8G8R8A8_SRGB
            && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                return available_format.clone();
            }
    }

    available_formats.first().unwrap().clone()
}

pub fn choose_swapchain_present_mode(
    available_present_modes: &Vec<vk::PresentModeKHR>,
) -> vk::PresentModeKHR {
    for &available_present_mode in available_present_modes.iter() {
        if available_present_mode == vk::PresentModeKHR::MAILBOX {
            return available_present_mode;
        }
    }

    vk::PresentModeKHR::FIFO
}

fn choose_swapchain_extent(
    device: Rc<InnerDevice>,
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::max_value() {
        capabilities.current_extent
    } else {
        use num::clamp;

	let window_size = device.window.inner_size();

        vk::Extent2D{
            width: clamp(
                window_size.width,
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: clamp(
                window_size.height,
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

struct FrameData {
    _frame_index: u32,
    _image: Image,
    _imageview: ImageView,
    framebuffer: Framebuffer,
}

struct Swapchain {
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    _swapchain_format: vk::Format,
    pub (in super) swapchain_extent: vk::Extent2D,
    frames: Vec<FrameData>,

    _color_image: Image,
    _color_image_view: ImageView,
    _depth_image: Image,
    _depth_image_view: ImageView,
}

impl Swapchain {
    fn new(
	device: Rc<InnerDevice>,
	msaa_samples: vk::SampleCountFlags,
	render_pass: &RenderPass,
    ) -> anyhow::Result<Self> {
	let swapchain_support = super::query_swapchain_support(
	    device.physical_device,
	    &device.surface_loader,
	    device.surface,
	);

	let surface_format = choose_swapchain_format(&swapchain_support.formats);
	let present_mode = choose_swapchain_present_mode(&swapchain_support.present_modes);
	let extent = choose_swapchain_extent(device.clone(), &swapchain_support.capabilities);

	let image_count = swapchain_support.capabilities.min_image_count + 1;
	let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
	} else {
            image_count
	};

	let (image_sharing_mode, queue_family_indices) =
            if device.get_default_graphics_queue() != device.get_default_present_queue() {
		(
                    vk::SharingMode::CONCURRENT,
                    vec![
			device.get_default_graphics_queue().family_idx,
			device.get_default_present_queue().family_idx,
                    ],
		)
            } else {
		(vk::SharingMode::EXCLUSIVE, vec![])
            };
	let queue_family_index_count = queue_family_indices.len() as u32;

	let swapchain_create_info = vk::SwapchainCreateInfoKHR{
            s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
            p_next: ptr::null(),
            flags: vk::SwapchainCreateFlagsKHR::empty(),
            surface: device.surface,
            min_image_count: image_count,
            image_color_space: surface_format.color_space,
            image_format: surface_format.format,
            image_extent: extent,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            queue_family_index_count,
            pre_transform: swapchain_support.capabilities.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: vk::SwapchainKHR::null(),
            image_array_layers: 1,
	};

	let swapchain_loader = ash::extensions::khr::Swapchain::new(&device.instance, &device.device);
	let swapchain = unsafe {
            swapchain_loader
		.create_swapchain(&swapchain_create_info, None)?
	};

	let swapchain_images = unsafe {
            swapchain_loader
		.get_swapchain_images(swapchain)?
	};

	let color_image = Image::new_internal(
	    device.clone(),
	    ImageBuilder::new2d(extent.width, extent.height)
		.with_mip_levels(1)
		.with_num_samples(msaa_samples)
		.with_format(surface_format.format)
		.with_tiling(vk::ImageTiling::OPTIMAL)
		.with_usage(
		    vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
		)
		.with_required_memory_properties(vk::MemoryPropertyFlags::DEVICE_LOCAL),
	)?;
	let color_image_view = ImageView::from_image(
	    &color_image,
	    vk::ImageAspectFlags::COLOR,
	    1,
	)?;

	let depth_format = super::utils::find_depth_format(&device.instance, device.physical_device);
	let depth_image = Image::new_internal(device.clone(),
	    ImageBuilder::new2d(extent.width, extent.height)
		.with_mip_levels(1)
		.with_num_samples(msaa_samples)
		.with_format(depth_format)
		.with_tiling(vk::ImageTiling::OPTIMAL)
		.with_usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
		.with_required_memory_properties(vk::MemoryPropertyFlags::DEVICE_LOCAL),
	)?;
	let depth_image_view = ImageView::from_image(
	    &depth_image,
            vk::ImageAspectFlags::DEPTH,
            1,
	)?;

	let mut frames = Vec::new();
	let mut frame_index: u32 = 0;
	for swapchain_image in swapchain_images.iter() {
	    let image = Image::from_vk_image(
		device.clone(),
		*swapchain_image,
		vk::Extent3D{
		    width: extent.width,
		    height: extent.height,
		    depth: 0,
		},
		surface_format.format,
		vk::ImageType::TYPE_2D,
	    );
	    let imageview = ImageView::from_image(
		&image,
		vk::ImageAspectFlags::COLOR,
		1,
	    )?;

	    let framebuffer = Framebuffer::new(
		device.clone(),
		extent.width,
		extent.height,
		&color_image_view,
		&depth_image_view,
		&imageview,
		render_pass.render_pass,
	    )?;

	    frames.push(FrameData{
		_frame_index: frame_index,
		_image: image,
		_imageview: imageview,
		framebuffer,
	    });
	    frame_index += 1;
	}

	Ok(Self{
	    swapchain_loader,
	    swapchain,
	    _swapchain_format: surface_format.format,
	    swapchain_extent: extent,
	    frames,

	    _color_image: color_image,
	    _color_image_view: color_image_view,
	    _depth_image: depth_image,
	    _depth_image_view: depth_image_view,
	})
    }

    fn get_num_images(&self) -> usize {
	self.frames.len()
    }

    fn get_dimensions(&self) -> (usize, usize) {
	(self.swapchain_extent.width as usize, self.swapchain_extent.height as usize)
    }

    /*fn replace(&mut self, other: Self) {
	unsafe {
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);

	    self.swapchain_loader = other.swapchain_loader;
	    self.swapchain = other.swapchain;
	    self.swapchain_format = other.swapchain_format;
	    self.swapchain_extent = other.swapchain_extent;
	    self.frames = other.frames;

	    self.color_image = other.color_image;
	    self.color_image_view = other.color_image_view;
	    self.depth_image = other.depth_image;
	    self.depth_image_view = other.depth_image_view;
	}
	std::mem::forget(other);
    }*/
}

impl Drop for Swapchain {
    fn drop(&mut self) {
	// Images, image views, and framebuffers  will be destroyed by
	// their Drop implementations.  There's no need to destroy them here.
	unsafe {
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
	}
    }
}

////////////////////////////////////////////////////////////////////////////////
#[allow(unused)]
pub const GLOBAL_UNIFORM_SET: u32 = 0;
//------------------------------------------------------------------------------
pub const GLOBAL_UNIFORM_BINDING: u32 = 0;

////////////////////////////////////////////////////////////////////////////////
#[allow(unused)]
pub const TYPE_UNIFORM_SET: u32 = 1;
//------------------------------------------------------------------------------
pub const TYPE_UNIFORM_BINDING: u32 = 0;
pub const TEXTURE_ARRAY_BINDING: u32 = 1;

////////////////////////////////////////////////////////////////////////////////
#[allow(unused)]
pub const INSTANCE_UNIFORM_SET: u32 = 2;
//------------------------------------------------------------------------------
pub const INSTANCE_UNIFORM_BINDING: u32 = 0;

pub struct UniformSet<T> {
    uniform_struct: T,
    uniform_buffers: Vec<UniformBuffer<T>>,
}

impl<T> UniformSet<T> {
    pub fn new(
	device: &Device,
	uniform_struct: T,
	num_swapchain_images: usize,
    ) -> anyhow::Result<Self> {
	let mut uniform_buffers = vec![];
	for _ in 0..num_swapchain_images {
	    uniform_buffers.push(UniformBuffer::new(device, Some(&uniform_struct))?);
	}
	Ok(Self{
	    uniform_struct,
	    uniform_buffers,
	})
    }

    // Alters the uniform data and uploads it to the selected GPU buffer
    pub fn update_and_upload<F, R>(&mut self, i: usize, update_fn: F) -> anyhow::Result<R>
    where
	F: Fn(&mut T) -> anyhow::Result<R>
    {
	if i > self.uniform_buffers.len() {
	    return Err(anyhow!(
		"Attempted to update uniform buffer #{} of {}",
		i+1,
		self.uniform_buffers.len(),
	    ));
	}
	let result = update_fn(&mut self.uniform_struct)?;
	self.uniform_buffers[i].update(&self.uniform_struct)?;
	Ok(result)
    }

    // Alters the uniform data without altering any GPU buffers
    pub fn update<F, R>(&mut self, update_fn: F) -> anyhow::Result<R>
    where
	F: Fn(&mut T) -> anyhow::Result<R>
    {
	update_fn(&mut self.uniform_struct)
    }

    // Uploads whatever is currently in the uniform data to the selected GPU buffer
    #[allow(unused)]
    pub fn sync(&mut self, i: usize) -> anyhow::Result<()> {
	if i > self.uniform_buffers.len() {
	    return Err(anyhow!(
		"Attempted to sync uniform buffer #{} of {}",
		i+1,
		self.uniform_buffers.len(),
	    ));
	}
	self.uniform_buffers[i].update(&self.uniform_struct)
    }

    pub fn len(&self) -> usize {
	self.uniform_buffers.len()
    }

    #[allow(unused)]
    fn get_buffer(&self, i: usize) -> anyhow::Result<&UniformBuffer<T>> {
	if i < self.uniform_buffers.len() {
	    Ok(&self.uniform_buffers[i])
	} else {
	    Err(anyhow!(
		"Index out of range ({} uniform buffers, index {})",
		self.uniform_buffers.len(),
		i,
	    ))
	}
    }

    fn get_buffer_unchecked(&self, i: usize) -> &UniformBuffer<T> {
	&self.uniform_buffers[i]
    }
}

pub struct ObjectType<T, I> {
    uniform_set: UniformSet<T>,
    uniform_layout: DescriptorSetLayout,
    textures: Vec<(Rc<Sampler>, Rc<Texture>)>,
    type_pool: DescriptorPool,
    instance_pool: DescriptorPool,
    _phantom_t: PhantomData<T>,
    _phantom_i: PhantomData<I>,
}

impl<T, I> ObjectType<T, I> {
    pub fn new(
	device: &Device,
	uniform_struct: T,
	textures: Vec<(Rc<Sampler>, Rc<Texture>)>,
	num_swapchain_images: usize,
    ) -> anyhow::Result<Self> {
	let type_uniform_bindings = [
	    // TODO: OH GOD THIS IS IN TWO PLACES
	    vk::DescriptorSetLayoutBinding{
		binding: TYPE_UNIFORM_BINDING,
		descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		descriptor_count: 1,
		// TODO: is ALL the right choice here?
		stage_flags: vk::ShaderStageFlags::ALL,
		p_immutable_samplers: ptr::null(),
	    },
	    vk::DescriptorSetLayoutBinding{
		binding: TEXTURE_ARRAY_BINDING,
		descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		descriptor_count: textures.len() as u32,
		stage_flags: vk::ShaderStageFlags::FRAGMENT,
		p_immutable_samplers: ptr::null(),
	    },
	];
	let type_layout_info = vk::DescriptorSetLayoutCreateInfo{
	    s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
	    binding_count: type_uniform_bindings.len() as u32,
	    p_bindings: type_uniform_bindings.as_ptr(),
	};
	let uniform_layout = DescriptorSetLayout{
	    device: device.inner.clone(),
	    layout: unsafe {
		device.inner.device.create_descriptor_set_layout(&type_layout_info, None)?
	    },
	};
	Ok(Self{
	    uniform_set: UniformSet::new(device, uniform_struct, num_swapchain_images)?,
	    uniform_layout,
	    textures,
	    // TODO: automatically allocate more pools when these are exhausted.
	    type_pool: DescriptorPool::from_layout_bindings(
		device,
		&type_uniform_bindings,
		// 10 was too few here, and I don't know why.
		20,
	    )?,
	    // TODO: this has to be kept in sync with the stuff in RenderObject, which is
	    //       error-prone.  Find a better way.
	    instance_pool: DescriptorPool::new(
		device,
		&vec![(vk::DescriptorType::UNIFORM_BUFFER, 1)],
		1000,
	    )?,
	    _phantom_t: PhantomData,
	    _phantom_i: PhantomData,
	})
    }

    #[allow(unused)]
    pub fn get_uniform_set(&mut self) -> &mut UniformSet<T> {
	&mut self.uniform_set
    }

    pub fn create_descriptor_sets(
	&self,
	device: &Device,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
	let num_swapchain_images = self.uniform_set.len();
	let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..num_swapchain_images {
            layouts.push(self.uniform_layout.layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.type_pool.pool,
            descriptor_set_count: num_swapchain_images as u32,
            p_set_layouts: layouts.as_ptr(),
        };

        let descriptor_sets = unsafe {
            device.inner.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?
        };

        for i in 0..num_swapchain_images {
	    // The provenance of num_swapchain_images means i is never out of range.
	    let uniform_buffer = self.uniform_set.get_buffer_unchecked(i);
	    let uniform_buffer_info = [vk::DescriptorBufferInfo{
		buffer: uniform_buffer.get_buffer(),
		offset: 0,
		range: uniform_buffer.len(),
	    }];

	    let mut image_descriptor_infos = vec![];
	    for (sampler, texture) in self.textures.iter() {
		image_descriptor_infos.push(vk::DescriptorImageInfo{
		    sampler: sampler.sampler,
		    image_view: texture.image_view.view,
		    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
		});
	    }

	    let descriptor_write_sets = vec![
		vk::WriteDescriptorSet{
		    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
		    p_next: ptr::null(),
		    dst_set: descriptor_sets[i],
		    dst_binding: TYPE_UNIFORM_BINDING,
		    dst_array_element: 0,
		    descriptor_count: 1,
		    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		    p_image_info: ptr::null(),
		    p_buffer_info: uniform_buffer_info.as_ptr(),
		    p_texel_buffer_view: ptr::null(),
		},
		vk::WriteDescriptorSet{
		    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
		    p_next: ptr::null(),
		    dst_set: descriptor_sets[i],
		    dst_binding: TEXTURE_ARRAY_BINDING,
		    dst_array_element: 0,
		    descriptor_count: image_descriptor_infos.len() as u32,
		    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		    p_image_info: image_descriptor_infos.as_ptr(),
		    p_buffer_info: ptr::null(),
		    p_texel_buffer_view: ptr::null(),
		},
	    ];
            unsafe {
                device.inner.device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
	}
	Ok(descriptor_sets)
    }
}

// Type parameters are uniform types for global, per-type, and per-instance data.
pub struct RenderObject<G, T, I> {
    _phantom: PhantomData<G>,
    obj_type: Rc<ObjectType<T, I>>,
    uniform_set: UniformSet<I>,
    uniform_layout: DescriptorSetLayout,
}

impl<G, T, I> RenderObject<G, T, I> {
    pub fn new(
	device: &Device,
	obj_type: Rc<ObjectType<T, I>>,
	obj_uniform: I,
    ) -> anyhow::Result<Self> {
	let instance_uniform_bindings = [vk::DescriptorSetLayoutBinding{
	    binding: INSTANCE_UNIFORM_BINDING,
	    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
	    descriptor_count: 1,
	    // TODO: is ALL the right choice here?
	    stage_flags: vk::ShaderStageFlags::ALL,
	    p_immutable_samplers: ptr::null(),
	}];
	let instance_layout_info = vk::DescriptorSetLayoutCreateInfo{
	    s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
	    binding_count: 1,
	    p_bindings: instance_uniform_bindings.as_ptr(),
	};
	let uniform_layout = DescriptorSetLayout{
	    device: device.inner.clone(),
	    layout: unsafe {
		device.inner.device.create_descriptor_set_layout(&instance_layout_info, None)?
	    },
	};

	Ok(Self{
	    _phantom: PhantomData,
	    obj_type: obj_type.clone(),
	    uniform_set: UniformSet::new(device, obj_uniform, obj_type.uniform_set.len())?,
	    uniform_layout,
	})
    }

    #[allow(unused)]
    pub fn get_uniform_set(&mut self) -> &mut UniformSet<I> {
	&mut self.uniform_set
    }

    pub fn create_descriptor_sets(
	&self,
	device: &Device,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
	let num_swapchain_images = self.uniform_set.len();
	let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..num_swapchain_images {
            layouts.push(self.uniform_layout.layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.obj_type.instance_pool.pool,
            descriptor_set_count: num_swapchain_images as u32,
            p_set_layouts: layouts.as_ptr(),
        };

        let descriptor_sets = unsafe {
            device.inner.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?
        };

        for i in 0..num_swapchain_images {
	    // The provenance of num_swapchain_images ensures this will never fail.
	    let uniform_buffer = self.uniform_set.get_buffer_unchecked(i);
	    let uniform_buffer_info = [vk::DescriptorBufferInfo{
		buffer: uniform_buffer.get_buffer(),
		offset: 0,
		range: uniform_buffer.len(),
	    }];

	    let descriptor_write_sets = vec![
		vk::WriteDescriptorSet{
		    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
		    p_next: ptr::null(),
		    dst_set: descriptor_sets[i],
		    dst_binding: TYPE_UNIFORM_BINDING,
		    dst_array_element: 0,
		    descriptor_count: 1,
		    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		    p_image_info: ptr::null(),
		    p_buffer_info: uniform_buffer_info.as_ptr(),
		    p_texel_buffer_view: ptr::null(),
		},
	    ];
            unsafe {
                device.inner.device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
	}
	Ok(descriptor_sets)
    }

    pub fn write_draw_command<V>(
	&self,
	writer: super::command_buffer::RenderPass,
	renderer: &ObjectTypeRenderer<V, G, T, I>,
	vbo: &VertexBuffer<V>,
	ibo: &IndexBuffer,
	descriptor_sets: &[vk::DescriptorSet],
    ) where V: Vertex {
	writer.bind_pipeline(renderer.pipeline.pipeline);
	writer.draw_indexed(
	    renderer.pipeline_layout,
	    vbo,
	    ibo,
	    &descriptor_sets,
	);
    }
}

pub struct GlobalUniform<G> {
    uniform_set: UniformSet<G>, 
    uniform_layout: DescriptorSetLayout,
    global_pool: DescriptorPool,
}

impl<G> GlobalUniform<G> {
    pub fn new(
	device: &Device,
	uniform_struct: G,
	num_swapchain_images: usize,
    ) -> anyhow::Result<Self> {
	let uniform_set = UniformSet::new(device, uniform_struct, num_swapchain_images)?;
	let global_uniform_bindings = [vk::DescriptorSetLayoutBinding{
	    binding: GLOBAL_UNIFORM_BINDING,
	    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
	    descriptor_count: 1,
	    // TODO: is ALL the right choice here?
	    stage_flags: vk::ShaderStageFlags::ALL,
	    p_immutable_samplers: ptr::null(),
	}];
	let global_layout_info = vk::DescriptorSetLayoutCreateInfo{
	    s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
	    binding_count: 1,
	    p_bindings: global_uniform_bindings.as_ptr(),
	};
	Ok(Self{
	    uniform_set,
	    uniform_layout: DescriptorSetLayout{
		device: device.inner.clone(),
		layout: unsafe {
		    device.inner.device.create_descriptor_set_layout(&global_layout_info, None)?
		},
	    },
	    global_pool: DescriptorPool::new(
		device,
		&vec![(vk::DescriptorType::UNIFORM_BUFFER, 1)],
		// 10 was too few here, and I don't know why.
		20,
	    )?,
	})
    }

    pub fn get_uniform_set(&mut self) -> &mut UniformSet<G> {
	&mut self.uniform_set
    }

    pub fn create_descriptor_sets(
	&self,
	device: &Device,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
	let num_swapchain_images = self.uniform_set.len();
	let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..num_swapchain_images {
            layouts.push(self.uniform_layout.layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: self.global_pool.pool,
            descriptor_set_count: num_swapchain_images as u32,
            p_set_layouts: layouts.as_ptr(),
        };

        let descriptor_sets = unsafe {
            device.inner.device
                .allocate_descriptor_sets(&descriptor_set_allocate_info)?
        };

        for i in 0..num_swapchain_images {
	    // The provenance of num_swapchain_images ensures this will never fail.
	    let uniform_buffer = self.uniform_set.get_buffer_unchecked(i);
	    let uniform_buffer_info = [vk::DescriptorBufferInfo{
		buffer: uniform_buffer.get_buffer(),
		offset: 0,
		range: uniform_buffer.len(),
	    }];

	    let descriptor_write_sets = vec![
		vk::WriteDescriptorSet{
		    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
		    p_next: ptr::null(),
		    dst_set: descriptor_sets[i],
		    dst_binding: INSTANCE_UNIFORM_BINDING,
		    dst_array_element: 0,
		    descriptor_count: 1,
		    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		    p_image_info: ptr::null(),
		    p_buffer_info: uniform_buffer_info.as_ptr(),
		    p_texel_buffer_view: ptr::null(),
		},
	    ];
            unsafe {
                device.inner.device.update_descriptor_sets(&descriptor_write_sets, &[]);
            }
	}
	Ok(descriptor_sets)
    }
}

// Type parameters are:
//  * V: vertex type
//  * G: global uniform type
//  * T: uniform type for type-specific globals
//  * I: uniform type for per-instance data
pub struct ObjectTypeRenderer<V, G, T, I> where V: Vertex {
    device: Rc<InnerDevice>,
    _global_uniform_layout: DescriptorSetLayout,
    _type_uniform_layout: DescriptorSetLayout,
    _instance_uniform_layout: DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: Pipeline,
    vert_shader: VertexShader<V>,
    frag_shader: FragmentShader,
    msaa_samples: vk::SampleCountFlags,
    _phantom_g: PhantomData<G>,
    _phantom_t: PhantomData<T>,
    _phantom_i: PhantomData<I>,
}

impl<V, G, T, I> ObjectTypeRenderer<V, G, T, I>  where V: Vertex {
    fn from_inner(
	device: Rc<InnerDevice>,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: &RenderPass,
	vert_shader: VertexShader<V>,
	frag_shader: FragmentShader,
	msaa_samples: vk::SampleCountFlags,
	num_textures: usize,
    ) -> anyhow::Result<Self> {
	let set_layouts = {
	    let global_uniform_bindings = [vk::DescriptorSetLayoutBinding{
		binding: GLOBAL_UNIFORM_BINDING,
		descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		descriptor_count: 1,
		// TODO: is ALL the right choice here?
		stage_flags: vk::ShaderStageFlags::ALL,
		p_immutable_samplers: ptr::null(),
	    }];

	    // TODO: OH GOD THIS IS IN TWO PLACES
	    let type_uniform_bindings = [
		vk::DescriptorSetLayoutBinding{
		    binding: TYPE_UNIFORM_BINDING,
		    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		    descriptor_count: 1,
		    // TODO: is ALL the right choice here?
		    stage_flags: vk::ShaderStageFlags::ALL,
		    p_immutable_samplers: ptr::null(),
		},
		vk::DescriptorSetLayoutBinding{
		    binding: TEXTURE_ARRAY_BINDING,
		    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		    descriptor_count: num_textures as u32,
		    stage_flags: vk::ShaderStageFlags::FRAGMENT,
		    p_immutable_samplers: ptr::null(),
		},
	    ];

	    let instance_uniform_bindings = [vk::DescriptorSetLayoutBinding{
		binding: INSTANCE_UNIFORM_BINDING,
		descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		descriptor_count: 1,
		// TODO: is ALL the right choice here?
		stage_flags: vk::ShaderStageFlags::ALL,
		p_immutable_samplers: ptr::null(),
	    }];

	    let global_layout_info = vk::DescriptorSetLayoutCreateInfo{
		s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::DescriptorSetLayoutCreateFlags::empty(),
		binding_count: global_uniform_bindings.len() as u32,
		p_bindings: global_uniform_bindings.as_ptr(),
	    };
	    let type_layout_info = vk::DescriptorSetLayoutCreateInfo{
		s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::DescriptorSetLayoutCreateFlags::empty(),
		binding_count: type_uniform_bindings.len() as u32,
		p_bindings: type_uniform_bindings.as_ptr(),
	    };
	    let instance_layout_info = vk::DescriptorSetLayoutCreateInfo{
		s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::DescriptorSetLayoutCreateFlags::empty(),
		binding_count: instance_uniform_bindings.len() as u32,
		p_bindings: instance_uniform_bindings.as_ptr(),
	    };
	    unsafe {
		vec![
		    device.device.create_descriptor_set_layout(&global_layout_info, None)?,
		    device.device.create_descriptor_set_layout(&type_layout_info, None)?,
		    device.device.create_descriptor_set_layout(&instance_layout_info, None)?,
		]
	    }
	};

	let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
	};

	let pipeline_layout = unsafe {
            device.device
		.create_pipeline_layout(&pipeline_layout_create_info, None)?
	};

	let vert_shader_module = vert_shader.get_shader().shader;
	let frag_shader_module = frag_shader.get_shader().shader;
	let pipeline = Self::create_pipeline(
	    device.clone(),
	    viewport_width,
	    viewport_height,
	    render_pass.render_pass,
	    pipeline_layout,
	    vert_shader_module,
	    frag_shader_module,
	    msaa_samples,
	)?;

	Ok(Self{
	    device: device.clone(),
	    _global_uniform_layout: DescriptorSetLayout::new(device.clone(), set_layouts[0]),
	    _type_uniform_layout: DescriptorSetLayout::new(device.clone(), set_layouts[1]),
	    _instance_uniform_layout: DescriptorSetLayout::new(device.clone(), set_layouts[2]),
	    pipeline_layout,
	    pipeline,
	    vert_shader,
	    frag_shader,
	    msaa_samples,
	    _phantom_g: PhantomData,
	    _phantom_t: PhantomData,
	    _phantom_i: PhantomData,
	})
    }

    pub fn new(
	device: &Device,
	viewport_width: usize,
	viewport_height: usize,
	presenter: &Presenter,
	vert_shader: VertexShader<V>,
	frag_shader: FragmentShader,
	msaa_samples: vk::SampleCountFlags,
	num_textures: usize,
    ) -> anyhow::Result<Self> {
	Self::from_inner(
	    device.inner.clone(),
	    viewport_width,
	    viewport_height,
	    &presenter.render_pass,
	    vert_shader,
	    frag_shader,
	    msaa_samples,
	    num_textures,
	)
    }

    fn create_pipeline(
	device: Rc<InnerDevice>,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: vk::RenderPass,
	pipeline_layout: vk::PipelineLayout,
	vert_shader_module: vk::ShaderModule,
	frag_shader_module: vk::ShaderModule,
	msaa_samples: vk::SampleCountFlags,
    ) -> anyhow::Result<Pipeline> {
	let main_function_name = CString::new("main").unwrap();

	let shader_stages = [
            vk::PipelineShaderStageCreateInfo{
		s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineShaderStageCreateFlags::empty(),
		module: vert_shader_module,
		p_name: main_function_name.as_ptr(),
		p_specialization_info: ptr::null(),
		stage: vk::ShaderStageFlags::VERTEX,
            },
            vk::PipelineShaderStageCreateInfo{
		s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
		p_next: ptr::null(),
		flags: vk::PipelineShaderStageCreateFlags::empty(),
		module: frag_shader_module,
		p_name: main_function_name.as_ptr(),
		p_specialization_info: ptr::null(),
		stage: vk::ShaderStageFlags::FRAGMENT,
            },
	];

	let binding_description = V::get_binding_description();
	let attribute_description = V::get_attribute_descriptions();

	let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineVertexInputStateCreateFlags::empty(),
            vertex_attribute_description_count: attribute_description.len() as u32,
            p_vertex_attribute_descriptions: attribute_description.as_ptr(),
            vertex_binding_description_count: binding_description.len() as u32,
            p_vertex_binding_descriptions: binding_description.as_ptr(),
	};
	let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
            primitive_restart_enable: vk::FALSE,
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
	};

	let viewports = [vk::Viewport{
            x: 0.0,
            y: 0.0,
            width: viewport_width as f32,
            height: viewport_height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
	}];

	let scissors = [vk::Rect2D{
            offset: vk::Offset2D{ x: 0, y: 0 },
            extent: vk::Extent2D{
		width: viewport_width as u32,
		height: viewport_height as u32,
	    },
	}];

	let viewport_state_create_info = vk::PipelineViewportStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineViewportStateCreateFlags::empty(),
            scissor_count: scissors.len() as u32,
            p_scissors: scissors.as_ptr(),
            viewport_count: viewports.len() as u32,
            p_viewports: viewports.as_ptr(),
	};

	let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineRasterizationStateCreateFlags::empty(),
            depth_clamp_enable: vk::FALSE,
            cull_mode: vk::CullModeFlags::BACK,
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            rasterizer_discard_enable: vk::FALSE,
            depth_bias_clamp: 0.0,
            depth_bias_constant_factor: 0.0,
            depth_bias_enable: vk::FALSE,
            depth_bias_slope_factor: 0.0,
	};

	let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineMultisampleStateCreateFlags::empty(),
            rasterization_samples: msaa_samples,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_one_enable: vk::FALSE,
            alpha_to_coverage_enable: vk::FALSE,
	};

	let stencil_state = vk::StencilOpState{
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            compare_mask: 0,
            write_mask: 0,
            reference: 0,
	};
	let depth_state_create_info = vk::PipelineDepthStencilStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineDepthStencilStateCreateFlags::empty(),
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: stencil_state,
            back: stencil_state,
            max_depth_bounds: 1.0,
            min_depth_bounds: 0.0,
	};

	let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState{
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::all(),
            src_color_blend_factor: vk::BlendFactor::ONE,
            dst_color_blend_factor: vk::BlendFactor::ZERO,
            color_blend_op: vk::BlendOp::ADD,
            src_alpha_blend_factor: vk::BlendFactor::ONE,
            dst_alpha_blend_factor: vk::BlendFactor::ZERO,
            alpha_blend_op: vk::BlendOp::ADD,
	}];
	let color_blend_state = vk::PipelineColorBlendStateCreateInfo{
            s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineColorBlendStateCreateFlags::empty(),
            logic_op_enable: vk::FALSE,
            logic_op: vk::LogicOp::COPY,
            attachment_count: color_blend_attachment_states.len() as u32,
            p_attachments: color_blend_attachment_states.as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
	};

	let pipeline_create_infos = [vk::GraphicsPipelineCreateInfo{
            s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineCreateFlags::empty(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state_create_info,
            p_input_assembly_state: &vertex_input_assembly_state_info,
            p_tessellation_state: ptr::null(),
            p_viewport_state: &viewport_state_create_info,
            p_rasterization_state: &rasterization_state_create_info,
            p_multisample_state: &multisample_state_create_info,
            p_depth_stencil_state: &depth_state_create_info,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: ptr::null(),
            layout: pipeline_layout,
            render_pass: render_pass,
            subpass: 0,
            base_pipeline_handle: vk::Pipeline::null(),
            base_pipeline_index: -1,
	}];

	let pipelines = unsafe {
            match device.device
		.create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &pipeline_create_infos,
                    None,
		) {
		    Ok(p) => p,
		    Err(_) => return Err(
			anyhow!("Pipeline creation failed; this API sucks balls, so no more info for you.")
		    ),
		}
	};

	Ok(Pipeline{
	    device: device,
	    pipeline: pipelines[0],
	})
    }

    pub fn update_viewport(
	&mut self,
	viewport_width: usize,
	viewport_height: usize,
	presenter: &Presenter,
    ) -> anyhow::Result<()> {
	let vert_shader_module = self.vert_shader.get_shader().shader;
	let frag_shader_module = self.frag_shader.get_shader().shader;
	let pipeline = Self::create_pipeline(
	    self.device.clone(),
	    viewport_width,
	    viewport_height,
	    presenter.render_pass.render_pass,
	    self.pipeline_layout,
	    vert_shader_module,
	    frag_shader_module,
	    self.msaa_samples,
	)?;
	self.pipeline = pipeline;
	Ok(())
    }
}

impl<V, G, T, I> Drop for ObjectTypeRenderer<V, G, T, I>  where V: Vertex {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_pipeline_layout(self.pipeline_layout, None);
	}
    }
}

struct DescriptorSetLayout {
    device: Rc<InnerDevice>,
    layout: vk::DescriptorSetLayout,
}

impl DescriptorSetLayout {
    fn new(device: Rc<InnerDevice>, layout: vk::DescriptorSetLayout) -> Self {
	//println!("Creating a DescriptorSetLayout...");
	Self{
	    device,
	    layout,
	}
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
	//println!("Dropping a DescriptorSetLayout...");
	unsafe {
	    self.device.device.destroy_descriptor_set_layout(self.layout, None);
	}
    }
}

struct Pipeline {
    device: Rc<InnerDevice>,
    pipeline: vk::Pipeline,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_pipeline(self.pipeline, None);
	}
    }
}

pub struct RenderPass {
    device: Rc<InnerDevice>,
    pub (in super) render_pass: vk::RenderPass,
}

impl RenderPass {
    fn new(
	device: &Device,
	surface_format: vk::Format,
	depth_format: vk::Format,
	msaa_samples: vk::SampleCountFlags,
    ) -> anyhow::Result<Self> {
	let color_attachment = vk::AttachmentDescription{
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
	};

	let depth_attachment = vk::AttachmentDescription{
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: depth_format,
            samples: msaa_samples,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	let color_attachment_resolve = vk::AttachmentDescription{
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: surface_format,
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::DONT_CARE,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
	};

	let color_attachment_ref = vk::AttachmentReference{
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
	};

	let depth_attachment_ref = vk::AttachmentReference{
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	};

	let color_attachment_resolve_ref = vk::AttachmentReference{
            attachment: 2,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
	};

	let subpasses = [vk::SubpassDescription{
            flags: vk::SubpassDescriptionFlags::empty(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: 1,
            p_color_attachments: &color_attachment_ref,
            p_resolve_attachments: &color_attachment_resolve_ref,
            p_depth_stencil_attachment: &depth_attachment_ref,
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
	}];

	let render_pass_attachments = [color_attachment, depth_attachment, color_attachment_resolve];

	let subpass_dependencies = [vk::SubpassDependency{
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
		| vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
	}];

	let render_pass_create_info = vk::RenderPassCreateInfo{
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: render_pass_attachments.len() as u32,
            p_attachments: render_pass_attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: subpass_dependencies.len() as u32,
            p_dependencies: subpass_dependencies.as_ptr(),
	};

	unsafe {
            Ok(Self{
		device: device.inner.clone(),
		render_pass: device.inner.device
		    .create_render_pass(&render_pass_create_info, None)?,
	    })
	}
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_render_pass(self.render_pass, None);
	}
    }
}

struct DescriptorPool {
    device: Rc<InnerDevice>,
    pool: vk::DescriptorPool,
}

impl DescriptorPool {
    fn new(
	device: &Device,
	descriptors: &Vec<(vk::DescriptorType, u32)>,
	capacity: u32,
    ) -> anyhow::Result<Self> {
	let mut unique_sizes = std::collections::HashMap::new();
	for (ty, descriptor_count) in descriptors.iter() {
	    match unique_sizes.get_mut(ty) {
		Some(val) => {
		    *val += *descriptor_count;
		},
		None => {
		    unique_sizes.insert(*ty, *descriptor_count);
		},
	    };
	}
	let mut sizes = vec![];
	for (ty, descriptor_count) in unique_sizes.iter() {
	    sizes.push(vk::DescriptorPoolSize{
		ty: *ty,
		descriptor_count: *descriptor_count,
	    });
	}
	let create_info = vk::DescriptorPoolCreateInfo{
	    s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DescriptorPoolCreateFlags::empty(),
	    max_sets: capacity,
	    pool_size_count: sizes.len() as u32,
	    p_pool_sizes: sizes.as_ptr(),
	};
	Ok(Self{
	    device: device.inner.clone(),
	    pool: unsafe {
		device.inner.device.create_descriptor_pool(&create_info, None)?
	    },
	})
    }

    fn from_layout_bindings(
	device: &Device,
	layout_bindings: &[vk::DescriptorSetLayoutBinding],
	capacity: u32,
    ) -> anyhow::Result<Self> {
	let mut descriptors = vec![];
	for binding in layout_bindings.iter() {
	    descriptors.push((binding.descriptor_type, binding.descriptor_count));
	}
	Self::new(device, &descriptors, capacity)
    }

    // TODO: consider making this error out if any descriptor sets are still
    //       "at large" (e.g. store sets in objects that implement Drop and keep count).
    #[allow(unused)]
    fn reset(&self) -> anyhow::Result<()> {
	unsafe {
	    self.device.device.reset_descriptor_pool(
		self.pool,
		vk::DescriptorPoolResetFlags::empty(),
	    )?;
	}
	Ok(())
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_descriptor_pool(self.pool, None);
	}
    }
}

pub struct Framebuffer {
    device: Rc<InnerDevice>,
    framebuffer: vk::Framebuffer,
}

impl Framebuffer {
    fn new(
	device: Rc<InnerDevice>,
	width: u32,
	height: u32,
	color_image_view: &ImageView,
	depth_image_view: &ImageView,
	resolve_image_view: &ImageView,
	render_pass: vk::RenderPass,
    ) -> anyhow::Result<Self> {
        let attachments = [color_image_view.view, depth_image_view.view, resolve_image_view.view];

        let framebuffer_create_info = vk::FramebufferCreateInfo{
	    s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::FramebufferCreateFlags::empty(),
	    render_pass: render_pass,
	    attachment_count: attachments.len() as u32,
	    p_attachments: attachments.as_ptr(),
	    width,
	    height,
	    layers: 1,
        };

        Ok(Self{
	    device: device.clone(),
	    framebuffer: unsafe {
		device.device
                    .create_framebuffer(&framebuffer_create_info, None)?
            },
	})
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_framebuffer(self.framebuffer, None);
	}
    }
}
