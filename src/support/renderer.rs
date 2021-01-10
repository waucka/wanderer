use ash::version::DeviceV1_0;
use ash::vk;
use anyhow::anyhow;

use std::ffi::CString;
use std::time::{Instant, Duration};
use std::rc::Rc;
use std::ptr;
use std::pin::Pin;

use super::{Device, InnerDevice, Queue};
use super::image::{Image, ImageView, ImageBuilder};
use super::shader::{VertexShader, FragmentShader, Vertex, GenericShader};
use super::descriptor::DescriptorSetLayout;

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

    present_queue: Rc<Queue>,
}

impl Presenter {
    pub fn get_swapchain_image_formats(device: &Device) -> (vk::Format, vk::Format) {
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
	(surface_format.format, depth_format)
    }

    pub fn new(
	device: &Device,
	render_pass_builder: RenderPassBuilder,
	msaa_samples: vk::SampleCountFlags,
	desired_fps: u32,
    ) -> anyhow::Result<Self> {
        let render_pass = RenderPass::new(
            &device,
            msaa_samples,
	    render_pass_builder,
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

	// TODO: figure this out!
	let max_frames_in_flight = swapchain.get_num_images();//std::cmp::min(2, swapchain.get_num_images());
	dbg!(max_frames_in_flight);

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
    // Fuck that.  It looks like I can't avoid it for the pipeline, so I'll
    // have to figure out how to signal the engine to do that.
    // Same deal with the command buffers.
    pub fn fit_to_window(&mut self) -> anyhow::Result<()> {
        unsafe {
            self.device.device
                .device_wait_idle()?;
        }

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

    pub fn get_sync_objects(&self) -> (vk::Semaphore, vk::Semaphore, vk::Fence) {
	(
	    self.image_available_semaphores[self.current_frame],
	    self.render_finished_semaphores[self.current_frame],
	    self.inflight_fences[self.current_frame],
	)
    }

    pub fn present_frame<F>(&mut self, image_index: u32, viewport_update: &mut F) -> anyhow::Result<()>
    where
        F: FnMut(usize, usize) -> anyhow::Result<()>
    {
	//println!("Presenting a frame...");
	//let start = std::time::Instant::now();
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

        self.current_frame = (self.current_frame + 1) % self.max_frames_in_flight;
	self.last_frame_duration = self.last_frame.elapsed();
        self.last_frame = Instant::now();
	//println!("Presented frame in {}ns", start.elapsed().as_nanos());
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
	dbg!(swapchain_support.capabilities.min_image_count);
	dbg!(swapchain_support.capabilities.max_image_count);
	dbg!(image_count);

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

	    // TODO: is this the right thing to do?
	    let attachments = vec![
		&color_image_view,
		&depth_image_view,
		&imageview,
	    ];

	    let framebuffer = Framebuffer::new(
		device.clone(),
		extent.width,
		extent.height,
		&attachments,
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

pub struct Pipeline<V>
where
    V: Vertex,
{
    device: Rc<InnerDevice>,
    pipeline_layout: vk::PipelineLayout,
    pub (in super) pipeline: vk::Pipeline,
    vert_shader: VertexShader<V>,
    frag_shader: FragmentShader,
    msaa_samples: vk::SampleCountFlags,
}

impl<V> Pipeline<V>
where
    V: Vertex,
{
    fn from_inner(
	device: Rc<InnerDevice>,
	viewport_width: usize,
	viewport_height: usize,
	render_pass: &RenderPass,
	vert_shader: VertexShader<V>,
	frag_shader: FragmentShader,
	msaa_samples: vk::SampleCountFlags,
	set_layouts: &[&DescriptorSetLayout],
    ) -> anyhow::Result<Self> {
	let mut vk_set_layouts = vec![];
	for layout in set_layouts.iter() {
	    vk_set_layouts.push(layout.layout);
	}

	let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo{
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::PipelineLayoutCreateFlags::empty(),
            set_layout_count: vk_set_layouts.len() as u32,
            p_set_layouts: vk_set_layouts.as_ptr(),
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
	    pipeline_layout,
	    pipeline,
	    vert_shader,
	    frag_shader,
	    msaa_samples,
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
	set_layouts: &[&DescriptorSetLayout],
    ) -> anyhow::Result<Self> {
	Self::from_inner(
	    device.inner.clone(),
	    viewport_width,
	    viewport_height,
	    &presenter.render_pass,
	    vert_shader,
	    frag_shader,
	    msaa_samples,
	    set_layouts,
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
    ) -> anyhow::Result<vk::Pipeline> {
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
		    Err((_, res)) => return Err(
			anyhow!("Pipeline creation failed: {:?}", res)
		    ),
		}
	};
	Ok(pipelines[0])
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

    pub fn get_layout(&self) -> vk::PipelineLayout {
	self.pipeline_layout
    }
}

impl<V> Drop for Pipeline<V>
where
    V: Vertex,
{
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_pipeline(self.pipeline, None);
	    self.device.device.destroy_pipeline_layout(self.pipeline_layout, None);
	}
    }
}

pub struct AttachmentDescription {
    is_multisampled: bool,
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    stencil_load_op: vk::AttachmentLoadOp,
    stencil_store_op: vk::AttachmentStoreOp,
    initial_layout: vk::ImageLayout,
    final_layout: vk::ImageLayout,
}

impl AttachmentDescription {
    pub fn new(
	is_multisampled: bool,
	format: vk::Format,
	load_op: vk::AttachmentLoadOp,
	store_op: vk::AttachmentStoreOp,
	stencil_load_op: vk::AttachmentLoadOp,
	stencil_store_op: vk::AttachmentStoreOp,
	initial_layout: vk::ImageLayout,
	final_layout: vk::ImageLayout,
    ) -> Self {
	Self{
	    is_multisampled,
	    format,
	    load_op,
	    store_op,
	    stencil_load_op,
	    stencil_store_op,
	    initial_layout,
	    final_layout,
	}
    }

    pub fn standard_color_intermediate(format: vk::Format, is_multisampled: bool) -> Self {
	Self{
	    is_multisampled,
	    format,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
	}
    }

    pub fn standard_color_final(format: vk::Format, should_clear: bool) -> Self {
	Self{
	    is_multisampled: false,
	    format,
            load_op: if should_clear {
		vk::AttachmentLoadOp::CLEAR
	    } else {
		vk::AttachmentLoadOp::DONT_CARE
	    },
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
	}
    }

    pub fn standard_depth(format: vk::Format, is_multisampled: bool) -> Self {
	Self{
	    is_multisampled,
	    format,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
	}
    }

    fn as_vk(&self, msaa_samples: vk::SampleCountFlags) -> vk::AttachmentDescription {
	vk::AttachmentDescription{
            flags: vk::AttachmentDescriptionFlags::empty(),
            format: self.format,
            samples: if self.is_multisampled {
		msaa_samples
	    } else {
		vk::SampleCountFlags::TYPE_1
	    },
            load_op: self.load_op,
            store_op: self.store_op,
            stencil_load_op: self.stencil_load_op,
            stencil_store_op: self.stencil_store_op,
            initial_layout: self.initial_layout,
            final_layout: self.final_layout,
	}
    }
}

pub struct Subpass {
    pipeline_bind_point: vk::PipelineBindPoint,
    input_attachments: Vec<vk::AttachmentReference>,
    color_attachments: Vec<vk::AttachmentReference>,
    depth_attachment: Option<vk::AttachmentReference>,
    resolve_attachments: Vec<vk::AttachmentReference>,
}

struct SubpassData {
    input_attachments: Pin<Vec<vk::AttachmentReference>>,
    color_attachments: Pin<Vec<vk::AttachmentReference>>,
    // TODO: I don't like making this a vec, but the alternative
    //       will likely be too obtuse/confusing/non-borrow-checker-friendly.
    depth_attachment: Pin<Vec<vk::AttachmentReference>>,
    resolve_attachments: Pin<Vec<vk::AttachmentReference>>,
}

impl Subpass {
    pub fn new(pipeline_bind_point: vk::PipelineBindPoint) -> Self {
	Self{
	    pipeline_bind_point,
	    input_attachments: Vec::new(),
	    color_attachments: Vec::new(),
	    depth_attachment: None,
	    resolve_attachments: Vec::new(),
	}
    }

    pub fn add_input_attachment(&mut self, att: vk::AttachmentReference) {
	self.input_attachments.push(att);
    }

    pub fn add_color_attachment(
	&mut self,
	att: vk::AttachmentReference,
	att_resolve: Option<vk::AttachmentReference>,
    ) {
	// This function is a bit tricky, since we need to have either one resolve
	// attachment per color attachment or none at all.
	if self.color_attachments.len() > self.resolve_attachments.len()
	    && att_resolve.is_some() {
	    panic!("This subpass already has color attachments without resolve attachments!");
	}
	if self.color_attachments.len() == self.resolve_attachments.len()
	    && att_resolve.is_none() {
	    panic!("This subpass requires a resolve attachment for every color attachment!");
	}
	match att.layout {
	    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL |
	    vk::ImageLayout::GENERAL => (),
	    _ => panic!(
		"Invalid image layout {:?} for a color attachment!",
		att.layout,
	    ),
	}
	self.color_attachments.push(att);
	if let Some(att_resolve) = att_resolve {
	    match att_resolve.layout {
		vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL |
		vk::ImageLayout::GENERAL => (),
		_ => panic!(
		    "Invalid image layout {:?} for a resolve attachment!",
		    att_resolve.layout,
		),
	    }
	    self.resolve_attachments.push(att_resolve);
	}
    }

    pub fn set_depth_attachment(&mut self, att: vk::AttachmentReference) {
	match att.layout {
	    vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL |
	    vk::ImageLayout::GENERAL => (),
	    _ => panic!(
		"Invalid image layout {:?} for a depth attachment!",
		att.layout,
	    ),
	}
	self.depth_attachment = Some(att);
    }

    fn to_vk(self) -> (SubpassData, vk::SubpassDescription) {
	let (pipeline_bind_point, subpass_data) = match self {
	    Self{
		pipeline_bind_point,
		input_attachments,
		color_attachments,
		depth_attachment,
		resolve_attachments,
	    } => {
		(
		    pipeline_bind_point,
		    SubpassData{
			input_attachments: Pin::new(input_attachments),
			color_attachments: Pin::new(color_attachments),
			depth_attachment: if let Some(att) = depth_attachment {
			    Pin::new(vec![att])
			} else {
			    Pin::new(Vec::new())
			},
			resolve_attachments: Pin::new(resolve_attachments),
		    },
		)
	    },
	};

	let input_attachment_count = subpass_data.input_attachments.len() as u32;
	let p_input_attachments = if subpass_data.input_attachments.len() == 0 {
	    ptr::null()
	} else {
	    subpass_data.input_attachments.as_ptr()
	};
	let color_attachment_count = subpass_data.color_attachments.len() as u32;
	let p_color_attachments = if subpass_data.color_attachments.len() == 0 {
	    ptr::null()
	} else {
	    subpass_data.color_attachments.as_ptr()
	};
	let p_resolve_attachments = if subpass_data.resolve_attachments.len() == 0 {
	    ptr::null()
	} else {
	    subpass_data.resolve_attachments.as_ptr()
	};
	let p_depth_stencil_attachment = if subpass_data.depth_attachment.len() == 0 {
	    ptr::null()
	} else {
	    subpass_data.depth_attachment.as_ptr()
	};

	(
	    subpass_data,
	    vk::SubpassDescription{
		flags: vk::SubpassDescriptionFlags::empty(),
		pipeline_bind_point,
		input_attachment_count,
		p_input_attachments,
		color_attachment_count,
		p_color_attachments,
		p_resolve_attachments,
		p_depth_stencil_attachment,
		preserve_attachment_count: 0,
		p_preserve_attachments: ptr::null(),
	    },
	)
    }
}

pub struct RenderPassBuilder {
    attachments: Vec<AttachmentDescription>,
    subpasses: Vec<Subpass>,
    deps: Vec<vk::SubpassDependency>,
}

impl RenderPassBuilder {
    pub fn new() -> Self {
	Self{
	    attachments: Vec::new(),
	    subpasses: Vec::new(),
	    deps: Vec::new(),
	}
    }

    pub fn with_attachment(mut self, att: AttachmentDescription) -> Self {
	self.attachments.push(att);
	self
    }

    pub fn with_subpass(mut self, subpass: Subpass) -> Self {
	let att_sets = [
	    ("input", &subpass.input_attachments),
	    ("color", &subpass.color_attachments),
	    ("resolve", &subpass.resolve_attachments),
	];
	for (att_type, att_set) in att_sets.iter() {
	    for (i, att_ref) in att_set.iter().enumerate() {
		if att_ref.attachment as usize > self.attachments.len() {
		    panic!(
			"Invalid {} attachment {}={} (we only have {})",
			att_type,
			i,
			att_ref.attachment,
			self.attachments.len(),
		    );
		}
	    }
	}

	if let Some(att_ref) = &subpass.depth_attachment {
	    if att_ref.attachment as usize > self.attachments.len() {
		panic!(
		    "Invalid depth attachment {} (we only have {})",
		    att_ref.attachment,
		    self.attachments.len(),
		);
	    }
	}

	self.subpasses.push(subpass);
	self
    }

    pub fn with_standard_entry_dep(self) -> Self {
	self.with_dep(vk::SubpassDependency{
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: vk::AccessFlags::empty(),
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
		| vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dependency_flags: vk::DependencyFlags::empty(),
	})
    }

    pub fn with_dep(mut self, dep: vk::SubpassDependency) -> Self {
	if dep.src_subpass != vk::SUBPASS_EXTERNAL
	    && dep.src_subpass as usize > self.subpasses.len() {
		panic!(
		    "Invalid source subpass {} (we only have {})",
		    dep.src_subpass,
		    self.subpasses.len(),
		);
	    }
	if dep.dst_subpass != vk::SUBPASS_EXTERNAL
	    && dep.dst_subpass as usize > self.subpasses.len() {
		panic!(
		    "Invalid destination subpass {} (we only have {})",
		    dep.dst_subpass,
		    self.subpasses.len(),
		);
	    }
	self.deps.push(dep);
	self
    }
}

pub struct RenderPass {
    device: Rc<InnerDevice>,
    pub (in super) render_pass: vk::RenderPass,
}

impl RenderPass {
    fn new(
	device: &Device,
	msaa_samples: vk::SampleCountFlags,
	builder: RenderPassBuilder,
    ) -> anyhow::Result<Self> {
	let (attachments, subpasses, _subpass_data, deps) = match builder {
	    RenderPassBuilder{
		attachments,
		mut subpasses,
		deps,
	    } => {
		let mut vk_attachments = Vec::new();
		let mut vk_subpasses = Vec::new();
		let mut _subpass_data = Vec::new();
		for att in attachments.iter() {
		    vk_attachments.push(att.as_vk(msaa_samples));
		}
		for subpass in subpasses.drain(..) {
		    let (subpass_data, vk_subpass) = subpass.to_vk();
		    _subpass_data.push(subpass_data);
		    vk_subpasses.push(vk_subpass);
		}
		(vk_attachments, vk_subpasses, _subpass_data, deps)
	    }
	};

	let render_pass_create_info = vk::RenderPassCreateInfo{
            s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::RenderPassCreateFlags::empty(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: deps.len() as u32,
            p_dependencies: deps.as_ptr(),
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

pub struct Framebuffer {
    device: Rc<InnerDevice>,
    framebuffer: vk::Framebuffer,
}

impl Framebuffer {
    fn new(
	device: Rc<InnerDevice>,
	width: u32,
	height: u32,
	attachments: &[&ImageView],
	render_pass: vk::RenderPass,
    ) -> anyhow::Result<Self> {
        let mut vk_attachments = vec![];
	for att in attachments {
	    vk_attachments.push(att.view);
	}
	

        let framebuffer_create_info = vk::FramebufferCreateInfo{
	    s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::FramebufferCreateFlags::empty(),
	    render_pass: render_pass,
	    attachment_count: vk_attachments.len() as u32,
	    p_attachments: vk_attachments.as_ptr(),
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
