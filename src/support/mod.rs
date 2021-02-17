use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::vk::{version_major, version_minor, version_patch};
use ash::vk;
use anyhow::anyhow;
use winit::event_loop::EventLoop;

use std::ptr;
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::rc::Rc;

use super::utils::vk_to_string;
use super::debug::VALIDATION;
use super::debug;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub mod utils;

macro_rules! impl_defaulted_setter {
    ( $fn_name:ident, $field_name:ident, str ) => {
    pub fn $fn_name(mut self, $field_name: &str) -> Self {
        self.$field_name.set_value($field_name.to_string());
        self
    }
    };
    ( $fn_name:ident, $field_name:ident, $type:ty, ref ) => {
    pub fn $fn_name(mut self, $field_name: &$type) -> Self {
        self.$field_name.set_value($field_name);
        self
    }
    };
    ( $fn_name:ident, $field_name:ident, $type:ty) => {
    pub fn $fn_name(mut self, $field_name: $type) -> Self {
        self.$field_name.set_value($field_name);
        self
    }
    };
}

pub mod buffer;
pub mod command_buffer;
pub mod image;
pub mod renderer;
pub mod shader;
pub mod texture;
pub mod descriptor;

use utils::Defaulted;
use command_buffer::CommandPool;

#[derive(Copy, Clone)]
pub struct FrameId {
    idx: usize,
}

impl FrameId {
    fn initial() -> Self {
	Self{
	    idx: 0,
	}
    }

    fn advance(&mut self) {
	self.idx = (self.idx + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn next(&self) -> Self {
	Self {
	    idx: (self.idx + 1) % MAX_FRAMES_IN_FLIGHT,
	}
    }
}

impl From<usize> for FrameId {
    fn from(idx: usize) -> Self {
	if idx >= MAX_FRAMES_IN_FLIGHT {
	    panic!(
		"Tried to create a FrameId with index {} (must be < {})",
		idx,
		MAX_FRAMES_IN_FLIGHT,
	    );
	}
	Self{
	    idx,
	}
    }
}

impl From<u32> for FrameId {
    fn from(idx: u32) -> Self {
	if idx as usize >= MAX_FRAMES_IN_FLIGHT {
	    panic!(
		"Tried to create a FrameId with index {} (must be < {})",
		idx,
		MAX_FRAMES_IN_FLIGHT,
	    );
	}
	Self{
	    idx: idx as usize,
	}
    }
}

impl std::fmt::Display for FrameId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.idx.fmt(f)
    }
}

pub struct PerFrameSet<T> {
    items: Vec<T>,
}

impl<T> PerFrameSet<T> {
    pub fn new<F>(mut item_generator: F) -> anyhow::Result<Self>
    where
        F: FnMut(FrameId) -> anyhow::Result<T>
    {
	let mut items = Vec::new();
	for i in 0..MAX_FRAMES_IN_FLIGHT {
	    items.push(item_generator(FrameId::from(i))?);
	}
	Ok(Self{
	    items,
	})
    }

    pub fn get(&self, frame: FrameId) -> &T {
	&self.items[frame.idx]
    }

    pub fn get_mut(&mut self, frame: FrameId) -> &mut T {
	&mut self.items[frame.idx]
    }

    // Extracts data from self and creates a new PerFrameSet containing the extracted data
    pub fn extract<F, R>(&self, extractor: F) -> anyhow::Result<PerFrameSet<R>>
    where
        F: Fn(&T) -> anyhow::Result<R>
    {
	let new_set: PerFrameSet<R> = PerFrameSet::new(|frame| {
	    extractor(self.get(FrameId::from(frame)))
	})?;
	Ok(new_set)
    }

    #[allow(unused)]
    pub fn foreach<F>(&mut self, mut action: F) -> anyhow::Result<()>
    where
        F: FnMut(FrameId, &mut T) -> anyhow::Result<()>
    {
	for i in 0..MAX_FRAMES_IN_FLIGHT {
	    let item = &mut self.items[i];
	    action(FrameId::from(i), item)?;
	}
	Ok(())
    }

    pub fn replace<F>(&mut self, mut constructor: F) -> anyhow::Result<()>
    where
        F: FnMut(FrameId, &T) -> anyhow::Result<T>
    {
	let mut new_items = Vec::new();
	for i in 0..MAX_FRAMES_IN_FLIGHT {
	    let old_item = &self.items[i];
	    new_items.push(constructor(FrameId::from(i), old_item)?);
	}
	self.items = new_items;
	Ok(())
    }
}

impl<T: Clone> Clone for PerFrameSet<T> {
    fn clone(&self) -> Self {
	Self{
	    items: self.items.clone(),
	}
    }
}

pub fn pick_physical_device(
    instance: &ash::Instance,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    required_extensions: &[String],
) -> vk::PhysicalDevice {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate physical devices")
    };
    println!(
        "{} devices (GPU) found with Vulkan support.",
        physical_devices.len()
    );

    for &physical_device in physical_devices.iter() {
        if is_physical_device_suitable(instance, physical_device, surface_loader, surface, required_extensions) {
            return physical_device;
        }
    }
    panic!("Failed to find a suitable GPU");
}

fn is_physical_device_suitable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    required_extensions: &[String],
) -> bool {
    let device_properties = unsafe{ instance.get_physical_device_properties(physical_device) };
    let device_features = unsafe{ instance.get_physical_device_features(physical_device) };
    let device_queue_families = unsafe{ instance.get_physical_device_queue_family_properties(physical_device) };

    let device_type = match device_properties.device_type {
        vk::PhysicalDeviceType::CPU => "cpu",
        vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
        vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
        vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
        vk::PhysicalDeviceType::OTHER => "Unknown",
        _ => "Unknown",
    };

    let device_name = vk_to_string(&device_properties.device_name);
    println!(
        "\tDevice name: {}, id: {}, type: {}",
        device_name, device_properties.device_id, device_type,
    );

    let major_version = version_major(device_properties.api_version);
    let minor_version = version_minor(device_properties.api_version);
    let patch_version = version_patch(device_properties.api_version);

    println!(
        "\tAPI version: {}.{}.{}",
        major_version, minor_version, patch_version,
    );

    println!("\tSupported queue families: {}", device_queue_families.len());
    println!("\t\tQueue count | Graphics, Compute, Transfer, Sparse Binding");
    for queue_family in device_queue_families.iter() {
        let is_graphics_supported = if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            "supported"
        } else  {
            "unsupported"
        };
        let is_compute_supported = if queue_family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            "supported"
        } else  {
            "unsupported"
        };
        let is_transfer_supported = if queue_family.queue_flags.contains(vk::QueueFlags::TRANSFER) {
            "supported"
        } else  {
            "unsupported"
        };
        let is_sparse_supported = if queue_family.queue_flags.contains(vk::QueueFlags::SPARSE_BINDING) {
            "supported"
        } else  {
            "unsupported"
        };
        println!(
            "\t\t{}\t    | {},  {},  {},  {}",
            queue_family.queue_count,
            is_graphics_supported,
            is_compute_supported,
            is_transfer_supported,
            is_sparse_supported,
        );
    }

    println!(
        "\tGeometry shader support: {}",
        if device_features.geometry_shader == 1 {
            "yes"
        } else {
            "no"
        },
    );

    let (graphics_queue_idx, present_queue_idx) = find_queue_family(instance, physical_device, surface_loader, surface);

    let is_queue_family_supported = graphics_queue_idx.is_some() && present_queue_idx.is_some();
    let is_device_extension_supported = check_device_extension_support(instance, physical_device, required_extensions);
    let is_swapchain_supported = if is_device_extension_supported {
        let swapchain_support = query_swapchain_support(physical_device, surface_loader, surface);
        !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
    } else {
        false
    };

    is_queue_family_supported && is_device_extension_supported && is_swapchain_supported
}

struct SwapChainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

fn query_swapchain_support(
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
) -> SwapChainSupport {
    unsafe {
        let capabilities = surface_loader
            .get_physical_device_surface_capabilities(physical_device, surface)
            .expect("Failed to query for surface capabilities.");
        let formats = surface_loader
            .get_physical_device_surface_formats(physical_device, surface)
            .expect("Failed to query for surface formats.");
        let present_modes = surface_loader
            .get_physical_device_surface_present_modes(physical_device, surface)
            .expect("Failed to query for surface present mode.");

        SwapChainSupport {
            capabilities,
            formats,
            present_modes,
        }
    }
}

fn find_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_loader: &ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
) -> (Option<u32>, Option<u32>) {
    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut graphics_queue_idx = None;
    let mut present_queue_idx = None;

    let mut index = 0;
    for queue_family in queue_families.iter() {
        if queue_family.queue_count > 0 && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            graphics_queue_idx = Some(index);
        }

        let is_present_supported = unsafe {
            surface_loader
                .get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface,
                )
                .expect("Failed to query present support")
        };
        if queue_family.queue_count > 0 && is_present_supported {
            present_queue_idx = Some(index)
        }

        if graphics_queue_idx.is_some() && present_queue_idx.is_some() {
            return (graphics_queue_idx, present_queue_idx);
        }

        index += 1;
    }
    return (None, None);
}

fn check_device_extension_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    required_extensions_list: &[String],
) -> bool {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to get device extension properties")
    };

    let mut available_extension_names = vec![];

    println!("\tAvailable device extensions:");
    for extension in available_extensions.iter() {
        let extension_name = vk_to_string(&extension.extension_name);
        println!(
            "\t\tName: {}, Version: {}",
            extension_name, extension.spec_version,
        );

        available_extension_names.push(extension_name);
    }

    let mut required_extensions: HashSet<String> = HashSet::new();
    for extension in required_extensions_list.iter() {
        required_extensions.insert(extension.to_string());
    }

    for extension_name in available_extension_names.iter() {
        required_extensions.remove(extension_name);
    }

    required_extensions.is_empty()
}

fn try_create_instance(
    entry: &ash::Entry,
    app_info: &vk::ApplicationInfo,
    required_extensions: &[*const i8],
    debug_utils_create_info: &vk::DebugUtilsMessengerCreateInfoEXT,
) -> anyhow::Result<ash::Instance> {
    let required_validation_layer_raw_names: Vec<CString> = VALIDATION
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const i8> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: if VALIDATION.is_enabled {
            debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                as *const c_void
        } else {
            ptr::null()
        },
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: app_info,
        pp_enabled_layer_names: if VALIDATION.is_enabled {
            enable_layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_layer_count: if VALIDATION.is_enabled {
            enable_layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_extension_names: required_extensions.as_ptr(),
        enabled_extension_count: required_extensions.len() as u32,
    };

    let instance = unsafe {
        entry
            .create_instance(&create_info, None)?
    };

    Ok(instance)
}

fn create_instance(
    entry: &ash::Entry,
    app_name: &str,
    engine_name: &str,
    app_version: u32,
    engine_version: u32,
    api_version: u32,
) -> anyhow::Result<ash::Instance> {
    if VALIDATION.is_enabled && !super::debug::check_validation_layer_support(entry) {
        panic!("Validation layers requested but not available!");
    }

    let c_app_name = CString::new(app_name).unwrap();
    let c_engine_name = CString::new(engine_name).unwrap();
    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        p_application_name: c_app_name.as_ptr(),
        application_version: app_version,
        p_engine_name: c_engine_name.as_ptr(),
        engine_version: engine_version,
        api_version: api_version,
    };

    let debug_utils_create_info = super::debug::populate_debug_messenger_create_info();
    let extension_names = super::platforms::required_extension_names();
    let mut try_extension_names = vec![];
    for ext in extension_names.iter() {
	try_extension_names.push(*ext);
    }
    for ext in super::platforms::optional_extension_names() {
	try_extension_names.push(ext);
    }
    let maybe_instance = try_create_instance(
	entry,
	&app_info,
	&try_extension_names,
	&debug_utils_create_info,
    );
    match maybe_instance {
	Ok(instance) => Ok(instance),
	Err(_) => try_create_instance(
	    entry,
	    &app_info,
	    &extension_names,
	    &debug_utils_create_info,
	),
    }
}

// Device

pub const ENGINE_NAME: &'static str = "Wanderer Engine";
pub const ENGINE_VERSION: u32 = vk::make_version(0, 1, 0);
pub const VULKAN_API_VERSION: u32 = vk::make_version(1, 2, 131);

pub struct Queue {
    family_idx: u32,
    queue_idx: u32,
    flags: vk::QueueFlags,
    can_present: bool,
    queue: vk::Queue,
}

impl Queue {
    fn new(
	device: Rc<InnerDevice>,
	family_idx: u32,
	queue_idx: u32,
	flags: vk::QueueFlags,
	can_present: bool,
    ) -> anyhow::Result<Self> {
	let queue = unsafe {
	    device.device.get_device_queue(family_idx, queue_idx)
	};

	Ok(Self{
	    family_idx,
	    queue_idx,
	    flags,
	    can_present,
	    queue,
	})
    }

    pub fn can_do_graphics(&self) -> bool {
        self.flags.contains(vk::QueueFlags::GRAPHICS)
    }

    pub fn can_present(&self) -> bool {
	self.can_present
    }

    #[allow(unused)]
    pub fn can_do_compute(&self) -> bool {
        self.flags.contains(vk::QueueFlags::COMPUTE)
    }

    pub fn can_do_transfer(&self) -> bool {
        self.flags.contains(vk::QueueFlags::TRANSFER)
    }

    #[allow(unused)]
    pub fn can_do_sparse_binding(&self) -> bool {
        self.flags.contains(vk::QueueFlags::SPARSE_BINDING)
    }

    fn get(&self) -> vk::Queue {
	self.queue
    }
}

impl std::cmp::PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
	self.family_idx == other.family_idx && self.queue_idx == other.queue_idx
    }
}
impl std::cmp::Eq for Queue {}

pub struct DeviceBuilder {
    window_title: Defaulted<String>,
    application_version: Defaulted<u32>,
    window_size: Defaulted<(u32, u32)>,
    extensions: Vec<String>,
    validation_enabled: Defaulted<bool>,
}

impl DeviceBuilder {
    pub fn new() -> Self {
        Self {
            window_title: Defaulted::new("Some Random Application".to_string()),
            application_version: Defaulted::new(vk::make_version(0, 1, 0)),
            window_size: Defaulted::new((640, 480)),
            extensions: Vec::new(),
	    validation_enabled: Defaulted::new(false),
        }
    }

    pub fn get_extensions(&self) -> &[String] {
	&self.extensions
    }

    impl_defaulted_setter!(with_window_title, window_title, str);
    impl_defaulted_setter!(with_validation, validation_enabled, bool);

    pub fn with_application_version(mut self, major: u32, minor: u32, patch: u32) -> Self {
        self.application_version.set_value(vk::make_version(major, minor, patch));
        self
    }

    pub fn with_window_size(mut self, width: usize, height: usize) -> Self {
        self.window_size.set_value((width as u32, height as u32));
        self
    }

    pub fn with_default_extensions(mut self) -> Self {
        self.extensions.push("VK_KHR_swapchain".to_string());
	self.extensions.push("VK_EXT_descriptor_indexing".to_string());
        self
    }

    #[allow(unused)]
    pub fn with_extension(mut self, extension_name: &str) -> Self {
        self.extensions.push(extension_name.to_string());
        self
    }
}

pub struct Device {
    inner: Rc<InnerDevice>,
}

impl Device {
    pub fn new(event_loop: &EventLoop<()>, builder: DeviceBuilder) -> anyhow::Result<Self> {
	Ok(Self {
	    inner: InnerDevice::new(event_loop, builder)?,
	})
    }

    pub fn check_mipmap_support(
	&self,
	image_format: vk::Format,
    ) -> anyhow::Result<()>{
	let format_properties = unsafe {
            self.inner.instance.get_physical_device_format_properties(self.inner.physical_device, image_format)
	};

	let is_sample_image_filter_linear_supported = format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
	if !is_sample_image_filter_linear_supported {
            Err(anyhow!("Texture image format does not support linear filtering"))
	} else {
	    Ok(())
	}
    }

    #[allow(unused)]
    pub fn get_max_usable_sample_count(
	&self,
    ) -> vk::SampleCountFlags {
	let physical_device_properties =
            unsafe { self.inner.instance.get_physical_device_properties(self.inner.physical_device) };
	let count = std::cmp::min(
            physical_device_properties
		.limits
		.framebuffer_color_sample_counts,
            physical_device_properties
		.limits
		.framebuffer_depth_sample_counts,
	);

	if count.contains(vk::SampleCountFlags::TYPE_64) {
            return vk::SampleCountFlags::TYPE_64;
	}
	if count.contains(vk::SampleCountFlags::TYPE_32) {
            return vk::SampleCountFlags::TYPE_32;
	}
	if count.contains(vk::SampleCountFlags::TYPE_16) {
            return vk::SampleCountFlags::TYPE_16;
	}
	if count.contains(vk::SampleCountFlags::TYPE_8) {
            return vk::SampleCountFlags::TYPE_8;
	}
	if count.contains(vk::SampleCountFlags::TYPE_4) {
            return vk::SampleCountFlags::TYPE_4;
	}
	if count.contains(vk::SampleCountFlags::TYPE_2) {
            return vk::SampleCountFlags::TYPE_2;
	}

	vk::SampleCountFlags::TYPE_1
    }

    pub fn wait_for_idle(&self) -> anyhow::Result<()>{
        unsafe {
            self.inner.device
                .device_wait_idle()?
        }
	Ok(())
    }

    pub fn window_ref(&self) -> &winit::window::Window {
        &self.inner.window
    }

    pub fn get_default_graphics_queue(&self) -> Rc<Queue> {
	self.inner.get_default_graphics_queue()
    }

    pub fn get_default_graphics_pool(&self) -> Rc<CommandPool> {
	self.inner.get_default_graphics_pool()
    }

    #[allow(unused)]
    pub fn get_default_present_queue(&self) -> Rc<Queue> {
	self.inner.get_default_present_queue()
    }

    pub fn get_default_transfer_queue(&self) -> Rc<Queue> {
	self.inner.get_default_transfer_queue()
    }

    pub fn get_default_transfer_pool(&self) -> Rc<CommandPool> {
	self.inner.get_default_transfer_pool()
    }

    #[allow(unused)]
    pub fn get_queues(&self) -> Vec<Rc<Queue>> {
	let mut queues = vec![];
	for q in &self.inner.queue_set.borrow().queues {
	    queues.push(q.clone());
	}
	queues
    }

    pub fn get_window_size(&self) -> (usize, usize) {
	let size = self.inner.window.inner_size();
	(size.width as usize, size.height as usize)
    }
}

struct QueueSet {
    queues: Vec<Rc<Queue>>,
    pools: Vec<Rc<CommandPool>>,
    // These three are indexes into the above vector ("queues").
    default_graphics_queue_idx: usize,
    default_present_queue_idx: usize,
    default_transfer_queue_idx: usize,
}

struct InnerDevice {
    window: winit::window::Window,
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    swapchain_loader: ash::extensions::khr::Swapchain,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_enabled: bool,

    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_set: RefCell<QueueSet>,
}

impl std::fmt::Debug for InnerDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
	f.debug_struct("InnerDevice")
	    .field("window", &self.window)
	    .field("surface", &self.surface)
	    .field("physical_device", &self.physical_device)
	    .finish()
    }
}

impl InnerDevice {
    pub fn new(event_loop: &EventLoop<()>, builder: DeviceBuilder) -> anyhow::Result<Rc<Self>> {
	let window_title = builder.window_title.get_value();
        let (window_width, window_height) = builder.window_size.get_value();
        let window = super::window::init_window(
            event_loop,
            window_title,
            *window_width,
            *window_height,
        );
        let entry = ash::Entry::new().unwrap();
        let instance = create_instance(
            &entry,
            window_title,
            ENGINE_NAME,
            *builder.application_version.get_value(),
            ENGINE_VERSION,
            VULKAN_API_VERSION,
        )?;
        let (debug_utils_loader, debug_messenger) = debug::setup_debug_utils(&entry, &instance);
        let surface = unsafe {
            super::platforms::create_surface(&entry, &instance, &window)?
        };
        let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
        let physical_device = pick_physical_device(
            &instance,
            &surface_loader,
            surface,
            builder.get_extensions(),
        );
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };
	let queue_infos = get_queue_info(
	    &instance,
	    physical_device,
	    surface,
	    &surface_loader,
	)?;
        let device = create_logical_device(
	    &instance,
	    physical_device,
	    &queue_infos,
	    builder.get_extensions(),
	);
	let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

        let this = Rc::new(Self{
            window,
            _entry: entry,
            instance,
            surface_loader,
            surface,
	    swapchain_loader,
            debug_utils_loader,
            debug_messenger,
	    validation_enabled: *builder.validation_enabled.get_value(),

            physical_device,
            memory_properties,
            device: device.clone(),
	    queue_set: RefCell::new(QueueSet {
		queues: Vec::new(),
		pools: Vec::new(),
		default_graphics_queue_idx: 0,
		default_present_queue_idx: 0,
		default_transfer_queue_idx: 0,
	    }),
        });

	{
	    let queues = get_queues_from_device(
		this.clone(),
		queue_infos,
	    )?;
	    let mut queue_set = this.queue_set.borrow_mut();

	    let mut maybe_graphics_queue_idx = None;
	    let mut maybe_present_queue_idx = None;
	    let mut maybe_transfer_queue_idx = None;
	    for (idx, queue) in queues.iter().enumerate() {
		if queue.can_do_graphics() {
		    maybe_graphics_queue_idx = Some(idx)
		}
		if queue.can_present() {
		    maybe_present_queue_idx = Some(idx)
		}
		if queue.can_do_transfer() {
		    maybe_transfer_queue_idx = Some(idx)
		}
	    }

	    let (default_graphics_queue_idx, default_present_queue_idx, default_transfer_queue_idx) =
		match (maybe_graphics_queue_idx, maybe_present_queue_idx, maybe_transfer_queue_idx) {
		    (Some(q1), Some(q2), Some(q3)) => (q1, q2, q3),
		    _ => panic!("Unable to create all three of: graphics queue, present queue, transfer queue!"),
		};

	    let mut pools = Vec::new();
	    for q in queues.iter() {
		pools.push(CommandPool::from_inner(
		    Rc::clone(&this),
		    Rc::clone(q),
		    false,
		    false,
		)?);
	    }

	    queue_set.queues = queues;
	    queue_set.pools = pools;
	    queue_set.default_graphics_queue_idx = default_graphics_queue_idx;
	    queue_set.default_present_queue_idx = default_present_queue_idx;
	    queue_set.default_transfer_queue_idx = default_transfer_queue_idx;
	}

	Ok(this)
    }

    fn get_default_graphics_queue(&self) -> Rc<Queue> {
	let queue_set = self.queue_set.borrow();
	Rc::clone(&queue_set.queues[queue_set.default_graphics_queue_idx])
    }

    fn get_default_graphics_pool(&self) -> Rc<CommandPool> {
	let queue_set = self.queue_set.borrow();
	Rc::clone(&queue_set.pools[queue_set.default_graphics_queue_idx])
    }

    #[allow(unused)]
    fn get_default_present_queue(&self) -> Rc<Queue> {
	let queue_set = self.queue_set.borrow();
	Rc::clone(&queue_set.queues[queue_set.default_present_queue_idx])
    }

    fn get_default_transfer_queue(&self) -> Rc<Queue> {
	let queue_set = self.queue_set.borrow();
	Rc::clone(&queue_set.queues[queue_set.default_transfer_queue_idx])
    }

    fn get_default_transfer_pool(&self) -> Rc<CommandPool> {
	let queue_set = self.queue_set.borrow();
	Rc::clone(&queue_set.pools[queue_set.default_transfer_queue_idx])
    }

    fn query_swapchain_support(&self) -> SwapChainSupport {
	query_swapchain_support(
	    self.physical_device,
	    &self.surface_loader,
	    self.surface,
	)
    }

    fn create_swapchain(&self, swapchain_create_info: &vk::SwapchainCreateInfoKHR) -> anyhow::Result<vk::SwapchainKHR> {
	Ok(unsafe {
            self.swapchain_loader
		.create_swapchain(swapchain_create_info, None)?
	})
    }

    fn get_swapchain_images(&self, swapchain: vk::SwapchainKHR) -> anyhow::Result<Vec<vk::Image>> {
	Ok(unsafe {
            self.swapchain_loader
		.get_swapchain_images(swapchain)?
	})
    }

    fn destroy_swapchain(&self, swapchain: vk::SwapchainKHR) {
	unsafe {
            self.swapchain_loader.destroy_swapchain(swapchain, None);
	}
    }

    fn acquire_next_image(
	&self,
	swapchain: vk::SwapchainKHR,
	timeout: u64,
	semaphore: vk::Semaphore,
	fence: vk::Fence
    ) -> ash::prelude::VkResult<(u32, bool)> {
	unsafe {
	    self.swapchain_loader
		.acquire_next_image(
                    swapchain,
                    timeout,
                    semaphore,
                    fence,
		)
	}
    }

    fn queue_present(
	&self,
	queue: Rc<Queue>,
	present_info: &vk::PresentInfoKHR
    ) -> ash::prelude::VkResult<bool> {
	unsafe {
            self.swapchain_loader
                .queue_present(queue.get(), present_info)
        }
    }
}

impl Drop for InnerDevice {
    fn drop(&mut self) {
	unsafe {
	    if let Ok(mut queue_set) = self.queue_set.try_borrow_mut() {
		for q in queue_set.queues.drain(..) {
		    if Rc::strong_count(&q) > 1 {
			panic!("We are destroying a Device, but a queue is still in use!");
		    }
		}
	    } else {
		panic!("We are destroying a Device, but its queue set is borrowed!");
	    }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            if self.validation_enabled {
		self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_messenger, None);
            }
            self.instance.destroy_instance(None);
	}
    }
}

fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_infos: &Vec<QueueInfo>,
    enabled_extensions: &[String],
) -> ash::Device {
    let mut queue_create_infos = vec![];
    // This needs to be outside the loop to avoid use-after-free problems with the pointer
    // stuff going on in DeviceQueueCreateInfo below.
    let mut priority = 1.0_f32;
    let mut queue_priorities = vec![];
    // This will fail horribly if the queue IDs are not consecutive.
    // Since the Vulkan API assumes they are, I don't think there are
    // any plausible cases where they won't be.
    for queue_info in queue_infos.iter() {
	for _ in queue_priorities.len()..queue_info.queues.len() {
	    queue_priorities.push(priority);
	    priority = priority / 2_f32;
	}
        queue_create_infos.push(vk::DeviceQueueCreateInfo{
	    s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DeviceQueueCreateFlags::empty(),
	    queue_family_index: queue_info.family_idx,
	    p_queue_priorities: queue_priorities.as_ptr(),
	    queue_count: queue_info.queues.len() as u32,
        });
    }

    let mut physical_device_features = vk::PhysicalDeviceFeatures{
        ..Default::default()
    };
    physical_device_features.independent_blend = vk::TRUE;

    let mut imageless_framebuffer_features = vk::PhysicalDeviceImagelessFramebufferFeatures{
	s_type: vk::StructureType::PHYSICAL_DEVICE_IMAGELESS_FRAMEBUFFER_FEATURES,
	p_next: ptr::null_mut(),
	..Default::default()
    };
    imageless_framebuffer_features.imageless_framebuffer = vk::TRUE;

    let descriptor_indexing_features = {
	let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures{
	    s_type: vk::StructureType::PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
	    p_next: (&mut imageless_framebuffer_features as *mut _) as *mut c_void,
	    ..Default::default()
	};
	descriptor_indexing_features.runtime_descriptor_array = vk::TRUE;
	//descriptor_indexing_features.shader_uniform_buffer_array_non_uniform_indexing = vk::TRUE;
	descriptor_indexing_features.shader_sampled_image_array_non_uniform_indexing = vk::TRUE;
	descriptor_indexing_features.shader_storage_buffer_array_non_uniform_indexing = vk::TRUE;
	descriptor_indexing_features.shader_storage_image_array_non_uniform_indexing = vk::TRUE;
	descriptor_indexing_features.descriptor_binding_partially_bound = vk::TRUE;
	descriptor_indexing_features
    };

    //physical_device_features.shader_uniform_buffer_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_sampled_image_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_storage_buffer_array_dynamic_indexing = vk::TRUE;
    physical_device_features.shader_storage_image_array_dynamic_indexing = vk::TRUE;
    physical_device_features.sampler_anisotropy = vk::TRUE;

    let required_validation_layer_raw_names: Vec<CString> = VALIDATION
        .required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let enable_layer_names: Vec<*const c_char> = required_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let mut enable_extension_names = vec![
        ash::extensions::khr::Swapchain::name().as_ptr(),
    ];
    // All this crap because a vk struct INSISTS on using *const c_char!
    let mut extension_names_list = vec![];
    for ext in enabled_extensions.iter() {
	extension_names_list.push(CString::new(ext.as_str()).unwrap());
    }
    for ext in extension_names_list.iter() {
	enable_extension_names.push(ext.as_ptr());
    }

    let p_next: *const c_void = &descriptor_indexing_features as *const _ as *const _;

    let device_create_info = vk::DeviceCreateInfo{
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next,
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_layer_count: if VALIDATION.is_enabled {
            enable_layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_layer_names: if VALIDATION.is_enabled {
            enable_layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_extension_count: enable_extension_names.len() as u32,
        pp_enabled_extension_names: enable_extension_names.as_ptr(),
        p_enabled_features: &physical_device_features,
    };

    let device: ash::Device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical device!")
    };

    device
}

struct QueueInfo {
    family_idx: u32,
    flags: vk::QueueFlags,
    queues: Vec<u32>,
    can_present: bool,
}

fn get_queue_info(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    surface_loader: &ash::extensions::khr::Surface,
) -> anyhow::Result<Vec<QueueInfo>> {
    let queue_families = unsafe {
	instance.get_physical_device_queue_family_properties(physical_device)
    };

    let mut infos = vec![];

    for (i, queue_family) in queue_families.iter().enumerate() {
	let mut queues = vec![];
	for j in 0..queue_family.queue_count {
	    queues.push(j as u32);
	}
	infos.push(QueueInfo{
	    family_idx: i as u32,
	    flags: queue_family.queue_flags,
	    queues: queues,
	    can_present: unsafe {
		surface_loader.get_physical_device_surface_support(
		    physical_device,
		    i as u32,
		    surface,
		)?
	    },
	});
    }

    Ok(infos)
}

fn get_queues_from_device(
    device: Rc<InnerDevice>,
    queue_infos: Vec<QueueInfo>
) -> anyhow::Result<Vec<Rc<Queue>>> {
    let mut queues = vec![];
    for queue_info in queue_infos.iter() {
	if queue_info.queues.len() == 0 {
	    println!("A queue family with no queues in it?  This driver is on crack!");
	    continue;
	}

	for queue_idx in queue_info.queues.iter() {
	    queues.push(Rc::new(Queue::new(
		device.clone(),
		queue_info.family_idx,
		*queue_idx,
		queue_info.flags,
		queue_info.can_present,
	    )?));
	}
    }

    Ok(queues)
}
