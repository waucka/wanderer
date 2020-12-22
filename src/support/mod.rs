use ash::version::{EntryV1_0, InstanceV1_0, DeviceV1_0};
use ash::vk::{version_major, version_minor, version_patch};
use ash::vk;
use anyhow::anyhow;
use winit::event_loop::EventLoop;

use std::ptr;
use std::collections::HashSet;
use std::ffi::CString;
use std::os::raw::{c_char, c_void};
use std::rc::Rc;

use super::utils::vk_to_string;
use super::debug::VALIDATION;
use super::debug;

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

use utils::Defaulted;

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

fn create_instance(
    entry: &ash::Entry,
    app_name: &str,
    engine_name: &str,
    app_version: u32,
    engine_version: u32,
    api_version: u32,
) -> ash::Instance {
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
            &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT
                as *const c_void
        } else {
            ptr::null()
        },
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
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
        pp_enabled_extension_names: extension_names.as_ptr(),
        enabled_extension_count: extension_names.len() as u32,
    };

    let instance: ash::Instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create Vulkan instance")
    };

    instance
}

// Device

pub const ENGINE_NAME: &'static str = "Wanderer Engine";
pub const ENGINE_VERSION: u32 = vk::make_version(0, 1, 0);
pub const VULKAN_API_VERSION: u32 = vk::make_version(1, 2, 131);

pub struct Queue {
    device: ash::Device,
    family_idx: u32,
    queue_idx: u32,
    flags: vk::QueueFlags,
    can_present: bool,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
}

impl Queue {
    fn new(
	device: ash::Device,
	family_idx: u32,
	queue_idx: u32,
	flags: vk::QueueFlags,
	can_present: bool,
    ) -> anyhow::Result<Self> {
	let queue = unsafe {
	    device.get_device_queue(family_idx, queue_idx)
	};
	let command_pool_create_info = vk::CommandPoolCreateInfo{
            s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::CommandPoolCreateFlags::empty(),
            queue_family_index: family_idx,
	};

	let command_pool = unsafe {
            device.create_command_pool(&command_pool_create_info, None)?
	};

	Ok(Self{
	    device,
	    family_idx,
	    queue_idx,
	    flags,
	    can_present,
	    queue,
	    command_pool,
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

    // Do not call this except in the drop() method for InnerDevice.
    fn discard(&mut self) {
	unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
	}
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
    pub inner: Rc<InnerDevice>,
}

impl Device {
    pub fn new(event_loop: &EventLoop<()>, builder: DeviceBuilder) -> anyhow::Result<Self> {
	Ok(Self {
	    inner: Rc::new(InnerDevice::new(event_loop, builder)?),
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
	self.inner.default_graphics_queue.clone()
    }

    #[allow(unused)]
    pub fn get_default_present_queue(&self) -> Rc<Queue> {
	self.inner.default_present_queue.clone()
    }

    #[allow(unused)]
    pub fn get_default_transfer_queue(&self) -> Rc<Queue> {
	self.inner.default_transfer_queue.clone()
    }

    #[allow(unused)]
    pub fn get_queues(&self) -> &[Rc<Queue>] {
	&self.inner.queues
    }
}


pub struct InnerDevice {
    window: winit::window::Window,
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_messenger: vk::DebugUtilsMessengerEXT,
    validation_enabled: bool,

    physical_device: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queues: Vec<Rc<Queue>>,
    default_graphics_queue: Rc<Queue>,
    default_present_queue: Rc<Queue>,
    default_transfer_queue: Rc<Queue>,
}

impl InnerDevice {
    pub fn new(event_loop: &EventLoop<()>, builder: DeviceBuilder) -> anyhow::Result<Self> {
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
        );
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
	let queues = get_queues_from_device(
	    device.clone(),
	    queue_infos,
	)?;

	let mut maybe_graphics_queue = None;
	let mut maybe_present_queue = None;
	let mut maybe_transfer_queue = None;
	for queue in queues.iter() {
            if queue.can_do_graphics() {
		maybe_graphics_queue = Some(queue.clone())
            }
	    if queue.can_present() {
		maybe_present_queue = Some(queue.clone())
	    }
	    // Not used yet
            //if queue.flags.contains(vk::QueueFlags::COMPUTE) {
            //}
            if queue.can_do_transfer() {
		maybe_transfer_queue = Some(queue.clone())
            }
	    // Not used yet
            //if queue.flags.contains(vk::QueueFlags::SPARSE_BINDING) {
            //}
	}

	let (default_graphics_queue, default_present_queue, default_transfer_queue) =
	    match (maybe_graphics_queue, maybe_present_queue, maybe_transfer_queue) {
		(Some(q1), Some(q2), Some(q3)) => (q1, q2, q3),
		_ => panic!("Unable to create all three of: graphics queue, present queue, transfer queue!"),
	    };

        Ok(Self{
            window,
            _entry: entry,
            instance,
            surface_loader,
            surface,
            debug_utils_loader,
            debug_messenger,
	    validation_enabled: *builder.validation_enabled.get_value(),

            physical_device,
            memory_properties,
            device: device.clone(),
	    queues,
	    default_graphics_queue,
	    default_present_queue,
	    default_transfer_queue,
        })
    }
}

impl Drop for InnerDevice {
    fn drop(&mut self) {
	unsafe {
	    if let Some(q) = Rc::get_mut(&mut self.default_graphics_queue) {
		q.discard();
	    } else {
		panic!("We are destroying a Device, but a graphics queue is still in use!");
	    }
	    if self.default_graphics_queue != self.default_present_queue {
		if let Some(q) = Rc::get_mut(&mut self.default_present_queue) {
		    q.discard();
		} else {
		    panic!("We are destroying a Device, but a present queue is still in use!");
		}
	    }
	    if self.default_graphics_queue != self.default_transfer_queue &&
		self.default_present_queue != self.default_transfer_queue {
		    if let Some(q) = Rc::get_mut(&mut self.default_transfer_queue) {
			q.discard();
		    } else {
			panic!("We are destroying a Device, but a transfer queue is still in use!");
		    }
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
    // This will fail horribly if the queue IDs are not consecutive.
    // Since the Vulkan API assumes they are, I don't think there are
    // any plausible cases where they won't be.
    for queue_info in queue_infos.iter() {
	let mut priority = 1.0_f32;
	let mut queue_priorities = vec![];
	for _ in queue_info.queues.iter() {
	    queue_priorities.push(priority);
	    priority = priority / 2_f32;
	}
        queue_create_infos.push(vk::DeviceQueueCreateInfo{
	    s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DeviceQueueCreateFlags::empty(),
	    queue_family_index: queue_info.family_idx,
	    p_queue_priorities: queue_priorities.as_ptr(),
	    queue_count: queue_priorities.len() as u32,
        });
    }

    let mut physical_device_features = vk::PhysicalDeviceFeatures{
        ..Default::default()
    };

    physical_device_features.shader_storage_image_array_dynamic_indexing = vk::TRUE;

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

    let device_create_info = vk::DeviceCreateInfo{
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: ptr::null(),
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
    device: ash::Device,
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
