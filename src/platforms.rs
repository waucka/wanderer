use ash::version::{EntryV1_0, InstanceV1_0};
use ash::vk;

#[cfg(target_os = "windows")]
use ash::extensions::khr::Win32Surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::extensions::khr::{XlibSurface, WaylandSurface};
#[cfg(target_os = "macos")]
use ash::extensions::mvk::MacOSSurface;

use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::Surface;

#[cfg(target_os = "macos")]
use cocoa::appkit::{NSView, NSWindow};
#[cfg(target_os = "macos")]
use cocoa::base::id as cocoa_id;
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;

// required extension ------------------------------------------------------
#[cfg(target_os = "macos")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        MacOSSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}
#[cfg(target_os = "macos")]
pub fn optional_extension_names() -> Vec<*const i8> {
    vec![]
}

#[cfg(all(windows))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        Win32Surface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}
#[cfg(all(windows))]
pub fn optional_extension_names() -> Vec<*const i8> {
    vec![]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        Surface::name().as_ptr(),
        XlibSurface::name().as_ptr(),
        DebugUtils::name().as_ptr(),
    ]
}
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn optional_extension_names() -> Vec<*const i8> {
    if std::env::var("WANDERER_FORCE_X11") == Ok("y".to_string()) {
        vec![]
    } else {
        vec![
            WaylandSurface::name().as_ptr(),
        ]
    }
}
// ------------------------------------------------------------------------

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
unsafe fn create_x11_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> anyhow::Result<vk::SurfaceKHR> {
    use std::ptr;
    use winit::platform::unix::WindowExtUnix;

    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR {
        s_type: vk::StructureType::XLIB_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        window: x11_window as vk::Window,
        dpy: x11_display as *mut vk::Display,
    };
    let xlib_surface_loader = XlibSurface::new(entry, instance);
    Ok(xlib_surface_loader.create_xlib_surface(&x11_create_info, None)?)
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
unsafe fn create_wayland_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> anyhow::Result<vk::SurfaceKHR> {
    use std::ptr;
    use winit::platform::unix::WindowExtUnix;

    let wayland_display = window.wayland_display().unwrap();
    let wayland_surface = window.wayland_surface().unwrap();
    let wayland_create_info = vk::WaylandSurfaceCreateInfoKHR {
        s_type: vk::StructureType::WAYLAND_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        display: wayland_display as *mut vk::wl_display,
        surface: wayland_surface as *mut vk::wl_surface,
    };
    let wayland_surface_loader = WaylandSurface::new(entry, instance);
    Ok(wayland_surface_loader.create_wayland_surface(&wayland_create_info, None)?)
}

// create surface ---------------------------------------------------------
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> anyhow::Result<vk::SurfaceKHR> {
    if std::env::var("WANDERER_FORCE_X11") == Ok("y".to_string()) {
        create_x11_surface(entry, instance, window)
    } else {
        match create_wayland_surface(entry, instance, window) {
            Ok(surface) => {
                println!("Created a Wayland surface");
                Ok(surface)
            },
            Err(_) => {
                println!("Failed to create Wayland surface.  Trying X11...");
                create_x11_surface(entry, instance, window)
            },
        }
    }
}

#[cfg(target_os = "macos")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::mem;
    use std::os::raw::c_void;
    use std::ptr;
    use winit::platform::macos::WindowExtMacOS;

    let wnd: cocoa_id = mem::transmute(window.ns_window());

    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK {
        s_type: vk::StructureType::MACOS_SURFACE_CREATE_INFO_M,
        p_next: ptr::null(),
        flags: Default::default(),
        p_view: window.ns_view() as *const c_void,
    };

    let macos_surface_loader = MacOSSurface::new(entry, instance);
    macos_surface_loader.create_mac_os_surface_mvk(&create_info, None)
}

#[cfg(target_os = "windows")]
pub unsafe fn create_surface<E: EntryV1_0, I: InstanceV1_0>(
    entry: &E,
    instance: &I,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::os::raw::c_void;
    use std::ptr;
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::platform::windows::WindowExtWindows;

    let hwnd = window.hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR {
        s_type: vk::StructureType::WIN32_SURFACE_CREATE_INFO_KHR,
        p_next: ptr::null(),
        flags: Default::default(),
        hinstance,
        hwnd: hwnd as *const c_void,
    };
    let win32_surface_loader = Win32Surface::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}
// ------------------------------------------------------------------------
