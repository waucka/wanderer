use ash::version::InstanceV1_0;
use ash::vk;

pub fn find_depth_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::Format {
    find_supported_format(
        instance,
        physical_device,
        &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ],
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

pub fn find_supported_format(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    candidate_formats: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> vk::Format {
    for &format in candidate_formats.iter() {
        let format_properties =
            unsafe { instance.get_physical_device_format_properties(physical_device, format) };
        let linear_acceptable = tiling == vk::ImageTiling::LINEAR
            && format_properties.linear_tiling_features.contains(features);
        let optimal_acceptable = tiling == vk::ImageTiling::OPTIMAL
            && format_properties.optimal_tiling_features.contains(features);
        if linear_acceptable || optimal_acceptable {
            return format;
        }
    }

    panic!("Failed to find supported format!")
}

#[derive(Clone)]
pub struct Defaulted<T> {
    value: Option<T>,
    default: T,
}

impl<T> Defaulted<T> {
    pub fn new(default: T) -> Defaulted<T> {
        Defaulted{
            value: None,
            default,
        }
    }

    pub fn get_value(&self) -> &T {
        match &self.value {
            Some(v) => v,
            None => &self.default,
        }
    }

    pub fn set_value(&mut self, value: T) {
        self.value = Some(value);
    }
}
