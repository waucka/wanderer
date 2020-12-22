use ash::version::DeviceV1_0;
use ash::vk;

use std::path::Path;
use std::rc::Rc;
use std::ptr;

use super::{Device, InnerDevice};

pub (in super) trait GenericShader {
    fn get_shader(&self) -> &Shader;
}

pub struct Shader {
    device: Rc<InnerDevice>,
    pub (in super) shader: vk::ShaderModule,
}

impl Shader {
    fn from_spv_file(device: Rc<InnerDevice>, spv_file: &Path) -> anyhow::Result<Self> {
	let spv_bytes = std::fs::read(spv_file)?;
	Shader::from_spv_bytes(device, spv_bytes)
    }

    fn from_spv_bytes(device: Rc<InnerDevice>, spv_bytes: Vec<u8>) -> anyhow::Result<Self> {
	let shader_module_create_info = vk::ShaderModuleCreateInfo{
            s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::ShaderModuleCreateFlags::empty(),
            code_size: spv_bytes.len(),
            p_code: spv_bytes.as_ptr() as *const u32,
	};

	Ok(Self{
	    device: device.clone(),
	    shader: unsafe {
		device.device
		    .create_shader_module(&shader_module_create_info, None)?
	    },
	})
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_shader_module(self.shader, None);
	}
    }
}

pub trait Vertex {
    fn get_binding_description() -> Vec<vk::VertexInputBindingDescription>;
    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

pub struct VertexShader<V> where V: Vertex {
    shader: Shader,
    _phantom: std::marker::PhantomData<V>,
}

impl<V> VertexShader<V> where V: Vertex {
    pub fn from_spv_file(device: &Device, spv_file: &Path) -> anyhow::Result<Self> {
	Ok(Self{
	    shader: Shader::from_spv_file(device.inner.clone(), spv_file)?,
	    _phantom: std::marker::PhantomData,
	})
    }

    #[allow(unused)]
    pub fn from_spv_bytes(device: &Device, spv_bytes: Vec<u8>) -> anyhow::Result<Self> {
	Ok(Self {
	    shader: Shader::from_spv_bytes(device.inner.clone(), spv_bytes)?,
	    _phantom: std::marker::PhantomData,
	})
    }

    // This always returns true because your code won't compile
    // if the shader isn't compatible.
    #[allow(unused)]
    pub fn is_compatible(&self, _vertices: &[V]) -> bool {
	return true;
    }
}

impl<V> GenericShader for VertexShader<V> where V: Vertex {
    fn get_shader(&self) -> &Shader {
	&self.shader
    }
}

pub struct FragmentShader {
    shader: Shader,
}

impl FragmentShader {
    pub fn from_spv_file(device: &Device, spv_file: &Path) -> anyhow::Result<Self> {
	Ok(Self{
	    shader: Shader::from_spv_file(device.inner.clone(), spv_file)?,
	})
    }

    #[allow(unused)]
    pub fn from_spv_bytes(device: &Device, spv_bytes: Vec<u8>) -> anyhow::Result<Self> {
	Ok(Self {
	    shader: Shader::from_spv_bytes(device.inner.clone(), spv_bytes)?
	})
    }
}

impl GenericShader for FragmentShader {
    fn get_shader(&self) -> &Shader {
	&self.shader
    }
}
