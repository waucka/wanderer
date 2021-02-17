use ash::vk;
use glsl_layout::AsStd140;
use cgmath::{Matrix4, Vector2, Vector4};

use std::os::raw::c_char;
use std::ffi::CStr;
use std::convert::{AsRef, AsMut};

pub fn vk_to_string(raw_string_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}

pub struct NullVertex {}
impl super::support::shader::Vertex for NullVertex {
    fn get_binding_description() -> Vec<vk::VertexInputBindingDescription> {
	Vec::new()
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
	Vec::new()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Matrix4f {
    inner: Matrix4<f32>,
}

impl AsRef<Matrix4<f32>> for Matrix4f {
    fn as_ref(&self) -> &Matrix4<f32> {
	&self.inner
    }
}

impl AsMut<Matrix4<f32>> for Matrix4f {
    fn as_mut(&mut self) -> &mut Matrix4<f32> {
	&mut self.inner
    }
}

impl From<Matrix4<f32>> for Matrix4f {
    fn from(other: Matrix4<f32>) -> Self {
	Self{
	    inner: other,
	}
    }
}

impl From<[[f32; 4]; 4]> for Matrix4f {
    fn from(other: [[f32; 4]; 4]) -> Self {
	Self{
	    inner: other.into(),
	}
    }
}

impl Into<Matrix4<f32>> for Matrix4f {
    fn into(self) -> Matrix4<f32> {
        self.inner
    }
}

impl Default for Matrix4f {
    fn default() -> Self {
	Self{
	    inner: Matrix4::from_scale(1.0),
	}
    }
}

unsafe impl AsStd140 for Matrix4f {
    type Align = <glsl_layout::mat4x4 as AsStd140>::Align;
    type Std140 = glsl_layout::mat4x4;

    fn std140(&self) -> Self::Std140
    where
	Self::Std140: Sized
    {
	let converted: glsl_layout::mat4x4 = [
	    [self.inner.x.x, self.inner.x.y, self.inner.x.z, self.inner.x.w],
	    [self.inner.y.x, self.inner.y.y, self.inner.y.z, self.inner.y.w],
	    [self.inner.z.x, self.inner.z.y, self.inner.z.z, self.inner.z.w],
	    [self.inner.w.x, self.inner.w.y, self.inner.w.z, self.inner.w.w],
	].into();
	converted
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vector4f {
    inner: Vector4<f32>,
}

impl AsRef<Vector4<f32>> for Vector4f {
    fn as_ref(&self) -> &Vector4<f32> {
	&self.inner
    }
}

impl AsMut<Vector4<f32>> for Vector4f {
    fn as_mut(&mut self) -> &mut Vector4<f32> {
	&mut self.inner
    }
}

impl From<Vector4<f32>> for Vector4f {
    fn from(other: Vector4<f32>) -> Self {
	Self{
	    inner: other,
	}
    }
}

impl From<[f32; 4]> for Vector4f {
    fn from(other: [f32; 4]) -> Self {
	Self{
	    inner: other.into(),
	}
    }
}

impl Into<Vector4<f32>> for Vector4f {
    fn into(self) -> Vector4<f32> {
        self.inner
    }
}

impl Default for Vector4f {
    fn default() -> Self {
	Self{
	    inner: Vector4::new(0.0, 0.0, 0.0, 0.0),
	}
    }
}

unsafe impl AsStd140 for Vector4f {
    type Align = <glsl_layout::vec4 as AsStd140>::Align;
    type Std140 = glsl_layout::vec4;

    fn std140(&self) -> Self::Std140
    where
	Self::Std140: Sized
    {
	let converted: glsl_layout::vec4 = [
	    self.inner.x,
	    self.inner.y,
	    self.inner.z,
	    self.inner.w,
	].into();
	converted
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Vector2f {
    inner: Vector2<f32>,
}

impl Vector2f {
    pub fn new(x: f32, y: f32) -> Self {
	Self{
	    inner: Vector2::new(x, y),
	}
    }
}

impl AsRef<Vector2<f32>> for Vector2f {
    fn as_ref(&self) -> &Vector2<f32> {
	&self.inner
    }
}

impl AsMut<Vector2<f32>> for Vector2f {
    fn as_mut(&mut self) -> &mut Vector2<f32> {
	&mut self.inner
    }
}

impl From<Vector2<f32>> for Vector2f {
    fn from(other: Vector2<f32>) -> Self {
	Self{
	    inner: other,
	}
    }
}

impl From<[f32; 2]> for Vector2f {
    fn from(other: [f32; 2]) -> Self {
	Self{
	    inner: other.into(),
	}
    }
}

impl Into<Vector2<f32>> for Vector2f {
    fn into(self) -> Vector2<f32> {
        self.inner
    }
}

impl Default for Vector2f {
    fn default() -> Self {
	Self{
	    inner: Vector2::new(0.0, 0.0),
	}
    }
}

unsafe impl AsStd140 for Vector2f {
    type Align = <glsl_layout::vec2 as AsStd140>::Align;
    type Std140 = glsl_layout::vec2;

    fn std140(&self) -> Self::Std140
    where
	Self::Std140: Sized
    {
	let converted: glsl_layout::vec2 = [
	    self.inner.x,
	    self.inner.y,
	].into();
	converted
    }
}
