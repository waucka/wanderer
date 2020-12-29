use ash::version::DeviceV1_0;
use ash::vk;
use anyhow::anyhow;

use std::collections::HashMap;
use std::pin::Pin;
use std::ptr;
use std::os::raw::c_void;
use std::rc::Rc;

use super::{Device, InnerDevice};
use super::buffer::{UniformBuffer, HasBuffer};
use super::texture::{Texture, Sampler};

pub struct WriteDescriptorSet {
    descriptor_type: vk::DescriptorType,
    buffer_info: Pin<Box<Vec<vk::DescriptorBufferInfo>>>,
    image_info: Pin<Box<Vec<vk::DescriptorImageInfo>>>,
    dst_set: vk::DescriptorSet,
    dst_binding: u32,
}

impl WriteDescriptorSet {
    fn for_buffers(
	descriptor_type: vk::DescriptorType,
	buffer_info: Vec<vk::DescriptorBufferInfo>,
	dst_set: vk::DescriptorSet,
	dst_binding: u32,
    ) -> Self {
	Self{
	    descriptor_type,
	    buffer_info: Pin::new(Box::new(buffer_info)),
	    image_info: Pin::new(Box::new(Vec::new())),
	    dst_set,
	    dst_binding,
	}
    }

    fn for_images(
	descriptor_type: vk::DescriptorType,
	image_info: Vec<vk::DescriptorImageInfo>,
	dst_set: vk::DescriptorSet,
	dst_binding: u32,
    ) -> Self {
	Self{
	    descriptor_type,
	    buffer_info: Pin::new(Box::new(Vec::new())),
	    image_info: Pin::new(Box::new(image_info)),
	    dst_set,
	    dst_binding,
	}
    }

    fn get(&self) -> vk::WriteDescriptorSet {
	let mut set = vk::WriteDescriptorSet{
	    s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
	    p_next: ptr::null(),
	    dst_set: self.dst_set,
	    dst_binding: self.dst_binding,
	    dst_array_element: 0,
	    descriptor_count: 0,
	    descriptor_type: self.descriptor_type,
	    p_image_info: ptr::null(),
	    p_buffer_info: ptr::null(),
	    p_texel_buffer_view: ptr::null(),
	};

	if self.buffer_info.len() > 0 {
	    set.descriptor_count = self.buffer_info.len() as u32;
	    set.p_buffer_info = self.buffer_info.as_ptr();
	} else if self.image_info.len() > 0 {
	    set.descriptor_count = self.image_info.len() as u32;
	    set.p_image_info = self.image_info.as_ptr();
	} else {
	    panic!("Coding error!");
	}
	if set.descriptor_count == 0 {
	    panic!("No descriptors set!");
	}

	set
    }
}

pub trait DescriptorRef {
    fn get_write(&self, dst_set: vk::DescriptorSet, dst_binding: u32) -> WriteDescriptorSet;
    fn get_type(&self) -> vk::DescriptorType;
}

pub struct UniformBufferRef<T> {
    uniform_buffers: Vec<Rc<UniformBuffer<T>>>,
}

impl<T> UniformBufferRef<T> {
    pub fn new(uniform_buffers: Vec<Rc<UniformBuffer<T>>>) -> Self {
	Self{
	    uniform_buffers,
	}
    }
}

impl<T> DescriptorRef for UniformBufferRef<T> {
    fn get_write(&self, dst_set: vk::DescriptorSet, dst_binding: u32) -> WriteDescriptorSet {
	let mut uniform_buffer_info = vec![];
	for buf in self.uniform_buffers.iter() {
	    uniform_buffer_info.push(vk::DescriptorBufferInfo{
		buffer: buf.get_buffer(),
		offset: 0,
		range: buf.len(),
	    });
	}
	WriteDescriptorSet::for_buffers(
	    vk::DescriptorType::UNIFORM_BUFFER,
	    uniform_buffer_info,
	    dst_set,
	    dst_binding,
	)
    }

    fn get_type(&self) -> vk::DescriptorType {
	vk::DescriptorType::UNIFORM_BUFFER
    }
}

pub struct TextureRef {
    textures: Vec<Rc<Texture>>,
}

impl TextureRef {
    pub fn new(textures: Vec<Rc<Texture>>) -> Self {
	Self{
	    textures,
	}
    }
}

impl DescriptorRef for TextureRef {
    fn get_write(&self, dst_set: vk::DescriptorSet, dst_binding: u32) -> WriteDescriptorSet {
	let mut texture_info = vec![];
	for tex in self.textures.iter() {
	    texture_info.push(vk::DescriptorImageInfo{
		sampler: vk::Sampler::null(),
		image_view: tex.image_view.view,
		image_layout: tex.image.layout,
	    });
	}
	WriteDescriptorSet::for_images(
	    vk::DescriptorType::SAMPLED_IMAGE,
	    texture_info,
	    dst_set,
	    dst_binding,
	)
    }

    fn get_type(&self) -> vk::DescriptorType {
	vk::DescriptorType::SAMPLED_IMAGE
    }
}

pub struct CombinedRef {
    samplers: Vec<Rc<Sampler>>,
    textures: Vec<Rc<Texture>>,
}

impl CombinedRef {
    pub fn new(sampler: Rc<Sampler>, textures: Vec<Rc<Texture>>) -> Self {
	Self{
	    samplers: vec![sampler],
	    textures,
	}
    }

    pub fn new_per(samplers: Vec<Rc<Sampler>>, textures: Vec<Rc<Texture>>) -> anyhow::Result<Self> {
	if samplers.len() != textures.len() {
	    Err(anyhow!("{} samplers provided for {} textures", samplers.len(), textures.len()))
	} else {
	    Ok(Self{
		samplers,
		textures,
	    })
	}
    }
}

impl DescriptorRef for CombinedRef {
    fn get_write(&self, dst_set: vk::DescriptorSet, dst_binding: u32) -> WriteDescriptorSet {
	let mut texture_info = vec![];
	for (i, tex) in self.textures.iter().enumerate() {
	    texture_info.push(vk::DescriptorImageInfo{
		sampler: if self.samplers.len() == 1 {
		    self.samplers[0].sampler
		} else {
		    self.samplers[i].sampler
		},
		image_view: tex.image_view.view,
		image_layout: tex.image.layout,
	    });
	}
	WriteDescriptorSet::for_images(
	    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
	    texture_info,
	    dst_set,
	    dst_binding,
	)
    }

    fn get_type(&self) -> vk::DescriptorType {
	vk::DescriptorType::COMBINED_IMAGE_SAMPLER
    }
}

pub struct DescriptorSetLayout {
    device: Rc<InnerDevice>,
    pub (in super) layout: vk::DescriptorSetLayout,
    bindings: Vec<vk::DescriptorSetLayoutBinding>,
}

impl DescriptorSetLayout {
    pub fn new(
	device: &Device,
	bindings: Vec<vk::DescriptorSetLayoutBinding>,
    ) -> anyhow::Result<Self> {
	/*let binding_flags = [
	    vk::DescriptorBindingFlags::PARTIALLY_BOUND,
	    vk::DescriptorBindingFlags::PARTIALLY_BOUND_EXT,
	];
	let binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo{
	    s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
	    p_next: ptr::null(),
	    binding_count: binding_flags.len() as u32,
	    p_binding_flags: binding_flags.as_ptr(),
	};
	let p_next: *const c_void = ((&binding_flags_info) as *const _) as *const c_void;*/
	let layout_info = vk::DescriptorSetLayoutCreateInfo{
	    s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DescriptorSetLayoutCreateFlags::empty(),
	    binding_count: bindings.len() as u32,
	    p_bindings: bindings.as_ptr(),
	};
	Ok(Self{
	    device: device.inner.clone(),
	    layout: unsafe {
		device.inner.device.create_descriptor_set_layout(&layout_info, None)?
	    },
	    bindings,
	})
    }
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
	unsafe {
	    self.device.device.destroy_descriptor_set_layout(self.layout, None);
	}
    }
}

pub struct UniformLayout {
    binding_sets: Vec<Rc<DescriptorSetLayout>>,
}

impl UniformLayout {
    pub fn contains(&self, other: &Self) -> bool {
	if self.binding_sets.len() < other.binding_sets.len() {
	    return false;
	}

	for set_idx in 0..self.binding_sets.len() {
	    if set_idx >= other.binding_sets.len() {
		break;
	    }

	    let self_set = &self.binding_sets[set_idx].bindings;
	    let other_set = &other.binding_sets[set_idx].bindings;
	    if self_set.len() < other_set.len() {
		return false;
	    }

	    for binding_idx in 0..self_set.len() {
		if binding_idx >= other_set.len() {
		    break;
		}

		let self_binding = self_set[binding_idx];
		let other_binding = other_set[binding_idx];
		if self_binding.descriptor_type != other_binding.descriptor_type {
		    return false;
		}
		if self_binding.descriptor_count < other_binding.descriptor_count {
		    return false;
		}
	    }
	}

	return true;
    }
}

pub struct DescriptorPool {
    device: Rc<InnerDevice>,
    pools: Vec<vk::DescriptorPool>,
    pool_sizes: HashMap<vk::DescriptorType, u32>,
    max_sets: u32,
}

impl DescriptorPool {
    pub fn new(
	device: &Device,
	pool_sizes: HashMap<vk::DescriptorType, u32>,
	max_sets: u32,
    ) -> anyhow::Result<Self> {
	let mut this = Self{
	    device: device.inner.clone(),
	    pools: Vec::new(),
	    pool_sizes,
	    max_sets,
	};
	this.grow()?;
	Ok(this)
    }

    // TODO: consider making this error out if any descriptor sets are still
    //       "at large" (e.g. store sets in objects that implement Drop and keep count).
    #[allow(unused)]
    pub fn reset(&mut self) -> anyhow::Result<()> {
	panic!("This shouldn't be happening yet!");
	unsafe {
	    if self.pools.len() > 1 {
		for pool in self.pools.drain(1..) {
		    self.device.device.destroy_descriptor_pool(pool, None);
		}
	    }
	    self.device.device.reset_descriptor_pool(
		self.pools[0],
		vk::DescriptorPoolResetFlags::empty(),
	    )?;
	}
	Ok(())
    }

    fn grow(&mut self) -> anyhow::Result<()> {
	let mut sizes = vec![];
	for (ty, descriptor_count) in self.pool_sizes.iter() {
	    sizes.push(vk::DescriptorPoolSize{
		ty: *ty,
		descriptor_count: *descriptor_count,
	    });
	}
	let create_info = vk::DescriptorPoolCreateInfo{
	    s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
	    p_next: ptr::null(),
	    flags: vk::DescriptorPoolCreateFlags::empty(),
	    max_sets: self.max_sets,
	    pool_size_count: sizes.len() as u32,
	    p_pool_sizes: sizes.as_ptr(),
	};
	self.pools.push(unsafe {
	    self.device.device.create_descriptor_pool(&create_info, None)?
	});
	Ok(())
    }

    pub fn create_descriptor_sets(
	&mut self,
	count: usize,
	layout: &DescriptorSetLayout,
	items: &Vec<Box<dyn DescriptorRef>>,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
	if items.len() != layout.bindings.len() {
	    return Err(anyhow!(
		"Provided {} items for a layout that takes {} items",
		items.len(),
		layout.bindings.len(),
	    ));
	}

	let sets = self.allocate(count, layout)?;
	for set in sets.iter() {
	    let mut writes = vec![];
	    // These have to stick around until the call to update_descriptor_sets() returns.
	    let mut writers = vec![];
	    for (binding, item) in items.iter().enumerate() {
		writers.push(item.get_write(*set, binding as u32));
		writes.push(writers.last().unwrap().get());
	    }
	    println!("Performing {} writes to descriptor set {:?}...", writes.len(), set);
	    for write in writes.iter() {
		println!(
		    "\tset: {:?}\n\tbinding: {}\n\ttype: {:?}\n\tcount: {}",
		    write.dst_set,
		    write.dst_binding,
		    write.descriptor_type,
		    write.descriptor_count,
		);
	    }
	    unsafe {
		self.device.device.update_descriptor_sets(&writes, &[]);
	    }
	}
	Ok(sets)
    }

    fn allocate(
	&mut self,
	count: usize,
	layout: &DescriptorSetLayout,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {
	let mut layouts: Vec<vk::DescriptorSetLayout> = vec![];
        for _ in 0..count {
            layouts.push(layout.layout);
        }

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo{
            s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
            p_next: ptr::null(),
            descriptor_pool: *self.pools.last().unwrap(),
            descriptor_set_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
        };
	self.allocate_internal(&layouts, &descriptor_set_allocate_info, true)
    }

    fn allocate_internal(
	&mut self,
	layouts: &[vk::DescriptorSetLayout],
	info: &vk::DescriptorSetAllocateInfo,
	retry: bool,
    ) -> anyhow::Result<Vec<vk::DescriptorSet>> {

        match unsafe { self.device.device.allocate_descriptor_sets(info) } {
	    Ok(sets) => Ok(sets),
	    Err(vk::Result::ERROR_FRAGMENTED_POOL) | Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
		if retry {
		    self.grow()?;
		    self.allocate_internal(layouts, info, false)
		} else {
		    Err(anyhow!("Cannot allocate descriptor set with given pool parameters"))
		}
	    },
	    Err(e) => Err(anyhow!("Cannot allocate descriptor set: {:?}", e)),
	}
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
	unsafe {
	    for pool in self.pools.iter() {
		self.device.device.destroy_descriptor_pool(*pool, None);
	    }
	}
    }
}
