use ash::vk;
use anyhow::anyhow;
use cgmath::Matrix4;
use glsl_layout::AsStd140;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr;
use std::rc::Rc;

use super::support::Device;
use super::support::buffer::{VertexBuffer, IndexBuffer, UniformBufferSet};
use super::support::command_buffer::RenderPassWriter;
use super::support::descriptor::{
    DescriptorPool,
    DescriptorSetLayout,
    CombinedRef,
    DescriptorRef,
    UniformBufferRef,
};
use super::support::renderer::Pipeline;
use super::support::shader::Vertex;
use super::support::texture::Material;
use super::scene::Renderable;
use super::utils::{NullVertex, Vector4f, Matrix4f};

const DEBUG_DESCRIPTOR_SETS: bool = false;

// TODO: get rid of static geometry type UBO after validating that
//       this approach works.

#[derive(Debug, Default, Clone, Copy, AsStd140)]
pub struct StaticGeometryTypeUBO {
    tint: Vector4f,
}

pub struct StaticGeometrySet<V: Vertex> {
    global_descriptor_sets: Vec<vk::DescriptorSet>,
    type_descriptor_set_layout: DescriptorSetLayout,
    type_descriptor_sets: Vec<vk::DescriptorSet>,
    instance_descriptor_set_layout: DescriptorSetLayout,
    vertex_buffer: Rc<VertexBuffer<V>>,
    index_buffer: Option<Rc<IndexBuffer>>,
    uniform_buffer_set: UniformBufferSet<StaticGeometryTypeUBO>,
    type_pool: DescriptorPool,
    instance_pool: DescriptorPool,
    objects: Vec<StaticGeometry>,
}

pub type StaticGeometryId = usize;

impl<V: Vertex> StaticGeometrySet<V> {
    const NUM_UNIFORM_BUFFERS_PER_INSTANCE: u32 = 1;
    const MAX_TEXTURES: u32 = 256;
    const NUM_IMAGES_PER_TEXTURE: u32 = 3;
    const POOL_SIZE: u32 = 1024;

    pub fn new(
	device: &Device,
	global_descriptor_sets: Vec<vk::DescriptorSet>,
	vertex_buffer: Rc<VertexBuffer<V>>,
	index_buffer: Option<Rc<IndexBuffer>>,
	materials: &Vec<Rc<Material>>,
	num_frames: usize,
    ) -> anyhow::Result<Self> {
	let uniform_buffer_set = UniformBufferSet::new(
	    device,
	    StaticGeometryTypeUBO{
		tint: [1.0, 1.0, 1.0, 1.0].into(),
	    },
	    num_frames,
	)?;

	let type_descriptor_set_layout = DescriptorSetLayout::new(
	    device,
	    vec![
		vk::DescriptorSetLayoutBinding{
		    binding: 0,
		    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		    descriptor_count: 1,
		    stage_flags: vk::ShaderStageFlags::ALL,
		    p_immutable_samplers: ptr::null(),
		},
		vk::DescriptorSetLayoutBinding{
		    binding: 1,
		    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		    descriptor_count: Self::MAX_TEXTURES,
		    stage_flags: vk::ShaderStageFlags::ALL,
		    p_immutable_samplers: ptr::null(),
		},
		vk::DescriptorSetLayoutBinding{
		    binding: 2,
		    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		    descriptor_count: Self::MAX_TEXTURES,
		    stage_flags: vk::ShaderStageFlags::ALL,
		    p_immutable_samplers: ptr::null(),
		},
		vk::DescriptorSetLayoutBinding{
		    binding: 3,
		    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		    descriptor_count: Self::MAX_TEXTURES,
		    stage_flags: vk::ShaderStageFlags::ALL,
		    p_immutable_samplers: ptr::null(),
		},
	    ],
	)?;

	let instance_descriptor_set_layout = DescriptorSetLayout::new(
	    device,
	    vec![vk::DescriptorSetLayoutBinding{
		binding: 0,
		descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
		descriptor_count: Self::NUM_UNIFORM_BUFFERS_PER_INSTANCE,
		stage_flags: vk::ShaderStageFlags::ALL,
		p_immutable_samplers: ptr::null(),
	    }],
	)?;

	let mut type_pool = {
	    let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
	    pool_sizes.insert(
		vk::DescriptorType::UNIFORM_BUFFER,
		Self::POOL_SIZE * Self::NUM_UNIFORM_BUFFERS_PER_INSTANCE,
	    );
	    pool_sizes.insert(
		vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
		Self::MAX_TEXTURES * Self::NUM_IMAGES_PER_TEXTURE,
	    );
	    DescriptorPool::new(
		device,
		pool_sizes,
		Self::POOL_SIZE,
	    )?
	};

	let (samplers, textures_color, textures_normal, textures_props) = {
	    let mut samplers = Vec::new();
	    let mut textures_color = Vec::new();
	    let mut textures_normal = Vec::new();
	    let mut textures_props = Vec::new();
	    for material in materials.iter() {
		let sampler = material.get_sampler();
		let color = material.get_color_texture();
		let normal = material.get_normal_texture();
		let props = material.get_properties_texture();
		samplers.push(sampler);
		textures_color.push(color);
		textures_normal.push(normal);
		textures_props.push(props);
	    }
	    for _ in materials.len()..(Self::MAX_TEXTURES as usize) {
		let material = &materials[0];
		let sampler = material.get_sampler();
		let color = material.get_color_texture();
		let normal = material.get_normal_texture();
		let props = material.get_properties_texture();
		samplers.push(sampler);
		textures_color.push(color);
		textures_normal.push(normal);
		textures_props.push(props);
	    }
	    if samplers.len() > (Self::MAX_TEXTURES as usize) {
		panic!("{} samplers for {} textures!", samplers.len(), Self::MAX_TEXTURES);
	    }
	    (samplers, textures_color, textures_normal, textures_props)
	};

	let mut type_descriptor_sets = vec![];
	for frame_idx in 0..num_frames {
	    let mut items: Vec<Box<dyn DescriptorRef>> = vec![
		Box::new(UniformBufferRef::new(vec![uniform_buffer_set.get_buffer(frame_idx)?])),
	    ];
	    items.push(Box::new(CombinedRef::new_per(
		samplers.clone(),
		textures_color.clone(),
	    )?));
	    items.push(Box::new(CombinedRef::new_per(
		samplers.clone(),
		textures_normal.clone(),
	    )?));
	    items.push(Box::new(CombinedRef::new_per(
		samplers.clone(),
		textures_props.clone(),
	    )?));

	    if DEBUG_DESCRIPTOR_SETS {
		println!("Creating type descriptor sets with {} items...", items.len());
	    }
	    let sets = type_pool.create_descriptor_sets(
		1,
		&type_descriptor_set_layout,
		&items,
	    )?;
	    type_descriptor_sets.push(sets[0]);
	}

	let instance_pool = {
	    let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
	    pool_sizes.insert(
		vk::DescriptorType::UNIFORM_BUFFER,
		Self::POOL_SIZE * Self::NUM_UNIFORM_BUFFERS_PER_INSTANCE,
	    );
	    DescriptorPool::new(
		device,
		pool_sizes,
		Self::POOL_SIZE,
	    )?
	};

	Ok(Self{
	    global_descriptor_sets,
	    type_descriptor_set_layout,
	    type_descriptor_sets,
	    instance_descriptor_set_layout,
	    vertex_buffer,
	    index_buffer,
	    uniform_buffer_set,
	    type_pool,
	    instance_pool,
	    objects: Vec::new(),
	})
    }

    pub fn add(
	&mut self,
	device: &Device,
	model_matrix: Matrix4<f32>,
    ) -> anyhow::Result<StaticGeometryId> {
	let num_frames = self.uniform_buffer_set.len();
	let uniform_buffer_set = UniformBufferSet::new(
	    device,
	    StaticGeometryInstanceUBO{
		model: model_matrix.into(),
	    },
	    num_frames,
	)?;

	let mut instance_descriptor_sets = vec![];
	for frame_idx in 0..num_frames {
	    let items: Vec<Box<dyn DescriptorRef>> = vec![
		Box::new(UniformBufferRef::new(vec![uniform_buffer_set.get_buffer(frame_idx)?])),
	    ];

	    if DEBUG_DESCRIPTOR_SETS {
		println!("Creating instance descriptor sets with {} items...", items.len());
	    }
	    let sets = self.instance_pool.create_descriptor_sets(
		1,
		&self.instance_descriptor_set_layout,
		&items,
	    )?;
	    instance_descriptor_sets.push(sets[0]);
	}

	let object_id = self.objects.len();
	self.objects.push(StaticGeometry{
	    instance_descriptor_sets,
	    uniform_buffer_set,
	});
	Ok(object_id)
    }

    pub fn clear(&mut self) {
	self.objects.clear();
    }

    pub fn get_mut(&mut self, object_id: StaticGeometryId) -> anyhow::Result<&mut StaticGeometry> {
	if object_id >= self.objects.len() {
	    Err(anyhow!("Tried to modify invalid static geometry object {}", object_id))
	} else {
	    Ok(&mut self.objects[object_id])
	}
    }

    pub fn get_type_layout(&self) -> &DescriptorSetLayout {
	&self.type_descriptor_set_layout
    }

    pub fn get_instance_layout(&self) -> &DescriptorSetLayout {
	&self.instance_descriptor_set_layout
    }
}

pub struct StaticGeometrySetRenderer<V: Vertex> {
    pipeline: Rc<RefCell<Pipeline<V>>>,
    static_geometry_set: StaticGeometrySet<V>,
}

impl<V: Vertex> StaticGeometrySetRenderer<V> {
    pub fn new(
	pipeline: Rc<RefCell<Pipeline<V>>>,
	static_geometry_set: StaticGeometrySet<V>,
    ) -> Self {
	Self{
	    pipeline,
	    static_geometry_set,
	}
    }
}

impl<V: Vertex> Renderable for StaticGeometrySetRenderer<V> {
    fn write_draw_command(&self, idx: usize, writer: &RenderPassWriter) -> anyhow::Result<()> {
	writer.bind_pipeline(self.pipeline.clone());
	for object in self.static_geometry_set.objects.iter() {
	    let descriptor_sets = [
		self.static_geometry_set.global_descriptor_sets[idx],
		self.static_geometry_set.type_descriptor_sets[idx],
		object.instance_descriptor_sets[idx],
	    ];
	    if DEBUG_DESCRIPTOR_SETS {
		println!("Binding descriptor sets...");
		println!("\tSet 0: {:?}", descriptor_sets[0]);
		println!("\tSet 1: {:?}", descriptor_sets[1]);
		println!("\tSet 2: {:?}", descriptor_sets[2]);
	    }
	    writer.bind_descriptor_sets(self.pipeline.borrow().get_layout(), &descriptor_sets);
	    match &self.static_geometry_set.index_buffer {
		Some(idx_buf) => writer.draw_indexed(
		    &self.static_geometry_set.vertex_buffer,
		    idx_buf,
		),
		None => writer.draw(
		    &self.static_geometry_set.vertex_buffer,
		),
	    };
	}
	Ok(())
    }

    fn sync_uniform_buffers(&self, idx: usize) -> anyhow::Result<()> {
	for object in self.static_geometry_set.objects.iter() {
	    object.uniform_buffer_set.sync(idx)?;
	}
	Ok(())
    }
}

#[derive(Debug, Default, Clone, Copy, AsStd140)]
pub struct StaticGeometryInstanceUBO {
    #[allow(unused)]
    model: Matrix4f,
}

pub struct StaticGeometry {
    instance_descriptor_sets: Vec<vk::DescriptorSet>,
    uniform_buffer_set: UniformBufferSet<StaticGeometryInstanceUBO>,
}

//TODO: This should probably be moved to a new module called "postprocessing" or something.

pub struct PostProcessingStep {
    global_descriptor_sets: Vec<vk::DescriptorSet>,
    pipeline: Rc<RefCell<Pipeline<NullVertex>>>,
}

impl PostProcessingStep {
    pub fn new(
	global_descriptor_sets: Vec<vk::DescriptorSet>,
	pipeline: Rc<RefCell<Pipeline<NullVertex>>>,
    ) -> Self {
	Self{
	    global_descriptor_sets,
	    pipeline,
	}
    }

    pub fn replace_descriptor_sets(&mut self, global_descriptor_sets: Vec<vk::DescriptorSet>) {
	self.global_descriptor_sets = global_descriptor_sets;
    }
}

impl Renderable for PostProcessingStep {
    fn write_draw_command(&self, idx: usize, writer: &RenderPassWriter) -> anyhow::Result<()> {
	writer.bind_pipeline(self.pipeline.clone());
	let descriptor_sets = [
	    self.global_descriptor_sets[idx],
	];
	if DEBUG_DESCRIPTOR_SETS {
	    println!("Binding descriptor sets...");
	    println!("\tSet 0: {:?}", descriptor_sets[0]);
	}
	writer.bind_descriptor_sets(self.pipeline.borrow().get_layout(), &descriptor_sets);
	writer.draw_no_vbo(3, 1);
	Ok(())
    }

    fn sync_uniform_buffers(&self, _idx: usize) -> anyhow::Result<()> {
	Ok(())
    }
}
