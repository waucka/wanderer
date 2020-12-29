use ash::vk;
use anyhow::anyhow;
use memoffset::offset_of;
use cgmath::{Vector4, Vector3, InnerSpace};

use std::path::Path;

pub struct Model {
    vertices: Vec<Vertex>,
    indices: Vec<u32>
}

impl Model {
    pub fn load(obj_path: &Path) -> anyhow::Result<Self> {
        let model_obj = tobj::load_obj(obj_path, true)?;

        let mut vertices = vec![];
        let mut indices = vec![];

        let (models, _) = model_obj;
        for m in models.iter() {
            let mesh = &m.mesh;

            if mesh.texcoords.len() == 0 {
                return Err(anyhow!("Missing texture coordinates for the model"));
            }
	    if mesh.indices.len() % 3 != 0 {
		return Err(anyhow!("Index count is not a multiple of 3; these are not triangles!"))
	    }

            let total_vertices_count = mesh.positions.len() / 3;
            for i in 0..total_vertices_count {
                let vertex = Vertex{
                    pos: [
                        mesh.positions[i * 3],
                        mesh.positions[i * 3 + 1],
                        mesh.positions[i * 3 + 2],
                        1.0,
                    ],
                    normal: [
                        mesh.normals[i * 3],
                        mesh.normals[i * 3 + 1],
                        mesh.normals[i * 3 + 2],
                        1.0,
                    ],
                    tangent: [
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    tex_coord: [
                        mesh.texcoords[i * 2],
                        mesh.texcoords[i * 2 +1],
                    ],
		    tex_idx: 0,//(mesh.positions[i * 3] * 10.0) as u32 % 2,
                };
                vertices.push(vertex);
            }
            indices = mesh.indices.clone();
	    let mut tangents = vec![];
	    let mut bitangents = vec![];
	    for _ in 0..vertices.len() {
		tangents.push(Vector3::new(0_f32, 0_f32, 0_f32));
		bitangents.push(Vector3::new(0_f32, 0_f32, 0_f32));
	    }

	    let total_faces_count = indices.len() / 3;
	    for idx in 0..total_faces_count {
		let i0 = indices[idx * 3] as usize;
		let i1 = indices[idx * 3 + 1] as usize;
		let i2 = indices[idx * 3 + 2] as usize;
		let v0 = vertices[i0];
		let v1 = vertices[i1];
		let v2 = vertices[i2];

		let e1 = Vector4::from(v1.pos).truncate() - Vector4::from(v0.pos).truncate();
		let e2 = Vector4::from(v2.pos).truncate() - Vector4::from(v0.pos).truncate();
		let x1 = v1.tex_coord[0] - v0.tex_coord[0];
		let y1 = v1.tex_coord[1] - v0.tex_coord[1];
		let x2 = v2.tex_coord[0] - v0.tex_coord[0];
		let y2 = v2.tex_coord[1] - v0.tex_coord[1];
		let r = 1.0 / (x1 * y2 - x2 * y1);
		let t = (e1 * y2 - e2 * y1) * r;
		let b = (e2 * x1 - e1 * x2) * r;

		tangents[i0] += t;
		tangents[i1] += t;
		tangents[i2] += t;
		bitangents[i0] += b;
		bitangents[i1] += b;
		bitangents[i2] += b;
	    }
	    for i in 0..vertices.len() {
		let t = tangents[i];
		let b = bitangents[i];
		let n = Vector4::from(vertices[i].normal).truncate();

		let tangent = (t - t.project_on(n)).normalize();
		let dot_product = t.cross(b).dot(n);
		vertices[i].tangent = [
		    tangent.x,
		    tangent.y,
		    tangent.z,
		    if dot_product > 0.0 {
			1.0
		    } else {
			-1.0
		    },
		];
	    }
        }

        Ok(Self {
            vertices,
            indices,
        })
    }

    pub fn get_vertices(&self) -> &[Vertex] {
        &self.vertices
    }

    pub fn get_indices(&self) -> &[u32] {
        &self.indices
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub pos: [f32; 4],
    pub normal: [f32; 4],
    pub tangent: [f32; 4],
    pub tex_coord: [f32; 2],
    pub tex_idx: u32,
}

impl super::support::shader::Vertex for Vertex {
    fn get_binding_description() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription{
            binding: 0,
            stride: std::mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }]
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
	vec![
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, normal) as u32,
            },
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 2,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, tangent) as u32,
            },
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 3,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, tex_coord) as u32,
            },
            vk::VertexInputAttributeDescription{
                binding: 0,
                location: 4,
                format: vk::Format::R32_UINT,
                offset: offset_of!(Self, tex_idx) as u32,
            },
        ]
    }
}
