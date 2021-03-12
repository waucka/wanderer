use ash::vk;
use glsl_layout::Uniform;

use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use super::support::{Device, PerFrameSet, FrameId, Queue};
use super::support::buffer::UniformBuffer;
use super::support::command_buffer::{SecondaryCommandBuffer, RenderPassWriter};
use super::support::descriptor::{
    DescriptorBindings,
    DescriptorPool,
    DescriptorSet,
    DescriptorSetLayout,
    DescriptorRef,
    UniformBufferRef,
};
use super::support::renderer::{Pipeline, RenderPass, SubpassRef};
use super::support::shader::{VertexShader, FragmentShader};
use super::support::texture::Texture;
use super::utils::{NullVertex};
use super::ui_app::{UniformTwiddler, UniformData, UniformDataItemRadio, UniformDataItemSliderSFloat, UniformDataVar};
use super::postproc::{PostProcStep, PostProcResources};

pub enum Algorithm {
    #[allow(unused)]
    NoOp,
    #[allow(unused)]
    Linear,
    #[allow(unused)]
    ReinhardSimple,
    #[allow(unused)]
    ReinhardEnhanced,
    #[allow(unused)]
    Uncharted2,
    #[allow(unused)]
    Aces,
}

impl std::convert::Into<u32> for Algorithm {
    fn into(self) -> u32 {
        use Algorithm::*;
        match self {
            NoOp => 0,
            Linear => 1,
            ReinhardSimple => 2,
            ReinhardEnhanced => 3,
            Uncharted2 => 4,
            Aces => 5,
        }
    }
}

#[derive(Debug, Default, Clone, Copy, Uniform)]
pub struct HdrControlUniform {
    #[allow(unused)]
    exposure: f32,
    #[allow(unused)]
    white_luminance: f32,
    #[allow(unused)]
    algo: u32,
}

impl HdrControlUniform {
    pub fn new(
        exposure: f32,
        white_luminance: f32,
        algo: Algorithm,
    ) -> Self {
        Self{
            exposure,
            white_luminance,
            algo: algo.into(),
        }
    }
}

impl HdrControlUniform {
    pub fn get_twiddler_data(&self) -> Rc<UniformData> {
        let mut data = UniformData::new();
        data.add(
            "exposure",
            "Exposure",
            UniformDataItemSliderSFloat::new(self.exposure, -20.0..=20.0, false),
        );
        data.add(
            "white_luminance",
            "White luminance",
            UniformDataItemSliderSFloat::new(self.exposure, 0.00001..=100000.0, true),
        );
        data.add(
            "algo",
            "Tonemapping algorithm",
            UniformDataItemRadio::new(
                self.algo,
                vec![
                    ("No-op".to_owned(), 0),
                    ("Linear".to_owned(), 1),
                    ("Reinhard simple".to_owned(), 2),
                    ("Reinhard enhanced".to_owned(), 3),
                    ("Uncharted 2".to_owned(), 4),
                    ("ACES".to_owned(), 5),
                    ("Invalid".to_owned(), 9001),
                ],
            ),
        );
        Rc::new(data)
    }

    pub fn set_data(&mut self, twiddler: Rc<UniformTwiddler>) {
        let uniform_data = twiddler.get_uniform_data();

        if let Some(UniformDataVar::SFloat(exposure)) = uniform_data.get_value("exposure") {
            self.exposure = exposure;
        }

        if let Some(UniformDataVar::SFloat(white_luminance)) = uniform_data.get_value("white_luminance") {
            self.white_luminance = white_luminance;
        }

        if let Some(UniformDataVar::UInt(algo)) = uniform_data.get_value("algo") {
            self.algo = algo;
        }
    }
}

pub struct HdrStep {
    _pool: DescriptorPool,
    _descriptor_set_layout: DescriptorSetLayout,
    descriptor_sets: PerFrameSet<Rc<DescriptorSet>>,
    uniform_buffers: PerFrameSet<Rc<UniformBuffer<HdrControlUniform>>>,
    pipeline: Rc<Pipeline<NullVertex>>,
    pp_res: PostProcResources,
}

impl HdrStep {
    pub fn new(
        device: &Device,
        queue: Rc<Queue>,
        pixel_sources: &PerFrameSet<Rc<Texture>>,
        uniform: &HdrControlUniform,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: &RenderPass,
        subpass: SubpassRef,
    ) -> anyhow::Result<Self> {
        let vert_shader: VertexShader<NullVertex> =
            VertexShader::from_spv_file(
                device,
                Path::new("./assets/shaders/hdr.vert.spv"),
            )?;
        let frag_shader = FragmentShader::from_spv_file(
            device,
            Path::new("./assets/shaders/hdr.frag.spv"),
        )?;

        let pp_res = PostProcResources::new(
            device,
            pixel_sources,
            queue,
        )?;

        let mut pool = {
            let mut pool_sizes: HashMap<vk::DescriptorType, u32> = HashMap::new();
            pool_sizes.insert(
                vk::DescriptorType::UNIFORM_BUFFER,
                2,
            );
            DescriptorPool::new(
                "HDR",
                device,
                pool_sizes,
                2,
            )?
        };

        let descriptor_bindings = DescriptorBindings::new()
            .with_binding(
                vk::DescriptorType::UNIFORM_BUFFER,
                1,
                vk::ShaderStageFlags::FRAGMENT,
                false,
            );
        let descriptor_set_layout = DescriptorSetLayout::new(
            device,
            descriptor_bindings,
        )?;

        let uniform_buffers = PerFrameSet::new(
            |_| {
                Ok(Rc::new(UniformBuffer::new(device, Some(uniform))?))
            }
        )?;

        let descriptor_sets = PerFrameSet::new(
            |frame| {
                let uniform_buffer = uniform_buffers.get(frame);
                let items: Vec<Rc<dyn DescriptorRef>> = vec![
                    Rc::new(UniformBufferRef::new(vec![
                        Rc::clone(uniform_buffer),
                    ])),
                ];
                let sets = pool.create_descriptor_sets(
                    1,
                    &descriptor_set_layout,
                    &items,
                )?;

                Ok(Rc::clone(&sets[0]))
            }
        )?;

        let set_layouts = [
            pp_res.get_descriptor_set_layout(),
            &descriptor_set_layout,
        ];

        let pipeline = Rc::new(Pipeline::new(
            &device,
            viewport_width,
            viewport_height,
            render_pass,
            vert_shader,
            frag_shader,
            &set_layouts,
            PostProcResources::get_pipeline_parameters(subpass),
        )?);


        Ok({
            let this = Self{
                _pool: pool,
                _descriptor_set_layout: descriptor_set_layout,
                descriptor_sets,
                uniform_buffers,
                pipeline,
                pp_res,
            };
            this.pp_res.rebuild_command_buffers(
                &this,
                render_pass,
                subpass,
            )?;
            this
        })
    }

    pub fn update(&self, frame: FrameId, uniform: &HdrControlUniform) -> anyhow::Result<()> {
        self.uniform_buffers.get(frame).update(uniform)
    }
}

impl PostProcStep for HdrStep {
    fn bind_resources(
        &self,
        frame: FrameId,
        rt_descriptor_set: Rc<DescriptorSet>,
        writer: &mut RenderPassWriter,
    ) -> anyhow::Result<()> {
        writer.bind_pipeline(Rc::clone(&self.pipeline));
        let descriptor_sets = [
            rt_descriptor_set,
            Rc::clone(&self.descriptor_sets.get(frame)),
        ];
        writer.bind_descriptor_sets(self.pipeline.get_layout(), &descriptor_sets);
        Ok(())
    }

    fn update_viewport(
        &mut self,
        pixel_sources: &PerFrameSet<Rc<Texture>>,
        viewport_width: usize,
        viewport_height: usize,
        render_pass: &RenderPass,
        subpass: SubpassRef,
    ) -> anyhow::Result<()> {
        self.pp_res.update_pixel_sources(pixel_sources)?;
        self.pipeline.update_viewport(
            viewport_width,
            viewport_height,
            render_pass,
        )?;
        self.pp_res.rebuild_command_buffers(
            self,
            render_pass,
            subpass,
        )
    }

    fn get_command_buffer(
        &self,
        frame: FrameId,
    ) -> Rc<SecondaryCommandBuffer> {
        self.pp_res.get_command_buffer(frame)
    }
}
