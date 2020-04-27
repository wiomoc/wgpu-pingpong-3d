extern crate bytemuck;
extern crate futures;
extern crate winit;

mod obj_reader;

use crate::obj_reader::{Material, Mesh, ObjFile, Vertex};
use bytemuck::{Pod, Zeroable};
use cgmath::{vec2, vec3, Deg, Matrix, Matrix4, Point3, Transform, Vector2, Vector3};
use core::mem;
use futures::executor;
use wgpu::{
    BindGroupLayout, Buffer, BufferUsage, Device, Extent3d, Queue, RenderPipeline,
    SamplerDescriptor, Surface, SwapChain, SwapChainOutput, Texture, TextureDescriptor,
    TextureFormat, TextureView, TextureViewDescriptor,
};
use winit::dpi::PhysicalSize;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);

fn main() {
    println!("Hello, world!");

    let renderer_delegate = RendererDelegate {};

    WindowEventManager::new(renderer_delegate).run_blocking()
}

struct RendererDelegate {}

struct WindowRenderTarget {
    surface: Surface,
    swap_chain: SwapChain,
    pub depth_texture: Texture,
}

impl WindowRenderTarget {
    fn new(device: &Device, surface: Surface, size: PhysicalSize<u32>) -> Self {
        let (swap_chain, depth_texture) =
            Self::create_swap_chain_and_depth_texture(device, &surface, size);

        WindowRenderTarget {
            surface,
            swap_chain,
            depth_texture,
        }
    }

    fn update_size(&mut self, device: &Device, size: PhysicalSize<u32>) {
        let (swap_chain, depth_texture) =
            Self::create_swap_chain_and_depth_texture(device, &self.surface, size);
        self.swap_chain = swap_chain;
        self.depth_texture = depth_texture;
    }

    fn create_swap_chain_and_depth_texture(
        device: &Device,
        surface: &Surface,
        size: PhysicalSize<u32>,
    ) -> (SwapChain, Texture) {
        let swap_chain = device.create_swap_chain(
            surface,
            &wgpu::SwapChainDescriptor {
                usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Mailbox,
            },
        );
        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            array_layer_count: 1,
            label: None,
        });

        (swap_chain, depth_texture)
    }

    fn get_texture(&mut self) -> SwapChainOutput {
        self.swap_chain.get_next_texture().unwrap()
    }
}

struct Renderer {
    counter: f32,
    render_target: WindowRenderTarget,
    device: Device,
    queue: Queue,
    pipeline: Option<RenderPipeline>,
    binding_layout: Option<BindGroupLayout>,
    size: PhysicalSize<u32>,
    mesh_buffer: Option<Buffer>,
    materials: Option<Vec<Material>>,
    vertices_count: u32,
}

impl Renderer {
    async fn init(delegate: RendererDelegate, window: &Window) -> Self {
        let mut renderer = Self::init_device_and_target(delegate, window).await;
        renderer.init_pipeline();
        renderer
    }

    async fn init_device_and_target(delegate: RendererDelegate, window: &Window) -> Self {
        let surface = wgpu::Surface::create(window);

        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::PRIMARY,
        )
        .await
        .unwrap();

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions {
                    anisotropic_filtering: false,
                },
                limits: wgpu::Limits::default(),
            })
            .await;

        let render_target = WindowRenderTarget::new(&device, surface, window.inner_size());

        Renderer {
            counter: 0f32,
            render_target,
            device,
            queue,
            pipeline: None,
            binding_layout: None,
            size: window.inner_size(),
            mesh_buffer: None,
            materials: None,
            vertices_count: 0,
        }
    }

    fn recreate_swap_chain(&mut self, size: PhysicalSize<u32>) {
        self.render_target.update_size(&self.device, size);
        self.size = size
    }

    fn init_pipeline(&mut self) {
        let device = &self.device;
        let object = ObjFile::read("minicooper.obj").unwrap();
        let mesh = object.flatten();
        self.materials = object.materials;
        //if self.materials == None {
        self.materials = Some(vec![Material {
            name: "".to_string(),
            texture: None,
            specular_exponent: Some(16),
            specular_color: Some([0.1, 0.4, 0.1]),
            diffuse_color: Some([0.5, 1.0, 0.5]),
            ambient_color: Some([1.0, 0.5, 0.5]),
        }]);
        //}
        let mesh_buffer =
            device.create_buffer_with_data(bytemuck::cast_slice(&mesh), BufferUsage::VERTEX);
        self.mesh_buffer = Some(mesh_buffer);
        self.vertices_count = mesh.len() as u32;
        let vs = include_bytes!("shader.vert.spv");
        let vs_module =
            device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&vs[..])).unwrap());

        let fs = include_bytes!("shader.frag.spv");
        let fs_module =
            device.create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&fs[..])).unwrap());

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::UniformBuffer { dynamic: false },
                },
            ],
            label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
        });

        self.binding_layout = Some(bind_group_layout);

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 2, // corresponds to bilinear filtering
                depth_bias_slope_scale: 2.0,
                depth_bias_clamp: 0.0,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                stencil_front: wgpu::StencilStateFaceDescriptor::IGNORE,
                stencil_back: wgpu::StencilStateFaceDescriptor::IGNORE,
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Float2,
                            offset: 24,
                            shader_location: 2,
                        },
                        wgpu::VertexAttributeDescriptor {
                            format: wgpu::VertexFormat::Uint,
                            offset: 32,
                            shader_location: 3,
                        },
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });
        self.pipeline = Some(render_pipeline);
    }

    fn draw(&mut self) {
        let device = &mut self.device;

        let aspect_ratio = self.size.width as f32 / self.size.height as f32;
        let projection_matrix = cgmath::perspective(Deg(45f32), aspect_ratio, 1.0, 10.0);

        let view_pos = Point3::new(1.2f32, 3.0, 0.5);
        let view_matrix = Matrix4::look_at(
            view_pos.clone(),
            Point3::new(0f32, 0f32, 0f32),
            cgmath::Vector3::unit_z(),
        );
        let view_projection_matrix = OPENGL_TO_WGPU_MATRIX * projection_matrix * view_matrix;

        let rot = self.counter;
        self.counter += 0.4;

        /*let model_matrix = Matrix4::from_angle_z(Deg(rot))
         * Matrix4::from_angle_x(Deg(90f32))
         * Matrix4::from_scale(1.5f32)
         * Matrix4::from_translation(vec3(0f32, -0.7f32, 0f32));*/

        let model_matrix = Matrix4::from_angle_z(Deg(rot))
            * Matrix4::from_scale(0.02f32)
            * Matrix4::from_translation(vec3(0f32, -0f32, -13f32));

        let normal_matrix = model_matrix.inverse_transform().unwrap().transpose();

        let vertex_uniforms = VertexUniforms {
            model_matrix,
            normal_matrix,
            view_projection_matrix,
        };

        let vertix_uniforms_buffer = device.create_buffer_with_data(
            bytemuck::bytes_of(&vertex_uniforms),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let mut styles = [unsafe { mem::zeroed() }; 10];

        let materials = self.materials.as_ref().unwrap();
        for i in 0..materials.len() {
            let m = &materials[i];
            styles[i] = Style {
                ambient_color: m.ambient_color.as_ref().unwrap().clone(),
                diffuse_color: m.diffuse_color.as_ref().unwrap().clone(),
                specular_color: m.specular_color.as_ref().unwrap().clone(),
                specular_exponent: m.specular_exponent.unwrap() as u32,
            }
        }

        let fragment_uniforms = FragmentUniforms {
            size: vec2(self.size.width as f32, self.size.height as f32),
            view_pos,
            styles,
        };
        let fragment_uniforms_buffer = device.create_buffer_with_data(
            bytemuck::bytes_of(&fragment_uniforms),
            wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: self.binding_layout.as_ref().unwrap(),
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &vertix_uniforms_buffer,
                        range: 0..(mem::size_of::<VertexUniforms>() as u64),
                    },
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &fragment_uniforms_buffer,
                        range: 0..(mem::size_of::<FragmentUniforms>() as u64),
                    },
                },
            ],
            label: None,
        });

        let out_texture = self.render_target.get_texture();
        let depth_texture = self.render_target.depth_texture.create_default_view();

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &out_texture.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::GREEN,
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &depth_texture,
                    depth_load_op: wgpu::LoadOp::Clear,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: wgpu::LoadOp::Clear,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
            });

            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_pipeline(self.pipeline.as_ref().unwrap());

            rpass.set_vertex_buffer(0, self.mesh_buffer.as_ref().unwrap(), 0, 0);
            rpass.draw(0..self.vertices_count, 0..1);
        }

        self.queue.submit(&[encoder.finish()]);
    }
}

struct WindowEventManager {
    event_loop: EventLoop<()>,
    renderer_delegate: RendererDelegate,
}

impl WindowEventManager {
    fn new(renderer_delegate: RendererDelegate) -> Self {
        let event_loop = EventLoop::new();

        WindowEventManager {
            event_loop,
            renderer_delegate,
        }
    }

    fn run_blocking(self) -> ! {
        let window = winit::window::Window::new(&self.event_loop).unwrap();
        let mut renderer = executor::block_on(Renderer::init(self.renderer_delegate, &window));
        self.event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::NewEvents(_) => {}
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => renderer.recreate_swap_chain(size),
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::DeviceEvent { .. } => {}
                Event::UserEvent(_) => {}
                Event::Suspended => {}
                Event::Resumed => {}
                Event::MainEventsCleared => window.request_redraw(),
                Event::RedrawRequested(_) => renderer.draw(),
                Event::RedrawEventsCleared => {}
                Event::LoopDestroyed => {}
                _ => {}
            }
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct VertexUniforms {
    view_projection_matrix: Matrix4<f32>,
    model_matrix: Matrix4<f32>,
    normal_matrix: Matrix4<f32>,
}

unsafe impl Pod for VertexUniforms {}

unsafe impl Zeroable for VertexUniforms {}

#[repr(C)]
#[derive(Clone, Copy)]
struct Style {
    pub ambient_color: [f32; 3],
    pub diffuse_color: [f32; 3],
    pub specular_color: [f32; 3],
    pub specular_exponent: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FragmentUniforms {
    size: Vector2<f32>,
    view_pos: Point3<f32>,
    styles: [Style; 10],
}

unsafe impl Pod for FragmentUniforms {}

unsafe impl Zeroable for FragmentUniforms {}
