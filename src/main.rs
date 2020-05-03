extern crate bytemuck;
extern crate futures;
extern crate winit;

mod obj_reader;

use crate::obj_reader::{Material, Mesh, ObjFile, Vertex};
use bytemuck::{Pod, Zeroable};
use cgmath::{
    dot, vec2, vec3, Deg, Matrix, Matrix4, Point3, SquareMatrix, Transform, Vector2, Vector3,
};
use core::mem;
use futures::executor;
use std::cmp::min;
use std::f32::consts::PI;
use std::time::Instant;
use wgpu::{
    BindGroupLayout, Buffer, BufferUsage, Device, Extent3d, Queue, RenderPipeline, Sampler,
    SamplerDescriptor, Surface, SwapChain, SwapChainOutput, Texture, TextureDescriptor,
    TextureFormat, TextureView, TextureViewDescriptor,
};
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::KeyboardInput;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, VirtualKeyCode};
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

    let renderer_delegate = PingPongRendererDelegate::new();

    WindowEventManager::run_blocking(renderer_delegate)
}

enum PingPongState {
    PlayerA,
    PlayerB,
    PlayerAtoB(f32),
    PlayerBtoA(f32),
}

struct PingPongRendererDelegate {
    state: PingPongState,
    ball_direction: Vector3<f32>,
    ball_position: Vector3<f32>,
    player_a_position: Vector2<f32>,
    player_b_position: Vector2<f32>,

    last_update: Instant,

    left_pressed: bool,
    right_pressed: bool,
    up_pressed: bool,
    down_pressed: bool,
}

fn generate_sphere() -> Mesh {
    let mut mesh = vec![];
    let steps = 20;
    let step_size = (2f32 * PI) / steps as f32;

    for r in 0..steps {
        let x = r as f32 * step_size;

        let rad_out_sin_start = x.sin();
        let rad_out_sin_end = (x + step_size).sin();

        let rad_out_cos_start = x.cos();
        let rad_out_cos_end = (x + step_size).cos();

        for i in 0..steps {
            let rad_start = i as f32 * step_size;
            let rad_end = rad_start + step_size;

            let a = [
                rad_start.sin() * rad_out_cos_start,
                rad_start.cos() * rad_out_cos_start,
                rad_out_sin_start,
            ];
            mesh.push(Vertex {
                texture: [0f32, 0f32],
                normal: a.clone(),
                position: a,
            });

            let b = [
                rad_start.sin() * rad_out_cos_end,
                rad_start.cos() * rad_out_cos_end,
                rad_out_sin_end,
            ];
            mesh.push(Vertex {
                texture: [0f32, 0f32],
                normal: b,
                position: b,
            });

            let c = [
                rad_end.sin() * rad_out_cos_start,
                rad_end.cos() * rad_out_cos_start,
                rad_out_sin_start,
            ];
            mesh.push(Vertex {
                texture: [0f32, 0f32],
                normal: c.clone(),
                position: c,
            });

            let a = [
                rad_end.sin() * rad_out_cos_start,
                rad_end.cos() * rad_out_cos_start,
                rad_out_sin_start,
            ];
            mesh.push(Vertex {
                texture: [0f32, 0f32],
                normal: a.clone(),
                position: a,
            });

            let b = [
                rad_start.sin() * rad_out_cos_end,
                rad_start.cos() * rad_out_cos_end,
                rad_out_sin_end,
            ];
            mesh.push(Vertex {
                texture: [0f32, 0f32],
                normal: b,
                position: b,
            });

            let c = [
                rad_end.sin() * rad_out_cos_end,
                rad_end.cos() * rad_out_cos_end,
                rad_out_sin_end,
            ];
            mesh.push(Vertex {
                texture: [0f32, 0f32],
                normal: c.clone(),
                position: c,
            });
        }
    }

    mesh
}

fn generate_cube() -> Mesh {
    ObjFile::read("cube.obj").unwrap().flatten()[0].clone()
}

impl PingPongRendererDelegate {
    fn new() -> Self {
        PingPongRendererDelegate {
            state: PingPongState::PlayerA,
            ball_direction: vec3(0.2f32.sin(), 0.5f32.sin() * 0.2f32.cos(), 0.5f32.cos()),
            ball_position: vec3(0.0, 0.0, 0.0),
            player_a_position: vec2(0.0, 0.0),
            player_b_position: vec2(0.0, 0.0),
            last_update: Instant::now(),
            left_pressed: false,
            right_pressed: false,
            up_pressed: false,
            down_pressed: false,
        }
    }
}

fn clamp(val: f32, min: f32, max: f32) -> f32 {
    if val < min {
        return min;
    }
    if val > max {
        return max;
    }
    val
}
fn clamp_vec2(vec: Vector2<f32>, min: Vector2<f32>, max: Vector2<f32>) -> Vector2<f32> {
    vec2(clamp(vec.x, min.x, max.x), clamp(vec.y, min.y, max.y))
}
fn clamp_vec3(vec: Vector3<f32>, min: Vector3<f32>, max: Vector3<f32>) -> Vector3<f32> {
    vec3(
        clamp(vec.x, min.x, max.x),
        clamp(vec.y, min.y, max.y),
        clamp(vec.z, min.z, max.z),
    )
}

fn reflect(vec: Vector3<f32>, normal: Vector3<f32>) -> Vector3<f32> {
    vec - 2.0 * dot(vec, normal) * normal
}

const PLAYER_Z: f32 = 1.5;

impl RenderDelegate for PingPongRendererDelegate {
    fn on_init(&mut self, elements: &mut Vec<RenderElement>) {
        elements.push(RenderElement {
            mesh: generate_sphere(),
            material: Material {
                texture: None,
                diffuse_color: Some([1.0f32, 1.0f32, 1.0f32]),
                ambient_color: Some([1.0f32, 1.0f32, 1.0f32]),
                specular_color: Some([1.0f32, 1.0f32, 1.0f32]),
                specular_exponent: Some(16),
                name: "".to_string(),
            },
            transformation: Matrix4::from_scale(0.1f32),
            mesh_buffer: None,
        });

        elements.push(RenderElement {
            mesh: generate_cube(),
            material: Material {
                texture: None,
                diffuse_color: Some([0f32, 1.0f32, 0f32]),
                ambient_color: Some([0f32, 1.0f32, 0f32]),
                specular_color: Some([0f32, 1.0f32, 0f32]),
                specular_exponent: Some(16),
                name: "".to_string(),
            },
            transformation: Matrix4::from_translation(vec3(-0.3f32, -0.25f32, PLAYER_Z))
                * Matrix4::from_nonuniform_scale(0.6f32, 0.5f32, 0.03f32),
            mesh_buffer: None,
        });

        elements.push(RenderElement {
            mesh: generate_cube(),
            material: Material {
                texture: None,
                diffuse_color: Some([0f32, 1.0f32, 0f32]),
                ambient_color: Some([0f32, 1.0f32, 0f32]),
                specular_color: Some([0f32, 1.0f32, 0f32]),
                specular_exponent: Some(16),
                name: "".to_string(),
            },
            transformation: Matrix4::from_translation(vec3(-0.3f32, -0.25f32, -PLAYER_Z))
                * Matrix4::from_nonuniform_scale(0.6f32, 0.5f32, 0.03f32),
            mesh_buffer: None,
        });

        elements.push(RenderElement {
            mesh: generate_cube(),
            material: Material {
                texture: None,
                diffuse_color: Some([0f32, 1.0f32, 0f32]),
                ambient_color: Some([1f32, 1.0f32, 1f32]),
                specular_color: Some([0f32, 1.0f32, 0f32]),
                specular_exponent: Some(16),
                name: "".to_string(),
            },
            transformation: Matrix4::from_translation(vec3(-0.75f32, -0.75f32, -PLAYER_Z))
                * Matrix4::from_nonuniform_scale(0.01f32, 0.01f32, 2.0 * PLAYER_Z),
            mesh_buffer: None,
        });

        elements.push(RenderElement {
            mesh: generate_cube(),
            material: Material {
                texture: None,
                diffuse_color: Some([0f32, 1.0f32, 0f32]),
                ambient_color: Some([1f32, 1.0f32, 1f32]),
                specular_color: Some([0f32, 1.0f32, 0f32]),
                specular_exponent: Some(16),
                name: "".to_string(),
            },
            transformation: Matrix4::from_translation(vec3(0.75f32, 0.75f32, -PLAYER_Z))
                * Matrix4::from_nonuniform_scale(0.01f32, 0.01f32, 2.0 * PLAYER_Z),
            mesh_buffer: None,
        });

        elements.push(RenderElement {
            mesh: generate_cube(),
            material: Material {
                texture: None,
                diffuse_color: Some([0f32, 1.0f32, 0f32]),
                ambient_color: Some([1f32, 1.0f32, 1f32]),
                specular_color: Some([0f32, 1.0f32, 0f32]),
                specular_exponent: Some(16),
                name: "".to_string(),
            },
            transformation: Matrix4::from_translation(vec3(0.75f32, -0.75f32, -PLAYER_Z))
                * Matrix4::from_nonuniform_scale(0.01f32, 0.01f32, 2.0 * PLAYER_Z),
            mesh_buffer: None,
        });

        elements.push(RenderElement {
            mesh: generate_cube(),
            material: Material {
                texture: None,
                diffuse_color: Some([0f32, 1.0f32, 0f32]),
                ambient_color: Some([1f32, 1.0f32, 1f32]),
                specular_color: Some([0f32, 1.0f32, 0f32]),
                specular_exponent: Some(16),
                name: "".to_string(),
            },
            transformation: Matrix4::from_translation(vec3(-0.75f32, 0.75f32, -PLAYER_Z))
                * Matrix4::from_nonuniform_scale(0.01f32, 0.01f32, 2.0 * PLAYER_Z),
            mesh_buffer: None,
        });
    }

    fn on_frame(&mut self, elements: &mut Vec<RenderElement>, view_matrix: &mut Matrix4<f32>) {
        let now = Instant::now();
        let duration = (now - self.last_update).as_millis() as f32;
        self.last_update = now;

        self.ball_position += self.ball_direction * duration * 0.0005f32;

        let max_ball = vec3(0.75, 0.75, PLAYER_Z - 0.1);
        let min_ball = vec3(-0.75, -0.75, -PLAYER_Z + 0.1);

        if self.ball_position.z > max_ball.z {
            self.ball_direction = reflect(self.ball_direction, vec3(0.0, 0.0, -1.0));
            self.ball_position.z = max_ball.z;

            if (self.ball_position.x - self.player_a_position.x).abs() > 0.25
                || (self.ball_position.y - self.player_a_position.y).abs() > 0.25
            {
                println!("Game Over")
            }
            self.state = PingPongState::PlayerAtoB(0.0)
        } else if self.ball_position.z < min_ball.z {
            self.ball_direction = reflect(self.ball_direction, vec3(0.0, 0.0, 1.0));
            self.ball_position.z = min_ball.z;
            self.state = PingPongState::PlayerBtoA(0.0)
        }

        if self.ball_position.y > max_ball.y {
            self.ball_direction = reflect(self.ball_direction, vec3(0.0, -1.0, 0.0));
            self.ball_position.y = max_ball.y;
        } else if self.ball_position.y < min_ball.y {
            self.ball_direction = reflect(self.ball_direction, vec3(0.0, 1.0, 0.0));
            self.ball_position.y = min_ball.y;
        }

        if self.ball_position.x > max_ball.x {
            self.ball_direction = reflect(self.ball_direction, vec3(-1.0, 0.0, 0.0));
            self.ball_position.x = max_ball.x;
        } else if self.ball_position.x < min_ball.y {
            self.ball_direction = reflect(self.ball_direction, vec3(1.0, 0.0, 0.0));
            self.ball_position.x = min_ball.x;
        }

        elements[0].transformation =
            Matrix4::from_translation(self.ball_position) * Matrix4::from_scale(0.1f32);

        if self.left_pressed || self.right_pressed || self.down_pressed || self.up_pressed {
            let (player_pos, index, z_pos) = match self.state {
                PingPongState::PlayerAtoB(_) | PingPongState::PlayerA => {
                    (&mut self.player_a_position, 1, PLAYER_Z)
                }
                PingPongState::PlayerBtoA(_) | PingPongState::PlayerB => {
                    (&mut self.player_b_position, 2, -PLAYER_Z)
                }
            };

            if self.left_pressed && player_pos.x > min_ball.x {
                player_pos.x -= duration * 0.001f32;
            } else if self.right_pressed && player_pos.x < max_ball.x {
                player_pos.x += duration * 0.001f32;
            }

            if self.down_pressed && player_pos.y > min_ball.y {
                player_pos.y -= duration * 0.001f32;
            } else if self.up_pressed && player_pos.y < max_ball.y {
                player_pos.y += duration * 0.001f32;
            }

            elements[index].transformation =
                Matrix4::from_translation(vec3(
                    -0.25f32 + player_pos.x,
                    -0.25f32 + player_pos.y,
                    z_pos,
                )) * Matrix4::from_nonuniform_scale(0.5f32, 0.5f32, 0.03f32);
        }

        let (eye, state) = match self.state {
            PingPongState::PlayerA => (Point3::new(0f32, 1.9f32, -2.1), PingPongState::PlayerA),
            PingPongState::PlayerB => (Point3::new(0f32, 1.9f32, 2.1), PingPongState::PlayerB),
            PingPongState::PlayerAtoB(rad) => (
                Point3::new(rad.sin(), 1.9f32, -2.1 * rad.cos()),
                if rad < PI {
                    PingPongState::PlayerAtoB(rad + 0.05)
                } else {
                    PingPongState::PlayerB
                },
            ),
            PingPongState::PlayerBtoA(rad) => (
                Point3::new(rad.sin(), 1.9f32, 2.1 * rad.cos()),
                if rad < PI {
                    PingPongState::PlayerBtoA(rad + 0.05)
                } else {
                    PingPongState::PlayerA
                },
            ),
        };

        self.state = state;
        *view_matrix = Matrix4::look_at(
            eye,
            Point3::new(0f32, 0f32, self.ball_position.z),
            cgmath::Vector3::unit_y(),
        );
    }

    fn on_key(&mut self, key_code: VirtualKeyCode, pressed: bool) {
        match key_code {
            VirtualKeyCode::Left => self.left_pressed = pressed,
            VirtualKeyCode::Right => self.right_pressed = pressed,
            VirtualKeyCode::Up => self.up_pressed = pressed,
            VirtualKeyCode::Down => self.down_pressed = pressed,
            _ => {}
        }
    }
}

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

struct RenderElement {
    mesh: Mesh,
    mesh_buffer: Option<Buffer>,
    transformation: Matrix4<f32>,
    material: Material,
}

trait RenderDelegate: Sized {
    fn on_init(&mut self, elements: &mut Vec<RenderElement>);
    fn on_frame(&mut self, elements: &mut Vec<RenderElement>, view_matrix: &mut Matrix4<f32>);
    fn on_key(&mut self, key_code: VirtualKeyCode, pressed: bool);
}

struct Renderer<D: RenderDelegate> {
    delegate: D,
    camera_move: PhysicalPosition<f64>,
    camera_zoom: f32,
    render_target: WindowRenderTarget,
    device: Device,
    queue: Queue,
    pipeline: Option<RenderPipeline>,
    binding_layout: Option<BindGroupLayout>,
    size: PhysicalSize<u32>,
    elements: Vec<RenderElement>,
    counter: f32,
}

impl<D: RenderDelegate> Renderer<D> {
    async fn init(delegate: D, window: &Window) -> Self {
        let mut renderer = Self::init_device_and_target(delegate, window).await;
        renderer.init_pipeline();
        renderer
    }

    async fn init_device_and_target(delegate: D, window: &Window) -> Self {
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
                    anisotropic_filtering: true,
                },
                limits: wgpu::Limits::default(),
            })
            .await;

        let render_target = WindowRenderTarget::new(&device, surface, window.inner_size());

        let mut r = Renderer {
            camera_move: PhysicalPosition::new(0f64, 0f64),
            camera_zoom: 0.02f32,
            render_target,
            device,
            queue,
            pipeline: None,
            binding_layout: None,
            size: window.inner_size(),
            elements: vec![],
            delegate,
            counter: 0.0f32,
        };

        r.delegate.on_init(&mut r.elements);
        r
    }

    fn recreate_swap_chain(&mut self, size: PhysicalSize<u32>) {
        self.render_target.update_size(&self.device, size);
        self.size = size
    }

    fn init_pipeline(&mut self) {
        let device = &self.device;

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
                depth_bias: 2,
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
                    ],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });
        self.pipeline = Some(render_pipeline);
    }

    fn add_element(&mut self, mut element: RenderElement) {
        element.mesh_buffer = Some(self.device.create_buffer_with_data(
            bytemuck::cast_slice(&element.mesh),
            wgpu::BufferUsage::VERTEX,
        ));
        self.elements.push(element);
    }

    fn draw(&mut self) {
        let mut view_matrix = Matrix4::identity();

        self.delegate.on_frame(&mut self.elements, &mut view_matrix);
        let device = &mut self.device;
        let out_texture = self.render_target.get_texture();
        let depth_texture = self.render_target.depth_texture.create_default_view();

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        for (i, element) in self.elements.iter_mut().enumerate() {
            if element.mesh_buffer.is_none() {
                element.mesh_buffer = Some(device.create_buffer_with_data(
                    bytemuck::cast_slice(&element.mesh),
                    wgpu::BufferUsage::VERTEX,
                ));
            }

            let m = &element.material;
            let style = Style {
                ambient_color: m.ambient_color.as_ref().unwrap().clone(),
                diffuse_color: m.diffuse_color.as_ref().unwrap().clone(),
                specular_color: m.specular_color.as_ref().unwrap().clone(),
                specular_exponent: m.specular_exponent.unwrap() as f32,
            };

            let aspect_ratio = self.size.width as f32 / self.size.height as f32;
            let projection_matrix = cgmath::perspective(Deg(45f32), aspect_ratio, 1.0, 10.0);

            let view_pos = Point3::new(0f32, 0f32, -3f32);
            let view_projection_matrix = OPENGL_TO_WGPU_MATRIX * projection_matrix * view_matrix;

            let model_matrix = element.transformation.clone();

            /*
                    let model_matrix = Matrix4::from_angle_z(Deg(rot))
                        * Matrix4::from_scale(0.02f32)
                        * Matrix4::from_translation(vec3(0f32, -0f32, -13f32));
            */
            let normal_matrix = model_matrix.inverse_transform().unwrap().transpose();

            let vertex_uniforms = VertexUniforms {
                model_matrix,
                normal_matrix,
                view_projection_matrix,
            };

            let vertex_uniforms_buffer = device.create_buffer_with_data(
                bytemuck::bytes_of(&vertex_uniforms),
                wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            );

            let fragment_uniforms_arr = [
                view_pos.x,
                view_pos.y,
                view_pos.z,
                style.ambient_color[0],
                style.ambient_color[1],
                style.ambient_color[2],
                style.diffuse_color[0],
                style.diffuse_color[1],
                style.diffuse_color[2],
                style.specular_color[0],
                style.specular_color[1],
                style.specular_color[2],
                style.specular_exponent,
                0f32,
                0f32,
            ];

            let fragment_uniforms_buffer = device.create_buffer_with_data(
                bytemuck::cast_slice(&fragment_uniforms_arr),
                wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            );

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: self.binding_layout.as_ref().unwrap(),
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &vertex_uniforms_buffer,
                            range: 0..(mem::size_of::<VertexUniforms>() as u64),
                        },
                    },
                    wgpu::Binding {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &fragment_uniforms_buffer,
                            range: 0..(mem::size_of::<[f32; 15]>() as u64),
                        },
                    },
                ],
                label: None,
            });

            let load_op = if i == 0 {
                wgpu::LoadOp::Clear
            } else {
                wgpu::LoadOp::Load
            };

            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &out_texture.view,
                    resolve_target: None,
                    load_op,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color::GREEN,
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachmentDescriptor {
                    attachment: &depth_texture,
                    depth_load_op: load_op,
                    depth_store_op: wgpu::StoreOp::Store,
                    stencil_load_op: load_op,
                    stencil_store_op: wgpu::StoreOp::Store,
                    clear_depth: 1.0,
                    clear_stencil: 0,
                }),
            });

            rpass.set_bind_group(0, &bind_group, &[]);
            rpass.set_pipeline(self.pipeline.as_ref().unwrap());

            rpass.set_vertex_buffer(0, element.mesh_buffer.as_ref().unwrap(), 0, 0);
            rpass.draw(0..(element.mesh.len() as u32), 0..1);
        }
        self.queue.submit(&[encoder.finish()]);
    }
}

struct WindowEventManager {}

impl WindowEventManager {
    fn run_blocking<D: RenderDelegate + 'static>(renderer_delegate: D) -> ! {
        let event_loop = EventLoop::new();
        let window = winit::window::Window::new(&event_loop).unwrap();
        let mut renderer = executor::block_on(Renderer::init(renderer_delegate, &window));
        let mut current_pos: Option<PhysicalPosition<f64>> = None;

        event_loop.run(move |event, _, control_flow| {
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
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { input, .. },
                    ..
                } => match input {
                    KeyboardInput {
                        state: ElementState::Pressed,
                        virtual_keycode: Some(virtual_keycode),
                        ..
                    } => renderer.delegate.on_key(virtual_keycode, true),
                    KeyboardInput {
                        state: ElementState::Released,
                        virtual_keycode: Some(virtual_keycode),
                        ..
                    } => renderer.delegate.on_key(virtual_keycode, false),
                    _ => {}
                },
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
    pub specular_exponent: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FragmentUniforms {
    size: Vector2<f32>,
    view_pos: Point3<f32>,
    style: Style,
}

unsafe impl Pod for FragmentUniforms {}

unsafe impl Zeroable for FragmentUniforms {}
