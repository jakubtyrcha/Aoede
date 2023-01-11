use std::borrow::Cow;
use std::iter;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use audio_script::build_audio_script_engine;
use audio_setup::stream_setup_for;
use audio_synthesis::AudioGraph;
use audio_synthesis::AudioGraphBuilder;
use bytemuck::{Pod, Zeroable};
use cpal::traits::StreamTrait;

use ::egui::FontDefinitions;
use chrono::Timelike;
use egui::plot;
use egui::Context;
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::util::DeviceExt;
use wgpu::{ImageCopyTexture, Origin3d};
use winit::event::Event::*;
use winit::event_loop::ControlFlow;

const INITIAL_WIDTH: u32 = 1920;
const INITIAL_HEIGHT: u32 = 1080;

mod audio_setup;
mod audio_synthesis;
mod audio_script;

use rhai::{Engine};
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    _pos: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniforms {
    size: [i32; 2],
}

//

struct MyApp {
    stream: cpal::Stream,
    queue: Arc<crossbeam_queue::ArrayQueue<f32>>,
    samples: Vec<f32>,
    write_index: usize,
    graph: Option<AudioGraph>,
    picked_path: Option<String>,
    engine: Engine,
    error: Option<Box<rhai::EvalAltResult>>
}

struct AudioSampleReader {
    queue: Arc<crossbeam_queue::ArrayQueue<f32>>,
}

const AUDIO_BUFFER_SEC: u32 = 10;
const SAMPLES_PRECOMPUTE_BUFFER_SEC: usize = 1;

impl MyApp {
    fn precompute_samples(&mut self, index: usize) {
        let buffer = SAMPLES_PRECOMPUTE_BUFFER_SEC * 48000; // 1 sec ahead
        while self.write_index < index + buffer {
            let sample = if let Some(graph) = &mut self.graph {
                graph.gen_next_sample()
            } else {
                0.0
            };

            let samples_num = self.samples.len();
            self.samples[self.write_index % samples_num] = sample;
            self.queue.push(sample).unwrap();

            self.write_index += 1;
        }
    }
}

impl Default for MyApp {
    fn default() -> Self {
        let (_host, device, config) = audio_setup::host_device_setup().unwrap();

        let engine = build_audio_script_engine();

        let queue_size = config.sample_rate().0 * AUDIO_BUFFER_SEC;
        let q = Arc::new(crossbeam_queue::ArrayQueue::new(queue_size as usize));

        let samples = Vec::from_iter(
            [0.0 as f32]
                .iter()
                .cycle()
                .take(queue_size as usize)
                .cloned(),
        );

        let mut result = MyApp {
            stream: stream_setup_for(
                &device,
                &config,
                move |o: &mut AudioSampleReader| o.queue.pop().unwrap_or(0.0),
                AudioSampleReader { queue: q.clone() },
            )
            .unwrap(),
            queue: q,
            samples,
            write_index: 0,
            graph: None,
            picked_path: None,
            engine,
            error: None
        };
        result.precompute_samples(0);
        result.stream.play().unwrap();
        result
    }
}

impl MyApp {
    pub fn recompile_graph(&mut self) {
        let graph_builder = self
            .engine
            .eval_file::<AudioGraphBuilder>(self.picked_path.clone().unwrap().into());
        if graph_builder.is_ok() {
            let mut graph = graph_builder.unwrap().extract_graph();
            graph.set_sample_rate(48000);
            self.graph = Some(graph);
            self.error = None;
        }
        else {
            self.error = graph_builder.err();
        }
    }

    pub fn ui(&mut self, ctx: &Context) {
        egui::Window::new("Aoede").show(ctx, |ui| {
            if ui.button("Open file…").clicked() {
                if let Some(path) = rfd::FileDialog::new().pick_file() {
                    self.picked_path = Some(path.display().to_string());

                    self.recompile_graph();
                }
            }

            if ui.button("Reload").clicked() && self.picked_path.is_some() {
                self.recompile_graph();
            }

            if let Some(error) = &self.error {
                ui.label(error.to_string());
            }

            if let Some(picked_path) = &self.picked_path {
                ui.horizontal(|ui| {
                    ui.label("Picked file:");
                    ui.monospace(picked_path);
                });
            }

            let data_clone = self.samples.clone();

            let sample_rate = 48000;

            let points = plot::PlotPoints::from_explicit_callback(
                move |x: f64| {
                    let index = if x >= 0.0 {
                        (x * sample_rate as f64) as usize
                    } else {
                        0
                    };
                    if index < data_clone.len() {
                        return data_clone[index] as f64;
                    }
                    0.0
                },
                std::f64::NEG_INFINITY..std::f64::INFINITY,
                1000,
            );

            plot::Plot::new("Audio plot")
                .data_aspect(1.0)
                .show(ui, |plot_ui| plot_ui.line(plot::Line::new(points)));
        });
    }
}

/// A custom event type for the winit app.
enum Event {
    RequestRedraw,
}

/// This is the repaint signal type that egui needs for requesting a repaint from another thread.
/// It sends the custom RequestRedraw event to the winit event loop.
struct ExampleRepaintSignal(std::sync::Mutex<winit::event_loop::EventLoopProxy<Event>>);

impl epi::backend::RepaintSignal for ExampleRepaintSignal {
    fn request_repaint(&self) {
        self.0.lock().unwrap().send_event(Event::RequestRedraw).ok();
    }
}

/// A simple egui + wgpu + winit based example.
fn main() {
    let event_loop = winit::event_loop::EventLoopBuilder::<Event>::with_user_event().build();
    let window = winit::window::WindowBuilder::new()
        .with_decorations(true)
        .with_resizable(true)
        .with_transparent(false)
        .with_title("egui-wgpu_winit example")
        .with_inner_size(winit::dpi::PhysicalSize {
            width: INITIAL_WIDTH,
            height: INITIAL_HEIGHT,
        })
        .build(&event_loop)
        .unwrap();

    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);
    let surface = unsafe { instance.create_surface(&window) };

    // WGPU 0.11+ support force fallback (if HW implementation not supported), set it to true or false (optional).
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    }))
    .unwrap();

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
            label: None,
        },
        None,
    ))
    .unwrap();

    let size = window.inner_size();
    let surface_format = surface.get_supported_formats(&adapter)[0];
    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: size.width as u32,
        height: size.height as u32,
        present_mode: wgpu::PresentMode::Fifo,
    };
    surface.configure(&device, &surface_config);

    // We use the egui_winit_platform crate as the platform.
    let mut platform = Platform::new(PlatformDescriptor {
        physical_width: size.width as u32,
        physical_height: size.height as u32,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    });

    // We use the egui_wgpu_backend crate as the render backend.
    let mut egui_rpass = RenderPass::new(&device, surface_format, 1);

    let mut app: MyApp = MyApp::default();
    //let mut reader = MtRingbufferReader::new(unsafe { AUDIO_RINGBUFFER.as_mut().unwrap() });

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Uniforms>() as _),
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: None,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let audio_texture_width = 480;
    let audio_texture_height = 100;

    let texture_extent = wgpu::Extent3d {
        width: audio_texture_width,
        height: audio_texture_height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Color Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let uniforms = Uniforms {
        size: [audio_texture_width as i32, audio_texture_height as i32],
    };
    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("uniforms"),
        contents: bytemuck::bytes_of(&uniforms),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
        label: None,
    });

    let swapchain_format = surface.get_supported_formats(&adapter)[0];

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(swapchain_format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let audio_samples_tex_buffer_size = (audio_texture_width * audio_texture_height) as i64;
    let audio_data = Vec::from_iter(
        [0.0 as f32]
            .iter()
            .cycle()
            .take(audio_samples_tex_buffer_size as usize)
            .cloned(),
    );

    // todo: this should match audio start time, not start of the loop
    let start_time = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        // Pass the winit events to the platform integration.
        platform.handle_event(&event);

        match event {
            RedrawRequested(..) => {
                platform.update_time(start_time.elapsed().as_secs_f64());
                let time_pointer = start_time.elapsed().as_millis() as usize * 48000 / 1000;
                app.precompute_samples(time_pointer);

                let output_frame = match surface.get_current_texture() {
                    Ok(frame) => frame,
                    Err(wgpu::SurfaceError::Outdated) => {
                        // This error occurs when the app is minimized on Windows.
                        // Silently return here to prevent spamming the console with:
                        // "The underlying surface has changed, and therefore the swap chain must be updated"
                        return;
                    }
                    Err(e) => {
                        eprintln!("Dropped frame with error: {}", e);
                        return;
                    }
                };
                let output_view = output_frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // Begin to draw the UI frame.
                platform.begin_frame();

                // Draw the demo application.
                //demo_app.ui(&platform.context());
                app.ui(&platform.context());

                // End the UI frame. We could now handle the output and draw the UI with the backend.
                let full_output = platform.end_frame(Some(&window));
                let paint_jobs = platform.context().tessellate(full_output.shapes);

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("encoder"),
                });

                let v_bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        audio_data.as_ptr() as *const u8,
                        audio_data.len() * std::mem::size_of::<f32>(),
                    )
                };

                queue.write_texture(
                    ImageCopyTexture {
                        texture: &texture,
                        mip_level: 0,
                        origin: Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    v_bytes,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: NonZeroU32::new(
                            audio_texture_width * std::mem::size_of::<f32>() as u32,
                        ),
                        rows_per_image: NonZeroU32::new(audio_texture_height),
                    },
                    wgpu::Extent3d {
                        width: audio_texture_width,
                        height: audio_texture_height,
                        depth_or_array_layers: 1,
                    },
                );

                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &output_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_bind_group(0, &bind_group, &[]);
                    rpass.draw(0..3, 0..1);
                }

                // Upload all resources for the GPU.
                let screen_descriptor = ScreenDescriptor {
                    physical_width: surface_config.width,
                    physical_height: surface_config.height,
                    scale_factor: window.scale_factor() as f32,
                };
                let tdelta: egui::TexturesDelta = full_output.textures_delta;
                egui_rpass
                    .add_textures(&device, &queue, &tdelta)
                    .expect("add texture ok");
                egui_rpass.update_buffers(&device, &queue, &paint_jobs, &screen_descriptor);

                // Record all render passes.
                egui_rpass
                    .execute(
                        &mut encoder,
                        &output_view,
                        &paint_jobs,
                        &screen_descriptor,
                        None,
                    )
                    .unwrap();

                // Submit the commands.
                queue.submit(iter::once(encoder.finish()));

                // Redraw egui
                output_frame.present();

                egui_rpass
                    .remove_textures(tdelta)
                    .expect("remove texture ok");

                // Support reactive on windows only, but not on linux.
                // if _output.needs_repaint {
                //     *control_flow = ControlFlow::Poll;
                // } else {
                //     *control_flow = ControlFlow::Wait;
                // }
            }
            MainEventsCleared | UserEvent(Event::RequestRedraw) => {
                window.request_redraw();
            }
            WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::Resized(size) => {
                    // Resize with 0 width and height is used by winit to signal a minimize event on Windows.
                    // See: https://github.com/rust-windowing/winit/issues/208
                    // This solves an issue where the app would panic when minimizing on Windows.
                    if size.width > 0 && size.height > 0 {
                        surface_config.width = size.width;
                        surface_config.height = size.height;
                        surface.configure(&device, &surface_config);
                    }
                }
                winit::event::WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            _ => (),
        }
    });
}

/// Time of day as seconds since midnight. Used for clock in demo app.
pub fn seconds_since_midnight() -> f64 {
    let time = chrono::Local::now().time();
    time.num_seconds_from_midnight() as f64 + 1e-9 * (time.nanosecond() as f64)
}
