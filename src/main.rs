use std::borrow::Cow;
use std::iter;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicI32, AtomicU32};
use std::sync::mpsc::channel;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use audio::{stream_setup_for, SampleRequestOptions};
use bytemuck::{Pod, Zeroable};
use cpal::traits::StreamTrait;
use cpal::Sample;

use ::egui::FontDefinitions;
use chrono::Timelike;
use egui::Context;
use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
use egui_winit_platform::{Platform, PlatformDescriptor};
use wgpu::util::DeviceExt;
use wgpu::{ImageCopyTexture, ImageDataLayout, Origin3d};
use winit::event::Event::*;
use winit::event_loop::ControlFlow;
const INITIAL_WIDTH: u32 = 1920;
const INITIAL_HEIGHT: u32 = 1080;

mod audio;

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

#[derive(Copy, Clone)]
struct SampleSetup {
    sample_rate: u32,
    nchannels: usize,
}

#[derive(Copy, Clone)]
struct AudioSampleGenerator {
    // hz
    sin0_freq: f32,
    sin0_phase: f32,
    sin0_vol: f32,
    sin1_freq: f32,
    sin1_phase: f32,
    sin1_vol: f32,
}

impl AudioSampleGenerator {
    fn generate(&self, setup: SampleSetup, sample_index: i32) -> f32 {
        let t_sec = sample_index as f32 / setup.sample_rate as f32;
        let wave0 = (t_sec * self.sin0_freq * 2.0 * std::f32::consts::PI + self.sin0_phase).sin();
        let wave1 = (t_sec * self.sin1_freq * 2.0 * std::f32::consts::PI + self.sin1_phase).sin();
        let sum = wave0 * self.sin0_vol + wave1 * self.sin1_vol;
        sum
    }
}

// API
// one buffer (array?)
// updated from the audio thread?
// audio thread evaluates new samples and writes to the array

// app reads from the buffer, also updates config!

// sample rate
// nchannels

struct AudioSampleRingbuffer {
    buffer: Vec<AtomicU32>,
    next_sample_write: AtomicI32,
}

impl AudioSampleRingbuffer {
    fn new(size: usize) -> AudioSampleRingbuffer {
        let mut buffer = Vec::new();
        buffer.resize_with(size, || AtomicU32::new(0));
        AudioSampleRingbuffer {
            buffer: buffer,
            next_sample_write: AtomicI32::new(0),
        }
    }

    fn write(&mut self, value: u32) {
        let N = self.buffer.len();
        let write_sample = self
            .next_sample_write
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let write_index = write_sample as usize % N;
        self.buffer[write_index].store(value, std::sync::atomic::Ordering::SeqCst);
    }
}

struct AudioSampleProducer<'a> {
    setup: SampleSetup,
    config: AudioSampleGenerator,
    config_tx: std::sync::mpsc::Receiver<AudioSampleGenerator>,
    write_buffer: &'a mut AudioSampleRingbuffer,
}

impl<'a> AudioSampleProducer<'a> {
    fn new(
        setup: SampleSetup,
        config: AudioSampleGenerator,
        config_tx: std::sync::mpsc::Receiver<AudioSampleGenerator>,
        write_buffer: &'a mut AudioSampleRingbuffer,
    ) -> AudioSampleProducer<'a> {
        AudioSampleProducer {
            setup,
            config,
            config_tx,
            write_buffer,
        }
    }

    fn write_sample(&mut self, value: u32) {
        self.write_buffer.write(value);
    }
}

struct AudioSampleViewer<'a> {
    buffer: &'a AudioSampleRingbuffer,
    next_sample_read: i32,
}

impl<'a> AudioSampleViewer<'a> {
    fn new(buffer: &'a AudioSampleRingbuffer) -> AudioSampleViewer<'a> {
        AudioSampleViewer {
            buffer,
            next_sample_read: 0,
        }
    }

    fn next(&mut self) -> Option<u32> {
        let buffer_sentinel = self
            .buffer
            .next_sample_write
            .load(std::sync::atomic::Ordering::SeqCst);
        if self.next_sample_read < buffer_sentinel {
            let buffer_index = self.next_sample_read % self.buffer.buffer.len() as i32;
            let value =
                self.buffer.buffer[buffer_index as usize].load(std::sync::atomic::Ordering::SeqCst);
            self.next_sample_read += 1;
            return Some(value);
        }
        None
    }
}

//

static mut AUDIO_RINGBUFFER: Option<AudioSampleRingbuffer> = None;
static mut AUDIO_SAMPLE_PRODUCER: Option<AudioSampleProducer> = None;

struct MyApp {
    stream: cpal::Stream,
    state: Arc<Mutex<SharedState>>,
    state1: AudioSampleGenerator,
    rx: std::sync::mpsc::Sender<AudioSampleGenerator>,
}

struct SharedState {
    freq: f32,
    volume: f32,
}

impl Default for MyApp {
    fn default() -> Self {
        let state = Arc::new(Mutex::new(SharedState {
            freq: 1000.0,
            volume: 1.0,
        }));

        let sample_setup = SampleSetup {
            sample_rate: 48000,
            nchannels: 2,
        };

        let samplegen = AudioSampleGenerator {
            sin0_freq: 1000.0,
            sin0_phase: 0.0,
            sin0_vol: 0.5,
            sin1_freq: 500.0,
            sin1_phase: 0.0,
            sin1_vol: 0.0,
        };

        let (rx, tx) = channel();

        //let state_clone = Arc::clone(&state);
        unsafe {
            AUDIO_RINGBUFFER = Some(AudioSampleRingbuffer::new(256));
            AUDIO_SAMPLE_PRODUCER = Some(AudioSampleProducer::new(
                sample_setup,
                samplegen,
                tx,
                unsafe { AUDIO_RINGBUFFER.as_mut().unwrap() },
            ));
        }
        let result = MyApp {
            stream: stream_setup_for(move |o: &mut SampleRequestOptions| {
                let mut sample = 0.0;
                unsafe {
                    let mut producer = AUDIO_SAMPLE_PRODUCER
                    .as_mut()
                    .unwrap();

                    if let Ok(config) = producer.config_tx.try_recv() {
                        producer.config = config;
                    }

                    sample = producer
                        .config
                        .generate(producer.setup, o.index as i32);
                        o.index += 1;
                    producer.write_sample(sample.to_bits());
                }
                sample
                // o.tick();
                // let state = state_clone.lock().unwrap();
                // o.tone(state.freq) * state.volume
            })
            .unwrap(),
            state1: samplegen,
            rx,
            state: state,
        };
        result.stream.play().unwrap();
        result
    }
}

impl MyApp {
    pub fn ui(&mut self, ctx: &Context) {
        egui::Window::new("Demo").show(ctx, |ui| {
            ui.heading("My egui Application");
            //let mut state = self.state.lock();
            let config = &mut self.state1;
            let mut changed = false;
            changed |= ui.add(
                egui::Slider::new(&mut config.sin0_freq, 10.0..=1000.0)
                    .text("Frequency 0"),
            ).changed();
            changed |= ui.add(
                egui::Slider::new(&mut config.sin0_phase, 0.0..=3.14159)
                    .text("Phase 0"),
            ).changed();
            changed |= ui.add(
                egui::Slider::new(&mut config.sin0_vol, 0.0..=2.0).text("Volume 0"),
            ).changed();

            changed |= ui.add(
                egui::Slider::new(&mut config.sin1_freq, 10.0..=1000.0)
                    .text("Frequency 1"),
            ).changed();
            changed |= ui.add(
                egui::Slider::new(&mut config.sin1_phase, 0.0..=3.14159)
                    .text("Phase 1"),
            ).changed();
            changed |= ui.add(
                egui::Slider::new(&mut config.sin1_vol, 0.0..=2.0).text("Volume 1"),
            ).changed();

            if(changed) {
                self.rx.send(self.state1).unwrap();
            }
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

    // Display the demo application that ships with egui.

    // let sample_setup = SampleSetup {
    //     sample_rate: 48000,
    //     nchannels: 2,
    // };

    // let samplegen = AudioSampleGenerator {
    //     sin0_freq: 1000.0,
    //     sin0_phase: 0.0,
    //     sin0_vol: 0.5,
    //     sin1_freq: 500.0,
    //     sin1_phase: 0.0,
    //     sin1_vol: 0.0,
    // };

    // let (rx, tx) = channel();

    // let mut audio_buffer = AudioSampleRingbuffer::new(256);

    // let mut producer = AudioSampleProducer::new(sample_setup, samplegen, tx, &mut audio_buffer);

    // producer.write_sample(0);
    // producer.write_sample(16);
    // producer.write_sample(256);

    // let mut reader = AudioSampleViewer::new(unsafe { &mut audio_buffer });

    // println!("{:?}", reader.next());
    // println!("{:?}", reader.next());
    // println!("{:?}", reader.next());

    //let mut demo_app = egui_demo_lib::DemoWindows::default();
    let mut app: MyApp = MyApp::default();

    let mut reader = AudioSampleViewer::new(unsafe { AUDIO_RINGBUFFER.as_mut().unwrap() });

    //

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

    let audio_samples_buffer_size = (audio_texture_width * audio_texture_height) as i64;
    let mut audio_data = Vec::from_iter(
        [0.0 as f32]
            .iter()
            .cycle()
            .take(audio_samples_buffer_size as usize)
            .cloned(),
    );
    let mut next_sample_index: i64 = 0;

    // let sample_rate = 480000 as f32;
    // let sample_clock = 0f32;
    // let nchannels = 1 as usize;
    // let mut request = SampleRequestOptions {
    //     sample_rate,
    //     sample_clock,
    //     nchannels,
    //     index: 0
    // };

    // todo: this should match audio start time, not start of the loop
    let start_time = Instant::now();
    event_loop.run(move |event, _, control_flow| {
        // Pass the winit events to the platform integration.
        platform.handle_event(&event);

        match event {
            RedrawRequested(..) => {
                platform.update_time(start_time.elapsed().as_secs_f64());

                let samples_per_sec = audio_texture_width;
                let current_sample_index =
                    start_time.elapsed().as_micros() as i64 * samples_per_sec as i64 / 1000000;

                while let Some(sample) = reader.next() {
                    audio_data[(next_sample_index % audio_samples_buffer_size) as usize] = f32::from_bits(sample as u32);
                    next_sample_index += 1;
                }
                // for i in next_sample_index..current_sample_index {
                //     request.tick();
                //     audio_data[(i % audio_samples_buffer_size) as usize] = request.tone(500.0);
                // }
                // next_sample_index = current_sample_index;

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
