
use cpal::traits::{DeviceTrait, HostTrait};

pub fn stream_setup_for<F, G>(on_sample: F, context: G) -> Result<cpal::Stream, anyhow::Error>
where
    F: FnMut(&mut G) -> f32 + std::marker::Send + 'static + Copy,
    G: std::marker::Send + 'static
{
    let (_host, device, config) = host_device_setup()?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => stream_make::<f32, _, G>(&device, &config.into(), on_sample, context),
        cpal::SampleFormat::I16 => stream_make::<i16, _, G>(&device, &config.into(), on_sample, context),
        cpal::SampleFormat::U16 => stream_make::<u16, _, G>(&device, &config.into(), on_sample, context),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {:?}", config);

    Ok((host, device, config))
}

pub fn stream_make<T, F, G>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    on_sample: F,
    mut context: G,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: cpal::Sample,
    F: FnMut(&mut G) -> f32 + std::marker::Send + 'static + Copy,
    G: std::marker::Send + 'static
{
    let err_fn = |err| eprintln!("Error building output sound stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            on_window(output, &mut context, on_sample)
        },
        err_fn,
    )?;

    Ok(stream)
}

fn on_window<T, F, G>(output: &mut [T], context: &mut G, mut on_sample: F)
where
    T: cpal::Sample,
    F: FnMut(&mut G) -> f32 + std::marker::Send + 'static,
{
    let nchannels = 2;
    for frame in output.chunks_mut(2) {
        let value: T = cpal::Sample::from::<f32>(&on_sample(context));
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}