pub struct AudioGenerator {
    sample_rate: u32
}

impl AudioGenerator {
    pub fn new(stream_config: &cpal::StreamConfig) -> AudioGenerator {
        AudioGenerator{ sample_rate: stream_config.sample_rate.0 }
    }

    pub fn get_next_sample(&self) -> f32 {
        0.0
    }
}