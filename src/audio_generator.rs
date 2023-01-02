use crate::{mt_ringbuffer::MtRingbufferReader};

pub struct AudioGenerator {
    reader: MtRingbufferReader<'static>,
}

impl AudioGenerator {
    pub fn new(reader: MtRingbufferReader<'static>,) -> AudioGenerator {
        AudioGenerator{ reader }
    }

    pub fn get_next_sample(&mut self) -> f32 {
        self.reader.next().unwrap_or(0) as f32
    }
}