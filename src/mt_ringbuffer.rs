// single producer, multi consumer
// if the reader doesn't consume the data by the time it's overwritten, it will be gone
// "lightweight" way of accessing data from multiple threads, mostly for non-blocking audio thread access


use std::sync::atomic::{AtomicI32, AtomicU32};
pub struct MtRingbuffer {
    buffer: Vec<AtomicU32>,
    next_sample_write: AtomicI32,
}

impl MtRingbuffer {
    pub fn new(size: usize) -> MtRingbuffer {
        let mut buffer = Vec::new();
        buffer.resize_with(size, || AtomicU32::new(0));
        MtRingbuffer {
            buffer: buffer,
            next_sample_write: AtomicI32::new(0),
        }
    }

    pub fn write(&mut self, value: u32) {
        let N = self.buffer.len();
        let write_sample = self
            .next_sample_write
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let write_index = write_sample as usize % N;
        self.buffer[write_index].store(value, std::sync::atomic::Ordering::SeqCst);
    }
}

pub struct MtRingbufferReader<'a> {
    buffer: &'a MtRingbuffer,
    next_sample_read: i32,
}

impl<'a> MtRingbufferReader<'a> {
    pub fn new(buffer: &'a MtRingbuffer) -> MtRingbufferReader<'a> {
        MtRingbufferReader {
            buffer,
            next_sample_read: 0,
        }
    }

    pub fn skip_n_samples(&mut self, jump: i32) {
        self.next_sample_read += jump;
    }

    pub fn next(&mut self) -> Option<u32> {
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
