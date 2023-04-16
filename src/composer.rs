/*
This is the precursor of composition scripting.
First iteration allows to play synthetic piano at variable bpm.
Piano keys are generated from sound scripts and the sound is cached.
*/

use std::collections::HashMap;
use std::rc::Rc;
use std::cell::RefCell;

use crate::audio_script::build_audio_script_engine;
use crate::audio_synthesis::AudioGraphBuilder;

// the composing API exposed via scripting
#[derive(Debug, Clone)]
pub struct Composer {
    composition: Rc<RefCell<Composition>>
}

// current composition state
#[derive(Debug, Clone)]
pub struct Composition {
    // beats per minute
    // TODO: can it vary?
    bpm: f32,
    // keeps track of the current head position in beats
    head_beats: f32,
    sounds: Vec<SoundQueued>
}

impl Composer {
    pub fn new() -> Composer {
        Composer { composition: Rc::new(RefCell::new(Composition::new())) }
    }
    
    pub fn play_sound(&mut self, id: i32) -> SoundBuilder {
        let index = self.composition.borrow().sounds.len() as i32;
        let offset = self.composition.borrow().head_beats;
        self.composition.borrow_mut().sounds.push(SoundQueued{ sound_id: id, duration_beats: 1.0, offset_beats: offset });
        SoundBuilder { index, composer: self.clone() }
    }

    pub fn set_sound_beats(&mut self, index: i32, beats: f32) {
        self.composition.borrow_mut().sounds[index as usize].duration_beats = beats;
    }

    pub fn move_head(&mut self, beats: f32) {
        self.composition.borrow_mut().head_beats += beats;
    }

    pub fn clone_composition(&self) -> Composition {
        self.composition.borrow().clone()
    }
}

#[derive(Debug, Clone)]
struct SoundQueued {
    pub sound_id: i32,
    pub duration_beats: f32,
    pub offset_beats: f32,
}

pub struct SoundBuilder {
    index: i32,
    composer: Composer
}

impl SoundBuilder {
    pub fn beats(&mut self, beats: f32) {
        self.composer.set_sound_beats(self.index, beats);
    }

    pub fn move_head(&mut self) {
        let dur = self.composer.composition.borrow().sounds[self.index as usize].duration_beats;
        self.composer.move_head(dur);
    } 
}

impl Composition {
    fn new() -> Composition {
        Composition { bpm: 60.0, head_beats: 0.0, sounds: Vec::new() }
    }

    pub fn get_beat_time(&self) -> f32 {
        1.0 / self.bpm
    }
}

fn piano_key_freq(key: i32) -> f32 {
    2.0_f32.powf((key as f32 - 49.0) / 12.0) * 440.0
}

fn cache_piano_key(key: i32, press_duration: f32) -> Vec<f32> {
    // key -> freq -> script -> soundwave
    let freq = piano_key_freq(key);
    let script = 
    format!("
    let g = new_graph();
    let main_freq = {:.4};
    let overtone_freq_0 = main_freq * 2.0;
    let overtone_freq_1 = main_freq * 4.0;
    
    let mix = (
        (g.sin().freq(main_freq) -> g.gain().volume(2.0)) +
        (g.sin().freq(overtone_freq_0) -> g.gain().volume(0.75)) +
        (g.sin().freq(overtone_freq_1) -> g.gain().volume(0.5))
        ) -> g.adsr().attack(0.05).decay(0.2).sustain(0.01).release(0.75);
    
    g.set_out(mix);
    g
    ", freq);

    let graph_builder = build_audio_script_engine().eval::<AudioGraphBuilder>(&script);
    let duration = 48000;
    let mut graph = graph_builder.unwrap().extract_graph();
    graph.set_sample_rate(48000);

    let mut cached = Vec::new();
    cached.reserve_exact(48000);
    for _ in 0..48000 {
        cached.push(graph.gen_next_sample());
    }
    cached
}

pub struct CompositionPlayer {
    composition: Composition,
    sound_cache: HashMap<i32, Vec<f32>>,
    sample_index: usize,
}

impl CompositionPlayer {
    pub fn new(composition: Composition) -> CompositionPlayer {
        CompositionPlayer { composition, sound_cache: HashMap::new(), sample_index: 0 }
    }

    pub fn gen_next_sample(&mut self) -> f32 {
        let sample_index = self.sample_index;
        let time = sample_index as f32 / 48000.0;

        let mut sample_acc: f32 = 0.0;

        for i in 0..self.composition.sounds.len() {
            let sound = &self.composition.sounds[i];
            let offset = sound.offset_beats * self.composition.get_beat_time();
            let duration = sound.duration_beats * self.composition.get_beat_time();
            if time >= offset && time < offset + duration {
                let sound_id = sound.sound_id;
                if !self.sound_cache.contains_key(&sound_id) {
                    self.sound_cache.insert(sound_id, cache_piano_key(sound_id, duration));
                }
                let sound_samples = self.sound_cache.get(&sound_id).unwrap();
                let sound_sample_index = ((time - offset) * 48000.0f32) as usize; 
                sample_acc += sound_samples[sound_sample_index];
            }
        }
        
        self.sample_index += 1;
        sample_acc
    }

}