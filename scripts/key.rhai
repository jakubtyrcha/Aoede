let g = new_graph();
let main_freq = 170.0;
let harmonic_freq_0 = main_freq * 0.5;
let harmonic_freq_1 = main_freq * 0.25;

let mix = (
    (g.sin().freq(main_freq) -> g.gain().volume(2.0)) +
    (g.sin().freq(harmonic_freq_0) -> g.gain().volume(0.5)) +
    (g.sin().freq(harmonic_freq_1) -> g.gain().volume(0.5))
    ) -> g.adsr().attack(0.05).decay(0.2).sustain(0.01).release(1.0);

g.set_out(mix);
g