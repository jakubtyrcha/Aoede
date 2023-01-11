let g = new_graph();
let main_freq = 140.0;
let overtone_freq_0 = main_freq * 2.0;
let overtone_freq_1 = main_freq * 4.0;

let mix = (
    (g.sin().freq(main_freq) -> g.gain().volume(2.0)) +
    (g.sin().freq(overtone_freq_0) -> g.gain().volume(0.75)) +
    (g.sin().freq(overtone_freq_1) -> g.gain().volume(0.5))
    ) -> g.adsr().attack(0.05).decay(0.2).sustain(0.01).release(1.0);

g.set_out(mix);
g