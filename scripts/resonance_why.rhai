let g = new_graph();
let main_freq = 50.0;
let overtones = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5].map(|offset| main_freq * offset);
let v = 1.0;

let mix = g.sin().freq(main_freq);
for tone in overtones {
    v = v * 0.95;
    mix = mix + (g.sin().freq(tone) -> g.gain().volume(v) );
}
mix = mix -> g.adsr()
    .attack(0.1)
    .decay(0.3) 
    .sustain(0.01)
    .release(0.9) -> g.gain().volume(4.0);

g.set_out(mix);
g