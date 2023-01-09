let g = new_graph();
let o = g.spawn_saw().freq(40.0);
let envp = g.spawn_adsr()
    .attack(0.5)
    .decay(0.1)
    .sustain(0.2)
    .release(0.4);
let r = g.spawn_rand();
let gain = g.spawn_gain().volume(0.1);
let mix = g.spawn_mix();
o -> envp;
envp -> mix;
r -> gain;
gain -> mix;
g.set_out(mix);
g