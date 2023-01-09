let g = new_graph();
let o = g.spawn_pulse().freq(100.0);
let envp = g.spawn_adsr()
    .attack(0.5)
    .decay(0.1)
    .sustain(0.2)
    .release(0.4);
let mix = g.spawn_mix();
o -> envp;
envp -> mix;
g.set_out(mix);
g