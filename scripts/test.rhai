let g = new_graph();
let o = g.saw().freq(40.0);
let envp = g.adsr()
    .attack(0.1)
    .decay(0.1)
    .sustain(0.2)
    .release(0.4);
let r = g.rand();
let gain = g.gain().volume(0.2);
g.set_out((o -> envp) + (r -> gain));
g