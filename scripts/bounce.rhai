let g = new_graph();
let m = g.mix();
(g.tri().freq(40.0) + (g.saw().freq(200.0) -> g.gain().volume(0.05)) ) 
    -> g.adsr()
        .attack(0.05)
        .decay(0.1) 
        .sustain(0.7)
        .release(0.3)
    -> m;
g.set_out(m);
g