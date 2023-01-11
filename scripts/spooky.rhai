let g = new_graph();
let o = g.rand().freq(80.0);
let envp = g.adsr()
    .attack(0.5)
    .decay(0.1)
    .sustain(0.2)
    .release(0.4);
let r = g.tri().freq(40.0);
let gain = g.gain().volume(0.5);
let mix = g.mix();
let delay = g.delay().delay(0.2);
let gain1 = g.gain().volume(0.5);
o -> envp;
envp -> mix;
r -> gain;
gain -> mix;
mix -> delay;
delay -> gain1;
gain1 -> mix;
g.set_out(mix);
g