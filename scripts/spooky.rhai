let g = new_graph();
let o = g.spawn_rand().freq(80.0);
let envp = g.spawn_adsr()
    .attack(0.5)
    .decay(0.1)
    .sustain(0.2)
    .release(0.4);
let r = g.spawn_tri().freq(40.0);
let gain = g.spawn_gain().volume(0.5);
let mix = g.spawn_mix();
let delay = g.spawn_delay().delay(0.2);
let gain1 = g.spawn_gain().volume(0.5);
o -> envp;
envp -> mix;
r -> gain;
gain -> mix;
mix -> delay;
delay -> gain1;
gain1 -> mix;
g.set_out(mix);
g