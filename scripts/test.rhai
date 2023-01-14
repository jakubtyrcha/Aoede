let g = new_graph();
g.set_out(
    g.sin().freq(400.0) +
    g.sin().freq(1000.0) +
    g.sin().freq(2000.0)
);
g