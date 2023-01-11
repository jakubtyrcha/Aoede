use rhai::Engine;

use crate::audio_synthesis::{AudioGraphBuilder, NodeBuilder};

pub fn build_audio_script_engine() -> Engine {
    let mut engine = Engine::new();
    engine
        .register_type_with_name::<AudioGraphBuilder>("AudioGraphBuilder")
        .register_fn("new_graph", AudioGraphBuilder::new)
        .register_fn("sin", AudioGraphBuilder::sin)
        .register_fn("square", AudioGraphBuilder::square)
        .register_fn("saw", AudioGraphBuilder::sawtooth)
        .register_fn("tri", AudioGraphBuilder::triangle)
        .register_fn("rand", AudioGraphBuilder::random)
        .register_fn("gain", AudioGraphBuilder::gain)
        .register_fn("delay", AudioGraphBuilder::delay)
        .register_fn("adsr", AudioGraphBuilder::adsr)
        .register_fn("mix", AudioGraphBuilder::mix)
        .register_fn("set_out", AudioGraphBuilder::set_out)
        .register_type_with_name::<NodeBuilder>("NodeBuilder")
        .register_fn("input", NodeBuilder::set_input)
        .register_fn("freq", NodeBuilder::set_freq)
        .register_fn("volume", NodeBuilder::set_volume)
        .register_fn("delay", NodeBuilder::set_delay)
        .register_fn("attack", NodeBuilder::set_attack)
        .register_fn("decay", NodeBuilder::set_decay)
        .register_fn("sustain", NodeBuilder::set_sustain)
        .register_fn("release", NodeBuilder::set_release)
        .register_custom_operator("->", 160)
        .unwrap()
        .register_fn("->", |l: NodeBuilder, mut r: NodeBuilder| {
            r.set_input(l);
            r
        })
        .register_fn("+", |mut l: NodeBuilder, r: NodeBuilder| {
            let mut g = l.get_graph();
            g.mix().set_input(l).set_input(r)
        });
    engine
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn can_compile_basic_script() {
        let graph_builder = build_audio_script_engine().eval::<AudioGraphBuilder>(
            "
            let g = new_graph();
            let o = g.square();
            g.set_out(o);
            g
            ",
        );
        assert!(graph_builder.is_ok());
    }

    #[test]
    fn can_detect_syntax_error() {
        let graph_builder = build_audio_script_engine().eval::<AudioGraphBuilder>(
            "let g = new_graph();
        let o = g.square();
        g.set_out(o)
        g",
        );
        assert!(graph_builder.is_err());
    }

    #[test]
    fn can_connect_nodes() {
        let graph_builder = build_audio_script_engine().eval::<AudioGraphBuilder>(
            "
            let g = new_graph();
            let o = g.square().freq(0.5);
            let mix = g.mix();
            mix.input(o);
            g.set_out(mix);
            g
            ",
        );
        assert!(graph_builder.is_ok());
        let mut graph = graph_builder.unwrap().extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), -1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn can_connect_nodes_using_operator() {
        let graph_builder = build_audio_script_engine().eval::<AudioGraphBuilder>(
            "
            let g = new_graph();
            g.set_out(g.square().freq(0.5) -> g.mix());
            g
            ",
        );
        assert!(graph_builder.is_ok());
        let mut graph = graph_builder.unwrap().extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), -1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn can_build_graph_with_math_operators() {
        let graph_builder = build_audio_script_engine().eval::<AudioGraphBuilder>(
            "
            let g = new_graph();
            let p = g.square().freq(0.5);
            let p1 = g.square().freq(1.0);
            let mix = p + p1;
            g.set_out(mix);
            g
            ",
        );
        assert!(graph_builder.is_ok());
        let mut graph = graph_builder.unwrap().extract_graph();
        graph.set_sample_rate(2);
        assert_eq!(graph.gen_next_sample(), 2.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), -2.0);
    }

    #[test]
    fn delay_can_have_disconnected_input() {
        let graph_builder = build_audio_script_engine().eval::<AudioGraphBuilder>(
            "
        let g = new_graph();
        let d = g.delay();
        g.set_out(d);
        g
        ",
        );
        assert!(graph_builder.is_ok());
        let mut graph = graph_builder.unwrap().extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
    }
}
