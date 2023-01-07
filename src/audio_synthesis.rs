use std::collections::HashSet;

pub struct Context<'a> {
    time: f32,
    input_nodes: &'a Vec<i32>,
    outputs: &'a Vec<f32>,
}

pub trait NodeBehaviour {
    fn gen_next_sample(&self, context: Context) -> f32;
    fn process_outputs(&mut self, _: Context) {}
    fn is_delay(&self) -> bool { false }
}

pub struct SineOscillator {
    pub freq: f32,
}

impl NodeBehaviour for SineOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        (context.time * self.freq * 2.0 * std::f32::consts::PI).sin()
    }
}

pub struct TestOscillator {
    pub phase: f32,
}

impl NodeBehaviour for TestOscillator {
    fn gen_next_sample(&self, _: Context) -> f32 {
        self.phase
    }

    fn process_outputs(&mut self, _: Context) {
        if self.phase == 1.0 {
            self.phase = 0.0
        }
        else {
            self.phase = 1.0
        }
    }
}

pub struct Gain {
    pub volume: f32,
}

impl NodeBehaviour for Gain {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let input_node = context.input_nodes[0];
        let input_sample = context.outputs[input_node as usize];
        input_sample * self.volume
    }
}

pub struct Add {}

impl NodeBehaviour for Add {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let mut acc = 0.0;
        for input_node in context.input_nodes {
            acc += context.outputs[*input_node as usize];
        }
        acc
    }
}

pub struct Delay {
    pub buffered_samples: Vec<f32>,
    pub delay_samples: u64,
}

impl NodeBehaviour for Delay {
    fn gen_next_sample(&self, _: Context) -> f32 {
        if self.buffered_samples.len() >= self.delay_samples as usize {
            return self.buffered_samples
                [self.buffered_samples.len() - self.delay_samples as usize];
        }
        0.0
    }

    fn process_outputs(&mut self, context: Context) {
        let input_node = context.input_nodes[0];
        let input_sample = context.outputs[input_node as usize];
        self.buffered_samples.push(input_sample);
    }

    fn is_delay(&self) -> bool {
        true
    }
}

struct Node {
    id: i32,
    behaviour: Box<dyn NodeBehaviour>,
}

pub struct NodeGraph {
    next_id: i32,
    nodes: Vec<Node>,
    node_input_nodes: Vec<Vec<i32>>,
    sample_rate: i32,
    current_sample: i32,
}

impl NodeGraph {
    pub fn new() -> NodeGraph {
        NodeGraph {
            next_id: 0,
            nodes: Vec::new(),
            node_input_nodes: Vec::new(),
            sample_rate: 0,
            current_sample: -1,
        }
    }

    pub fn add_node(&mut self, behaviour: Box<dyn NodeBehaviour>) -> i32 {
        self.nodes.push(Node {
            id: self.next_id,
            behaviour,
        });
        self.node_input_nodes.push(Vec::new());
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn link(&mut self, node_from: i32, node_to: i32) {
        self.node_input_nodes[node_to as usize].push(node_from);
    }

    pub fn set_sample_rate(&mut self, num: i32) {
        self.sample_rate = num;
    }

    pub fn gen_next_sample(&mut self, node: i32) -> f32 {
        let mut topo_sort = Vec::<i32>::new();
        let mut next_topo_index = 0;
        // we build per node list of output indices
        let mut node_outputs = Vec::<Vec<i32>>::new();
        // for the topo sort, we maintain a number of inputs
        let mut node_input_count = Vec::<i32>::new();
        node_outputs.resize_with(self.nodes.len(), || Vec::new());
        node_input_count.resize(self.nodes.len(), 0);
        let mut fantom_delay_nodes = HashSet::<i32>::new();
        for node in &self.nodes {
            for input_node_index in &self.node_input_nodes[node.id as usize] {
                //
                if !self.nodes[*input_node_index as usize].behaviour.is_delay() {
                    node_outputs[*input_node_index as usize].push(node.id);
                    node_input_count[node.id as usize] += 1;
                }
                else {
                    // we don't treat the delay node as input node, but we 
                    // make a "fantom" input node with 0 inputs, that we process before the topo 
                    // ordered processing
                    fantom_delay_nodes.insert(*input_node_index);
                }
            }

            if node_input_count[node.id as usize] == 0 {
                topo_sort.push(node.id);
            }
        }

        while next_topo_index < topo_sort.len() {
            // pop from set
            let v = topo_sort[next_topo_index];
            next_topo_index += 1;

            // iter outputs
            for output in &node_outputs[v as usize] {
                // reduce input count
                node_input_count[*output as usize] -= 1;
                if node_input_count[*output as usize] == 0 {
                    topo_sort.push(*output);
                }
            }
        }

        //
        let mut outputs = Vec::<f32>::new();
        outputs.resize(self.nodes.len(), 0.0);

        self.current_sample += 1;
        let time = self.current_sample as f32 / self.sample_rate as f32;

        for v in &fantom_delay_nodes {
            let context = Context {
                time: time,
                outputs: &outputs,
                input_nodes: &self.node_input_nodes[*v as usize],
            };

            let sample = self.nodes[*v as usize].behaviour.gen_next_sample(context);
            outputs[*v as usize] = sample;
        }

        for v in &topo_sort {
            let context = Context {
                time: time,
                outputs: &outputs,
                input_nodes: &self.node_input_nodes[*v as usize],
            };

            if !fantom_delay_nodes.contains(v) {
                let sample = self.nodes[*v as usize].behaviour.gen_next_sample(context);
                outputs[*v as usize] = sample;
            }
        }

        for v in topo_sort {
            let context = Context {
                time: time,
                outputs: &outputs,
                input_nodes: &self.node_input_nodes[v as usize],
            };
            self.nodes[v as usize].behaviour.process_outputs(context);
        }

        outputs[node as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn can_build_a_path() {
        let mut graph = NodeGraph::new();
        let sine = graph.add_node(Box::new(SineOscillator { freq: 1.0 }));
        let gain = graph.add_node(Box::new(Gain { volume: 0.5 }));
        graph.link(sine, gain);
        graph.set_sample_rate(4);
        assert_eq!(graph.gen_next_sample(gain), 0.0);
        assert_eq!(graph.gen_next_sample(gain) <= 0.5, true);
        assert_eq!(graph.gen_next_sample(gain) <= 0.5, true);
        assert_eq!(graph.gen_next_sample(gain) <= 0.5, true);
    }

    #[test]
    fn can_mix_signals() {
        let mut graph = NodeGraph::new();
        let sine = graph.add_node(Box::new(SineOscillator { freq: 1.0 }));
        let sine2 = graph.add_node(Box::new(SineOscillator { freq: 2.0 }));
        let mix = graph.add_node(Box::new(Add {}));
        graph.link(sine, mix);
        graph.link(sine2, mix);
        graph.set_sample_rate(8);

        assert_eq!(graph.gen_next_sample(mix), 0.0);
        assert_eq!((graph.gen_next_sample(mix) - 1.7071).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample(mix) - 1.0).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample(mix) - -0.2929).abs() < 0.01, true);
    }

    #[test]
    fn can_evaluate_nodes_in_topological_order() {
        let mut graph = NodeGraph::new();
        let sine = graph.add_node(Box::new(SineOscillator { freq: 1.0 }));
        let mix = graph.add_node(Box::new(Add {}));
        let gain = graph.add_node(Box::new(Gain { volume: 0.5 }));
        graph.link(sine, mix);
        graph.link(sine, gain);
        graph.link(gain, mix);
        graph.set_sample_rate(4);

        assert_eq!(graph.gen_next_sample(mix), 0.0);
        assert_eq!((graph.gen_next_sample(mix) - 1.5).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample(mix)).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample(mix) - -1.5).abs() < 0.01, true);
    }

    #[test]
    fn can_run_a_delay_node() {
        let mut graph = NodeGraph::new();
        let sine = graph.add_node(Box::new(SineOscillator { freq: 1.0 }));
        let delay = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        graph.link(sine, delay);
        graph.set_sample_rate(4);
        graph.gen_next_sample(delay);
        assert_eq!(graph.gen_next_sample(delay), 0.0);
        assert_eq!(graph.gen_next_sample(delay), 1.0);
    }

    #[test]
    fn can_run_a_queue_of_delay_nodes() {
        let mut graph = NodeGraph::new();
        let test = graph.add_node(Box::new(TestOscillator { phase: 1.0 }));
        let delay = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        let delay1 = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        let delay2 = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        graph.link(test, delay);
        graph.link(delay, delay1);
        graph.link(delay1, delay2);
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 1.0);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 1.0);
    }

    #[test]
    fn can_run_a_queue_of_delay_nodes_constructed_out_of_order() {
        let mut graph = NodeGraph::new();
        let test = graph.add_node(Box::new(TestOscillator { phase: 1.0 }));
        let delay2 = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        let delay1 = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        let delay = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        graph.link(test, delay);
        graph.link(delay, delay1);
        graph.link(delay1, delay2);
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 1.0);
        assert_eq!(graph.gen_next_sample(delay2), 0.0);
        assert_eq!(graph.gen_next_sample(delay2), 1.0);
    }

    #[test]
    fn can_run_a_delay_to_mix_graph() {
        let mut graph = NodeGraph::new();
        let test = graph.add_node(Box::new(TestOscillator { phase: 1.0 }));
        let delay = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        let mix = graph.add_node(Box::new(Add {}));
        graph.link(test, delay);
        graph.link(delay, mix);
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(mix), 0.0);
        assert_eq!(graph.gen_next_sample(mix), 1.0);
        assert_eq!(graph.gen_next_sample(mix), 0.0);
    }

    #[test]
    fn can_run_a_delay_gain_graph() {
        let mut graph = NodeGraph::new();
        let test = graph.add_node(Box::new(TestOscillator { phase: 1.0 }));
        let delay = graph.add_node(Box::new(Delay {
            delay_samples: 4,
            buffered_samples: Vec::new(),
        }));
        let gain = graph.add_node(Box::new(Gain { volume: 0.5 }));
        let mix = graph.add_node(Box::new(Add {}));
        graph.link(test, delay);
        graph.link(delay, gain);
        graph.link(gain, mix);
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(mix), 0.0);
        assert_eq!(graph.gen_next_sample(mix), 0.0);
        assert_eq!(graph.gen_next_sample(mix), 0.0);
        assert_eq!(graph.gen_next_sample(mix), 0.0);
        assert_eq!(graph.gen_next_sample(mix), 0.5);
        assert_eq!(graph.gen_next_sample(mix), 0.0);
    }

    #[test]
    fn can_run_a_delay_gain_loop() {
        let mut graph = NodeGraph::new();
        let test = graph.add_node(Box::new(TestOscillator { phase: 1.0 }));
        let delay = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        let gain = graph.add_node(Box::new(Gain { volume: 0.5 }));
        let mix = graph.add_node(Box::new(Add {}));
        graph.link(test, mix);
        graph.link(mix, delay);
        graph.link(delay, gain);
        graph.link(gain, mix);
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(mix), 1.0);
        assert_eq!(graph.gen_next_sample(mix), 0.5);
        assert_eq!(graph.gen_next_sample(mix), 1.25);
        assert_eq!(graph.gen_next_sample(mix), 0.625);
        assert_eq!(graph.gen_next_sample(mix), 1.3125);
    }
}
