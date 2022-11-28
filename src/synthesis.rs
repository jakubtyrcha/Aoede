use std::collections::HashSet;

struct Context<'a> {
    time: f32,
    inputs: &'a Vec<i32>,
    outputs: &'a Vec<f32>,
}

pub trait NodeBehaviour {
    fn gen_next_sample(&self, context: Context) -> f32;
    fn process_outputs(&mut self, context: Context);
}

pub struct SineOscillator {
    pub freq: f32,
}

impl NodeBehaviour for SineOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        (context.time * self.freq * 2.0 * std::f32::consts::PI).sin()
    }

    fn process_outputs(&mut self, context: Context) {}
}

pub struct Gain {
    pub volume: f32,
}

impl NodeBehaviour for Gain {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let input_node = context.inputs[0];
        let input_sample = context.outputs[input_node as usize];
        input_sample * self.volume
    }

    fn process_outputs(&mut self, context: Context) {}
}

pub struct Add {}

impl NodeBehaviour for Add {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let mut acc = 0.0;
        for input_node in context.inputs {
            acc += context.outputs[*input_node as usize];
        }
        acc
    }

    fn process_outputs(&mut self, context: Context) {}
}

pub struct Delay {
    pub buffered_samples: Vec<f32>,
    pub delay_samples: u64,
}

impl NodeBehaviour for Delay {
    fn gen_next_sample(&self, context: Context) -> f32 {
        if self.buffered_samples.len() >= self.delay_samples as usize {
            return self.buffered_samples
                [self.buffered_samples.len() - self.delay_samples as usize];
        }
        0.0
    }

    fn process_outputs(&mut self, context: Context) {
        let input_node = context.inputs[0];
        let input_sample = context.outputs[input_node as usize];
        self.buffered_samples.push(input_sample);
    }
}

struct Node {
    id: i32,
    behaviour: Box<dyn NodeBehaviour>,
}

pub struct NodeGraph {
    next_id: i32,
    nodes: Vec<Node>,
    node_inputs: Vec<Vec<i32>>,
    sample_rate: i32,
    current_sample: i32,
}

impl NodeGraph {
    pub fn new() -> NodeGraph {
        NodeGraph {
            next_id: 0,
            nodes: Vec::new(),
            node_inputs: Vec::new(),
            sample_rate: 0,
            current_sample: -1,
        }
    }

    pub fn add_node(&mut self, behaviour: Box<dyn NodeBehaviour>) -> i32 {
        self.nodes.push(Node {
            id: self.next_id,
            behaviour,
        });
        self.node_inputs.push(Vec::new());
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    pub fn link(&mut self, node_from: i32, node_to: i32) {
        self.node_inputs[node_to as usize].push(node_from);
    }

    pub fn set_sample_rate(&mut self, num: i32) {
        self.sample_rate = num;
    }

    pub fn gen_next_sample(&mut self, node: i32) -> f32 {
        let mut topo_sort = Vec::<i32>::new();
        let mut next_index = 0;
        // find initial nodes

        let mut outputs = Vec::<Vec<i32>>::new();
        let mut node_inputs_count = Vec::<i32>::new();
        outputs.resize_with(self.nodes.len(), || Vec::new());
        node_inputs_count.resize(self.nodes.len(), 0);
        for node in &self.nodes {
            let inputs_num = self.node_inputs[node.id as usize].len() as i32;
            node_inputs_count[node.id as usize] = inputs_num;

            if inputs_num == 0 {
                topo_sort.push(node.id);
            }

            for input in &self.node_inputs[node.id as usize] {
                outputs[*input as usize].push(node.id);
            }
        }

        while next_index < topo_sort.len() {
            println!("topo {} {:?}", next_index, topo_sort[next_index]);
            // pop from set
            let v = topo_sort[next_index];
            next_index += 1;

            // iter outputs
            for output in &outputs[v as usize] {
                // reduce input count
                node_inputs_count[*output as usize] -= 1;
                if node_inputs_count[*output as usize] == 0 {
                    topo_sort.push(*output);
                }
            }
        }

        //
        let mut outputs = Vec::<f32>::new();
        outputs.resize(self.nodes.len(), 0.0);

        self.current_sample += 1;
        let time = self.current_sample as f32 / self.sample_rate as f32;

        for v in &topo_sort {
            let context = Context {
                time: time,
                outputs: &outputs,
                inputs: &self.node_inputs[*v as usize],
            };

            let sample = self.nodes[*v as usize].behaviour.gen_next_sample(context);
            outputs[*v as usize] = sample;
        }

        for v in topo_sort {
            for input in &self.node_inputs[v as usize] {
                let context = Context {
                    time: time,
                    outputs: &outputs,
                    inputs: &self.node_inputs[v as usize],
                };
                self.nodes[v as usize].behaviour.process_outputs(context);
            }
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
    fn can_run_a_delay_cycle() {
        let mut graph = NodeGraph::new();
        let sine = graph.add_node(Box::new(SineOscillator { freq: 1.0 }));
        let delay = graph.add_node(Box::new(Delay {
            delay_samples: 1,
            buffered_samples: Vec::new(),
        }));
        let mix = graph.add_node(Box::new(Add{}));
        graph.link(sine, mix);
        graph.link(mix, delay);
        graph.link(delay, mix);
        graph.set_sample_rate(4);
        graph.gen_next_sample(mix);
        assert_eq!(graph.gen_next_sample(delay), 0.0);
        assert_eq!(graph.gen_next_sample(delay), 1.0);

    }
}
