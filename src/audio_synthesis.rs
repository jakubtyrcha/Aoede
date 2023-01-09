use std::collections::{HashSet, HashMap};
use std::rc::Rc;
use std::cell::RefCell;


#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum InputSlotEnum
{
    Input,
    Freq,
    Attack,
    Delay,
    Sustain,
    Release,
    Volume
}
pub struct Context<'a> {
    time: f32,
    sample_rate: i32,
    input_nodes: &'a Vec<i32>,
    input_slots: &'a HashMap<InputSlotEnum, NodeParamInput>,
    outputs: &'a Vec<f32>,
}

impl Context<'_> {
    fn read_input(&self, slot: InputSlotEnum) -> Option<f32> {
        let input = self.input_slots.get(&slot).copied();
        let value = 
        match input {
            None => None,
            Some(NodeParamInput::Node(id)) => {
                // TODO: read from node output
                None
            },
            Some(NodeParamInput::Constant(value)) => Some(value),
        };
        value
    }
}

pub trait NodeBehaviour {
    fn gen_next_sample(&self, context: Context) -> f32;
    fn process_outputs(&mut self, _: Context) {}
    fn is_delay(&self) -> bool { false }
}

impl std::fmt::Debug for dyn NodeBehaviour {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NodeBehaviour")
    }
}

pub struct SineOscillator {
}

impl NodeBehaviour for SineOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let freq = context.read_input(InputSlotEnum::Freq).unwrap_or(1000.0);
        (context.time * freq * 2.0 * std::f32::consts::PI).sin()
    }
}

pub struct PulseOscillator {
}

impl NodeBehaviour for PulseOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let freq = context.read_input(InputSlotEnum::Freq).unwrap_or(1000.0);
        let cycle = 1.0 / freq;
        if context.time.rem_euclid(cycle) < cycle * 0.5 { 1.0 } else { 0.0 }
    }
}

pub struct Gain {
}

impl NodeBehaviour for Gain {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let input_node = context.input_nodes[0];
        let input_sample = context.outputs[input_node as usize];
        input_sample * context.read_input(InputSlotEnum::Volume).unwrap_or(1.0)
    }
}

// pub struct ADSR {
// }

// fn lerp(a: f32, b: f32, t: f32) -> f32 {
//     a + (b - a) * t
// }

// impl NodeBehaviour for ADSR {
//     fn gen_next_sample(&self, context: Context) -> f32 {
//         let input_node = context.input_nodes[0];
//         let input_sample = context.outputs[input_node as usize];
//         let total_duration = self.attack + self.decay + self.release;
//         let t = context.time.rem_euclid(total_duration);

//         let factor = if t < self.attack {
//             lerp(0.0, 1.0, t / self.attack)
//         }
//         else if t < self.attack + self.decay {
//             lerp(1.0, self.sustain, (t - self.attack) / self.decay)
//         }
//         else {
//             lerp(self.sustain, 0.0, (t - self.attack - self.decay) / self.release)
//         };
//         factor * input_sample
//     }
// }

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
}

impl Delay {
    fn new() -> Delay {
        Delay{ buffered_samples: Vec::new() }
    }
}

impl NodeBehaviour for Delay {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let delay_samples = (context.read_input(InputSlotEnum::Delay).unwrap_or(1.0) * (context.sample_rate as f32)) as usize;

        if self.buffered_samples.len() >= delay_samples as usize {
            return self.buffered_samples
                [self.buffered_samples.len() - delay_samples as usize];
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

#[derive(Debug, Clone)]
struct Node {
    id: i32,
    behaviour: Rc<RefCell<dyn NodeBehaviour>>,
}

#[derive(Debug, Clone, Copy)]
enum NodeParamInput {
    Constant(f32),
    Node(i32)
}

#[derive(Debug, Clone)]
pub struct AudioGraph {
    next_id: i32,
    nodes: Vec<Node>,
    node_input_nodes: Vec<Vec<i32>>,
    node_input_slots: Vec<HashMap<InputSlotEnum, NodeParamInput>>,
    sample_rate: i32,
    current_sample: i32,
    out_node: Option<i32>
}

#[derive(Debug, Clone)]
pub struct NodeBuilder {
    graph: Rc<RefCell<AudioGraph>>,
    id: i32
}

impl NodeBuilder {
    pub fn set_input(&mut self, node: NodeBuilder) -> NodeBuilder {
        self.graph.borrow_mut().link_node(self.id, InputSlotEnum::Input, node.id);
        self.clone()
    }

    pub fn set_freq(&mut self, value: f64) -> NodeBuilder {
        self.graph.borrow_mut().link_constant_f64(self.id, InputSlotEnum::Freq, value);
        self.clone()
    }

    pub fn set_volume(&mut self, value: f64) -> NodeBuilder {
        self.graph.borrow_mut().link_constant_f64(self.id, InputSlotEnum::Volume, value);
        self.clone()
    }

    pub fn set_delay(&mut self, value: f64) -> NodeBuilder {
        self.graph.borrow_mut().link_constant_f64(self.id, InputSlotEnum::Delay, value);
        self.clone()
    }
}

#[derive(Debug, Clone)]
pub struct AudioGraphBuilder 
{
    internal: Rc<RefCell<AudioGraph>>
}

impl AudioGraphBuilder {
    pub fn new() -> AudioGraphBuilder {
        AudioGraphBuilder{ internal: Rc::new(RefCell::new(AudioGraph::new())) }
    }
    
    pub fn spawn_sin(&mut self) -> NodeBuilder {
        let id = self.internal.borrow_mut().add_node(Rc::new(RefCell::new(SineOscillator{})));
        NodeBuilder{ graph: self.internal.clone(), id }
    }

    pub fn spawn_pulse(&mut self) -> NodeBuilder {
        let id = self.internal.borrow_mut().add_node(Rc::new(RefCell::new(PulseOscillator{})));
        NodeBuilder{ graph: self.internal.clone(), id }
    }

    pub fn spawn_gain(&mut self) -> NodeBuilder {
        let id = self.internal.borrow_mut().add_node(Rc::new(RefCell::new(Gain{})));
        NodeBuilder{ graph: self.internal.clone(), id }
    }

    pub fn spawn_delay(&mut self) -> NodeBuilder {
        let id = self.internal.borrow_mut().add_node(Rc::new(RefCell::new(Delay::new())));
        NodeBuilder{ graph: self.internal.clone(), id }
    }

    // pub fn spawn_adsr(&mut self) -> NodeBuilder {
    //     let id = self.internal.borrow_mut().add_node(Rc::new(RefCell::new(ADSR{})));
    //     NodeBuilder{ graph: self.internal.clone(), id }
    // }

    pub fn spawn_mix(&mut self) -> NodeBuilder {
        let id = self.internal.borrow_mut().add_node(Rc::new(RefCell::new(Add{})));
        NodeBuilder{ graph: self.internal.clone(), id }
    }

    pub fn set_out(&mut self, node: NodeBuilder) {
        self.internal.borrow_mut().set_out(node.id);
    }

    pub fn extract_graph(&mut self) -> AudioGraph {
        // TODO: move out of the Rc<RefCell>?
        self.internal.borrow().clone()
    }
}

impl AudioGraph {
    pub fn new() -> AudioGraph {
        AudioGraph {
            next_id: 0,
            nodes: Vec::new(),
            node_input_nodes: Vec::new(),
            node_input_slots: Vec::new(),
            sample_rate: 0,
            current_sample: -1,
            out_node: None
        }
    }

    fn add_node(&mut self, behaviour: Rc<RefCell<dyn NodeBehaviour>>) -> i32 {
        self.nodes.push(Node {
            id: self.next_id,
            behaviour,
        });
        self.node_input_nodes.push(Vec::new());
        self.node_input_slots.push(HashMap::new());
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    // pub fn spawn_adsr_node(&mut self) -> i32 {
    //     self.add_node(Rc::new(RefCell::new(ADSR{ attack: 0.1,
    //         decay: 0.1,
    //         sustain: 0.2,
    //         release: 0.3, })))
    // }

    // pub fn spawn_mix_node(&mut self) -> i32 {
    //     self.add_node(Rc::new(RefCell::new(Add{})))
    // }

    pub fn set_out(&mut self, sink: i32) {
        self.out_node = Some(sink);
    }

    pub fn link_node(&mut self, node_to: i32, slot: InputSlotEnum, node_from: i32) {
        match slot {
            InputSlotEnum::Input => self.node_input_nodes[node_to as usize].push(node_from),
            other => self.node_input_slots[node_to as usize].insert(other, NodeParamInput::Node(node_from)).map_or((), |_|()),
        }
    }

    pub fn link_constant_f64(&mut self, node_to: i32, slot: InputSlotEnum, value: f64) {
        self.node_input_slots[node_to as usize].insert(slot, NodeParamInput::Constant(value as f32));
    }

    pub fn set_sample_rate(&mut self, num: i32) {
        self.sample_rate = num;
    }

    pub fn gen_next_sample(&mut self) -> f32 {
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
                if !self.nodes[*input_node_index as usize].behaviour.borrow().is_delay() {
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
                sample_rate: self.sample_rate,
                outputs: &outputs,
                input_nodes: &self.node_input_nodes[*v as usize],
                input_slots: &self.node_input_slots[*v as usize],
            };

            let sample = self.nodes[*v as usize].behaviour.borrow().gen_next_sample(context);
            outputs[*v as usize] = sample;
        }

        for v in &topo_sort {
            let context = Context {
                time: time,
                sample_rate: self.sample_rate,
                outputs: &outputs,
                input_nodes: &self.node_input_nodes[*v as usize],
                input_slots: &self.node_input_slots[*v as usize],
            };

            if !fantom_delay_nodes.contains(v) {
                let sample = self.nodes[*v as usize].behaviour.borrow().gen_next_sample(context);
                outputs[*v as usize] = sample;
            }
        }

        for v in topo_sort {
            let context = Context {
                time: time,
                sample_rate: self.sample_rate,
                outputs: &outputs,
                input_nodes: &self.node_input_nodes[v as usize],
                input_slots: &self.node_input_slots[v as usize],
            };
            self.nodes[v as usize].behaviour.borrow_mut().process_outputs(context);
        }

        outputs[self.out_node.unwrap() as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn can_build_a_path() {
        let mut graph_builder = AudioGraphBuilder::new();
        let sine = graph_builder.spawn_sin();
        let mut gain = graph_builder.spawn_gain();
        gain.set_volume(0.5);
        gain.set_input(sine);
        graph_builder.set_out(gain);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(4);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample() <= 0.5, true);
        assert_eq!(graph.gen_next_sample() <= 0.5, true);
        assert_eq!(graph.gen_next_sample() <= 0.5, true);
    }

    #[test]
    fn can_mix_signals() {
        let mut graph_builder = AudioGraphBuilder::new();
        let sine = graph_builder.spawn_sin().set_freq(1.0);
        let sine2 = graph_builder.spawn_sin().set_freq(2.0);
        let mut mix = graph_builder.spawn_mix();
        mix.set_input(sine);
        mix.set_input(sine2);

        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(8);

        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!((graph.gen_next_sample() - 1.7071).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample() - 1.0).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample() - -0.2929).abs() < 0.01, true);
    }

    #[test]
    fn can_evaluate_nodes_in_topological_order() {
        let mut graph_builder = AudioGraphBuilder::new();
        let sine = graph_builder.spawn_sin().set_freq(1.0);
        let mut mix = graph_builder.spawn_mix();
        let mut gain = graph_builder.spawn_gain().set_volume(0.5);
        mix.set_input(sine.clone());
        gain.set_input(sine);
        mix.set_input(gain);
        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(4);

        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!((graph.gen_next_sample() - 1.5).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample()).abs() < 0.01, true);
        assert_eq!((graph.gen_next_sample() - -1.5).abs() < 0.01, true);
    }

    #[test]
    fn can_run_a_delay_node() {
        let mut graph_builder = AudioGraphBuilder::new();
        let sine = graph_builder.spawn_sin().set_freq(1.0);
        let mut delay = graph_builder.spawn_delay().set_delay(0.25);
        delay.set_input(sine);
        graph_builder.set_out(delay);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(4);
        graph.gen_next_sample();
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn can_run_a_queue_of_delay_nodes() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.spawn_pulse().set_freq(0.5);
        let mut delay = graph_builder.spawn_delay().set_delay(1.0);
        let mut delay1 = graph_builder.spawn_delay().set_delay(1.0);
        let mut delay2 = graph_builder.spawn_delay().set_delay(1.0);
        delay.set_input(pulse);
        delay1.set_input(delay);
        delay2.set_input(delay1);
        graph_builder.set_out(delay2);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn can_run_a_queue_of_delay_nodes_constructed_out_of_order() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.spawn_pulse().set_freq(0.5);
        let mut delay2 = graph_builder.spawn_delay().set_delay(1.0);
        let mut delay1 = graph_builder.spawn_delay().set_delay(1.0);
        let mut delay = graph_builder.spawn_delay().set_delay(1.0);
        delay.set_input(pulse);
        delay1.set_input(delay);
        delay2.set_input(delay1);
        graph_builder.set_out(delay2);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn can_run_a_delay_to_mix_graph() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.spawn_pulse().set_freq(0.5);
        let mut delay = graph_builder.spawn_delay().set_delay(1.0);
        let mut mix = graph_builder.spawn_mix();
        delay.set_input(pulse);
        mix.set_input(delay);
        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
    }

    #[test]
    fn can_run_a_delay_gain_graph() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.spawn_pulse().set_freq(0.5);
        let mut delay = graph_builder.spawn_delay().set_delay(4.0);
        let mut gain = graph_builder.spawn_gain().set_volume(0.5);
        let mut mix = graph_builder.spawn_mix();
        delay.set_input(pulse);
        gain.set_input(delay);
        mix.set_input(gain);
        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.5);
        assert_eq!(graph.gen_next_sample(), 0.0);
    }

    #[test]
    fn can_run_a_delay_gain_loop() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.spawn_pulse().set_freq(0.5);
        let mut delay = graph_builder.spawn_delay().set_delay(1.0);
        let mut gain = graph_builder.spawn_gain().set_volume(0.5);
        let mut mix = graph_builder.spawn_mix();
        mix.set_input(pulse);
        delay.set_input(mix.clone());
        gain.set_input(delay);
        mix.set_input(gain);
        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 0.5);
        assert_eq!(graph.gen_next_sample(), 1.25);
        assert_eq!(graph.gen_next_sample(), 0.625);
        assert_eq!(graph.gen_next_sample(), 1.3125);
    }
}
