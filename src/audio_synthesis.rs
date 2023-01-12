use rand::{thread_rng, Rng};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NamedInputEnum {
    Freq,
    Delay,
    Attack,
    Decay,
    Sustain,
    Release,
    Volume,
}
pub struct Context<'a> {
    time: f32,
    sample_rate: i32,
    inputs: &'a Vec<NodeParamInput>,
    named_inputs: &'a HashMap<NamedInputEnum, NodeParamInput>,
    outputs: &'a Vec<f32>,
}

impl Context<'_> {
    fn read_named_input(&self, slot: NamedInputEnum) -> Option<f32> {
        self.handle_slot(self.named_inputs.get(&slot).copied())
    }

    fn read_input(&self, index: i32) -> Option<f32> {
        self.handle_slot(self.inputs.get(index as usize).cloned())
    }

    fn handle_slot(&self, input: Option<NodeParamInput>) -> Option<f32> {
        match input {
            None => None,
            Some(NodeParamInput::Node(id)) => self.outputs.get(id as usize).cloned(),
            Some(NodeParamInput::Constant(value)) => Some(value),
        }
    }
}

pub trait NodeBehaviour {
    fn gen_next_sample(&self, context: Context) -> f32;
    fn process_outputs(&mut self, _: Context) {}
    // Phantom input means the node is replaced by a temporary node with no inputs
    // for the purpose of dependency tracking. This is done for delay nodes, so
    // that we can break the dependency cycles. The phantom provides the "past"
    // data without dragging the dependencies that are meaningless in the context
    // of current evaluation.
    fn is_phantom_input(&self, _: Context) -> bool {
        false
    }
}

impl std::fmt::Debug for dyn NodeBehaviour {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "NodeBehaviour")
    }
}

pub struct SineOscillator {}

impl NodeBehaviour for SineOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let freq = context
            .read_named_input(NamedInputEnum::Freq)
            .unwrap_or(1000.0);
        (context.time * freq * 2.0 * std::f32::consts::PI).sin()
    }
}

pub struct SquareOscillator {}

impl NodeBehaviour for SquareOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let freq = context
            .read_named_input(NamedInputEnum::Freq)
            .unwrap_or(1000.0);
        let cycle = 1.0 / freq;
        if context.time.rem_euclid(cycle) < cycle * 0.5 {
            1.0
        } else {
            -1.0
        }
    }
}

pub struct SawtoothOscillator {}

impl NodeBehaviour for SawtoothOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let freq = context
            .read_named_input(NamedInputEnum::Freq)
            .unwrap_or(1000.0);
        let cycle = 1.0 / freq;
        lerp(-1.0, 1.0, context.time.rem_euclid(cycle) / cycle)
    }
}

pub struct TriangleOscillator {}

impl NodeBehaviour for TriangleOscillator {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let freq = context
            .read_named_input(NamedInputEnum::Freq)
            .unwrap_or(1000.0);
        if freq == 0.0 {
            return -1.0;
        }
        let cycle = 1.0 / freq;
        let rem = context.time.rem_euclid(cycle);
        lerp(-1.0, 1.0, (rem - cycle * 0.5).abs() / (cycle * 0.5))
    }
}

pub struct RandomOscillator {}

impl NodeBehaviour for RandomOscillator {
    fn gen_next_sample(&self, _: Context) -> f32 {
        let mut rng = thread_rng();
        rng.gen_range(-1.0..1.0)
    }
}

pub struct Gain {}

impl NodeBehaviour for Gain {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let input_sample = context.read_input(0).unwrap_or(0.0);
        input_sample
            * context
                .read_named_input(NamedInputEnum::Volume)
                .unwrap_or(1.0)
    }
}

pub struct ADSR {}

impl NodeBehaviour for ADSR {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let input_sample = context.read_input(0).unwrap_or(0.0);
        let attack = context
            .read_named_input(NamedInputEnum::Attack)
            .unwrap_or(1.0);
        let decay = context
            .read_named_input(NamedInputEnum::Decay)
            .unwrap_or(0.0);
        let release = context
            .read_named_input(NamedInputEnum::Release)
            .unwrap_or(0.0);
        let sustain = context
            .read_named_input(NamedInputEnum::Sustain)
            .unwrap_or(0.0);
        let total_duration = attack + decay + release;
        let t = context.time.rem_euclid(total_duration);

        let factor = if t < attack {
            lerp(0.0, 1.0, t / attack)
        } else if t < attack + decay {
            lerp(1.0, sustain, (t - attack) / decay)
        } else {
            lerp(sustain, 0.0, (t - attack - decay) / release)
        };
        factor * input_sample
    }
}

pub struct Add {}

impl NodeBehaviour for Add {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let mut acc = 0.0;
        let mut index = 0;
        while let Some(input) = context.read_input(index) {
            acc += input;
            index += 1;
        }
        acc
    }
}

pub struct Delay {
    pub buffered_samples: VecDeque<f32>,
}

impl Delay {
    fn new() -> Delay {
        Delay {
            buffered_samples: VecDeque::new(),
        }
    }
}

impl NodeBehaviour for Delay {
    fn gen_next_sample(&self, context: Context) -> f32 {
        let delay_samples = (context
            .read_named_input(NamedInputEnum::Delay)
            .unwrap_or(1.0)
            * (context.sample_rate as f32)) as i32;

        if delay_samples <= 0 {
            return context.read_input(0).unwrap_or(0.0);
        }

        if self.buffered_samples.len() >= delay_samples as usize {
            return self.buffered_samples.front().copied().unwrap_or(0.0);
        }
        0.0
    }

    fn process_outputs(&mut self, context: Context) {
        let delay_samples = (context
            .read_named_input(NamedInputEnum::Delay)
            .unwrap_or(1.0)
            * (context.sample_rate as f32)) as i32;

        if self.buffered_samples.len() >= delay_samples as usize {
            self.buffered_samples.pop_front();
        }

        self.buffered_samples
            .push_back(context.read_input(0).unwrap_or(0.0));
    }

    fn is_phantom_input(&self, context: Context) -> bool {
        let delay_samples = (context
            .read_named_input(NamedInputEnum::Delay)
            .unwrap_or(1.0)
            * (context.sample_rate as f32)) as usize;

        if delay_samples <= 0 {
            return false;
        }
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
    Node(i32),
}

#[derive(Debug, Clone)]
pub struct AudioGraph {
    next_id: i32,
    nodes: Vec<Node>,
    node_input_nodes: Vec<Vec<NodeParamInput>>,
    node_input_slots: Vec<HashMap<NamedInputEnum, NodeParamInput>>,
    sample_rate: i32,
    current_sample: i32,
    out_node: Option<i32>,
}

#[derive(Debug, Clone)]
pub struct NodeBuilder {
    graph_builder: AudioGraphBuilder,
    id: i32,
}

impl NodeBuilder {
    pub fn set_input_node(&mut self, node: NodeBuilder) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_node(self.id, node.id);
        self.clone()
    }

    pub fn set_input_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_constant_f64(self.id, value);
        self.clone()
    }

    pub fn set_freq_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_constant_f64(self.id, NamedInputEnum::Freq, value);
        self.clone()
    }

    pub fn set_freq_node(&mut self, node: NodeBuilder) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_node(self.id, NamedInputEnum::Freq, node.id);
        self.clone()
    }

    pub fn set_volume_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_constant_f64(self.id, NamedInputEnum::Volume, value);
        self.clone()
    }

    pub fn set_volume_node(&mut self, node: NodeBuilder) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_node(self.id, NamedInputEnum::Volume, node.id);
        self.clone()
    }

    pub fn set_delay_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_constant_f64(self.id, NamedInputEnum::Delay, value);
        self.clone()
    }

    pub fn set_delay_node(&mut self, node: NodeBuilder) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_node(self.id, NamedInputEnum::Delay, node.id);
        self.clone()
    }

    pub fn set_attack_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_constant_f64(self.id, NamedInputEnum::Attack, value);
        self.clone()
    }

    pub fn set_decay_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_constant_f64(self.id, NamedInputEnum::Decay, value);
        self.clone()
    }

    pub fn set_sustain_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_constant_f64(self.id, NamedInputEnum::Sustain, value);
        self.clone()
    }

    pub fn set_release_constant(&mut self, value: f64) -> NodeBuilder {
        self.graph_builder
            .internal
            .borrow_mut()
            .link_named_constant_f64(self.id, NamedInputEnum::Release, value);
        self.clone()
    }

    pub fn get_graph(&mut self) -> AudioGraphBuilder {
        self.graph_builder.clone()
    }
}

#[derive(Debug, Clone)]
pub struct AudioGraphBuilder {
    internal: Rc<RefCell<AudioGraph>>,
}

impl AudioGraphBuilder {
    pub fn new() -> AudioGraphBuilder {
        AudioGraphBuilder {
            internal: Rc::new(RefCell::new(AudioGraph::new())),
        }
    }

    pub fn sin(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(SineOscillator {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn square(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(SquareOscillator {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn sawtooth(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(SawtoothOscillator {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn triangle(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(TriangleOscillator {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn random(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(RandomOscillator {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn gain(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(Gain {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn delay(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(Delay::new())));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn adsr(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(ADSR {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
    }

    pub fn mix(&mut self) -> NodeBuilder {
        let id = self
            .internal
            .borrow_mut()
            .add_node(Rc::new(RefCell::new(Add {})));
        NodeBuilder {
            graph_builder: self.clone(),
            id,
        }
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
            out_node: None,
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

    pub fn set_out(&mut self, sink: i32) {
        self.out_node = Some(sink);
    }

    pub fn link_named_node(&mut self, node_to: i32, slot: NamedInputEnum, node_from: i32) {
        self.node_input_slots[node_to as usize]
            .insert(slot, NodeParamInput::Node(node_from))
            .map_or((), |_| ());
    }

    pub fn link_node(&mut self, node_to: i32, node_from: i32) {
        self.node_input_nodes[node_to as usize].push(NodeParamInput::Node(node_from));
    }

    pub fn link_named_constant_f64(&mut self, node_to: i32, slot: NamedInputEnum, value: f64) {
        self.node_input_slots[node_to as usize]
            .insert(slot, NodeParamInput::Constant(value as f32));
    }

    pub fn link_constant_f64(&mut self, node_to: i32, value: f64) {
        self.node_input_nodes[node_to as usize].push(NodeParamInput::Constant(value as f32));
    }

    pub fn set_sample_rate(&mut self, num: i32) {
        self.sample_rate = num;
    }

    pub fn gen_next_sample(&mut self) -> f32 {
        if self.out_node.is_none() {
            return 0.0;
        }

        let mut topo_sort = Vec::<i32>::new();
        let mut next_topo_index = 0;
        // we build per node list of output indices
        let mut node_outputs = Vec::<Vec<i32>>::new();
        // for the topo sort, we maintain a number of inputs
        let mut node_input_count = Vec::<i32>::new();
        node_outputs.resize_with(self.nodes.len(), || Vec::new());
        node_input_count.resize(self.nodes.len(), 0);
        let mut phantom_delay_nodes = HashSet::<i32>::new();

        //
        let mut outputs = Vec::<f32>::new();
        outputs.resize(self.nodes.len(), 0.0);

        self.current_sample += 1;
        let time = self.current_sample as f32 / self.sample_rate as f32;
        // first pass
        for node in &self.nodes {
            // figure out the dependency graph
            for indexed_input in &self.node_input_nodes[node.id as usize] {
                if let NodeParamInput::Node(input_node_index) = indexed_input {
                    //
                    let context = Context {
                        time: time,
                        sample_rate: self.sample_rate,
                        outputs: &outputs,
                        inputs: &self.node_input_nodes[*input_node_index as usize],
                        named_inputs: &self.node_input_slots[*input_node_index as usize],
                    };

                    if !self.nodes[*input_node_index as usize]
                        .behaviour
                        .borrow()
                        .is_phantom_input(context)
                    {
                        node_outputs[*input_node_index as usize].push(node.id);
                        node_input_count[node.id as usize] += 1;
                    } else {
                        // we don't treat the delay node as input node, but we
                        // make a "phantom" input node with 0 inputs, that we process before the topo
                        // ordered processing
                        phantom_delay_nodes.insert(*input_node_index);
                    }
                }
            }

            for named_input in &self.node_input_slots[node.id as usize] {
                if let NodeParamInput::Node(input_node_index) = named_input.1 {
                    //
                    let context = Context {
                        time: time,
                        sample_rate: self.sample_rate,
                        outputs: &outputs,
                        inputs: &self.node_input_nodes[*input_node_index as usize],
                        named_inputs: &self.node_input_slots[*input_node_index as usize],
                    };

                    if !self.nodes[*input_node_index as usize]
                        .behaviour
                        .borrow()
                        .is_phantom_input(context)
                    {
                        node_outputs[*input_node_index as usize].push(node.id);
                        node_input_count[node.id as usize] += 1;
                    } else {
                        // we don't treat the delay node as input node, but we
                        // make a "phantom" input node with 0 inputs, that we process before the topo
                        // ordered processing
                        phantom_delay_nodes.insert(*input_node_index);
                    }
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

        // phantom nodes have no inputs, so they can be safely processed before the regular nodes
        for v in &phantom_delay_nodes {
            let context = Context {
                time: time,
                sample_rate: self.sample_rate,
                outputs: &outputs,
                inputs: &self.node_input_nodes[*v as usize],
                named_inputs: &self.node_input_slots[*v as usize],
            };

            let sample = self.nodes[*v as usize]
                .behaviour
                .borrow()
                .gen_next_sample(context);
            outputs[*v as usize] = sample;
        }

        for v in &topo_sort {
            let context = Context {
                time: time,
                sample_rate: self.sample_rate,
                outputs: &outputs,
                inputs: &self.node_input_nodes[*v as usize],
                named_inputs: &self.node_input_slots[*v as usize],
            };

            if !phantom_delay_nodes.contains(v) {
                let sample = self.nodes[*v as usize]
                    .behaviour
                    .borrow()
                    .gen_next_sample(context);
                outputs[*v as usize] = sample;
            }
        }

        for v in topo_sort {
            let context = Context {
                time: time,
                sample_rate: self.sample_rate,
                outputs: &outputs,
                inputs: &self.node_input_nodes[v as usize],
                named_inputs: &self.node_input_slots[v as usize],
            };
            self.nodes[v as usize]
                .behaviour
                .borrow_mut()
                .process_outputs(context);
        }

        outputs[self.out_node.unwrap() as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_output_returns_zero() {
        let mut graph_builder = AudioGraphBuilder::new();
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
    }

    #[test]
    fn can_build_a_path() {
        let mut graph_builder = AudioGraphBuilder::new();
        let sine = graph_builder.sin();
        let mut gain = graph_builder.gain();
        gain.set_volume_constant(0.5);
        gain.set_input_node(sine);
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
        let sine = graph_builder.sin().set_freq_constant(1.0);
        let sine2 = graph_builder.sin().set_freq_constant(2.0);
        let mut mix = graph_builder.mix();
        mix.set_input_node(sine);
        mix.set_input_node(sine2);

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
        let sine = graph_builder.sin().set_freq_constant(1.0);
        let mut mix = graph_builder.mix();
        let mut gain = graph_builder.gain().set_volume_constant(0.5);
        mix.set_input_node(sine.clone());
        gain.set_input_node(sine);
        mix.set_input_node(gain);
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
        let sine = graph_builder.sin().set_freq_constant(1.0);
        let mut delay = graph_builder.delay().set_delay_constant(0.25);
        delay.set_input_node(sine);
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
        let pulse = graph_builder.square().set_freq_constant(0.5);
        let mut delay = graph_builder.delay().set_delay_constant(1.0);
        let mut delay1 = graph_builder.delay().set_delay_constant(1.0);
        let mut delay2 = graph_builder.delay().set_delay_constant(1.0);
        delay.set_input_node(pulse);
        delay1.set_input_node(delay);
        delay2.set_input_node(delay1);
        graph_builder.set_out(delay2);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), -1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn can_run_a_queue_of_delay_nodes_constructed_out_of_order() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.square().set_freq_constant(0.5);
        let mut delay2 = graph_builder.delay().set_delay_constant(1.0);
        let mut delay1 = graph_builder.delay().set_delay_constant(1.0);
        let mut delay = graph_builder.delay().set_delay_constant(1.0);
        delay.set_input_node(pulse);
        delay1.set_input_node(delay);
        delay2.set_input_node(delay1);
        graph_builder.set_out(delay2);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), -1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn can_run_a_delay_to_mix_graph() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.square().set_freq_constant(0.5);
        let mut delay = graph_builder.delay().set_delay_constant(1.0);
        let mut mix = graph_builder.mix();
        delay.set_input_node(pulse);
        mix.set_input_node(delay);
        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), -1.0);
    }

    #[test]
    fn can_run_a_delay_gain_graph() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.square().set_freq_constant(0.5);
        let mut delay = graph_builder.delay().set_delay_constant(4.0);
        let mut gain = graph_builder.gain().set_volume_constant(0.5);
        let mut mix = graph_builder.mix();
        delay.set_input_node(pulse);
        gain.set_input_node(delay);
        mix.set_input_node(gain);
        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.5);
        assert_eq!(graph.gen_next_sample(), -0.5);
    }

    #[test]
    fn can_run_a_delay_gain_loop() {
        let mut graph_builder = AudioGraphBuilder::new();
        let pulse = graph_builder.square().set_freq_constant(0.5);
        let mut delay = graph_builder.delay().set_delay_constant(1.0);
        let mut gain = graph_builder.gain().set_volume_constant(0.5);
        let mut mix = graph_builder.mix();
        mix.set_input_node(pulse);
        delay.set_input_node(mix.clone());
        gain.set_input_node(delay);
        mix.set_input_node(gain);
        graph_builder.set_out(mix);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), -0.5);
        assert_eq!(graph.gen_next_sample(), 0.75);
        assert_eq!(graph.gen_next_sample(), -0.625);
        assert_eq!(graph.gen_next_sample(), 0.6875);
    }

    #[test]
    fn delay_can_have_disconnected_input() {
        let mut graph_builder = AudioGraphBuilder::new();
        let delay = graph_builder.delay().set_delay_constant(1.0);
        graph_builder.set_out(delay);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 0.0);
        assert_eq!(graph.gen_next_sample(), 0.0);
    }

    #[test]
    fn ssquare_freq_can_be_zero() {
        let mut graph_builder = AudioGraphBuilder::new();
        let o = graph_builder.square().set_freq_constant(0.0);
        graph_builder.set_out(o);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }

    #[test]
    fn triangle_freq_can_be_zero() {
        let mut graph_builder = AudioGraphBuilder::new();
        let o = graph_builder.triangle().set_freq_constant(0.0);
        graph_builder.set_out(o);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), -1.0);
        assert_eq!(graph.gen_next_sample(), -1.0);
    }

    #[test]
    fn delay_can_be_zero() {
        let mut graph_builder = AudioGraphBuilder::new();
        let o = graph_builder.square().set_freq_constant(0.5);
        let mut delay = graph_builder.delay().set_delay_constant(0.0);
        delay.set_input_node(o);
        graph_builder.set_out(delay);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), -1.0);
    }

    #[test]
    fn can_connect_node_to_volume() {
        let mut graph_builder = AudioGraphBuilder::new();
        let o = graph_builder.square().set_freq_constant(0.5);
        let o1 = graph_builder.square().set_freq_constant(0.5);
        let mut gain = graph_builder.gain().set_volume_node(o);
        gain.set_input_node(o1);
        graph_builder.set_out(gain);
        let mut graph = graph_builder.extract_graph();
        graph.set_sample_rate(1);

        // negative value will be multiplied by negative volume
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
        assert_eq!(graph.gen_next_sample(), 1.0);
    }
}
