use crate::Float64;
use rand::distributions::{Distribution, Uniform};

pub trait Entity {
    fn params(&self) -> Vec<Float64>;
}

#[derive(Debug)]
pub struct Neuron {
    pub ws: Vec<Float64>,
    pub b: Float64,
    pub lin: bool,
}

fn gen_weights(n: usize) -> Vec<Float64> {
    let mut rng = rand::thread_rng();
    let range = Uniform::from(-1.0..=1.0);

    let mut ws = Vec::with_capacity(n);
    for _ in 0..n {
        ws.push(Float64::from(range.sample(&mut rng)));
    }
    ws
}

impl Neuron {
    pub fn new(n_in: usize, linear: bool) -> Self {
        Self {
            ws: gen_weights(n_in),
            b: Float64::from(0.0),
            lin: linear,
        }
    }
}

impl Entity for Neuron {
    fn params(&self) -> Vec<Float64> {
        let mut ps = self.ws.clone();
        ps.push(self.b.clone());
        ps
    }
}

#[derive(Debug)]
pub struct Layer {
    pub n_in: usize,
    pub ns: Vec<Neuron>,
}

impl Layer {
    pub fn new(n_in: usize, n_out: usize, linear: bool) -> Self {
        let mut ns = Vec::with_capacity(n_out);
        for _ in 0..n_out {
            ns.push(Neuron::new(n_in, linear));
        }

        Self { n_in, ns }
    }
}

impl Entity for Layer {
    fn params(&self) -> Vec<Float64> {
        let mut ps = Vec::<Float64>::with_capacity(self.n_in * self.ns.len());
        for n in self.ns.iter() {
            ps.append(&mut n.params());
        }
        ps
    }
}

pub struct Mlp {
    ls: Vec<Layer>,
}

impl Mlp {
    pub fn new(n_in: usize, mut layer_out: Vec<usize>) -> Self {
        let sz = layer_out.len();
        let mut n_in_out = Vec::<usize>::with_capacity(1 + sz);
        n_in_out.push(n_in);
        n_in_out.append(&mut layer_out);

        let mut ls = Vec::with_capacity(sz);
        for i in 0..sz {
            let lin = if i < sz - 1 { false } else { true };
            ls.push(Layer::new(n_in_out[i], n_in_out[i + 1], lin));
        }

        Mlp { ls }
    }
}
