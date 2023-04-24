use std::cell::{Ref, RefCell};
use std::cmp::PartialEq;
use std::collections::{HashSet, VecDeque};
use std::fmt::{self, Debug};
use std::hash::Hash;
use std::ops::{Add, Deref, Div, Mul, Neg, Sub};
use std::rc::Rc;

type BackwardFn = fn(v: &Ref<Float64Inner>);

pub struct Float64Inner {
    pub v: f64,
    pub g: f64,
    pub op: &'static str,
    pub children: Vec<Float64>,
    pub bwd: Option<BackwardFn>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Float64(Rc<RefCell<Float64Inner>>);

impl Float64Inner {
    pub fn new(v: f64, op: &'static str, children: Vec<Float64>, bwd: Option<BackwardFn>) -> Self {
        Self {
            v,
            g: 0.0,
            op,
            children,
            bwd,
        }
    }
}

impl Debug for Float64Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        f.debug_struct("Float64Inner")
            .field("v", &self.v)
            .field("g", &self.g)
            .field("op", &self.op)
            .finish()
    }
}

impl PartialEq for Float64Inner {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Eq for Float64Inner {}

impl Hash for Float64Inner {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let ptr = format!("{:p}", self);

        // ptr alone is not enough
        ptr.hash(state);
        self.children.hash(state);
    }
}

impl Float64 {
    pub fn new(v: Float64Inner) -> Self {
        Self(Rc::new(RefCell::new(v)))
    }

    pub fn from<T>(v: T) -> Self
    where
        T: Into<Self>,
    {
        v.into()
    }

    pub fn pow(&self, p: f64) -> Self {
        let v = self.borrow().v.powf(p);

        // Tracked in graph only for computing gradient, otherwise
        // `bwd` have to be a closure capturing the value of `p`.
        let p = Self::from(p);

        let bwd: BackwardFn = |val| {
            let mut b = val.children[0].borrow_mut();
            let p = val.children[1].borrow();
            b.g += val.g * p.v * b.v.powf(p.v - 1.0);
        };

        Self::new(Float64Inner::new(
            v,
            "^",
            vec![self.clone(), p.clone()],
            Some(bwd),
        ))
    }

    pub fn relu(&self) -> Self {
        let inner = self.borrow();
        let v = inner.v.max(0.0);

        let bwd: BackwardFn = |val| {
            let mut origin = val.children[0].borrow_mut();
            origin.g += val.g * (((val.v > 0.0) as u8) as f64);
        };

        Float64::new(Float64Inner::new(v, "ReLU", vec![self.clone()], Some(bwd)))
    }

    pub fn backward(&self) {
        let mut sorted = VecDeque::<Float64>::new();
        let mut visited = HashSet::<Float64>::new();

        // sort dag in topological order
        self._backward(&mut visited, &mut sorted);

        // head gradient
        self.borrow_mut().g = 1.0;

        // backprop
        for v in sorted {
            if let Some(bwd) = v.borrow().bwd {
                bwd(&v.borrow());
            }
        }
    }

    fn _backward(&self, visited: &mut HashSet<Float64>, sorted: &mut VecDeque<Float64>) {
        if !visited.contains(self) {
            visited.insert(self.clone());

            for child in &self.borrow().children {
                child._backward(visited, sorted);
            }

            sorted.push_front(self.clone())
        }
    }
}

impl<T> From<T> for Float64
where
    T: Into<f64>,
{
    fn from(v: T) -> Float64 {
        Float64::new(Float64Inner {
            v: v.into(),
            g: 0.0,
            op: "",
            children: vec![],
            bwd: None,
        })
    }
}

impl Deref for Float64 {
    type Target = Rc<RefCell<Float64Inner>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Float64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.borrow().hash(state);
    }
}

impl Neg for Float64 {
    type Output = Float64;
    fn neg(self) -> Self::Output {
        &self * -1.0
    }
}

impl Neg for &Float64 {
    type Output = Float64;
    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

fn add(lhs: &Float64, rhs: &Float64) -> Float64 {
    let bwd: BackwardFn = |val| {
        // CORNER CASE: vx + vx
        val.children[0].borrow_mut().g += val.g;
        val.children[1].borrow_mut().g += val.g;
    };

    Float64::new(Float64Inner::new(
        lhs.borrow().v + rhs.borrow().v,
        "+",
        vec![lhs.clone(), rhs.clone()],
        Some(bwd),
    ))
}

fn mul(lhs: &Float64, rhs: &Float64) -> Float64 {
    let bwd: BackwardFn = |val| {
        // CORNER CASE: vx * vx
        let mut lhs = val.children[0].borrow_mut();
        let mut rhs = val.children[1].borrow_mut();

        lhs.g += val.g * rhs.v;
        rhs.g += val.g * lhs.v;
    };

    Float64::new(Float64Inner::new(
        lhs.borrow().v * rhs.borrow().v,
        "*",
        vec![lhs.clone(), rhs.clone()],
        Some(bwd),
    ))
}

fn sub(lhs: &Float64, rhs: &Float64) -> Float64 {
    lhs + -rhs
}

fn div(lhs: &Float64, rhs: &Float64) -> Float64 {
    lhs * rhs.pow(-1.0)
}

macro_rules! impl_op_with_fn {
    ($T:path, $Trait:tt, $op:ident) => {
        impl $Trait<$T> for $T {
            type Output = $T;
            fn $op(self, rhs: $T) -> Self::Output {
                $op(&self, &rhs)
            }
        }

        impl $Trait<&$T> for $T {
            type Output = $T;
            fn $op(self, rhs: &$T) -> Self::Output {
                $op(&self, rhs)
            }
        }

        impl $Trait<f64> for $T {
            type Output = $T;
            fn $op(self, rhs: f64) -> Self::Output {
                $op(&self, &rhs.into())
            }
        }

        impl $Trait<$T> for &$T {
            type Output = $T;
            fn $op(self, rhs: $T) -> Self::Output {
                $op(self, &rhs)
            }
        }

        impl<'a, 'b> $Trait<&'a $T> for &'b $T {
            type Output = $T;
            fn $op(self, rhs: &'a $T) -> Self::Output {
                $op(self, rhs)
            }
        }

        impl $Trait<f64> for &$T {
            type Output = $T;
            fn $op(self, rhs: f64) -> Self::Output {
                $op(self, &rhs.into())
            }
        }

        impl $Trait<$T> for f64 {
            type Output = $T;
            fn $op(self, rhs: $T) -> Self::Output {
                $op(&self.into(), &rhs)
            }
        }

        impl $Trait<&$T> for f64 {
            type Output = $T;
            fn $op(self, rhs: &$T) -> Self::Output {
                $op(&self.into(), rhs)
            }
        }
    };
}

impl_op_with_fn! {Float64, Add, add}
impl_op_with_fn! {Float64, Mul, mul}
impl_op_with_fn! {Float64, Sub, sub}
impl_op_with_fn! {Float64, Div, div}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add() {
        let v1 = Float64::from(1.5);

        assert_eq!(Float64::from(1.5) + Float64::from(1.5), Float64::from(3.0));
        assert_eq!(Float64::from(1.5) + 1.5, Float64::from(3.0));
        assert_eq!(1.5 + Float64::from(1.5), Float64::from(3.0));
        assert_eq!(Float64::from(1.5) + &v1, Float64::from(3.0));
        assert_eq!(&v1 + Float64::from(1.5), Float64::from(3.0));
        assert_eq!(&v1 + &v1, Float64::from(3.0));
        assert_eq!(1.5 + &v1, Float64::from(3.0));
        assert_eq!(&v1 + 1.5, Float64::from(3.0));
    }

    #[test]
    fn mul() {
        let v1 = Float64::from(1.5);

        assert_eq!(Float64::from(1.5) * Float64::from(1.5), Float64::from(2.25));
        assert_eq!(Float64::from(1.5) * 1.5, Float64::from(2.25));
        assert_eq!(1.5 * Float64::from(1.5), Float64::from(2.25));
        assert_eq!(Float64::from(1.5) * &v1, Float64::from(2.25));
        assert_eq!(&v1 * Float64::from(1.5), Float64::from(2.25));
        assert_eq!(&v1 * &v1, Float64::from(2.25));
        assert_eq!(1.5 * &v1, Float64::from(2.25));
        assert_eq!(&v1 * 1.5, Float64::from(2.25));
    }

    #[test]
    fn sub() {
        let v1 = Float64::from(1.5);

        assert_eq!(Float64::from(1.5) - Float64::from(1.5), Float64::from(0.0));
        assert_eq!(Float64::from(1.5) - 1.5, Float64::from(0.0));
        assert_eq!(1.5 - Float64::from(1.5), Float64::from(0.0));
        assert_eq!(Float64::from(1.5) - &v1, Float64::from(0.0));
        assert_eq!(&v1 - Float64::from(1.5), Float64::from(0.0));
        assert_eq!(&v1 - &v1, Float64::from(0.0));
        assert_eq!(1.5 - &v1, Float64::from(0.0));
        assert_eq!(&v1 - 1.5, Float64::from(0.0));
    }

    #[test]
    fn div() {
        let v1 = Float64::from(1.5);

        assert_eq!(Float64::from(1.5) / Float64::from(1.5), Float64::from(1.0));
        assert_eq!(Float64::from(1.5) / 1.5, Float64::from(1.0));
        assert_eq!(1.5 / Float64::from(1.5), Float64::from(1.0));
        assert_eq!(Float64::from(1.5) / &v1, Float64::from(1.0));
        assert_eq!(&v1 / Float64::from(1.5), Float64::from(1.0));
        assert_eq!(&v1 / &v1, Float64::from(1.0));
        assert_eq!(1.5 / &v1, Float64::from(1.0));
        assert_eq!(&v1 / 1.5, Float64::from(1.0));
    }

    #[test]
    fn backpropagation() {
        let x = Float64::from(2.0);
        let y = Float64::from(3.0);
        let z = Float64::from(4.0);
        let l = &x * &y + &x * &z;

        assert_eq!(x.borrow().g, 0.0);
        assert_eq!(y.borrow().g, 0.0);
        assert_eq!(z.borrow().g, 0.0);
        assert_eq!(l.borrow().g, 0.0);

        l.backward();

        assert_eq!(l.borrow().g, 1.0);
        assert_eq!(z.borrow().g, 2.0);
        assert_eq!(y.borrow().g, 2.0);
        assert_eq!(x.borrow().g, 7.0); // L = x * (y + z)
    }
}
