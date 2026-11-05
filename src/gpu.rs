//! The mark on ferros's GPU (feature `gpu`; ferros GPU.md, "The resident service, and the attest mark's path"; owner, 2026-11-05).
//! The same shading code as the CPU's, run once with valhall's recording lane type: `Lane` for `valhall::ssa::F`, whose arithmetic records a graph that valhall compiles to Mali-G715 words.
//! One GPU invocation is one sample, but the lane type has several lanes' meaning (`live` selects), as the F4 path the CPU runs: the orbit's loop leaves through `Lane::iterate`, whose stopping pass runs the tail with the state frozen.

use crate::lanes::{Bits, Lane, Mask};
use crate::{Button, Look, Params};
use valhall::ssa::{self, Compiled, F, M, U};

impl Mask for M {
    fn splat(b: bool) -> M {
        M::splat(b)
    }
    fn any(self) -> bool {
        M::any(self)
    }
}

impl Bits for U {
    fn splat(v: u32) -> U {
        U::splat(v)
    }
    fn wrapping_mul(self, o: U) -> U {
        U::wrapping_mul(self, o)
    }
}

impl Lane for F {
    // Several lanes' meaning, though an invocation is one sample: `live` then selects on `running`, so a stopped invocation's state stays as it was through the tail, which valhall's `iterate` needs (the pass that stops runs the tail too); the F4 code the CPU runs does exactly this.
    const LANES: usize = 4;
    type M = M;
    type U = U;
    fn splat(v: f32) -> F {
        F::splat(v)
    }
    fn sqrt(self) -> F {
        F::sqrt(self)
    }
    fn sqrt_rounded(self) -> F {
        F::sqrt_rounded(self)
    }
    fn abs(self) -> F {
        F::abs(self)
    }
    fn min(self, o: F) -> F {
        F::min(self, o)
    }
    fn max(self, o: F) -> F {
        F::max(self, o)
    }
    fn floor(self) -> F {
        F::floor(self)
    }
    fn powi(self, n: i32) -> F {
        F::powi(self, n)
    }
    fn gt(self, o: F) -> M {
        F::gt(self, o)
    }
    fn lt(self, o: F) -> M {
        F::lt(self, o)
    }
    fn ge(self, o: F) -> M {
        F::ge(self, o)
    }
    fn select(m: M, a: F, b: F) -> F {
        F::select(m, a, b)
    }
    fn to_bits(self) -> U {
        F::to_bits(self)
    }
    fn from_bits(u: U) -> F {
        F::from_bits(u)
    }
    fn to_u32(self) -> U {
        F::to_u32(self)
    }
    fn to_i32(self) -> U {
        F::to_i32(self)
    }
    fn from_u32(u: U) -> F {
        F::from_u32(u)
    }
    fn iterate<S: Copy>(max: u32, init: S, head: impl FnMut(F, S) -> (S, M), tail: impl FnMut(F, S) -> S) -> S {
        ssa::iterate(max, init, head, tail)
    }
}

/// The attest button's quick pass (`render_button_pack16` with k = 1) as a GPU kernel: invocation g is pixel g of the button's square (side 2r + 4, as ahau sizes it; its centre at side / 2), one sample at the pixel's centre, its `pack16` word to the address in uniform 0 + 8g.
/// Pixels outside the disc (r + 1 out) come out 0, as the CPU writes them. Run it with (side² / 16) workgroups of 16.
pub fn button_quick_kernel(r: f32, p: &Params, look: &Look) -> Compiled {
    let bt = Button::new(p, look, r);
    let side = 2. * r + 4.;
    let c = side * 0.5;
    ssa::begin();
    let g = F::from_u32(ssa::gid());
    // Row and column from the index in floats: exact below 2²⁴, and (g + ½) / side is never a whole number, so its floor is the row.
    let py = ((g + 0.5) / side).floor();
    let px = g - py * side;
    let (dx, dy) = (px + 0.5 - c, py + 0.5 - c);
    let inside = (dx * dx + dy * dy).lt(F::splat((r + 1.) * (r + 1.)));
    let (rgb, a) = bt.sample(dx, dy);
    let a = F::select(inside, a, F::splat(0.));
    let q = (1u32 << 14) as f32;
    // pack16: each channel (c · α · 2¹⁴ + ½) truncated, R G B α in 16 bits each.
    let ch = |v: F| (v * a * q + 0.5).to_u32();
    let lo = ch(rgb[0]) | ch(rgb[1]) << 16;
    let hi = ch(rgb[2]) | (a * q + 0.5).to_u32() << 16;
    ssa::store(0, 3, 0, lo);
    ssa::store(0, 3, 4, hi);
    ssa::compile()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quick pass records and compiles (on the host as on the device: the recorder is plain Rust): its size and its registers.
    #[test]
    fn the_quick_pass_compiles() {
        let salt = *blake3::hash(b"gpu").as_bytes();
        let p = Params::new_placed(&salt, "handle", true);
        let k = button_quick_kernel(448., &p, &Look::attest());
        println!("QUICK_KERNEL {} words, 64 registers {}, {} nodes recorded, {} kept, {} registers at the peak", k.words.len(), k.regs64, k.nodes, k.kept, k.peak);
        assert!(!k.words.is_empty());
    }
}
