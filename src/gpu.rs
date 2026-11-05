//! The mark on ferros's GPU (feature `gpu`; ferros GPU.md, "The resident service, and the attest mark's path"; owner, 2026-11-05).
//! The same shading code as the CPU's, run once with valhall's recording lane type: `Lane` for `valhall::ssa::F`, whose arithmetic records a graph that valhall compiles to Mali-G715 words.
//! One GPU invocation is one sample, but the lane type has several lanes' meaning (`live` selects), as the F4 path the CPU runs: the orbit's loop leaves through `Lane::iterate`, whose stopping pass runs the tail with the state frozen.

use crate::lanes::{Bits, Lane, Mask};
use crate::{Button, Look, Params};
use valhall::ssa::{self, F, M, U};
/// What a kernel compiles to (valhall's), for a caller that runs it.
pub use valhall::ssa::Compiled;

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

/// The quick pass over the circle's staged coordinates (`button_stage`): invocation g draws staged pixel g as `button_quick_kernel` draws it, and nothing outside the circle is computed at all (~22 % of the square; the caller zeroes those pixels once). Uniforms: 0 the frame, 1 the stage. Run it with (staged count / 16) workgroups of 16.
pub fn button_quick_staged_kernel(r: f32, p: &Params, look: &Look) -> Compiled {
    let bt = Button::new(p, look, r);
    let side = 2. * r + 4.;
    let c = side * 0.5;
    ssa::begin();
    ssa::reserve(2);
    let g = ssa::gid();
    let idx = ssa::load(1, g, 2, 0);
    let fi = F::from_u32(idx);
    let py = ((fi + 0.5) / side).floor();
    let px = fi - py * side;
    let (rgb, a) = bt.sample(px + 0.5 - c, py + 0.5 - c);
    let q = (1u32 << 14) as f32;
    let ch = |v: F| (v * a * q + 0.5).to_u32();
    ssa::store_at(0, idx, 3, 0, ch(rgb[0]) | ch(rgb[1]) << 16);
    ssa::store_at(0, idx, 3, 4, ch(rgb[2]) | (a * q + 0.5).to_u32() << 16);
    ssa::compile()
}

/// The circle's staged coordinates (owner, 2026-11-05: "a stage of all the coordinates in the circle, then only execute the ones in the circle"): the index (py · side + px) of every pixel of the button's square that `render_button_pack16` draws (its centre within r + 1 of the square's), in row order. Every other pixel stays as the quick pass left it, 0.
pub fn button_stage(r: f32, mut f: impl FnMut(u32)) {
    let side = 2 * r as usize + 4;
    let c = side as f32 * 0.5;
    for py in 0..side {
        for px in 0..side {
            let (dx, dy) = (px as f32 + 0.5 - c, py as f32 + 0.5 - c);
            if dx * dx + dy * dy < (r + 1.) * (r + 1.) {
                f((py * side + px) as u32);
            }
        }
    }
}

/// Pass `m` (0 to 15) of the button's 4 × 4 pass, each pass distinct (owner, 2026-11-05): invocation g takes staged pixel g (`button_stage`), its m-th stratified, jittered sample exactly as `render_button_pack16` places it, adds c² · a and a to that pixel's sums in sample order (pass 0 starts them), and writes the pixel as the sums so far stand, coverage over m + 1 samples: after pass 15, the CPU's 4 × 4 pixel.
/// Uniforms: 0 the frame (side × side pack16 words), 1 the stage (u32 indices), 2 the sums (four f32 a staged pixel: R, G, B, coverage). Run it with (staged count / 16) workgroups of 16.
pub fn button_fine_kernel(r: f32, p: &Params, look: &Look, m: u32) -> Compiled {
    let bt = Button::new(p, look, r);
    let side = 2. * r + 4.;
    let c = side * 0.5;
    let (k, kf) = (4u32, 4f32);
    ssa::begin();
    ssa::reserve(3);
    let g = ssa::gid();
    let idx = ssa::load(1, g, 2, 0);
    let fi = F::from_u32(idx);
    // Row and column from the index in floats: exact below 2²⁴, and (i + ½) / side is never a whole number.
    let py = ((fi + 0.5) / side).floor();
    let px = fi - py * side;
    let (jx, jy) = crate::jitter::<F>(px.to_u32(), py.to_u32(), U::splat(m));
    let (sx, sy) = ((F::splat((m % k) as f32) + jx) / kf, (F::splat((m / k) as f32) + jy) / kf);
    let (rgb, a) = bt.sample(px + sx - c, py + sy - c);
    let zero = F::splat(0.);
    let (mut sum, mut cover) = if m == 0 {
        ([zero; 3], zero)
    } else {
        ([0i16, 4, 8].map(|o| F::from_bits(ssa::load(2, g, 4, o))), F::from_bits(ssa::load(2, g, 4, 12)))
    };
    for i in 0..3 {
        sum[i] = sum[i] + rgb[i] * rgb[i] * a;
    }
    cover = cover + a;
    for (i, o) in [0i16, 4, 8].iter().enumerate() {
        ssa::store_at(2, g, 4, *o, sum[i].to_bits());
    }
    ssa::store_at(2, g, 4, 12, cover.to_bits());
    let has = cover.gt(zero);
    let out = [0, 1, 2].map(|i| F::select(has, (sum[i] / cover).sqrt_rounded(), zero));
    let alpha = cover / ((m + 1) as f32);
    let q = (1u32 << 14) as f32;
    let ch = |v: F| (v * alpha * q + 0.5).to_u32();
    ssa::store_at(0, idx, 3, 0, ch(out[0]) | ch(out[1]) << 16);
    ssa::store_at(0, idx, 3, 4, ch(out[2]) | (alpha * q + 0.5).to_u32() << 16);
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
        println!("QUICK_KERNEL {} words, 64 registers {}, {} nodes recorded, {} kept, {} registers at the peak; the loop {} words, {} of them constant loads; {} uniform slots of constants", k.words.len(), k.regs64, k.nodes, k.kept, k.peak, k.loop_words, k.loop_consts, k.consts.len());
        assert!(!k.words.is_empty());
    }

    /// A fine pass records and compiles too.
    #[test]
    fn a_fine_pass_compiles() {
        let salt = *blake3::hash(b"gpu").as_bytes();
        let p = Params::new_placed(&salt, "handle", true);
        for m in [0, 7, 15] {
            let k = button_fine_kernel(448., &p, &Look::attest(), m);
            println!("FINE_KERNEL pass {m}: {} words, 64 registers {}, peak {}, {} constant slots", k.words.len(), k.regs64, k.peak, k.consts.len());
        }
    }
}
