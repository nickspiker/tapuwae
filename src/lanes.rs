//! Four samples at a time: the shading is written once, generic over `Lane`, and runs on `f32` (one sample, the scalar path) or on `F4` (four samples in the lanes of one 128-bit register: NEON on the phone's cores, SSE on a desktop).
//! Every `F4` operation is a plain loop over its four elements, inlined, which LLVM turns into one vector instruction; no `core::arch`, so the same source builds and is tested anywhere, and is ready to become a GPU shader later.
//! That takes LLVM's SLP vectoriser, which rustc runs at opt-level 2 and 3 but not at "s" or "z"; under LTO it runs after the link, at the level of the crate being linked (a payload), not this one's.
//! The generic code keeps the lane on the left of every operation (`x * 0.5`, `L::splat(1.) - x`): a bound for an f32 on the left (`f32: Mul<L>`) would shadow f32's own arithmetic in every generic function, since Rust prefers a where-clause to the built-in impl; and with + and × commuting exactly in IEEE, the order changes no bit.
//! A choice that depends on the sample is a `select` between two computed values (a lane mask picks per lane), never a branch; a choice the same for every lane (the look's style, the view's kind) stays an ordinary `if`.

use core::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Shl, Shr, Sub};

/// A lane mask: one truth per lane (`bool` for one lane, `M4` for four).
pub trait Mask: Copy + BitAnd<Output = Self> + BitOr<Output = Self> + Not<Output = Self> {
    /// Every lane the same truth.
    fn splat(b: bool) -> Self;
    /// Whether any lane holds.
    fn any(self) -> bool;
}

/// A value that can be chosen lane by lane under a mask `M`: a lane, or a tuple or array of them.
pub trait Sel<M>: Copy {
    fn sel(m: M, a: Self, b: Self) -> Self;
}

/// Per lane, `a` where the mask holds and `b` where it does not.
#[inline(always)]
pub fn select<M, T: Sel<M>>(m: M, a: T, b: T) -> T {
    T::sel(m, a, b)
}

/// Per lane, `then()` where the mask holds and `otherwise` where it does not, with `then` worked out only when some lane needs it: a branch for one lane (the scalar path skips the work as it always did), a select for four.
#[inline(always)]
pub fn pick<M: Mask, T: Sel<M>>(m: M, then: impl FnOnce() -> T, otherwise: T) -> T {
    if m.any() { T::sel(m, then(), otherwise) } else { otherwise }
}

/// An unsigned 32-bit integer per lane (`u32` for one lane, `U4` for four): a float's bits, and the integer hashes.
pub trait Bits: Copy + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Shl<u32, Output = Self> + Shr<u32, Output = Self> {
    fn splat(v: u32) -> Self;
    fn wrapping_mul(self, o: Self) -> Self;
}

/// One or more f32 samples side by side.
pub trait Lane: Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self> + Add<f32, Output = Self> + Sub<f32, Output = Self> + Mul<f32, Output = Self> + Div<f32, Output = Self> {
    /// The lane count.
    const LANES: usize;
    type M: Mask;
    type U: Bits;
    fn splat(v: f32) -> Self;
    /// √x, within an ulp (for four lanes, `sqrt1`; for one, the platform's).
    fn sqrt(self) -> Self;
    /// √x correctly rounded, as the platform gives it, lane by lane: for a value that feeds a chaotic orbit (a sample's position), where an ulp moves the whole escape.
    fn sqrt_rounded(self) -> Self;
    fn abs(self) -> Self;
    fn min(self, o: Self) -> Self;
    fn max(self, o: Self) -> Self;
    fn floor(self) -> Self;
    /// selfⁿ.
    fn powi(self, n: i32) -> Self;
    fn gt(self, o: Self) -> Self::M;
    fn lt(self, o: Self) -> Self::M;
    fn ge(self, o: Self) -> Self::M;
    fn select(m: Self::M, a: Self, b: Self) -> Self;
    fn to_bits(self) -> Self::U;
    fn from_bits(u: Self::U) -> Self;
    /// `as u32` (toward zero, negatives to 0).
    fn to_u32(self) -> Self::U;
    /// `as i32 as u32` (toward zero, negatives in two's complement).
    fn to_i32(self) -> Self::U;
    /// `u as f32`.
    fn from_u32(u: Self::U) -> Self;
}

impl<L: Lane> Sel<L::M> for L {
    #[inline(always)]
    fn sel(m: L::M, a: L, b: L) -> L {
        L::select(m, a, b)
    }
}

impl<M: Copy, A: Sel<M>, B: Sel<M>> Sel<M> for (A, B) {
    #[inline(always)]
    fn sel(m: M, a: Self, b: Self) -> Self {
        (A::sel(m, a.0, b.0), B::sel(m, a.1, b.1))
    }
}

impl<M: Copy, A: Sel<M>, B: Sel<M>, C: Sel<M>> Sel<M> for (A, B, C) {
    #[inline(always)]
    fn sel(m: M, a: Self, b: Self) -> Self {
        (A::sel(m, a.0, b.0), B::sel(m, a.1, b.1), C::sel(m, a.2, b.2))
    }
}

impl<M: Copy, A: Sel<M>> Sel<M> for [A; 3] {
    #[inline(always)]
    fn sel(m: M, a: Self, b: Self) -> Self {
        [A::sel(m, a[0], b[0]), A::sel(m, a[1], b[1]), A::sel(m, a[2], b[2])]
    }
}

/// The new value where a lane is still running and the old where it has stopped, for state carried round a loop that four lanes leave at different times (the orbit's escape).
/// A single lane leaves the loop the moment it stops, so inside the loop it is always running and takes the new value without the select.
#[inline(always)]
pub fn live<L: Lane, T: Sel<L::M>>(running: L::M, new: T, old: T) -> T {
    if L::LANES > 1 { T::sel(running, new, old) } else { new }
}

impl Mask for bool {
    #[inline(always)]
    fn splat(b: bool) -> bool {
        b
    }
    #[inline(always)]
    fn any(self) -> bool {
        self
    }
}

impl Bits for u32 {
    #[inline(always)]
    fn splat(v: u32) -> u32 {
        v
    }
    #[inline(always)]
    fn wrapping_mul(self, o: u32) -> u32 {
        u32::wrapping_mul(self, o)
    }
}

impl Lane for f32 {
    const LANES: usize = 1;
    type M = bool;
    type U = u32;
    #[inline(always)]
    fn splat(v: f32) -> f32 {
        v
    }
    #[inline(always)]
    fn sqrt(self) -> f32 {
        rounded_sqrt(self)
    }
    #[inline(always)]
    fn sqrt_rounded(self) -> f32 {
        rounded_sqrt(self)
    }
    #[inline(always)]
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    #[inline(always)]
    fn min(self, o: f32) -> f32 {
        f32::min(self, o)
    }
    #[inline(always)]
    fn max(self, o: f32) -> f32 {
        f32::max(self, o)
    }
    #[cfg(feature = "std")]
    #[inline(always)]
    fn floor(self) -> f32 {
        f32::floor(self)
    }
    #[cfg(not(feature = "std"))]
    #[inline(always)]
    fn floor(self) -> f32 {
        libm::floorf(self)
    }
    #[cfg(feature = "std")]
    #[inline(always)]
    fn powi(self, n: i32) -> f32 {
        f32::powi(self, n)
    }
    #[cfg(not(feature = "std"))]
    #[inline(always)]
    fn powi(self, n: i32) -> f32 {
        libm::powf(self, n as f32)
    }
    #[inline(always)]
    fn gt(self, o: f32) -> bool {
        self > o
    }
    #[inline(always)]
    fn lt(self, o: f32) -> bool {
        self < o
    }
    #[inline(always)]
    fn ge(self, o: f32) -> bool {
        self >= o
    }
    #[inline(always)]
    fn select(m: bool, a: f32, b: f32) -> f32 {
        if m { a } else { b }
    }
    #[inline(always)]
    fn to_bits(self) -> u32 {
        f32::to_bits(self)
    }
    #[inline(always)]
    fn from_bits(u: u32) -> f32 {
        f32::from_bits(u)
    }
    #[inline(always)]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline(always)]
    fn to_i32(self) -> u32 {
        self as i32 as u32
    }
    #[inline(always)]
    fn from_u32(u: u32) -> f32 {
        u as f32
    }
}

/// The platform's square root, correctly rounded: std's, or libm's without std.
#[cfg(feature = "std")]
#[inline(always)]
fn rounded_sqrt(x: f32) -> f32 {
    f32::sqrt(x)
}
#[cfg(not(feature = "std"))]
#[inline(always)]
fn rounded_sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

/// Four f32 lanes.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct F4(pub [f32; 4]);

/// Four lane truths, each all ones (true) or all zeros, as a vector compare leaves them, so a select is a bitwise blend.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct M4(pub [u32; 4]);

/// Four u32 lanes.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct U4(pub [u32; 4]);

/// The four lanes of `a` through `f`.
#[inline(always)]
fn map(a: F4, f: impl Fn(f32) -> f32) -> F4 {
    let mut r = [0f32; 4];
    for i in 0..4 {
        r[i] = f(a.0[i]);
    }
    F4(r)
}

/// The four lanes of `a` and `b` through `f`.
#[inline(always)]
fn zip(a: F4, b: F4, f: impl Fn(f32, f32) -> f32) -> F4 {
    let mut r = [0f32; 4];
    for i in 0..4 {
        r[i] = f(a.0[i], b.0[i]);
    }
    F4(r)
}

/// A lane compare as a mask.
#[inline(always)]
fn cmp(a: F4, b: F4, f: impl Fn(f32, f32) -> bool) -> M4 {
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = if f(a.0[i], b.0[i]) { !0 } else { 0 };
    }
    M4(r)
}

/// The four lanes of `a` and `b` through `f`, as integers.
#[inline(always)]
fn zip_u(a: U4, b: U4, f: impl Fn(u32, u32) -> u32) -> U4 {
    let mut r = [0u32; 4];
    for i in 0..4 {
        r[i] = f(a.0[i], b.0[i]);
    }
    U4(r)
}

macro_rules! f4_op {
    ($tr:ident, $f:ident, $op:tt) => {
        impl $tr for F4 {
            type Output = F4;
            #[inline(always)]
            fn $f(self, o: F4) -> F4 {
                zip(self, o, |a, b| a $op b)
            }
        }
        impl $tr<f32> for F4 {
            type Output = F4;
            #[inline(always)]
            fn $f(self, o: f32) -> F4 {
                map(self, |a| a $op o)
            }
        }
        impl $tr<F4> for f32 {
            type Output = F4;
            #[inline(always)]
            fn $f(self, o: F4) -> F4 {
                map(o, |b| self $op b)
            }
        }
    };
}
f4_op!(Add, add, +);
f4_op!(Sub, sub, -);
f4_op!(Mul, mul, *);
f4_op!(Div, div, /);

impl Neg for F4 {
    type Output = F4;
    #[inline(always)]
    fn neg(self) -> F4 {
        map(self, |a| -a)
    }
}

macro_rules! bit_op {
    ($ty:ident, $tr:ident, $f:ident, $op:tt) => {
        impl $tr for $ty {
            type Output = $ty;
            #[inline(always)]
            fn $f(self, o: $ty) -> $ty {
                let mut r = [0u32; 4];
                for i in 0..4 {
                    r[i] = self.0[i] $op o.0[i];
                }
                $ty(r)
            }
        }
    };
}
bit_op!(M4, BitAnd, bitand, &);
bit_op!(M4, BitOr, bitor, |);
bit_op!(U4, BitAnd, bitand, &);
bit_op!(U4, BitOr, bitor, |);
bit_op!(U4, BitXor, bitxor, ^);

impl Not for M4 {
    type Output = M4;
    #[inline(always)]
    fn not(self) -> M4 {
        M4(self.0.map(|a| !a))
    }
}

impl Shl<u32> for U4 {
    type Output = U4;
    #[inline(always)]
    fn shl(self, s: u32) -> U4 {
        U4(self.0.map(|a| a << s))
    }
}

impl Shr<u32> for U4 {
    type Output = U4;
    #[inline(always)]
    fn shr(self, s: u32) -> U4 {
        U4(self.0.map(|a| a >> s))
    }
}

impl Mask for M4 {
    #[inline(always)]
    fn splat(b: bool) -> M4 {
        M4([if b { !0 } else { 0 }; 4])
    }
    #[inline(always)]
    fn any(self) -> bool {
        (self.0[0] | self.0[1] | self.0[2] | self.0[3]) != 0
    }
}

impl Bits for U4 {
    #[inline(always)]
    fn splat(v: u32) -> U4 {
        U4([v; 4])
    }
    #[inline(always)]
    fn wrapping_mul(self, o: U4) -> U4 {
        zip_u(self, o, u32::wrapping_mul)
    }
}

/// Round to the nearest integer as a float, by adding and taking away 1.5·2²³ (exact for |x| < 2²²).
#[inline(always)]
fn round(x: f32) -> f32 {
    let magic = 12_582_912f32;
    (x + magic) - magic
}

/// √x for one lane, from plain arithmetic: a reciprocal square root from the bits (the exponent halved by a shift, against Lomont's constant, within 0.18%), two Newton steps to f32's precision, times x, and one Newton step on the root itself; within one ulp of the correctly rounded root for every normal x (`sqrt_within_an_ulp`; every normal float was checked once, 2026-10-29: 13% one ulp off, none further), exact at 0, and NaN below 0 as IEEE's is, so a negative argument still shows instead of quietly rounding to a colour. Subnormal x (below 2⁻¹²⁶) come out wrong: the shading's roots are of sums of squares that would need an orbit within 2⁻⁶³ of 0 to get there.
/// WHY not the platform's: without std there is no square root in core but libm's, which is one `fsqrt` in inline assembly per lane, and LLVM cannot vectorise through it; this one is four lanes of multiplies, and gives the same bits on every platform, as fm's functions do.
#[inline(always)]
fn sqrt1(x: f32) -> f32 {
    let y = f32::from_bits(0x5f37_5a86u32.wrapping_sub(x.to_bits() >> 1));
    let h = x * 0.5;
    let y = y * (1.5 - h * y * y);
    let y = y * (1.5 - h * y * y);
    let s = x * y;
    let s = s + y * 0.5 * (x - s * s);
    if x < 0. { f32::NAN } else { s }
}

impl Lane for F4 {
    const LANES: usize = 4;
    type M = M4;
    type U = U4;
    #[inline(always)]
    fn splat(v: f32) -> F4 {
        F4([v; 4])
    }
    #[inline(always)]
    fn sqrt(self) -> F4 {
        map(self, sqrt1)
    }
    /// One vector instruction with std (LLVM's square root); without it libm's `fsqrt` four times, once per sample, never inside the orbit's loop.
    #[inline(always)]
    fn sqrt_rounded(self) -> F4 {
        map(self, rounded_sqrt)
    }
    #[inline(always)]
    fn abs(self) -> F4 {
        map(self, f32::abs)
    }
    #[inline(always)]
    fn min(self, o: F4) -> F4 {
        zip(self, o, f32::min)
    }
    #[inline(always)]
    fn max(self, o: F4) -> F4 {
        zip(self, o, f32::max)
    }
    /// floor by rounding and stepping down where that rounded up; exact for |x| < 2²², and every use (the enamel's cell indices, at most a few thousand) is.
    #[inline(always)]
    fn floor(self) -> F4 {
        map(self, |x| {
            let r = round(x);
            if r > x { r - 1. } else { r }
        })
    }
    /// By squaring and multiplying, as compiler-rt's `__powisf2` does, so a power of two (every power the shading takes) is the same chain of squares as the scalar's.
    #[inline(always)]
    fn powi(self, n: i32) -> F4 {
        let (mut a, mut b, mut r) = (self, n.unsigned_abs(), F4::splat(1.));
        loop {
            if b & 1 == 1 {
                r = r * a;
            }
            b >>= 1;
            if b == 0 {
                break;
            }
            a = a * a;
        }
        if n < 0 { 1. / r } else { r }
    }
    #[inline(always)]
    fn gt(self, o: F4) -> M4 {
        cmp(self, o, |a, b| a > b)
    }
    #[inline(always)]
    fn lt(self, o: F4) -> M4 {
        cmp(self, o, |a, b| a < b)
    }
    #[inline(always)]
    fn ge(self, o: F4) -> M4 {
        cmp(self, o, |a, b| a >= b)
    }
    #[inline(always)]
    fn select(m: M4, a: F4, b: F4) -> F4 {
        let mut r = [0f32; 4];
        for i in 0..4 {
            r[i] = f32::from_bits((a.0[i].to_bits() & m.0[i]) | (b.0[i].to_bits() & !m.0[i]));
        }
        F4(r)
    }
    #[inline(always)]
    fn to_bits(self) -> U4 {
        U4(self.0.map(f32::to_bits))
    }
    #[inline(always)]
    fn from_bits(u: U4) -> F4 {
        F4(u.0.map(f32::from_bits))
    }
    #[inline(always)]
    fn to_u32(self) -> U4 {
        U4(self.0.map(|a| a as u32))
    }
    #[inline(always)]
    fn to_i32(self) -> U4 {
        U4(self.0.map(|a| a as i32 as u32))
    }
    #[inline(always)]
    fn from_u32(u: U4) -> F4 {
        F4(u.0.map(|a| a as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lanes' square root against the scalar's, correctly rounded one, at 0 and across the normal floats: never more than one ulp apart.
    #[test]
    fn sqrt_within_an_ulp() {
        assert_eq!(sqrt1(0.), 0.);
        let mut worst = 0i64;
        assert!(sqrt1(-1.).is_nan());
        // Every 2⁴th float runs quickly in a debug build and still crosses every exponent and mantissa pattern many times.
        let mut b = 1u32 << 23;
        while b < 0x7F80_0000 {
            let x = f32::from_bits(b);
            worst = worst.max((sqrt1(x).to_bits() as i64 - <f32 as Lane>::sqrt(x).to_bits() as i64).abs());
            b += 1 << 4;
        }
        assert!(worst <= 1, "worst {worst} ulp");
    }

    /// Four lanes agree with four scalars bit for bit on fm's functions over a spread of arguments: every function is the same arithmetic in each lane.
    #[test]
    fn lanes_match_scalars() {
        use crate::fm::Fast;
        for i in 0..(1u32 << 12) {
            let u = i as f32 / (1u32 << 12) as f32;
            let x = F4([(u - 0.5) * 64., u * 4096. + 1e-6, -u, u * u - 0.5]);
            let y = F4([u - 0.5, 0.5 - u, u, -u]);
            let lanes = [x.fexp2(), x.flog2(), x.fsin(), x.fcos(), x.ftanh(), x.fexp(), x.fln(), x.fatan2(y)];
            for l in 0..4 {
                let (a, b) = (x.0[l], y.0[l]);
                let scalars = [a.fexp2(), a.flog2(), a.fsin(), a.fcos(), a.ftanh(), a.fexp(), a.fln(), a.fatan2(b)];
                for k in 0..scalars.len() {
                    assert_eq!(lanes[k].0[l].to_bits(), scalars[k].to_bits(), "function {k}, lane {l}, x {a}");
                }
            }
        }
    }
}
