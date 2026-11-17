//! Four samples at a time: the shading is written once, generic over `Lane`, and runs on `f32` (one sample, the scalar path) or on `F4` (four samples in the lanes of one 128-bit register: NEON on the phone's cores, SSE2 on a desktop).
//! Every `F4` operation is one vector instruction through `core::arch` (or a short fixed sequence where SSE2 lacks one), the same IEEE operation the scalar does in each lane, so four lanes give the scalar path's bits (`lanes_match_scalars`, `lane_ops_are_the_scalars`); on any other target the lanes are an array and a loop. Nothing in the shading names an instruction, so it is still ready to become a GPU shader (`gpu`).
//! Until 2026-11-17 the operations were plain loops left to LLVM's SLP vectoriser, which never vectorised the shading (x86-64: `shade::<F4>` was all `mulss`, no `mulps`): four lanes ran as four scalars.
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
    /// √x, within an ulp (on the CPU correctly rounded, one lane or four; the GPU's unit is an ulp from it).
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
    /// 2ˣ and log₂ x, as hooks: the render's own polynomials (`fm`) by default, identical on every CPU; a GPU lane type puts its special function unit there (ferros valhall: within an ulp, a fraction of the instructions).
    #[inline(always)]
    fn exp2_lane(self) -> Self {
        crate::fm::exp2_poly(self)
    }
    #[inline(always)]
    fn log2_lane(self) -> Self {
        crate::fm::log2_poly(self)
    }
    /// The orbit's loop, as a hook (owner, 2026-11-05: the attest mark on the GPU, one source): pass n (0 to `max` − 1, as a lane) through `head`, leave with its state when no lane is still running, otherwise through `tail`. On the CPU this is the loop the orbit always ran; a GPU lane type records `head` and `tail` once and makes the leaving a select (valhall::ssa::iterate).
    #[inline(always)]
    fn iterate<S: Copy>(max: u32, init: S, mut head: impl FnMut(Self, S) -> (S, Self::M), mut tail: impl FnMut(Self, S) -> S) -> S {
        let mut s = init;
        for n in 0..max {
            let (h, running) = head(Self::splat(n as f32), s);
            if !running.any() {
                return h;
            }
            s = tail(Self::splat(n as f32), h);
        }
        s
    }
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


/// Four f32 lanes in one 128-bit register: NEON's on aarch64, SSE2's on x86-64, an array elsewhere (`v`).
#[derive(Clone, Copy)]
pub struct F4(v::F);

/// Four lane truths, each all ones (true) or all zeros, as a vector compare leaves them, so a select is a bitwise blend.
#[derive(Clone, Copy)]
pub struct M4(v::U);

/// Four u32 lanes.
#[derive(Clone, Copy)]
pub struct U4(v::U);

impl F4 {
    /// The four lanes from an array, lane i from element i.
    #[inline(always)]
    pub fn new(a: [f32; 4]) -> F4 {
        F4(v::from_array(a))
    }
    /// The four lanes as an array.
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        v::to_array(self.0)
    }
}

/// The lane operations on the platform's own vectors. Every one is the IEEE operation the scalar `f32` does in each lane (no fused multiply-add, no estimate), so four lanes give the scalar path's bits; `lanes_match_scalars` holds them to it.
/// WHY not plain loops over four elements (until 2026-11-17): LLVM's SLP vectoriser never turned the shading's loops into vector instructions (measured on x86-64: `shade::<F4>` was 2845 `mulss` and no `mulps`), so four lanes ran as four scalars, slower than one.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod v {
    //! SAFETY, for every intrinsic called below: each needs NEON alone, which the cfg on this module requires the target to have.
    use core::arch::aarch64::*;
    pub type F = float32x4_t;
    pub type U = uint32x4_t;
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> F {
        // SAFETY: [f32; 4] and float32x4_t are both 16 bytes of four f32s, lane i at element i, and every bit pattern is valid in both.
        unsafe { core::mem::transmute(a) }
    }
    #[inline(always)]
    pub fn to_array(a: F) -> [f32; 4] {
        // SAFETY: as `from_array`.
        unsafe { core::mem::transmute(a) }
    }
    #[inline(always)]
    pub fn splat(x: f32) -> F {
        unsafe { vdupq_n_f32(x) }
    }
    #[inline(always)]
    pub fn add(a: F, b: F) -> F {
        unsafe { vaddq_f32(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: F, b: F) -> F {
        unsafe { vsubq_f32(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: F, b: F) -> F {
        unsafe { vmulq_f32(a, b) }
    }
    #[inline(always)]
    pub fn div(a: F, b: F) -> F {
        unsafe { vdivq_f32(a, b) }
    }
    #[inline(always)]
    pub fn neg(a: F) -> F {
        unsafe { vnegq_f32(a) }
    }
    #[inline(always)]
    pub fn sqrt(a: F) -> F {
        unsafe { vsqrtq_f32(a) }
    }
    #[inline(always)]
    pub fn abs(a: F) -> F {
        unsafe { vabsq_f32(a) }
    }
    /// fminnm and fmaxnm: what `f32::min` and `f32::max` compile to here.
    #[inline(always)]
    pub fn min(a: F, b: F) -> F {
        unsafe { vminnmq_f32(a, b) }
    }
    #[inline(always)]
    pub fn max(a: F, b: F) -> F {
        unsafe { vmaxnmq_f32(a, b) }
    }
    #[inline(always)]
    pub fn gt(a: F, b: F) -> U {
        unsafe { vcgtq_f32(a, b) }
    }
    #[inline(always)]
    pub fn lt(a: F, b: F) -> U {
        unsafe { vcltq_f32(a, b) }
    }
    #[inline(always)]
    pub fn ge(a: F, b: F) -> U {
        unsafe { vcgeq_f32(a, b) }
    }
    #[inline(always)]
    pub fn select(m: U, a: F, b: F) -> F {
        unsafe { vbslq_f32(m, a, b) }
    }
    #[inline(always)]
    pub fn to_bits(a: F) -> U {
        unsafe { vreinterpretq_u32_f32(a) }
    }
    #[inline(always)]
    pub fn from_bits(u: U) -> F {
        unsafe { vreinterpretq_f32_u32(u) }
    }
    /// fcvtzu and fcvtzs saturate and take NaN to 0, as Rust's `as` does.
    #[inline(always)]
    pub fn to_u32(a: F) -> U {
        unsafe { vcvtq_u32_f32(a) }
    }
    #[inline(always)]
    pub fn to_i32(a: F) -> U {
        unsafe { vreinterpretq_u32_s32(vcvtq_s32_f32(a)) }
    }
    #[inline(always)]
    pub fn from_u32(u: U) -> F {
        unsafe { vcvtq_f32_u32(u) }
    }
    #[inline(always)]
    pub fn usplat(x: u32) -> U {
        unsafe { vdupq_n_u32(x) }
    }
    #[inline(always)]
    pub fn and(a: U, b: U) -> U {
        unsafe { vandq_u32(a, b) }
    }
    #[inline(always)]
    pub fn or(a: U, b: U) -> U {
        unsafe { vorrq_u32(a, b) }
    }
    #[inline(always)]
    pub fn xor(a: U, b: U) -> U {
        unsafe { veorq_u32(a, b) }
    }
    #[inline(always)]
    pub fn not(a: U) -> U {
        unsafe { vmvnq_u32(a) }
    }
    #[inline(always)]
    pub fn shl(a: U, s: u32) -> U {
        unsafe { vshlq_u32(a, vdupq_n_s32(s as i32)) }
    }
    /// A right shift is ushl by the negated count.
    #[inline(always)]
    pub fn shr(a: U, s: u32) -> U {
        unsafe { vshlq_u32(a, vdupq_n_s32(-(s as i32))) }
    }
    #[inline(always)]
    pub fn umul(a: U, b: U) -> U {
        unsafe { vmulq_u32(a, b) }
    }
    #[inline(always)]
    pub fn any(m: U) -> bool {
        unsafe { vmaxvq_u32(m) != 0 }
    }
    #[inline(always)]
    #[cfg(test)]
    pub fn uto_array(a: U) -> [u32; 4] {
        // SAFETY: as `from_array`, four u32s.
        unsafe { core::mem::transmute(a) }
    }
    #[cfg(test)]
    pub fn ufrom_array(a: [u32; 4]) -> U {
        // SAFETY: as `from_array`, four u32s.
        unsafe { core::mem::transmute(a) }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod v {
    //! SAFETY, for every intrinsic called below: each needs SSE2 alone, which every x86-64 has and the cfg on this module requires.
    use core::arch::x86_64::*;
    pub type F = __m128;
    pub type U = __m128i;
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> F {
        // SAFETY: [f32; 4] and __m128 are both 16 bytes of four f32s, lane i at element i, and every bit pattern is valid in both.
        unsafe { core::mem::transmute(a) }
    }
    #[inline(always)]
    pub fn to_array(a: F) -> [f32; 4] {
        // SAFETY: as `from_array`.
        unsafe { core::mem::transmute(a) }
    }
    #[inline(always)]
    pub fn splat(x: f32) -> F {
        unsafe { _mm_set1_ps(x) }
    }
    #[inline(always)]
    pub fn add(a: F, b: F) -> F {
        unsafe { _mm_add_ps(a, b) }
    }
    #[inline(always)]
    pub fn sub(a: F, b: F) -> F {
        unsafe { _mm_sub_ps(a, b) }
    }
    #[inline(always)]
    pub fn mul(a: F, b: F) -> F {
        unsafe { _mm_mul_ps(a, b) }
    }
    #[inline(always)]
    pub fn div(a: F, b: F) -> F {
        unsafe { _mm_div_ps(a, b) }
    }
    #[inline(always)]
    pub fn neg(a: F) -> F {
        unsafe { _mm_xor_ps(a, _mm_set1_ps(-0.)) }
    }
    #[inline(always)]
    pub fn sqrt(a: F) -> F {
        unsafe { _mm_sqrt_ps(a) }
    }
    #[inline(always)]
    pub fn abs(a: F) -> F {
        unsafe { _mm_andnot_ps(_mm_set1_ps(-0.), a) }
    }
    /// `f32::min` as LLVM lowers it here: minps returns its second operand when either is NaN (or both are zeros), so min(b, a) gives a then, and a NaN a gives way to b.
    #[inline(always)]
    pub fn min(a: F, b: F) -> F {
        unsafe { select(_mm_castps_si128(_mm_cmpunord_ps(a, a)), b, _mm_min_ps(b, a)) }
    }
    #[inline(always)]
    pub fn max(a: F, b: F) -> F {
        unsafe { select(_mm_castps_si128(_mm_cmpunord_ps(a, a)), b, _mm_max_ps(b, a)) }
    }
    #[inline(always)]
    pub fn gt(a: F, b: F) -> U {
        unsafe { _mm_castps_si128(_mm_cmpgt_ps(a, b)) }
    }
    #[inline(always)]
    pub fn lt(a: F, b: F) -> U {
        unsafe { _mm_castps_si128(_mm_cmplt_ps(a, b)) }
    }
    #[inline(always)]
    pub fn ge(a: F, b: F) -> U {
        unsafe { _mm_castps_si128(_mm_cmpge_ps(a, b)) }
    }
    #[inline(always)]
    pub fn select(m: U, a: F, b: F) -> F {
        unsafe {
            let m = _mm_castsi128_ps(m);
            _mm_or_ps(_mm_and_ps(m, a), _mm_andnot_ps(m, b))
        }
    }
    #[inline(always)]
    pub fn to_bits(a: F) -> U {
        unsafe { _mm_castps_si128(a) }
    }
    #[inline(always)]
    pub fn from_bits(u: U) -> F {
        unsafe { _mm_castsi128_ps(u) }
    }
    /// `as u32`, saturating with NaN to 0, from SSE2's signed conversion: NaN and negatives to 0 by the max (maxps gives its second operand, 0, for a NaN), from 2³¹ by converting x − 2³¹ (exact there, every f32 that large being a multiple of 2⁸) and setting the top bit, and from 2³² all ones.
    #[inline(always)]
    pub fn to_u32(a: F) -> U {
        unsafe {
            let x = _mm_max_ps(a, _mm_setzero_ps());
            let two31 = _mm_set1_ps(2_147_483_648.);
            let lo = _mm_cvttps_epi32(x);
            let hi = _mm_xor_si128(_mm_cvttps_epi32(_mm_sub_ps(x, two31)), _mm_set1_epi32(i32::MIN));
            let big = _mm_castps_si128(_mm_cmpge_ps(x, two31));
            let over = _mm_castps_si128(_mm_cmpge_ps(x, _mm_set1_ps(4_294_967_296.)));
            or(over, or(and(big, hi), _mm_andnot_si128(big, lo)))
        }
    }
    /// `as i32`: cvttps2dq gives i32::MIN for NaN and for every overflow; Rust's gives 0 for NaN and i32::MAX past the top.
    #[inline(always)]
    pub fn to_i32(a: F) -> U {
        unsafe {
            let c = _mm_cvttps_epi32(a);
            let c = and(c, _mm_castps_si128(_mm_cmpord_ps(a, a)));
            let top = _mm_castps_si128(_mm_cmpge_ps(a, _mm_set1_ps(2_147_483_648.)));
            or(and(top, _mm_set1_epi32(i32::MAX)), _mm_andnot_si128(top, c))
        }
    }
    /// `u as f32`, correctly rounded: the top and bottom 16 bits converted exactly (both under 2²⁴), the top scaled by 2¹⁶ exactly, and the one rounding in their sum.
    #[inline(always)]
    pub fn from_u32(u: U) -> F {
        unsafe {
            let hi = _mm_cvtepi32_ps(_mm_srli_epi32::<16>(u));
            let lo = _mm_cvtepi32_ps(_mm_and_si128(u, _mm_set1_epi32(0xFFFF)));
            _mm_add_ps(_mm_mul_ps(hi, _mm_set1_ps(65_536.)), lo)
        }
    }
    #[inline(always)]
    pub fn usplat(x: u32) -> U {
        unsafe { _mm_set1_epi32(x as i32) }
    }
    #[inline(always)]
    pub fn and(a: U, b: U) -> U {
        unsafe { _mm_and_si128(a, b) }
    }
    #[inline(always)]
    pub fn or(a: U, b: U) -> U {
        unsafe { _mm_or_si128(a, b) }
    }
    #[inline(always)]
    pub fn xor(a: U, b: U) -> U {
        unsafe { _mm_xor_si128(a, b) }
    }
    #[inline(always)]
    pub fn not(a: U) -> U {
        unsafe { _mm_xor_si128(a, _mm_set1_epi32(-1)) }
    }
    #[inline(always)]
    pub fn shl(a: U, s: u32) -> U {
        unsafe { _mm_sll_epi32(a, _mm_cvtsi32_si128(s as i32)) }
    }
    #[inline(always)]
    pub fn shr(a: U, s: u32) -> U {
        unsafe { _mm_srl_epi32(a, _mm_cvtsi32_si128(s as i32)) }
    }
    /// SSE2 has no 32-bit lane multiply: lanes 0 and 2, then 1 and 3, as 64-bit products, their low halves put back in order.
    #[inline(always)]
    pub fn umul(a: U, b: U) -> U {
        unsafe {
            let even = _mm_mul_epu32(a, b);
            let odd = _mm_mul_epu32(_mm_srli_epi64::<32>(a), _mm_srli_epi64::<32>(b));
            _mm_unpacklo_epi32(_mm_shuffle_epi32::<0b00_00_10_00>(even), _mm_shuffle_epi32::<0b00_00_10_00>(odd))
        }
    }
    #[inline(always)]
    pub fn any(m: U) -> bool {
        unsafe { _mm_movemask_ps(_mm_castsi128_ps(m)) != 0 }
    }
    #[inline(always)]
    #[cfg(test)]
    pub fn uto_array(a: U) -> [u32; 4] {
        // SAFETY: as `from_array`, four u32s.
        unsafe { core::mem::transmute(a) }
    }
    #[cfg(test)]
    pub fn ufrom_array(a: [u32; 4]) -> U {
        // SAFETY: as `from_array`, four u32s.
        unsafe { core::mem::transmute(a) }
    }
}

/// Elsewhere, four elements and a loop per operation, each the scalar's own.
#[cfg(not(any(all(target_arch = "aarch64", target_feature = "neon"), all(target_arch = "x86_64", target_feature = "sse2"))))]
mod v {
    pub type F = [f32; 4];
    pub type U = [u32; 4];
    #[inline(always)]
    fn zip(a: F, b: F, f: impl Fn(f32, f32) -> f32) -> F {
        [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
    }
    #[inline(always)]
    fn uzip(a: U, b: U, f: impl Fn(u32, u32) -> u32) -> U {
        [f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]), f(a[3], b[3])]
    }
    #[inline(always)]
    fn cmp(a: F, b: F, f: impl Fn(f32, f32) -> bool) -> U {
        [0, 1, 2, 3].map(|i| if f(a[i], b[i]) { !0 } else { 0 })
    }
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> F {
        a
    }
    #[inline(always)]
    pub fn to_array(a: F) -> [f32; 4] {
        a
    }
    #[inline(always)]
    pub fn splat(x: f32) -> F {
        [x; 4]
    }
    #[inline(always)]
    pub fn add(a: F, b: F) -> F {
        zip(a, b, |x, y| x + y)
    }
    #[inline(always)]
    pub fn sub(a: F, b: F) -> F {
        zip(a, b, |x, y| x - y)
    }
    #[inline(always)]
    pub fn mul(a: F, b: F) -> F {
        zip(a, b, |x, y| x * y)
    }
    #[inline(always)]
    pub fn div(a: F, b: F) -> F {
        zip(a, b, |x, y| x / y)
    }
    #[inline(always)]
    pub fn neg(a: F) -> F {
        a.map(|x| -x)
    }
    #[inline(always)]
    pub fn sqrt(a: F) -> F {
        a.map(super::rounded_sqrt)
    }
    #[inline(always)]
    pub fn abs(a: F) -> F {
        a.map(f32::abs)
    }
    #[inline(always)]
    pub fn min(a: F, b: F) -> F {
        zip(a, b, f32::min)
    }
    #[inline(always)]
    pub fn max(a: F, b: F) -> F {
        zip(a, b, f32::max)
    }
    #[inline(always)]
    pub fn gt(a: F, b: F) -> U {
        cmp(a, b, |x, y| x > y)
    }
    #[inline(always)]
    pub fn lt(a: F, b: F) -> U {
        cmp(a, b, |x, y| x < y)
    }
    #[inline(always)]
    pub fn ge(a: F, b: F) -> U {
        cmp(a, b, |x, y| x >= y)
    }
    #[inline(always)]
    pub fn select(m: U, a: F, b: F) -> F {
        [0, 1, 2, 3].map(|i| f32::from_bits((a[i].to_bits() & m[i]) | (b[i].to_bits() & !m[i])))
    }
    #[inline(always)]
    pub fn to_bits(a: F) -> U {
        a.map(f32::to_bits)
    }
    #[inline(always)]
    pub fn from_bits(u: U) -> F {
        u.map(f32::from_bits)
    }
    #[inline(always)]
    pub fn to_u32(a: F) -> U {
        a.map(|x| x as u32)
    }
    #[inline(always)]
    pub fn to_i32(a: F) -> U {
        a.map(|x| x as i32 as u32)
    }
    #[inline(always)]
    pub fn from_u32(u: U) -> F {
        u.map(|x| x as f32)
    }
    #[inline(always)]
    pub fn usplat(x: u32) -> U {
        [x; 4]
    }
    #[inline(always)]
    pub fn and(a: U, b: U) -> U {
        uzip(a, b, |x, y| x & y)
    }
    #[inline(always)]
    pub fn or(a: U, b: U) -> U {
        uzip(a, b, |x, y| x | y)
    }
    #[inline(always)]
    pub fn xor(a: U, b: U) -> U {
        uzip(a, b, |x, y| x ^ y)
    }
    #[inline(always)]
    pub fn not(a: U) -> U {
        a.map(|x| !x)
    }
    #[inline(always)]
    pub fn shl(a: U, s: u32) -> U {
        a.map(|x| x << s)
    }
    #[inline(always)]
    pub fn shr(a: U, s: u32) -> U {
        a.map(|x| x >> s)
    }
    #[inline(always)]
    pub fn umul(a: U, b: U) -> U {
        uzip(a, b, u32::wrapping_mul)
    }
    #[inline(always)]
    pub fn any(m: U) -> bool {
        (m[0] | m[1] | m[2] | m[3]) != 0
    }
    #[inline(always)]
    #[cfg(test)]
    pub fn uto_array(a: U) -> [u32; 4] {
        a
    }
    #[cfg(test)]
    pub fn ufrom_array(a: [u32; 4]) -> U {
        a
    }
}

macro_rules! f4_op {
    ($tr:ident, $f:ident, $op:ident) => {
        impl $tr for F4 {
            type Output = F4;
            #[inline(always)]
            fn $f(self, o: F4) -> F4 {
                F4(v::$op(self.0, o.0))
            }
        }
        impl $tr<f32> for F4 {
            type Output = F4;
            #[inline(always)]
            fn $f(self, o: f32) -> F4 {
                F4(v::$op(self.0, v::splat(o)))
            }
        }
        impl $tr<F4> for f32 {
            type Output = F4;
            #[inline(always)]
            fn $f(self, o: F4) -> F4 {
                F4(v::$op(v::splat(self), o.0))
            }
        }
    };
}
f4_op!(Add, add, add);
f4_op!(Sub, sub, sub);
f4_op!(Mul, mul, mul);
f4_op!(Div, div, div);

impl Neg for F4 {
    type Output = F4;
    #[inline(always)]
    fn neg(self) -> F4 {
        F4(v::neg(self.0))
    }
}

macro_rules! bit_op {
    ($ty:ident, $tr:ident, $f:ident, $op:ident) => {
        impl $tr for $ty {
            type Output = $ty;
            #[inline(always)]
            fn $f(self, o: $ty) -> $ty {
                $ty(v::$op(self.0, o.0))
            }
        }
    };
}
bit_op!(M4, BitAnd, bitand, and);
bit_op!(M4, BitOr, bitor, or);
bit_op!(U4, BitAnd, bitand, and);
bit_op!(U4, BitOr, bitor, or);
bit_op!(U4, BitXor, bitxor, xor);

impl Not for M4 {
    type Output = M4;
    #[inline(always)]
    fn not(self) -> M4 {
        M4(v::not(self.0))
    }
}

impl Shl<u32> for U4 {
    type Output = U4;
    #[inline(always)]
    fn shl(self, s: u32) -> U4 {
        U4(v::shl(self.0, s))
    }
}

impl Shr<u32> for U4 {
    type Output = U4;
    #[inline(always)]
    fn shr(self, s: u32) -> U4 {
        U4(v::shr(self.0, s))
    }
}

impl Mask for M4 {
    #[inline(always)]
    fn splat(b: bool) -> M4 {
        M4(v::usplat(if b { !0 } else { 0 }))
    }
    #[inline(always)]
    fn any(self) -> bool {
        v::any(self.0)
    }
}

impl Bits for U4 {
    #[inline(always)]
    fn splat(x: u32) -> U4 {
        U4(v::usplat(x))
    }
    #[inline(always)]
    fn wrapping_mul(self, o: U4) -> U4 {
        U4(v::umul(self.0, o.0))
    }
}

impl Lane for F4 {
    const LANES: usize = 4;
    type M = M4;
    type U = U4;
    #[inline(always)]
    fn splat(x: f32) -> F4 {
        F4(v::splat(x))
    }
    /// One vector instruction (fsqrt, sqrtps), correctly rounded as the scalar's is, with or without std.
    /// WHY not the bit trick it was until 2026-11-17 (a reciprocal root from Lomont's constant and three Newton steps, within an ulp): it was there because without std the lanes had no vector root (libm's is one `fsqrt` a lane), which `core::arch` gives; the instruction is faster (measured on x86-64: the 480 px avatar 198 ms to 127 ms), and correctly rounded it gives the scalar path's bits, so Photon's avatars drawn four lanes at a time hash as they did. It moved ahau's CPU attest mark by one Q14 step on 0.54% of its pixels (the rest identical).
    #[inline(always)]
    fn sqrt(self) -> F4 {
        F4(v::sqrt(self.0))
    }
    #[inline(always)]
    fn sqrt_rounded(self) -> F4 {
        F4(v::sqrt(self.0))
    }
    #[inline(always)]
    fn abs(self) -> F4 {
        F4(v::abs(self.0))
    }
    #[inline(always)]
    fn min(self, o: F4) -> F4 {
        F4(v::min(self.0, o.0))
    }
    #[inline(always)]
    fn max(self, o: F4) -> F4 {
        F4(v::max(self.0, o.0))
    }
    /// floor by rounding (adding and taking away 1.5·2²³) and stepping down where that rounded up; exact for |x| < 2²², and every use (the enamel's cell indices, at most a few thousand) is.
    #[inline(always)]
    fn floor(self) -> F4 {
        let magic = 12_582_912f32;
        let r = (self + magic) - magic;
        F4::select(r.gt(self), r - 1., r)
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
        M4(v::gt(self.0, o.0))
    }
    #[inline(always)]
    fn lt(self, o: F4) -> M4 {
        M4(v::lt(self.0, o.0))
    }
    #[inline(always)]
    fn ge(self, o: F4) -> M4 {
        M4(v::ge(self.0, o.0))
    }
    #[inline(always)]
    fn select(m: M4, a: F4, b: F4) -> F4 {
        F4(v::select(m.0, a.0, b.0))
    }
    #[inline(always)]
    fn to_bits(self) -> U4 {
        U4(v::to_bits(self.0))
    }
    #[inline(always)]
    fn from_bits(u: U4) -> F4 {
        F4(v::from_bits(u.0))
    }
    #[inline(always)]
    fn to_u32(self) -> U4 {
        U4(v::to_u32(self.0))
    }
    #[inline(always)]
    fn to_i32(self) -> U4 {
        U4(v::to_i32(self.0))
    }
    #[inline(always)]
    fn from_u32(u: U4) -> F4 {
        F4(v::from_u32(u.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lanes' square root is the scalar's, correctly rounded, bit for bit, at both zeros, across the floats (subnormals too), at infinity, and NaN below 0.
    #[test]
    fn sqrt_is_the_scalars() {
        // Every 2⁴th float runs quickly in a debug build and still crosses every exponent and mantissa pattern many times.
        let mut b = 0u32;
        while b <= 0x7F80_0000 {
            let x = f32::from_bits(b);
            let four = F4::new([x, x * 0.25, x * 4., -x - 1.]).sqrt().to_array();
            for (l, x) in [x, x * 0.25, x * 4.].into_iter().enumerate() {
                assert_eq!(four[l].to_bits(), <f32 as Lane>::sqrt(x).to_bits(), "√{x}");
            }
            assert!(four[3].is_nan());
            b += 1 << 4;
        }
        assert_eq!(F4::splat(-0.).sqrt().to_array()[0].to_bits(), (-0f32).to_bits());
    }

    /// Four lanes agree with four scalars bit for bit on fm's functions over a spread of arguments: every function is the same arithmetic in each lane.
    #[test]
    fn lanes_match_scalars() {
        use crate::fm::Fast;
        for i in 0..(1u32 << 12) {
            let u = i as f32 / (1u32 << 12) as f32;
            let x = F4::new([(u - 0.5) * 64., u * 4096. + 1e-6, -u, u * u - 0.5]);
            let y = F4::new([u - 0.5, 0.5 - u, u, -u]);
            let lanes = [x.fexp2(), x.flog2(), x.fsin(), x.fcos(), x.ftanh(), x.fexp(), x.fln(), x.fatan2(y)].map(F4::to_array);
            for l in 0..4 {
                let (a, b) = (x.to_array()[l], y.to_array()[l]);
                let scalars = [a.fexp2(), a.flog2(), a.fsin(), a.fcos(), a.ftanh(), a.fexp(), a.fln(), a.fatan2(b)];
                for k in 0..scalars.len() {
                    assert_eq!(lanes[k][l].to_bits(), scalars[k].to_bits(), "function {k}, lane {l}, x {a}");
                }
            }
        }
    }

    /// Each lane operation against the scalar's, over values that reach every edge it has: NaN, both zeros, infinities, negatives, past 2³¹ and 2³², and ordinary ones.
    #[test]
    fn lane_ops_are_the_scalars() {
        let edge = [f32::NAN, -f32::NAN, 0., -0., 1., -1., 0.5, -2.5, 3.75, f32::INFINITY, f32::NEG_INFINITY, 2_147_483_648., 2_147_483_520., 4_294_967_296., 4_294_967_040., -2_147_483_904., 1e-40, 12_582_911.5, f32::MAX, f32::MIN_POSITIVE];
        let same = |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        for &x in &edge {
            for &y in &edge {
                let (a, b) = (F4::splat(x), F4::splat(y));
                let one = |f: F4| f.to_array()[0];
                let u = |m: U4| v::uto_array(m.0)[0];
                // Of two zeros Rust leaves which one min and max give unspecified (its own build gives either), and no use here can tell them apart: the clamp of exp2's argument and atan2's ratio come out the same from either.
                let zeros = x == 0. && y == 0.;
                assert!(zeros || same(one(a.min(b)), x.min(y)), "min {x} {y}");
                assert!(zeros || same(one(a.max(b)), x.max(y)), "max {x} {y}");
                assert!(same(one(a + b), x + y) && same(one(a - b), x - y) && same(one(a * b), x * y) && same(one(a / b), x / y), "arith {x} {y}");
                assert_eq!(v::uto_array(a.gt(b).0)[0] != 0, x > y);
                assert_eq!(v::uto_array(a.lt(b).0)[0] != 0, x < y);
                assert_eq!(v::uto_array(a.ge(b).0)[0] != 0, x >= y);
                assert!(same(one(a.sqrt_rounded()), <f32 as Lane>::sqrt_rounded(x)), "sqrt {x}");
                assert!(same(one(a.abs()), x.abs()) && same(one(-a), -x), "abs/neg {x}");
                assert_eq!(u(a.to_u32()), x as u32, "to_u32 {x}");
                assert_eq!(u(a.to_i32()), x as i32 as u32, "to_i32 {x}");
            }
        }
        for k in 0..(1u32 << 12) {
            let n = k.wrapping_mul(0x9E37_79B1) ^ (k << 20);
            assert_eq!(F4::from_u32(U4::splat(n)).to_array()[0].to_bits(), (n as f32).to_bits(), "from_u32 {n}");
            let m = U4::splat(n).wrapping_mul(U4(v::ufrom_array([n ^ 0xC2B2_AE3D, 3, n, 7])));
            assert_eq!(v::uto_array(m.0), [n.wrapping_mul(n ^ 0xC2B2_AE3D), n.wrapping_mul(3), n.wrapping_mul(n), n.wrapping_mul(7)]);
            assert_eq!(v::uto_array((U4::splat(n) >> 15).0)[0], n >> 15);
            assert_eq!(v::uto_array((U4::splat(n) << 23).0)[0], n << 23);
        }
    }
}
