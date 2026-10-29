//! The render's own float maths (f32): the transcendentals the shading takes, branch-free (a compare only ever picks between two computed values), from bit tricks and short polynomials, accurate to about 10⁻⁷, which is far below the 2⁻⁸ steps of the output.
//! Written this way so the same code runs four samples at a time in NEON lanes now and in a GPU's shader later (README "Speed"); and identical on every platform, where libm and the platforms' own differ in the last bits.
//! The polynomial coefficients are the series' own (Taylor, atanh), not tuning knobs.

use core::f32::consts::{FRAC_PI_2, FRAC_PI_4, LN_2, LOG2_E, PI, SQRT_2};

/// Round to the nearest integer (ties to even), as a float: adding and taking away 1.5·2²³ leaves no fraction bits. Exact for |x| < 2²²; every caller has |x| ≤ 2⁷.
#[inline(always)]
fn round(x: f32) -> f32 {
    let magic = 12_582_912f32;
    (x + magic) - magic
}

/// The polynomial for 2^f on |f| ≤ ½: e^(f·ln 2) to the sixth power of f (error under 2⁻²³).
#[inline(always)]
fn exp2_frac(f: f32) -> f32 {
    let t = f * LN_2;
    1. + t * (1. + t * (1. / 2. + t * (1. / 6. + t * (1. / 24. + t * (1. / 120. + t * (1. / 720.))))))
}

/// 2ˣ: the integer part straight into the exponent bits, the fraction from the polynomial.
/// WHY the bounds on x: the exponent field holds −126 to 127; past them the bits would wrap into garbage, so x is held to [−126, 127], where 2ˣ is already 0 or the largest power to every use here (the shading's exp2(−d²) falls off to nothing long before).
#[inline(always)]
pub fn exp2(x: f32) -> f32 {
    let x = x.max(-126.).min(127.);
    let n = round(x);
    let scale = f32::from_bits(((n as i32 + 127) as u32) << 23);
    exp2_frac(x - n) * scale
}

/// eˣ, as 2^(x·log₂e).
#[inline(always)]
pub fn exp(x: f32) -> f32 {
    exp2(x * LOG2_E)
}

/// log₂ x for x > 0 (and −∞ at 0, as the exponent field gives it): the exponent from the bits, the mantissa folded into [√½, √2) and its log from atanh, log₂ m = (2/ln 2)·(t + t³/3 + t⁵/5 + t⁷/7) with t = (m − 1)/(m + 1), |t| ≤ 0.172.
#[inline(always)]
pub fn log2(x: f32) -> f32 {
    let bits = x.to_bits();
    let e = ((bits >> 23) & 255) as i32 - 127;
    let m = f32::from_bits((bits & 0x007F_FFFF) | 0x3F80_0000);
    // Fold [√2, 2) down by half, one more power of two.
    let big = m > SQRT_2;
    let (m, e) = if big { (m * 0.5, e + 1) } else { (m, e) };
    let t = (m - 1.) / (m + 1.);
    let t2 = t * t;
    let s = t * (1. + t2 * (1. / 3. + t2 * (1. / 5. + t2 * (1. / 7. + t2 * (1. / 9.)))));
    let l = e as f32 + s * (2. * LOG2_E);
    // x = 0: −∞, as the caller's arithmetic expects (log₂ 0 is never asked of the shading except where its floor holds it off; this keeps the libm answer).
    if x == 0. { f32::NEG_INFINITY } else { l }
}

/// ln x.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
    log2(x) * LN_2
}

/// sin(2π·t), t in turns: reduced to [−½, ½] by rounding, folded to a quarter wave [−¼, ¼] (sin(π − a) = sin a), and the odd series to the 11th power on |x| ≤ π/2 (error under 6·10⁻⁸).
#[inline(always)]
pub fn sin_turns(t: f32) -> f32 {
    let r = t - round(t);
    let r = if r > 0.25 { 0.5 - r } else if r < -0.25 { -0.5 - r } else { r };
    let x = r * (2. * PI);
    let x2 = x * x;
    x * (1. - x2 * (1. / 6. - x2 * (1. / 120. - x2 * (1. / 5040. - x2 * (1. / 362_880. - x2 * (1. / 39_916_800.))))))
}

/// sin x (radians).
#[inline(always)]
pub fn sin(x: f32) -> f32 {
    sin_turns(x * (1. / (2. * PI)))
}

/// cos x (radians), as the sine a quarter turn on.
#[inline(always)]
pub fn cos(x: f32) -> f32 {
    sin_turns(x * (1. / (2. * PI)) + 0.25)
}

/// (sin x, cos x).
#[inline(always)]
pub fn sin_cos(x: f32) -> (f32, f32) {
    (sin(x), cos(x))
}

/// atan a for a in [0, 1]: past tan(π/8) by atan a = π/4 + atan((a − 1)/(a + 1)), so the series argument stays within ±0.4142, then t − t³/3 + … to the 15th power (error under 10⁻⁸).
#[inline(always)]
fn atan01(a: f32) -> f32 {
    let far = a > 0.414_213_57;
    let t = if far { (a - 1.) / (a + 1.) } else { a };
    let t2 = t * t;
    let s = t * (1. - t2 * (1. / 3. - t2 * (1. / 5. - t2 * (1. / 7. - t2 * (1. / 9. - t2 * (1. / 11. - t2 * (1. / 13. - t2 * (1. / 15.))))))));
    if far { FRAC_PI_4 + s } else { s }
}

/// atan2(y, x), radians in (−π, π]: the smaller of |x|, |y| over the larger into atan01, then the octant put back.
/// WHY the smallest positive float under the division: at the origin both are 0 and 0/0 is NaN; libm's atan2(0, 0) is 0, which this gives (0 over the smallest float), and nowhere else does it move the result.
#[inline(always)]
pub fn atan2(y: f32, x: f32) -> f32 {
    let (ax, ay) = (x.abs(), y.abs());
    let a = atan01(ax.min(ay) / (ax.max(ay) + f32::MIN_POSITIVE));
    let a = if ay > ax { FRAC_PI_2 - a } else { a };
    let a = if x < 0. { PI - a } else { a };
    if y < 0. { -a } else { a }
}

/// tanh x, from e^(2x): (e − 1)/(e + 1), which exp2's bounds carry to ±1 far out.
#[inline(always)]
pub fn tanh(x: f32) -> f32 {
    let e = exp2(x * (2. * LOG2_E));
    (e - 1.) / (e + 1.)
}

#[cfg(test)]
mod tests {
    /// Every function against the platform's own over a wide spread of arguments: within 2⁻²⁰ absolute (or relative, for the ones that grow), far below an 8-bit output step.
    #[test]
    fn close_to_the_platforms() {
        let mut worst = [0f32; 7];
        for i in 0..(1u32 << 16) {
            let u = i as f32 / (1u32 << 16) as f32;
            let x = (u - 0.5) * 64.;
            let p = u * 4096. + 1e-6;
            let rel = |a: f32, b: f32| ((a - b) / b.abs().max(1.)).abs();
            worst[0] = worst[0].max(rel(super::exp2(x * 0.25), (x * 0.25).exp2()));
            worst[1] = worst[1].max(rel(super::log2(p), p.log2()));
            worst[2] = worst[2].max((super::sin(x) - x.sin()).abs());
            worst[3] = worst[3].max((super::cos(x) - x.cos()).abs());
            worst[4] = worst[4].max((super::tanh(x * 0.125) - (x * 0.125).tanh()).abs());
            worst[5] = worst[5].max((super::atan2(x, u - 0.5) - x.atan2(u - 0.5)).abs());
            worst[6] = worst[6].max(rel(super::exp(x * 0.125), (x * 0.125).exp()));
        }
        let bound = 1. / (1u32 << 20) as f32;
        // sin and cos of radians up to ±32 carry the f32 rounding of x/2π itself (libm reduces exactly): about 32·2⁻²⁴ of a turn, 2⁻¹⁸ absolute; the shading's sines take turns (`sin_turns`) and skip it.
        let bound_rad = 1. / (1u32 << 17) as f32;
        assert!(worst.iter().enumerate().all(|(i, &w)| w < if i == 2 || i == 3 { bound_rad } else { bound }), "worst errors (exp2, log2, sin, cos, tanh, atan2, exp): {worst:?}");
        assert_eq!(super::atan2(0., 0.), 0.);
        assert_eq!(super::log2(0.), f32::NEG_INFINITY);
    }
}

/// The same functions as methods on f32 (`x.fln()` for `x.ln()` and so on), so the render's expressions read as they did with the platform's maths.
pub trait Fast {
    fn fln(self) -> f32;
    fn flog2(self) -> f32;
    fn fexp(self) -> f32;
    fn fexp2(self) -> f32;
    fn fsin(self) -> f32;
    fn fcos(self) -> f32;
    fn fsin_cos(self) -> (f32, f32);
    fn ftanh(self) -> f32;
    fn fatan2(self, x: f32) -> f32;
}

impl Fast for f32 {
    #[inline(always)]
    fn fln(self) -> f32 {
        ln(self)
    }
    #[inline(always)]
    fn flog2(self) -> f32 {
        log2(self)
    }
    #[inline(always)]
    fn fexp(self) -> f32 {
        exp(self)
    }
    #[inline(always)]
    fn fexp2(self) -> f32 {
        exp2(self)
    }
    #[inline(always)]
    fn fsin(self) -> f32 {
        sin(self)
    }
    #[inline(always)]
    fn fcos(self) -> f32 {
        cos(self)
    }
    #[inline(always)]
    fn fsin_cos(self) -> (f32, f32) {
        sin_cos(self)
    }
    #[inline(always)]
    fn ftanh(self) -> f32 {
        tanh(self)
    }
    #[inline(always)]
    fn fatan2(self, x: f32) -> f32 {
        atan2(self, x)
    }
}
