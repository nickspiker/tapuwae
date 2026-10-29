//! tapuwae — a footprint: a mark that is recognisably one source's, drawn from a hash.
//!
//! The mark: the Julia set of z² + c seen through an inversion about its root spiral's eye, with c placed by the descent and the eye found on J(c), both drawn from a hash of (salt, text).
//!
//! The salt (on the phone, the device's own, sealed under its ira; here a stand-in) fixes the colouring method and the theme: one soft, low-contrast scheme for every device, varied widely in tone, polarity, edge, hue relations and saturation (see `Theme`).
//!
//! Placement, the descent: pick x in [−2, X_RIGHT ≈ 0.4715], the set's whole real extent, slanted toward the right (density rising linearly from 0 at the antenna's tip, as x = −2 + span·√u, since the left third is all needles), and come straight down the line Re c = x from Im c = 2, each step the proven lower bound on the distance to the Mandelbrot set (see `descend`), until within 2⁻¹² of the set — the set's skyline seen from above, its outermost contact on that line, from the antenna through the bulb tops and seahorse valley to the cusp.
//! Lines between the cusp (0.25) and the cardioid's own rightmost point (0.375) that pass beside the bulbs reach the real axis just outside the set, in the parabolic-implosion region, which is kept; past 0.375 they land on the decorations of the bulbs near the top right.
//! c is the landing point raised by 2^−(4..12) back up the line it came down (see HEIGHT_STOPS). Every point of that line above the landing was proven outside the set by the steps that crossed it, so c is always outside, J(c) is a Cantor dust with no inside to go flat, and near the boundary J(c) resembles the Mandelbrot set around c.
//!
//! The eye: a repelling periodic point of J(c) is the eye of a spiral whose multiplier λ = (f^p)′ says how it repeats (zooming by |λ| and turning by arg λ maps it onto itself), and the ROOT whorl, the largest, is the cycle closest to neutral (smallest ln|λ|).
//! Candidates: the cycle that is attracting in the Mandelbrot component beneath the descent's landing, Newton-closed at c (near-neutral when that component is close); the α fixed point in closed form; and 2⁵ sprayed cycles. The eye is one of the winner's own points (the hash picks which), never a preimage, which would be a sub-spiral.
//!
//! The view: dead centre on the eye, the whorl's radius R filling the plain window (`render`), or spherized into a disc (`render_disc`): a screen point p in the unit disc shows z = eye + R·p/√(1 − |p|²).
//! That is the plane wrapped onto a hemisphere seen from above (gnomonic onto the sphere, orthographic to the eye): the centre is nearly the plain view, and the rim is the plane's infinity, the outer field fading to black, so J(c) closes into a ball with no edge to crop.
//! It cannot also be conformal (no conformal map takes the disc onto the whole plane), so near the rim shapes stretch radially; the distance estimate converts to pixels through the map's area scale, R/(1 − |p|²) per disc unit, so the rim of the set keeps about its width.
//! A thumbnail still rejects views that show too little of the set, redrawing with the next counter.
//!
//! Precision: placement and the eye are f64 (a few dozen points per key, and the descent, the Newton classifier and the landing work near 2⁻¹²); the per-pixel render is f32 (the whorl's radius is kept at least 2⁻⁸, so even at the screen's edge, where the inversion packs z tightest, a pixel spans many f32 steps).
//!
//! Shading: the distance estimate (|z|·ln|z| over the derivative's magnitude, in pixels) separates the set from the field around it, because near the set the escape count changes chaotically from pixel to pixel and reads as static.
//! The set's body is dark with the colouring faintly inside it, its edge is a luminous rim a few pixels wide, and the field outside is the colouring method, dimmed and fading to black far out.
//! The colouring method is one of four: smooth iteration bands, a swirl of the potential's gradient direction, an orbit trap (distance to a hash-chosen point: rings and petals), or the flame (mandelbrot-exploder's weighted orbit means); every method also carries soft contours of the real distance estimate.
//! Nothing compares the text against anything, so the picture says nothing about whether the text is right.

use rayon::prelude::*;
use std::f64::consts::TAU;

/// Iteration cap for the render: deep enough for the windows used here.
const MAX_ITER: u32 = 1 << 10;
/// The placement's escape (f64): |z|² past √(f64::MAX), as the render's is past √(f32::MAX), so the descent's distance estimates use every clean bit.
const BAIL_SQ: f64 = 1.340_780_792_994_259_7e154;
/// The render's escape: |z|² past 2⁶², so the next square stays finite with room to spare (at most 2¹²⁴, against f32's 2¹²⁸), and the smooth count, the distance estimate and the flames get nearly every clean bit of the escape.
/// WHY not √(f32::MAX) (mandelbrot-exploder's rule, used here until 2026-10-29): an orbit just under it squares to just under f32::MAX, and the + c and the rounding of x² + y² carried |z|² to ∞, which made the smooth count −∞ and the flame NaN; about one sample in 2²⁶ (measured), each poisoning its pixel.
const BAIL_SQ_F32: f32 = (1u64 << 62) as f32;
/// A jump of 2² iterations between neighbouring thumbnail pixels marks an edge of the set.
const JUMP: i32 = 1 << 2;
/// A framing is kept when at least a quarter of the thumbnail is edge.
const DETAIL_MIN: f64 = 1. / (1 << 2) as f64;
/// Stops of height for c above its landing point, from 2⁻⁴ down: much above 2⁻⁴ J(c) thins into sparse dust, and below 2⁻¹² the landing's own tolerance (2⁻¹²) dominates, so the 2³ stops between are the ones that matter.
const HEIGHT_STOPS: u32 = 1 << 3;
/// Periods sprayed for a spiral's cycle: 1 to 2⁴.
const PERIODS: u32 = 1 << 4;
/// Cycles sprayed as candidates for the root whorl.
const SPIRAL_TRIES: u32 = 1 << 5;
/// The smallest window half-span kept, so the f32 render still has many representable steps per pixel.
const MIN_HALF: f64 = 1. / (1 << 8) as f64;
/// The engraving's radius, squared: level lines and rays are read where the orbit first passes |z| = 2⁸, close enough to the set that the rays' angle has not yet turned too fast to draw.
const ENGRAVE_SQ: f32 = (1u32 << 16) as f32;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// The Julia set of z² + c.
    Julia,
    /// The parameter plane of z² + c, z starting at 0 (the locator panes).
    Plane,
}

/// Everything one picture needs, in the plane's terms (f64: placement and framing).
#[derive(Clone, Copy)]
pub struct Params {
    kind: Kind,
    c: (f64, f64),
    centre: (f64, f64),
    /// Half the window's span, in the plane.
    half: f64,
    /// The view's rotation as (cos, sin): plane = centre + rot·(u + iv) for screen offsets (u, v), v downward.
    rot: (f64, f64),
    /// The device's theme within the one soft scheme (see `Theme`), from the salt.
    theme: Theme,
    /// Colouring method: 0 smooth bands, 1 escape angle, 2 orbit trap, 3 flame.
    mode: u8,
    /// The orbit trap's point, for mode 2.
    trap: (f64, f64),
    /// The key's nudged features (see `Keyed`).
    keyed: Keyed,
    /// The device's attest-only choices (see `Attest`), drawn under their own domain so they never move an avatar.
    attest: Attest,
    /// The root whorl's radius in the plane: the plain window's half-span, and the disc's scale.
    radius: f64,
}

/// The same picture in the render's precision (f32).
#[derive(Clone, Copy)]
struct View {
    kind: Kind,
    c: (f32, f32),
    centre: (f32, f32),
    rot: (f32, f32),
    theme: Theme,
    mode: u8,
    trap: (f32, f32),
    keyed: Keyed,
    attest: Attest,
}

impl Params {
    fn view(&self) -> View {
        let f = |v: (f64, f64)| (v.0 as f32, v.1 as f32);
        View { kind: self.kind, c: f(self.c), centre: f(self.centre), rot: f(self.rot), theme: self.theme, mode: self.mode, trap: f(self.trap), keyed: self.keyed, attest: self.attest }
    }
}

/// 32 bits of a hash as a fraction in [0, 1).
fn unit(b: &[u8]) -> f64 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64 / (1u64 << 32) as f64
}

/// A keyed BLAKE3 of the salt over a domain byte, a counter and any extra bytes.
fn draw(salt: &[u8; 32], domain: u8, counter: u32, extra: &[u8]) -> [u8; 32] {
    *blake3::Hasher::new_keyed(salt).update(&[domain]).update(&counter.to_le_bytes()).update(extra).finalize().as_bytes()
}

fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// What one orbit did: iterations until escape (MAX_ITER if it never did), z at escape, the derivative's magnitude, and the closest approach to the trap point (squared).
struct Orbit {
    n: u32,
    x: f32,
    y: f32,
    dm: f32,
    trap_sq: f32,
    /// |dz/dz₀| = dm · 2^dm_exp: the magnitude is rescaled by 2⁻⁶⁴ whenever it passes 2⁶⁴, so it never overflows f32 however long the orbit stays near the set (an overflowed derivative read as distance 0, a band of false "on the set" pixels).
    dm_exp: i32,
    /// The flames, from mandelbrot-exploder (compute/mandelbrot_f64.rs, OrbitTrackerf64): weighted means over the orbit, `fq` of z (weight iter²/|z|, tapered as the orbit escapes) and `fc` of 1/z (weight 1/|z|).
    fq: (f32, f32),
    fc: (f32, f32),
    /// The direction of z / (dz/dz₀) at escape, a unit complex number: the outward normal of the potential's level line, the "slope" the set's relief would have there (Julia sets only; (0, 0) in the parameter plane).
    ux: f32,
    uy: f32,
    /// The orbit where it first passed |z| = 2⁸ (`ENGRAVE_SQ`): its level `lv` (that crossing's smooth count, m + 1 − log₂(ln|z_m|² / ln 2¹⁶), an integer exactly on the crossing), its angle `ray` in turns, and ln of how fast that angle turns per plane unit, ln(|dz_m/dz₀| / |z_m|). All 0 if it never got there.
    lv: f32,
    ray: f32,
    ln_turn: f32,
}

/// Iterate z² + c from the plane point (px, py): z₀ = that point in a Julia set, c = that point in the parameter plane.
/// The derivative is tracked by magnitude only (|dz| ← 2·|z|·|dz|, plus 1 in the parameter plane), which is all the distance estimate needs.
/// It can overflow to infinity near the set, which the distance estimate reads, correctly, as "on the set".
#[inline]
fn orbit(px: f32, py: f32, v: &View) -> Orbit {
    let (mut z, c, mut plus, mut dm) = match v.kind {
        Kind::Plane => ((0f32, 0f32), (px, py), 1f32, 0f32),
        Kind::Julia => ((px, py), v.c, 0., 1.),
    };
    let julia = v.kind == Kind::Julia;
    let mut trap_sq = f32::MAX;
    // The derivative's direction, a unit complex number: dz ← 2·z·dz turns it by arg z each step. Kept apart from its magnitude (dm) so it never overflows, however close to the set.
    let mut dir = (1f32, 0f32);
    // The flame sums and weights.
    let (mut fq, mut fq_w, mut fc, mut fc_w) = ((0f32, 0f32), 0f32, (0f32, 0f32), 0f32);
    // The taper's scale: ln ln of the escape.
    let lnln_bail = BAIL_SQ_F32.ln().ln();
    let ln_2ln2 = (2. * 2f32.ln()).ln();
    let mut dm_exp = 0i32;
    let (mut lv, mut ray, mut ln_turn, mut engraved) = (0f32, 0f32, 0f32, false);
    let finish = |fq: (f32, f32), fq_w: f32, fc: (f32, f32), fc_w: f32| ((fq.0 / fq_w, fq.1 / fq_w), (fc.0 / fc_w, fc.1 / fc_w));
    for n in 0..MAX_ITER {
        let (x2, y2) = (z.0 * z.0, z.1 * z.1);
        let r2 = x2 + y2;
        if r2 > BAIL_SQ_F32 {
            // u = z / dz, as a direction: z/|z| times the conjugate of dz's direction.
            let rz = r2.sqrt();
            let (zx, zy) = (z.0 / rz, z.1 / rz);
            let (ux, uy) = if julia { (zx * dir.0 + zy * dir.1, zy * dir.0 - zx * dir.1) } else { (0., 0.) };
            let (fq, fc) = finish(fq, fq_w, fc, fc_w);
            return Orbit { n, x: z.0, y: z.1, dm, dm_exp, trap_sq, fq, fc, ux, uy, lv, ray, ln_turn };
        }
        if !engraved && r2 > ENGRAVE_SQ {
            engraved = true;
            lv = n as f32 + 1. - (r2.ln() / ENGRAVE_SQ.ln()).log2();
            ray = z.1.atan2(z.0) * (1. / TAU as f32);
            ln_turn = dm.ln() + dm_exp as f32 * std::f32::consts::LN_2 - 0.5 * r2.ln();
        }
        let (tx, ty) = (z.0 - v.trap.0, z.1 - v.trap.1);
        trap_sq = trap_sq.min(tx * tx + ty * ty);
        let rz = r2.sqrt();
        // Flames: iter² weights; past |z|² = 4 the z-flame tapers as (1 − ((ln ln |z|² − ln(2 ln 2)) / ln ln bail)²)⁶, which is in [0, 1] there because |z|² > 4 makes ln ln |z|² > ln(2 ln 2).
        let it = (n + 1) as f32;
        let it2 = it * it;
        let taper = if r2 > 4. { let t = (r2.ln().ln() - ln_2ln2) / lnln_bail; let t = 1. - t * t; let t = t * t; t * t * t } else { 1. };
        let wq = taper * it2;
        fq_w += wq;
        fq = (fq.0 + z.0 * wq / rz, fq.1 + z.1 * wq / rz);
        fc_w += 1. / rz;
        fc = (fc.0 + z.0 / r2, fc.1 - z.1 / r2);
        dm = 2. * rz * dm + plus;
        if dm > (1u64 << 63) as f32 * 2. {
            let s = 1. / ((1u64 << 63) as f32 * 2.);
            dm *= s;
            plus *= s;
            dm_exp += 64;
        }
        if julia {
            let d = (dir.0 * z.0 - dir.1 * z.1, dir.0 * z.1 + dir.1 * z.0);
            let dl = (d.0 * d.0 + d.1 * d.1).sqrt();
            dir = (d.0 / dl, d.1 / dl);
        }
        z = (x2 - y2 + c.0, 2. * z.0 * z.1 + c.1);
    }
    let (fq, fc) = finish(fq, fq_w, fc, fc_w);
    Orbit { n: MAX_ITER, x: z.0, y: z.1, dm, dm_exp, trap_sq, fq, fc, ux: 0., uy: 0., lv, ray, ln_turn }
}

/// The plane point under pixel (px, py) of a `w × h` window, `k` plane units per pixel: centre + rot·(u + is).
#[inline]
fn plane(v: &View, px: usize, py: usize, w: usize, h: usize, k: f32) -> (f32, f32) {
    let (u, s) = ((px as f32 + 0.5 - w as f32 * 0.5) * k, (py as f32 + 0.5 - h as f32 * 0.5) * k);
    (v.centre.0 + u * v.rot.0 - s * v.rot.1, v.centre.1 + u * v.rot.1 + s * v.rot.0)
}

/// Plane units per pixel for a `w × h` window: half the span covers `half`.
fn scale(half: f64, w: usize, h: usize) -> f32 {
    let span = 2. * (w * h) as f64 / (w + h) as f64;
    (half / (span * 0.5)) as f32
}

/// How much edge a 2⁵ × 2⁶ thumbnail shows: the share of pixels whose iteration count jumps by at least 2² to the neighbour on the right or below.
/// Only filaments of the set make such jumps; the smooth field between them makes none, so the score reads "how much of the view is set" at every zoom.
fn detail(p: &Params) -> f64 {
    const W: usize = 1 << 5;
    const H: usize = 1 << 6;
    let v = p.view();
    let k = scale(p.half, W, H);
    let mut n = [0i32; W * H];
    for py in 0..H {
        for px in 0..W {
            let (x, y) = plane(&v, px, py, W, H, k);
            n[py * W + px] = orbit(x, y, &v).n as i32;
        }
    }
    let mut edges = 0;
    for py in 0..H - 1 {
        for px in 0..W - 1 {
            let i = py * W + px;
            edges += ((n[i] - n[i + 1]).abs().max((n[i] - n[i + W]).abs()) >= JUMP) as u32;
        }
    }
    edges as f64 / ((W - 1) * (H - 1)) as f64
}

fn framed(p: &Params) -> bool {
    detail(p) >= DETAIL_MIN
}

/// One inverse branch of z² + c: ±√(z − c), the principal root, negated when `negate`.
fn inverse(c: (f64, f64), z: (f64, f64), negate: bool) -> (f64, f64) {
    let (ax, ay) = (z.0 - c.0, z.1 - c.1);
    let r = (ax * ax + ay * ay).sqrt();
    let (mut sx, mut sy) = (((r + ax) * 0.5).sqrt(), ((r - ax) * 0.5).sqrt());
    if ay < 0. {
        sy = -sy;
    }
    if negate {
        (sx, sy) = (-sx, -sy);
    }
    (sx, sy)
}

/// Address bit i (bit 0 is the coarsest: which half of J).
fn bit(bits: &[u8; 32], i: usize) -> bool {
    (bits[i >> 3] >> (i & 7)) & 1 == 1
}

/// The point of J(c) whose address is `bits`: inverse iteration applies the branches finest first, so bit 0, applied last, picks the half of J.
fn address_point(c: (f64, f64), bits: &[u8; 32]) -> (f64, f64) {
    let mut z = (0.5f64, 0.);
    for i in (0..(1 << 8)).rev() {
        z = inverse(c, z, bit(bits, i));
    }
    z
}

/// J(c)'s half-extent along its principal axis, from 2⁷ address points (J = −J, so its centre is 0).
fn j_extent(c: (f64, f64)) -> f64 {
    let pts: Vec<(f64, f64)> = (0u32..(1 << 7)).map(|i| address_point(c, blake3::hash(&i.to_le_bytes()).as_bytes())).collect();
    let (mut sxx, mut syy, mut sxy) = (0., 0., 0.);
    for p in &pts {
        sxx += p.0 * p.0;
        syy += p.1 * p.1;
        sxy += p.0 * p.1;
    }
    let axis = 0.5 * (2. * sxy).atan2(sxx - syy);
    let (ca, sa) = (axis.cos(), axis.sin());
    pts.iter().map(|p| (p.0 * ca + p.1 * sa).abs()).fold(0., f64::max)
}

/// Newton on f^period(w) = w from `guess`: the cycle point it closes on, with ln|λ| and arg λ of the multiplier λ = (f^period)′ there, kept only if f^period carries it back onto itself and it repels (every cycle of a Cantor Julia set does; a non-finite derivative means Newton wandered off to infinity).
fn close_cycle(c: (f64, f64), guess: (f64, f64), period: u32) -> Option<((f64, f64), f64, f64)> {
    // f^period and its derivative at w.
    let run = |w: (f64, f64)| {
        let (mut u, mut du) = (w, (1f64, 0f64));
        for _ in 0..period {
            du = cmul((2. * du.0, 2. * du.1), u);
            let uu = cmul(u, u);
            u = (uu.0 + c.0, uu.1 + c.1);
        }
        (u, du)
    };
    let mut w = guess;
    for _ in 0..NEWTON_STEPS {
        let (u, du) = run(w);
        // Newton step: w ← w − (u − w) / (du − 1).
        let (nr, ni) = (u.0 - w.0, u.1 - w.1);
        let (dr, di) = (du.0 - 1., du.1);
        let dd = dr * dr + di * di;
        let step = ((nr * dr + ni * di) / dd, (ni * dr - nr * di) / dd);
        w = (w.0 - step.0, w.1 - step.1);
        if step.0 * step.0 + step.1 * step.1 < (-80f64).exp2() {
            break;
        }
    }
    let (u, du) = run(w);
    let closed = (u.0 - w.0).abs() + (u.1 - w.1).abs() < 1. / (1 << 24) as f64;
    let mag_sq = du.0 * du.0 + du.1 * du.1;
    (closed && mag_sq > 1. && mag_sq.is_finite()).then(|| (w, 0.5 * mag_sq.ln(), du.1.atan2(du.0)))
}

/// A repelling cycle point of J(c) of period dividing `period`, Newton-closed from the point whose address repeats the first `period` bits of `bits`.
/// The repeated address is only the starting guess: the bits choose principal or negated square roots, and the principal root's cut can cross J, so a repeating bit pattern is not a true itinerary there.
fn cycle_point(c: (f64, f64), bits: &[u8; 32], period: u32) -> Option<((f64, f64), f64, f64)> {
    let mut rep = [0u8; 32];
    for i in 0..(1usize << 8) {
        if bit(bits, i % period as usize) {
            rep[i >> 3] |= 1 << (i & 7);
        }
    }
    close_cycle(c, address_point(c, &rep), period)
}

/// The root whorl of J(c): the cycle that is attracting inside the hyperbolic component beneath c's landing, now just repelling at c (|λ| a little over 1, arg λ near the component's internal angle), which makes it the slowest and largest spiral in the picture.
/// Found from the Mandelbrot side: keep going down the descent's line past the landing, which is usually a hair or decoration, until the Newton classifier proves a point inside; the component found is the one the hairs hang from. Below the landing the line crosses gaps (stepped by the same proven distance bound as the descent, at least 2⁻¹²) and hair the classifier cannot decide (stepped 2⁻¹⁰); at most 2¹⁰ steps, and not past the real axis.
/// The attracting cycle the classifier finds there is then Newton-closed at c itself. None when nothing inside is reached.
pub fn root_cycle(c: (f64, f64), land: (f64, f64)) -> Option<((f64, f64), f64, f64, u32)> {
    let x = land.0;
    let mut y = land.1;
    for _ in 0..(1 << 10) {
        match classify((x, y)) {
            Class::Inside { period, attractor } => {
                let (w, ln_mag, arg) = close_cycle(c, attractor, period)?;
                // The classifier's period can be a multiple of the cycle's own: walk the cycle until it returns.
                let mut z = w;
                let mut own = period;
                for q in 1..=period {
                    let zz = cmul(z, z);
                    z = (zz.0 + c.0, zz.1 + c.1);
                    if (z.0 - w.0).abs() + (z.1 - w.1).abs() < 1. / (1 << 20) as f64 {
                        own = q;
                        break;
                    }
                }
                return Some((w, ln_mag, arg, own));
            }
            Class::Outside { n, r, dm } => {
                let d = r * r.ln() / dm;
                let g = r.ln() * (-(n as f64)).exp2();
                let factor = if g > 0. { -(-2. * g).exp_m1() / (4. * g) } else { 0.5 };
                // WHY/PROOF: below the landing the line runs between hairs, where the proven step can shrink without bound; the floor keeps the probe moving, and at 2⁻¹² it is the descent's own landing tolerance.
                y -= (d * factor).max(1. / (1 << 12) as f64);
            }
            Class::Undecided => y -= 1. / (1 << 10) as f64,
        }
        if y < 0. {
            return None;
        }
    }
    None
}

/// The Mandelbrot set's rightmost real extent, measured with `descend` (the last line that lands, on a 2⁻¹² grid, is Re c = 0.47132, on the bulbs' decorations near Im 0.36), rounded up a hair: past the cardioid's own rightmost point (0.375) by the decorations of the 1/5, 1/6 and 1/7 bulbs.
pub const X_RIGHT: f64 = 0.4715;

/// Newton steps when solving for an attracting cycle.
const NEWTON_STEPS: u32 = 1 << 5;
/// Iteration cap for classifying a point of the parameter plane.
const CLASSIFY_ITER: u32 = 1 << 12;

/// Is there an attracting cycle of period `period` near `guess`? Newton on f^period(w) = w; a converged cycle with |multiplier| < 1 proves c is inside the set.
/// Ported from mandelbrot-exploder (compute/mandelbrot_f64.rs, find_interior_attractor).
fn attracting_cycle(c: (f64, f64), guess: (f64, f64), period: u32) -> Option<(f64, f64)> {
    let mut w = guess;
    for _ in 0..NEWTON_STEPS {
        let (mut u, mut du) = (w, (1., 0.));
        for _ in 0..period {
            du = cmul((2. * du.0, 2. * du.1), u);
            let uu = cmul(u, u);
            u = (uu.0 + c.0, uu.1 + c.1);
        }
        // Newton step: w ← w − (u − w) / (du − 1).
        let (nr, ni) = (u.0 - w.0, u.1 - w.1);
        let (dr, di) = (du.0 - 1., du.1);
        let dd = dr * dr + di * di;
        let step = ((nr * dr + ni * di) / dd, (ni * dr - nr * di) / dd);
        w = (w.0 - step.0, w.1 - step.1);
        if step.0 * step.0 + step.1 * step.1 < (-53f64).exp2() {
            return (du.0 * du.0 + du.1 * du.1 < 1.).then_some(w);
        }
    }
    None
}

/// Where c is with respect to the Mandelbrot set.
enum Class {
    /// Escaped at iteration n with |z| = r and |dz/dc| = dm.
    Outside { n: u32, r: f64, dm: f64 },
    /// Proven inside: an attracting cycle of this period was found, through this point.
    Inside { period: u32, attractor: (f64, f64) },
    /// Neither within the cap.
    Undecided,
}

/// Classify c, ported from mandelbrot-exploder's classify_point: iterate from z = c, and whenever |z|² falls below a quarter of its previous minimum, try Newton for an attracting cycle of the period reached so far.
fn classify(c: (f64, f64)) -> Class {
    let (mut z, mut dz) = (c, (1f64, 0f64));
    let mut smallest = f64::MAX;
    for n in 1..CLASSIFY_ITER {
        let r2 = z.0 * z.0 + z.1 * z.1;
        if r2 > BAIL_SQ {
            // hypot, not √(x² + y²): near the escape dz's components reach about 10¹⁶⁰, whose squares overflow f64 though dz does not, and an infinite derivative reads as distance 0, a false landing (it put the "set" at Re 0.49).
            return Class::Outside { n, r: r2.sqrt(), dm: dz.0.hypot(dz.1) };
        }
        if r2 < smallest {
            let ratio = r2 / smallest;
            smallest = r2;
            if ratio < 0.25 {
                if let Some(attractor) = attracting_cycle(c, z, n) {
                    return Class::Inside { period: n, attractor };
                }
            }
        }
        let zdz = cmul(z, dz);
        dz = (2. * zdz.0 + 1., 2. * zdz.1);
        let zz = cmul(z, z);
        z = (zz.0 + c.0, zz.1 + c.1);
    }
    Class::Undecided
}

/// The descent: down the line Re c = x from Im c = 2, until within 2⁻¹² of the set, onto it (proven inside or undecided), or onto the real axis; at most 2⁹ steps.
/// Each step is the PROVEN lower bound on the distance to the set (Koebe, via Milnor): with G = ln|z|/2ⁿ the Green's function and d = |z|·ln|z|/|dc| the estimate (= G/|G′|), the true distance is at least sinh G / (2·e^G·|G′|) = d·(1 − e^(−2G))/(4G).
/// That factor is ½ next to the set and smaller far out (about 0.22 at G = 1), so the step never jumps past anything, not even a hair-thin filament, and the descent stops at the first, outermost contact along the line.
/// Returns the landing point, whether it is on the set (false when the line missed it and reached the axis), and every point it checked, in order.
pub fn descend(x: f64) -> ((f64, f64), bool, Vec<(f64, f64)>) {
    let mut y = 2.;
    let mut trail = Vec::new();
    for _ in 0..(1 << 9) {
        trail.push((x, y));
        let Class::Outside { n, r, dm } = classify((x, y)) else {
            break;
        };
        let d = r * r.ln() / dm;
        if d < 1. / (1 << 12) as f64 {
            break;
        }
        // G underflows to 0 for orbits longer than about 2¹⁰ steps, where the factor's limit, ½, is exact enough.
        let g = r.ln() * (-(n as f64)).exp2();
        let factor = if g > 0. { -(-2. * g).exp_m1() / (4. * g) } else { 0.5 };
        y -= d * factor;
        if y <= 0. {
            trail.push((x, 0.));
            return ((x, 0.), false, trail);
        }
    }
    ((x, y), true, trail)
}

/// A progressive walk in [0, 1): the device's own starting value (the hash of the empty text), then for the i-th character (from 1) a nudge of up to ±2^−(i+1), from the hash of the text up to and including it, so the first characters move the picture a lot and each later one half as much as the one before; by the 25th a nudge is under 2⁻²⁵.
/// A retry (`n`) redraws only the newest character's nudge (or the start, with no text), so it never undoes the characters before it.
/// The walk folds back at the ends (a triangle wave), so it stays continuous and never jumps from one end to the other; `wraps` instead wraps it round, for a value that is an angle.
fn walk(salt: &[u8; 32], domain: u8, text: &str, n: u32, wraps: bool) -> f64 {
    walk_at(salt, domain, text, n, wraps, 0.5)
}

/// As `walk`, with the i-th character's nudge up to ±½·rate^i instead of ±2^−(i+1): a slower rate keeps a feature moving for more characters, a faster one settles it within the first few.
fn walk_at(salt: &[u8; 32], domain: u8, text: &str, n: u32, wraps: bool, rate: f64) -> f64 {
    let ends: Vec<usize> = text.char_indices().map(|(i, ch)| i + ch.len_utf8()).collect();
    let mut v = unit(&draw(salt, domain, if ends.is_empty() { n } else { 0 }, &[])[0..4]);
    for (i, &end) in ends.iter().enumerate() {
        let last = i + 1 == ends.len();
        let h = unit(&draw(salt, domain, if last { n } else { 0 }, &text.as_bytes()[..end])[0..4]);
        v += (h - 0.5) * rate.powi(i as i32 + 1);
    }
    if wraps {
        v.rem_euclid(1.)
    } else {
        let r = v.rem_euclid(2.);
        if r > 1. { 2. - r } else { r }
    }
}

/// The features a key nudges beside the placement, each walking from the device's own with its own falloff (`walk_at`'s rate), so some settle within a few characters and others keep shifting for longer.
#[derive(Clone, Copy)]
struct Keyed {
    /// The ripple's four phases (turns): the field's contour waves slide as you type. Rate ¾: they keep moving the longest.
    phase: [f32; 4],
    /// A complex offset of the colouring coordinate, added to the look's. Rate 1/√2.
    offset: (f32, f32),
    /// A turn of the lamps across the screen, up to ±1/16 of a turn from the device's own. Rate ¼: it settles within the first few characters.
    light_turn: f32,
    /// A scale of the relief, 2^−½ to 2^½. Rate ¼.
    relief: f32,
}

impl Keyed {
    fn new(salt: &[u8; 32], text: &str, progressive: bool) -> Self {
        let pick = |domain: u8, wraps: bool, rate: f64| (if progressive { walk_at(salt, domain, text, 0, wraps, rate) } else { unit(&draw(salt, domain, 0, text.as_bytes())[0..4]) }) as f32;
        let slow = 0.75;
        let mid = std::f64::consts::FRAC_1_SQRT_2;
        let fast = 0.25;
        Keyed {
            phase: [pick(32, true, slow), pick(33, true, slow), pick(34, true, slow), pick(35, true, slow)],
            offset: (pick(36, true, mid), pick(37, true, mid)),
            light_turn: (pick(38, false, fast) - 0.5) * 0.125,
            relief: (pick(39, false, fast) - 0.5).exp2(),
        }
    }
}

/// A device's attest-only features, all from the salt alone (never the text), so they hold still while the handle is typed and the device is recognisable from the first frame: banded into a few nameable kinds each.
#[derive(Clone, Copy)]
struct Attest {
    /// The ball's material: 0 matte, 1 satin, 2 gloss, 3 glass.
    material: u8,
    /// The engraving: 0 none, 1 level lines, 2 rays, 3 both (a net).
    engrave: u8,
    /// Mirrored (the conjugate Julia set, J(c̄)): the whorl turns the other way.
    mirror: bool,
    /// Level lines per level of the potential: ¼, ½ or 1.
    levels: f32,
    /// Rays per turn at each level: 2², 2³ or 2⁴.
    rays: f32,
    /// How the picture is made: 0 the avatars' soft colours, 1 a medal (the set in bas-relief, struck in metal), 2 enamel (cloisonné: the engraving's cells filled with colour, the set and the wires in metal), 3 lacquer (the avatars' colouring, deeper and glazed, the set raised in metal). A device draws 1, 2 or 3, never the avatars' 0.
    style: u8,
    /// The metals (indices into `METALS`): `metal_a` for the set and the wires, `metal_b` for a medal's field and the button's bezel; the same metal for half of devices, two for the rest (a bimetal).
    metal_a: u8,
    metal_b: u8,
    /// The enamel's cell colours: a seed for which of the four each cell takes.
    cells: u32,
    /// How densely the electric flame arcs: its contours at 2², 2³ or 2⁴ times the flame's scale (sparse, medium, a web).
    arcs: f32,
}

impl Attest {
    fn new(salt: &[u8; 32]) -> Self {
        let a = draw(salt, 64, 0, &[]);
        let metal_a = a[6] & 7;
        Attest {
            material: a[0] & 3,
            engrave: a[1] & 3,
            mirror: a[2] & 1 == 1,
            levels: ((a[3] % 3) as f32 - 2.).exp2(),
            rays: ((2 + a[4] % 3) as f32).exp2(),
            style: 1 + a[5] % 3,
            metal_a,
            metal_b: if a[7] & 1 == 0 { metal_a } else { (metal_a + 1 + (a[7] >> 1) % 7) & 7 },
            cells: u32::from_le_bytes([a[9], a[10], a[11], a[12]]),
            arcs: ((2 + a[13] % 3) as f32).exp2(),
        }
    }
}

/// How a pixel is sampled (the button; the plain disc keeps `Linear`).
#[derive(Clone, Copy, PartialEq)]
pub enum Aa {
    /// One sample at a time: two always, then another while the newest moved more than m − 1 levels from the mean of the m before it, up to `MAX_SAMPLES`.
    Linear,
    /// Passes that double the count (1, 1, 2, 4, … samples, so totals 1, 2, 4, 8, …): after each, the new pass's mean against the mean of all before it, in 8-bit display levels; stop once they agree within `threshold`, or at 2^`max_log2` samples; never before 2^`min_log2` (a pixel's first two samples agreeing by luck on either side of a thin arc stopped it early, the scheme's worst errors). A pass's mean is far steadier than one sample, so the test can be tight and still stop early where the pixel is plain.
    Doubling { threshold: f32, min_log2: u32, max_log2: u32 },
    /// Exactly n samples (a ground truth to measure the others against).
    Fixed(u32),
    /// k × k samples, one at a jittered point in each cell of a k × k grid over the pixel (stratified): the pixel covered evenly, with no two samples crowding the same corner, and no regular pattern to alias.
    Stratified(u32),
}

/// Whether a look shows an attest feature: never (Photon's avatars), as the device drew it, or one kind forced (for comparing kinds side by side).
#[derive(Clone, Copy, PartialEq)]
pub enum Pick {
    Off,
    Device,
    Force(u8),
}

impl Pick {
    fn of(self, device: u8) -> u8 {
        match self {
            Pick::Off => 0,
            Pick::Device => device,
            Pick::Force(k) => k,
        }
    }
}

impl Params {
    /// The picture for this device salt and this textbox content.
    pub fn new(salt: &[u8; 32], text: &str) -> Self {
        Self::new_placed(salt, text, true)
    }

    /// As `new`, choosing whether the key's continuous choices (the descent's x, c's height, the whorl's zoom and the turn) walk progressively from the device's own (`progressive`, see `walk`) or are drawn afresh for every key; either way x is slanted toward the right (see the crate's doc).
    pub fn new_placed(salt: &[u8; 32], text: &str, progressive: bool) -> Self {
        let f = draw(salt, 1, 0, &[]);
        let span = X_RIGHT + 2.;
        // A key's continuous choices: progressive (`walk`) or each key free.
        let pick = |domain: u8, n: u32, wraps: bool| if progressive { walk(salt, domain, text, n, wraps) } else { unit(&draw(salt, domain, n, text.as_bytes())[0..4]) };
        let theme = Theme::from_bytes(&f);
        let mode = f[12] & 3;
        let trap = ((unit(&f[16..20]) - 0.5) * 2., (unit(&f[20..24]) - 0.5) * 2.);
        let keyed = Keyed::new(salt, text, progressive);
        let attest = Attest::new(salt);
        let base = Params { kind: Kind::Julia, c: (0., 0.), centre: (0., 0.), half: 2., rot: (1., 0.), theme, mode, trap, keyed, attest, radius: 1. };

        // Each try draws a fresh c and framing from (salt, text, counter); the first framing that shows enough of the set is taken, and after 2⁵ tries the last one stands.
        let mut n = 0;
        loop {
            // The discrete choices (which sprayed cycles, which point of the winning cycle) are the device's own, so only c moves them.
            let t = draw(salt, 3, n, if progressive { &[] } else { text.as_bytes() });
            // √u: density rising linearly toward the right, so fewer keys land among the antenna's needles.
            let x = -2. + pick(16, n, false).sqrt() * span;
            n += 1;
            let (land, _, _) = descend(x);
            let c = (land.0, land.1 + (-(4. + pick(17, n - 1, false) * HEIGHT_STOPS as f64)).exp2());
            let along = j_extent(c);
            // Every key sits on the ROOT whorl's eye, dead centre, never a preimage of it (a preimage is a smaller copy: a sub-spiral).
            // The root is the cycle closest to neutral (smallest ln|λ|): it turns slowest and grows slowest, so its whorl is the largest in the picture. Candidates: the cycle of the component beneath the landing (`root_cycle`, near-neutral when that component is close, and losing on its own when it is far), the α fixed point in closed form (the root near the main cardioid), and 2⁵ sprayed cycles of periods 1 to 2⁴.
            // The hash picks which of the cycle's own points to sit on: each is a root, mapped onto the next by f.
            let spiral = {
                // The α fixed point, z = (1 − √(1 − 4c))/2 with λ = 2z: the root near the main cardioid.
                let alpha = {
                    let s = {
                        let (ax, ay) = (1. - 4. * c.0, -4. * c.1);
                        let r = (ax * ax + ay * ay).sqrt();
                        let (sx, sy) = (((r + ax) * 0.5).sqrt(), ((r - ax) * 0.5).sqrt());
                        (sx, if ay < 0. { -sy } else { sy })
                    };
                    let z = ((1. - s.0) * 0.5, -s.1 * 0.5);
                    close_cycle(c, z, 1).map(|(w, l, g)| (w, l, g, 1))
                };
                let sprayed = (0..SPIRAL_TRIES).filter_map(|i| {
                    let period = 1 + (i & (PERIODS - 1));
                    cycle_point(c, &draw(salt, 4, (n << 5) | i, if progressive { &[] } else { text.as_bytes() }), period).map(|(w, l, g)| (w, l, g, period))
                });
                root_cycle(c, land).into_iter().chain(alpha).chain(sprayed).fold(None, |best: Option<((f64, f64), f64, f64, u32)>, e| if best.map_or(true, |b| e.1 < b.1) { Some(e) } else { best })
            };
            // α always closes for c outside the set (it repels there); if even that failed, the next try draws a new c, and after 2⁵ tries the whole of J(c), straight, stands.
            let Some((first, _, _, period)) = spiral else {
                if n >= 1 << 5 {
                    return Params { c, half: along * 1.125, radius: along * 1.125, ..base };
                }
                continue;
            };
            {
                let mut eye = first;
                for _ in 0..(t[6] as u32 % period) {
                    let ee = cmul(eye, eye);
                    eye = (ee.0 + c.0, ee.1 + c.1);
                }
                // The whorl's radius R: 2⁻¹ to 2⁻³ of J's half-extent, or larger until the f32 render resolves it.
                let mut stops = 1. + pick(18, n - 1, false) * 2.;
                let radius = loop {
                    let radius = along * (-stops).exp2();
                    if radius >= MIN_HALF || stops <= 1. {
                        break radius;
                    }
                    stops -= 1.;
                };
                let turn = pick(19, n - 1, true) * TAU;
                let p = Params { c, centre: eye, half: radius, radius, rot: (turn.cos(), turn.sin()), ..base };
                if framed(&p) || n >= 1 << 5 {
                    return p;
                }
            }
        }
    }
}

/// A device's theme within the one soft scheme: every colour is OKLab lightness L at a hue with chroma C = sat·L, sat at most 2⁻³ + 2⁻⁵ (0.156), which keeps it soft and inside the sRGB gamut at every lightness (no hue goes negative below about 0.17·L; `soft_colours_stay_in_gamut` checks it).
/// The salt varies far more than the colours: the ground's tone (pale, light, mid or dusk), which way the set sits against it (darker on pale grounds, lighter on dusk ones, either on mid), how strongly, whether its edge glows or is inked, how far the set's hue turns from the ground's (none, a sixth, a quarter, a third, a half, either way), how far the ground's hue drifts from the centre to the far field, and how saturated each layer is.
#[derive(Clone, Copy)]
struct Theme {
    /// The ground's base hue, radians.
    hue: f32,
    /// How far the colouring swings the hue either side of the base, radians.
    spread: f32,
    /// The set's hue turn from the ground's, radians.
    accent: f32,
    /// The ground's hue drift from near the set to the far field, radians.
    drift: f32,
    /// Lightness of the ground, the set's body and its edge.
    ground_l: f32,
    body_l: f32,
    rim_l: f32,
    /// Saturation (C/L) of the ground, the body and the edge.
    ground_s: f32,
    body_s: f32,
    rim_s: f32,
    /// The device's lighting (see `Lighting`).
    lighting: Lighting,
    /// The field's ripple: four waves across log₂ of the distance to the set, `ripple_f` turns per doubling and `ripple_a` amplitude; waves 0 and 1 move the lightness, 2 and 3 the OKLab a and b (green–red and blue–yellow), so the contours shift colour as well as tone.
    ripple_f: [f32; 4],
    ripple_a: [f32; 4],
}

/// One lamp: where it shines from (x right, y down across the screen, z up out of the picture, a unit vector), how strongly it moves the lightness, and the tint it gives the faces it lights (an OKLab (a, b) direction, unit length, scaled at use).
#[derive(Clone, Copy)]
struct Lamp {
    dir: (f32, f32, f32),
    strength: f32,
    tint: (f32, f32),
}

/// A device's lighting, all from its salt: a key lamp somewhere in the top-left quarter (from straight left to straight above, 2⁻² to ¾ of the way up), an optional fill lamp from roughly the opposite side, weaker and lower, a tint for each, a tint for the shadows, and how deep the relief is.
#[derive(Clone, Copy)]
struct Lighting {
    key: Lamp,
    fill: Lamp,
    shadow: (f32, f32),
    /// The relief's scale s in pixels (2⁰ to 2³): the surface is H = −s·ln G.
    relief: f32,
}

impl Lighting {
    fn from_bytes(f: &[u8; 32]) -> Self {
        // A second 32 bytes for the rest, a third for the band jitter.
        let g = *blake3::hash(f).as_bytes();
        let j = *blake3::Hasher::new().update(f).update(b"lighting jitter").finalize().as_bytes();
        let u = |i: usize, n: u32| band(unit(&f[i..i + 4]) as f32, unit(&j[i..i + 4]) as f32, n);
        let u2 = |i: usize, n: u32| band(unit(&g[i..i + 4]) as f32, unit(&j[(i + 16)..(i + 20)]) as f32, n);
        let quarter = TAU as f32 * 0.25;
        let tint = |h: f32| (h.cos(), h.sin());
        let lamp = |azimuth: f32, elevation: f32, strength: f32, hue: f32| Lamp {
            dir: (azimuth.cos() * elevation.cos(), azimuth.sin() * elevation.cos(), elevation.sin()),
            strength,
            tint: tint(hue),
        };
        // Screen angles run clockwise from +x (y is down): straight left is half a turn, straight above is three quarters.
        // Bands: four azimuths across the quarter, three elevations, three strengths, twelve tint hues.
        let key_az = 2. * quarter + u(0, 4) * quarter;
        let key_el = quarter * (0.25 + u(4, 3) * 0.5);
        let key_s = 0.0625 + u(8, 3) * 0.0375;
        let key_hue = u(12, 12) * TAU as f32;
        // The fill: from the opposite side ± ⅙ of a turn, 2⁻³ to ½ of the way up, a quarter to a half the key's strength (none for a quarter of devices), its hue near the key's complement.
        let fill_az = key_az + 2. * quarter + (u(16, 3) - 0.5) * quarter * (4. / 3.);
        let fill_el = quarter * (0.125 + u(20, 3) * 0.375);
        let fill_s = if f[24] & 3 == 0 { 0. } else { key_s * (0.25 + u(25, 2) * 0.25) };
        let fill_hue = key_hue + 2. * quarter + (u2(0, 3) - 0.5) * quarter;
        // Shadows lean toward the key's complement or the fill's hue.
        let shadow_hue = if f[24] & 4 == 4 { key_hue + 2. * quarter } else { fill_hue };
        Lighting {
            key: lamp(key_az, key_el, key_s, key_hue),
            fill: lamp(fill_az, fill_el, fill_s, fill_hue),
            shadow: tint(shadow_hue),
            // Relief in four bands, 2⁰ to 2³ px.
            relief: (u2(4, 4) * 3.).exp2(),
        }
    }

    /// The one fixed lighting, for comparison with the per-device kind: a key lamp from the top left at 45° up, no tint, no fill, relief 2¹ px.
    fn fixed() -> Self {
        let e = std::f32::consts::FRAC_1_SQRT_2;
        let none = Lamp { dir: (0., 0., 1.), strength: 0., tint: (0., 0.) };
        Lighting { key: Lamp { dir: (-0.5, -0.5, e), strength: 0.08, tint: (0., 0.) }, fill: none, shadow: (0., 0.), relief: 2. }
    }
}

/// A banded draw: `n` bands across [0, 1), the band picked by `pick` and a jitter within it of a quarter of the band (by `jit`), so devices cluster into recognisable families instead of spreading evenly.
fn band(pick: f32, jit: f32, n: u32) -> f32 {
    ((pick * n as f32).floor() + 0.5 + (jit - 0.5) * 0.25) / n as f32
}

impl Theme {
    /// The theme a salt's bytes give.
    fn from_bytes(f: &[u8; 32]) -> Self {
        // Every continuous choice is banded (see `band`), the jitter from a second 32 bytes.
        let j = *blake3::Hasher::new().update(f).update(b"theme jitter").finalize().as_bytes();
        let u = |i: usize, n: u32| band(unit(&f[i..i + 4]) as f32, unit(&j[i..i + 4]) as f32, n);
        let tau = TAU as f32;
        // Ground tone: pale, light, mid or dusk.
        let ground_l = [0.9, 0.82, 0.64, 0.44][(f[28] & 3) as usize];
        // The set against it: darker on pale and light grounds, lighter on dusk, either way on mid; by 2⁻⁴ to 2⁻³ + 2⁻⁴ of lightness, low contrast.
        let contrast = 0.0625 + u(8, 4) * 0.125;
        let lighter = match f[28] & 3 {
            0 | 1 => false,
            3 => true,
            _ => f[29] & 1 == 1,
        };
        let body_l = if lighter { ground_l + contrast } else { ground_l - contrast };
        // The edge: a glow lighter than both, or an ink line darker than both (the lightest a glow gets is 0.9 + 2⁻⁴ ≈ 0.96, the darkest ink 0.44 − 2⁻⁴ ≈ 0.38).
        let (hi, lo) = if lighter { (body_l, ground_l) } else { (ground_l, body_l) };
        let rim_l = if f[29] & 2 == 2 { lo - 0.0625 } else { hi + 0.0625 };
        // The set's hue turn: none, ±⅙, ±¼, ±⅓ or ½ of a turn.
        let accent = [0., 1. / 6., -1. / 6., 0.25, -0.25, 1. / 3., -1. / 3., 0.5][(f[30] & 7) as usize] * tau;
        // The ripple, from a third hash: each wave's frequency one of four bands (½, 1, 2, 4 turns per doubling of distance, jittered) and its amplitude banded too; a fifth of the waves are silent, so some devices have a plain field and others a rich one.
        let rh = *blake3::Hasher::new().update(f).update(b"ripple").finalize().as_bytes();
        let rj = *blake3::Hasher::new().update(f).update(b"ripple jitter").finalize().as_bytes();
        let ru = |i: usize, n: u32| band(unit(&rh[i..i + 4]) as f32, unit(&rj[i..i + 4]) as f32, n);
        let ripple = {
            let mut fq = [0f32; 4];
            let mut am = [0f32; 4];
            for k in 0..4 {
                fq[k] = (ru(k * 4, 4) * 4. - 1.).exp2();
                let silent = rh[k + 28] % 5 == 0;
                // Lightness waves up to 2⁻⁵ each; colour waves up to 2⁻⁶ of chroma per unit lightness (the gamut cap in `soft` still holds them).
                let top = if k < 2 { 1. / (1 << 5) as f32 } else { 1. / (1 << 6) as f32 };
                am[k] = if silent { 0. } else { top * (0.25 + 0.75 * ru(k * 4 + 2, 3)) };
            }
            (fq, am)
        };
        Theme {
            // Twelve hue families.
            hue: u(0, 12) * tau,
            spread: 0.0625 + u(4, 4) * 0.5,
            accent,
            drift: (u(12, 4) - 0.5) * 1.25,
            ground_l,
            body_l,
            rim_l,
            ground_s: 0.015 + u(16, 4) * 0.0625,
            body_s: 0.03 + u(20, 4) * 0.09375,
            rim_s: 0.015 + u(24, 4) * 0.09375,
            lighting: Lighting::from_bytes(blake3::hash(f).as_bytes()),
            ripple_f: ripple.0,
            ripple_a: ripple.1,
        }
    }
}

/// OKLab (L, a, b) to display RGB in [0, 1]: OKLab's own matrices (Björn Ottosson's published constants, not tuning knobs) to linear sRGB, then gamma 2 (√) for display.
/// Chroma held at most 0.156·L keeps every colour inside the sRGB gamut, so no channel goes negative (`soft_colours_stay_in_gamut` checks every hue).
#[inline]
fn oklab(l: f32, a: f32, b: f32) -> [f32; 3] {
    let l_ = l + 0.396_337_78 * a + 0.215_803_76 * b;
    let m_ = l - 0.105_561_346 * a - 0.063_854_17 * b;
    let s_ = l - 0.089_484_18 * a - 1.291_485_5 * b;
    let (l3, m3, s3) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    [
        4.076_741_7 * l3 - 3.307_711_6 * m3 + 0.230_969_94 * s3,
        -1.268_438 * l3 + 2.609_757_4 * m3 - 0.341_319_38 * s3,
        -0.004_196_086 * l3 - 0.703_418_6 * m3 + 1.707_614_7 * s3,
    ]
    .map(|c| c.sqrt())
}

/// The soft colour at lightness `l`, saturation `sat` (chroma = sat·l), hue offset `dh` from the theme's base, plus a tint (an OKLab (a, b) offset in units of l), always inside the display's gamut, smoothly.
/// Lightness passes a shoulder: unchanged up to 0.85, then easing toward 0.95 as 0.85 + 0.1·tanh((L − 0.85)/0.1), so the brightest grounds, glows and lit faces never wash out to white.
/// Chroma (the theme's and the tint's together) is then held under the gamut at that lightness, which is about 0.17·L at the dark end (where a channel would go negative) and about 0.49·(1 − L) at the light end (where one would pass 1), measured over every hue: the cap is the smooth minimum of 0.16·L and 0.48·(1 − L), and the chroma eases into it as cap·tanh(C/cap), C∞ and nearly unchanged well inside it.
#[inline]
fn soft(th: &Theme, l: f32, sat: f32, dh: f32, tint: (f32, f32)) -> [f32; 3] {
    let (knee, room) = (0.85f32, 0.1f32);
    let l = if l > knee { knee + room * ((l - knee) / room).tanh() } else { l };
    let (h, c) = (th.hue + dh, sat * l);
    let (a, b) = (c * h.cos() + tint.0 * l, c * h.sin() + tint.1 * l);
    let cm = (a * a + b * b).sqrt();
    let (lo, hi) = (0.16 * l, 0.48 * (1. - l));
    let d = 1. / (1 << 6) as f32;
    let cap = 0.5 * (lo + hi - ((lo - hi) * (lo - hi) + d * d).sqrt());
    let scale = if cm > 0. && cap > 0. { cap * (cm / cap).tanh() / cm } else { 0. };
    oklab(l, a * scale, b * scale)
}

/// A smooth (C∞) split of a signed amount into its positive part, (x + √(x² + δ²))/2 with δ = 2⁻⁶: about x when well positive, about 0 when well negative, and no corner at 0.
#[inline]
fn positive_part(x: f32) -> f32 {
    let d = 1. / (1 << 6) as f32;
    0.5 * (x + (x * x + d * d).sqrt())
}

/// The metals' colours in linear light, each its reflectance at normal incidence (published F0 measurements; rose gold and blued titanium as their alloys and oxide show): gold, silver, copper, bronze, steel, rose gold, brass, titanium.
const METALS: [[f32; 3]; 8] = [
    [1.00, 0.71, 0.29],
    [0.95, 0.93, 0.88],
    [0.95, 0.64, 0.54],
    [0.80, 0.55, 0.34],
    [0.56, 0.57, 0.58],
    [0.97, 0.70, 0.60],
    [0.91, 0.78, 0.42],
    [0.52, 0.56, 0.70],
];

/// A studio around the picture, as a metal sees it: the brightness reflected along `rr` (a unit vector, screen x right, y down, z out), a bright sky above falling smoothly to a dim floor below, and a faint soft-box at the key lamp `l`.
#[inline]
fn studio(rr: (f32, f32, f32), l: (f32, f32, f32)) -> f32 {
    let sky = 0.5 - 0.5 * (3. * rr.1).tanh();
    let lamp = positive_part(rr.0 * l.0 + rr.1 * l.1 + rr.2 * l.2).powi(1 << 5);
    0.0625 + 0.5 * sky + 0.375 * lamp
}

/// A metal of colour `base` (linear) with unit normal `n`, lit by the studio: the view straight down the z axis reflected about n, tinted by the metal, in linear light.
#[inline]
fn metal(base: [f32; 3], n: (f32, f32, f32), l: (f32, f32, f32)) -> [f32; 3] {
    let rr = (2. * n.2 * n.0, 2. * n.2 * n.1, 2. * n.2 * n.2 - 1.);
    let e = studio(rr, l);
    base.map(|b| b * e)
}

/// Linear light to display, through the exposure curve 1 − e^(−c) (C∞, every brightness lands below white, highlights roll off instead of clipping), then gamma 2.
#[inline]
fn expose(lin: [f32; 3]) -> [f32; 3] {
    lin.map(|c| (1. - (-c).exp()).sqrt())
}

/// A unit vector.
#[inline]
fn unit3(v: (f32, f32, f32)) -> (f32, f32, f32) {
    let l = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    (v.0 / l, v.1 / l, v.2 / l)
}

/// The look's tweaks, applied at render time: `offset` is a complex offset of the colouring coordinate (see `shade`); `light` is the direction the disc is lit from, (x right, y down, z toward the viewer), normalised by the renderer.
#[derive(Clone, Copy)]
pub struct Look {
    pub offset: (f32, f32),
    /// A multiplier on the device's relief depth (1 = as the salt chose it).
    pub relief: f32,
    /// One fixed light for every device (`Lighting::fixed`) instead of each device's own.
    pub fixed_light: bool,
    /// The rim vignette's radial power k: the disc is multiplied by √(1 − |p|^k) (2 for Photon's avatars; ferros's attest screen uses 12, √(1 − |p|¹²), flat almost to the rim and black in a narrow band there).
    pub vignette: i32,
    /// The attest features (see `Attest`): the ball's material, the engraving, and the mirror.
    pub material: Pick,
    pub engrave: Pick,
    pub mirror: Pick,
    /// How the picture is made (see `Attest::style`): 0 soft colours, 1 medal, 2 enamel, 3 lacquer.
    pub style: Pick,
    /// Wrap the picture as a round button: the ball a little smaller, over a soft drop shadow.
    pub button: bool,
    /// The flame as electricity (attest): arcs along the flame's contours, white-hot at their cores, haloed in the device's colour, added as light; 0 off (the avatars), 1 as designed, more or less to taste.
    pub electric: f32,
    /// How the button samples each pixel (see `Aa`); the plain disc always samples linearly.
    pub aa: Aa,
}

impl Default for Look {
    /// Photon's avatar: no offset, the device's own relief and lighting, √(1 − |p|²), no attest features.
    fn default() -> Self {
        Look { offset: (0., 0.), relief: 1., fixed_light: false, vignette: 2, material: Pick::Off, engrave: Pick::Off, mirror: Pick::Off, style: Pick::Off, button: false, electric: 0., aa: Aa::Linear }
    }
}

impl Look {
    /// Photon's avatar (the default).
    pub fn avatar() -> Self {
        Look::default()
    }

    /// ferros's attest mark: √(1 − |p|¹²), every attest feature as the device drew it, wrapped as a button.
    pub fn attest() -> Self {
        Look { vignette: 12, material: Pick::Device, engrave: Pick::Device, mirror: Pick::Device, style: Pick::Device, button: true, electric: 1., ..Look::default() }
    }
}

/// A lamp's direction in the plane: its horizontal part turned by the view's `rot` (plane = centre + rot·screen), its height unchanged.
#[inline]
fn in_plane(d: (f32, f32, f32), rot: (f32, f32)) -> (f32, f32, f32) {
    (d.0 * rot.0 - d.1 * rot.1, d.0 * rot.1 + d.1 * rot.0, d.2)
}
/// One point of the plane as a visible colour in [0, 1]; `k` is plane units per pixel; `sphere` is the ball's unit normal under it (screen x right, y down, z out; (0, 0, 1) for a flat picture), and `hair` the engraving's line width in pixels.
/// One point of the plane as a visible colour in [0, 1]; `k` is plane units per pixel; `lp` is the light's direction in the plane.
/// One soft scheme, low contrast: the field is the theme's ground, the set's body its second tone, the edge a glow or an ink line; nothing goes near black or white, and nothing is saturated.
/// The colouring coordinate is complex, u = (the colouring method's value) + i·(the orbit's escape angle in turns), plus the look's offset: Re u swings the hue either side of the theme's base (by the theme's spread) and Im u breathes the lightness a little, so both halves of the offset move every method.
/// Lighting, from the slope itself: the picture is the surface H = −s·ln G (s = `look.relief` pixels, G the escape potential), a ridge along the set falling away into the field.
/// Its gradient points along u = z/(dz/dz₀) (the potential's own direction, outward) with magnitude s/d per pixel at d pixels from the set (d being the distance estimate, G/|∇G| up to a constant), so its unit normal is (sin α·u, cos α) with sin α = s/√(s² + d²), cos α = d/√(s² + d²), exact and finite even on the set (d = 0).
/// Each lamp's Lambert n·l, taken relative to a flat surface (which sees l_z), moves the lightness by strength·(n·l − l_z): faces turned to it rise, faces turned away fall, the flat field far out is untouched, and nothing is clipped. The faces a lamp lights take its tint, and the faces left in shade take the shadow tint, each split off smoothly (`positive_part`).
#[inline]
fn shade(x: f32, y: f32, v: &View, k: f32, look: &Look, sphere: (f32, f32, f32), hair: f32) -> ([f32; 3], f32) {
    let o = orbit(x, y, v);
    let r = (o.x * o.x + o.y * o.y).sqrt();
    let escaped = o.n < MAX_ITER;
    // Smooth iteration count, normalised to the escape: n + 1 − log₂(ln|z|² / ln bail).
    let nu = if escaped { o.n as f32 + 1. - ((r * r).ln() / BAIL_SQ_F32.ln()).log2() } else { MAX_ITER as f32 };
    let tau = TAU as f32;
    // The flame, as mandelbrot-exploder colours it: (conj(fc) − fq)·ln(smooth count)/2⁴, a complex number.
    let flame = { let g = nu.ln() * (1. / (1 << 4) as f32); ((o.fc.0 - o.fq.0) * g, (-o.fc.1 - o.fq.1) * g) };
    // The potential's gradient direction (u = z/(dz/dz₀)) in turns: continuous everywhere in the field. The angle of z itself is not usable at a fixed radius (it jumps at every level line, banding the field) nor at the far escape (it turns 2ⁿ times faster and aliases).
    let gangle = o.uy.atan2(o.ux) / tau;
    // Every method carries the flame on top (the owner likes the look, 2026-10-29); the fourth is the flame alone.
    let (re, im) = match v.mode {
        0 => (nu * (1. / (1 << 4) as f32) + flame.0, gangle + flame.1),
        // The potential's gradient direction, turned by the smooth count: a continuous swirl around the set.
        1 => (gangle + nu * (1. / (1 << 5) as f32) + flame.0, gangle + flame.1),
        // How close the orbit came to the trap, on a log scale: rings and petals around each preimage of the trap.
        2 => (-(o.trap_sq.ln()) * (1. / (1 << 3) as f32) + flame.0, gangle + flame.1),
        // The flame itself: its real part along the colouring, its imaginary part breathing.
        _ => (flame.0, flame.1),
    };
    let (re, im) = (re + look.offset.0 + v.keyed.offset.0, im + look.offset.1 + v.keyed.offset.1);
    let swing = (tau * re).sin();
    // Distance to the set: the real distance estimate, |z|·ln|z| / |dz/dz₀|, in plane units and in pixels (0 on it).
    let dist = if escaped { r * r.ln() / o.dm * (-(o.dm_exp as f32)).exp2() } else { 0. };
    let dist_px = dist / k;
    // Distance contours: the theme's ripple, four waves across log₂ of the distance from the set, the field's own topography, their phases nudged by the key.
    // Near the set a wave of f turns per doubling has f/(d·ln 2) contours per pixel at d pixels out and would alias into hatching, so it fades in as d²/(d² + (4f/ln 2)²): gone within a few pixels of the set, full once its contours are over 4 pixels apart.
    let crowd = 4. / std::f32::consts::LN_2;
    // The faster a wave, the more closely its contours crowd near the set, so each fades in by its own frequency: d²/(d² + (crowd·f)²).
    let ld = dist.log2();
    let th0 = &v.theme;
    let wave = |k: usize| {
        let f = th0.ripple_f[k];
        let fade = dist_px * dist_px / (dist_px * dist_px + (crowd * f) * (crowd * f));
        th0.ripple_a[k] * (tau * (f * ld + v.keyed.phase[k])).sin() * fade
    };
    let (rl, ra, rb) = if escaped { (wave(0) + wave(1), wave(2), wave(3)) } else { (0., 0., 0.) };
    // The soft flame (every look): along the flame's contours at 2² times its own scale, a gentle lightness wave (2⁻⁴ either way, from its imaginary part) and a hue wave (¼ rad either way, from its real part), so its tongues read in the open field in the theme's own tones; it fades in away from the set by its contours' crowding, as the ripple's waves do.
    let (flame_l, flame_h) = if escaped {
        let fade = dist_px * dist_px / (dist_px * dist_px + (1 << 4) as f32);
        ((tau * 4. * flame.1).sin() * (1. / (1 << 4) as f32) * fade, (tau * 4. * flame.0).sin() * 0.25 * fade)
    } else {
        (0., 0.)
    };
    let breathe = (tau * im).cos() * (1. / (1 << 6) as f32) + rl + flame_l;
    // Body: within about a pixel of the set, falling smoothly (C∞) to nothing a few pixels out.
    let body = 1. / (1. + dist_px * dist_px);
    // Rim: strongest about 2¹ pixels out, a smooth bump in log distance.
    let ll = (dist_px + 1. / (1 << 4) as f32).log2() - 1.;
    let rim = (-(ll * ll)).exp2();
    // Far: how far out in the field, 0 near the set to 1 far away (the disc's own rim, the plane's infinity, is fully far).
    let far = (-nu * 0.125).exp2();
    let th = &v.theme;
    let fixed = Lighting::fixed();
    let lt = if look.fixed_light { &fixed } else { &th.lighting };
    // The surface's tilt from the slope (the parameter plane has no u, so it lights flat).
    let sr = lt.relief * look.relief * v.keyed.relief;
    // The key's turn of the lamps, applied with the view's own.
    let turn = (TAU as f32 * v.keyed.light_turn).sin_cos();
    let lrot = (v.rot.0 * turn.1 - v.rot.1 * turn.0, v.rot.0 * turn.0 + v.rot.1 * turn.1);
    let hyp = (sr * sr + dist_px * dist_px).sqrt();
    let (sin_a, cos_a) = (sr / hyp, dist_px / hyp);
    let julia = v.kind == Kind::Julia;
    let lit = |lamp: &Lamp| {
        let d = in_plane(lamp.dir, lrot);
        let n_dot_l = if julia { sin_a * (o.ux * d.0 + o.uy * d.1) + cos_a * d.2 } else { d.2 };
        lamp.strength * (n_dot_l - d.2)
    };
    let (key, fill) = (lit(&lt.key), lit(&lt.fill));
    let shading = key + fill;
    // Tints: each lamp's on the faces it lights, the shadow's on the faces in shade, each at most 2⁻⁶ of chroma per unit lightness (4 · 2⁻⁶ for the key) and saturating smoothly as w/(w + 2⁻⁵).
    let soft_w = |w: f32| w / (w + 1. / (1 << 5) as f32);
    let (kw, fw, sw) = (soft_w(positive_part(key)), soft_w(positive_part(fill)), soft_w(positive_part(-shading)));
    let (k_t, f_t, s_t) = (0.02, 0.01, 0.01);
    let tint = (
        lt.key.tint.0 * kw * k_t + lt.fill.tint.0 * fw * f_t + lt.shadow.0 * sw * s_t + ra,
        lt.key.tint.1 * kw * k_t + lt.fill.tint.1 * fw * f_t + lt.shadow.1 * sw * s_t + rb,
    );
    // Nacre: the hue turns with the surface's facing across the key lamp (the normal's horizontal part crossed with it), up to ¼ rad either way, the way mother-of-pearl shifts colour with angle; only where the surface tilts.
    let kd = in_plane(lt.key.dir, lrot);
    let kh = (kd.0 * kd.0 + kd.1 * kd.1).sqrt();
    let nacre = if julia { 0.25 * sin_a * (o.ux * kd.1 - o.uy * kd.0) / kh } else { 0. };
    let edge = rim * (1. - body);
    // Lightness and saturation: the ground's, moving to the body's on the set and the edge's around it, settling back to the ground far out; the slope's shading on top, everywhere it tilts.
    let lightness = th.ground_l + (th.body_l - th.ground_l) * body + (th.rim_l - th.ground_l) * edge * (1. - far) + shading + breathe;
    let sat = th.ground_s + (th.body_s - th.ground_s) * body + (th.rim_s - th.ground_s) * edge * (1. - far);
    // How the picture is made: soft colours (the avatars'), a medal, or enamel (see `Attest::style`).
    let style = look.style.of(v.attest.style);
    // The engraving (attest): hairlines `hair` pixels wide on the level lines of the potential and on the rays, both read where the orbit first passed 2⁸ (see `Orbit`), so each level's rays meet the next level's twice as many exactly on a level line. Enamel needs its wires, so a device with no engraving gets level lines there.
    // A line's distance in pixels is its coordinate's distance to the nearest whole number over the coordinate's change per pixel, with |sin(πx)|/π standing in for that distance (C∞, and the same near a line); each family fades in as its lines get more than 2³ hairs apart (`apart`), so they never crowd into hatching near the set.
    let engrave = look.engrave.of(v.attest.engrave);
    let engrave = if style == 2 && engrave == 0 { 1 } else { engrave };
    let pi = std::f32::consts::PI;
    let (m_lv, n_ray) = (v.attest.levels, v.attest.rays);
    let lv_px = m_lv / (std::f32::consts::LN_2 * dist_px);
    let ray_px = n_ray * o.ln_turn.exp() * k / tau;
    let apart = |per_px: f32| {
        let s = 1. / per_px;
        let crowd = 8. * hair;
        s * s / (s * s + crowd * crowd)
    };
    let ink = |x: f32, per_px: f32| {
        let d = (pi * x).sin() / (pi * per_px * hair);
        (-(d * d)).exp2() * apart(per_px)
    };
    let line = if escaped {
        let levels = if engrave & 1 == 1 { ink(o.lv * m_lv, lv_px) } else { 0. };
        let rays = if engrave & 2 == 2 { ink(o.ray * n_ray, ray_px) } else { 0. };
        1. - (1. - levels) * (1. - rays)
    } else {
        0.
    };
    // The surface for the attest styles, in screen terms: the relief's horizontal tilt brought back from the plane (the conjugate of the view's turn), and the key lamp turned by the key.
    let (px, py) = if julia { (sin_a * o.ux, sin_a * o.uy) } else { (0., 0.) };
    let (rx, ry) = (px * v.rot.0 + py * v.rot.1, -px * v.rot.1 + py * v.rot.0);
    let l = { let d = lt.key.dir; (d.0 * turn.1 - d.1 * turn.0, d.0 * turn.0 + d.1 * turn.1, d.2) };
    let half = unit3((l.0, l.1, l.2 + 1.));
    let facing = |n: (f32, f32, f32)| positive_part(n.0 * half.0 + n.1 * half.1 + n.2 * half.2);
    // How raised the set stands for the metal styles: 1 on it, falling smoothly (C∞) to nothing about 2 hairs out, a width in the picture's own terms, so a dense whorl still shows its field between the dust.
    let raised = { let d = dist_px / (2. * hair); 1. / (1. + d * d) };
    let rgb = match style {
        0 => {
            // A line moves the lightness 2⁻³ the way the theme's edge goes from its ground (ink darker, glow lighter) and takes the edge's saturation.
            let way = if th.rim_l > th.ground_l { 1. } else { -1. };
            let lightness = lightness + way * 0.125 * line;
            let sat = sat + (th.rim_s - sat) * line;
            // Hue: the ground's base, drifting toward the far field, swung by the colouring, turned by the nacre, and turned by the theme's accent on the set.
            let rgb = soft(th, lightness, sat, th.spread * swing + flame_h + th.drift * far + nacre + th.accent * body, tint);
            // The material (attest): a highlight of the key lamp on the ball, its normal the sphere's (`sphere`, screen x right, y down, z out) tilted by a quarter of the relief's slope, and for glass a Fresnel sheen toward the rim. Added as a screen blend, c + s·(1 − c), so it brightens without clipping.
            let material = look.material.of(v.attest.material);
            if material == 0 {
                rgb
            } else {
                let n = unit3((sphere.0 + rx * 0.25, sphere.1 + ry * 0.25, sphere.2));
                let nh = facing(n);
                let spec = match material {
                    1 => 0.25 * nh.powi(1 << 3),
                    2 => 0.125 * nh.powi(1 << 3) + 0.75 * nh.powi(1 << 6),
                    _ => 0.875 * nh.powi(1 << 8) + 0.375 * (1. - n.2).powi(1 << 2),
                };
                rgb.map(|c| c + spec * (1. - c))
            }
        }
        1 => {
            // A medal: the relief struck in metal on a faintly domed field (a quarter of the ball's curve), the set raised in `metal_a` and the field in `metal_b` (one metal or two), the engraving cut in as darker grooves.
            let n = unit3((sphere.0 * 0.25 + rx, sphere.1 * 0.25 + ry, 1.));
            let (a, b) = (METALS[v.attest.metal_a as usize], METALS[v.attest.metal_b as usize]);
            let base = [0, 1, 2].map(|i| b[i] + (a[i] - b[i]) * raised);
            expose(metal(base, n, l).map(|c| c * (1. - 0.5 * line)))
        }
        2 => {
            // Enamel (cloisonné): each cell between the engraving's lines filled with one of four colours around the theme's hue (richer than the avatars' but held under the same gamut cap), the set's dust and the wires in `metal_a`, under a faint glaze.
            // A cell's colour comes from a hash of its level and ray indices; where the cells crowd near the set (as their lines fade) the colours fade to the first of the four, the theme's own hue, so they never speckle.
            let cl = (o.lv * m_lv).floor() as i32;
            let cr = if engrave & 2 == 2 { (o.ray * n_ray).floor() as i32 } else { 0 };
            let which = (jitter(cl as u32, cr as u32, v.attest.cells).0 * 4.) as usize;
            let lift = (th.ground_l - 0.67) * 0.5;
            let pal = [(0.62, 0.), (0.52, tau / 3.), (0.72, -tau / 3.), (0.44, tau / 2.)].map(|(pl, dh)| soft(th, pl + lift, 0.15, dh, (0., 0.)).map(|c| c * c * 1.25));
            let fade = if escaped { apart(lv_px) * if engrave & 2 == 2 { apart(ray_px) } else { 1. } } else { 0. };
            let cell = [0, 1, 2].map(|i| pal[0][i] + (pal[which][i] - pal[0][i]) * fade);
            let n = unit3((sphere.0 * 0.5 + rx, sphere.1 * 0.5 + ry, 1.));
            let gold = metal(METALS[v.attest.metal_a as usize], n, l);
            let w = 1. - (1. - raised) * (1. - line);
            let lin = [0, 1, 2].map(|i| cell[i] + (gold[i] - cell[i]) * w);
            // The glaze: a faint sheen of the key lamp on the relief's own slope (a quarter of the ball's curve under it), so it catches the set's walls instead of sitting on the ball like glass.
            let spec = 0.125 * facing(unit3((sphere.0 * 0.25 + rx, sphere.1 * 0.25 + ry, 1.))).powi(1 << 5);
            expose(lin).map(|c| c + spec * (1. - c))
        }
        _ => {
            // Lacquer: the avatars' own colouring, a step darker and twice as saturated (the gamut cap in `soft` still holds it), the engraving inked in, the set raised in `metal_a`, under the same glaze as enamel.
            let way = if th.rim_l > th.ground_l { 1. } else { -1. };
            let lacquer = soft(th, lightness * 0.75 + way * 0.125 * line, sat * 2., th.spread * swing + flame_h + th.drift * far + nacre + th.accent * body, tint).map(|c| c * c * 1.25);
            let n = unit3((sphere.0 * 0.5 + rx, sphere.1 * 0.5 + ry, 1.));
            let gold = metal(METALS[v.attest.metal_a as usize], n, l);
            let lin = [0, 1, 2].map(|i| lacquer[i] + (gold[i] - lacquer[i]) * raised);
            let spec = 0.125 * facing(unit3((sphere.0 * 0.25 + rx, sphere.1 * 0.25 + ry, 1.))).powi(1 << 5);
            expose(lin).map(|c| c + spec * (1. - c))
        }
    };
    // The electric flame (attest): arcs where the flame's real or imaginary part, at the device's `arcs` times its own scale, crosses a whole number — the contours that curl off the set as tongues. Each arc is a core (2⁻⁴ of a contour wide) inside a halo (¼ of one wide, falling off as 1/(1 + x²)), and between them wisps: the same contours at 2² times the density, a thin core 2⁻⁵ wide in the arc's colour, a touch toward white, and no halo.
    // Chromatic: an arc along one part takes its colour from the other part, the hue swinging up to a third of a turn either side of the theme's as the arc runs, so one arc passes through neighbouring hues; its core is that colour run toward white, hot but not blank.
    // Added as light with a screen blend (on a ground dimmed by ⅜, a plasma globe's dark glass), so it brightens without clipping. Like the ripple, it fades in away from the set, where its contours would crowd into noise, and the wisps, four times denser, fade in further out.
    let rgb = if look.electric > 0. && escaped {
        let line = |x: f32, core_w: f32| {
            let s = (pi * x).sin();
            let (c, h) = (s / core_w, s * 4.);
            ((-(c * c)).exp2(), 1. / (1. + h * h))
        };
        let gain = v.attest.arcs;
        let (c0, h0) = line(flame.0 * gain, 1. / (1 << 4) as f32);
        let (c1, h1) = line(flame.1 * gain, 1. / (1 << 4) as f32);
        let (w0, _) = line(flame.0 * gain * 4., 1. / (1 << 5) as f32);
        let (w1, _) = line(flame.1 * gain * 4., 1. / (1 << 5) as f32);
        let hue = |other: f32| soft(th, 0.7, 0.2, (tau / 3.) * (tau * other * gain * 0.25).sin(), (0., 0.));
        let (k0, k1) = (hue(flame.1), hue(flame.0));
        let d2 = dist_px * dist_px;
        let fade = d2 / (d2 + (1 << 4) as f32) * look.electric;
        let wisp_fade = d2 / (d2 + (1 << 6) as f32) * look.electric;
        [0, 1, 2].map(|i| {
            let arc = |k: [f32; 3], c: f32, h: f32| (k[i] * (h * 0.625 + c * 0.5) + c * 0.375) * fade;
            let wisp = |k: [f32; 3], w: f32| (k[i] * 0.625 + 0.125) * w * wisp_fade;
            let e = 1. - (1. - arc(k0, c0, h0)) * (1. - arc(k1, c1, h1)) * (1. - wisp(k0, w0)) * (1. - wisp(k1, w1));
            let base = rgb[i] * (1. - 0.375 * look.electric);
            base + e * (1. - base)
        })
    } else {
        rgb
    };
    (rgb, dist_px)
}


/// A visible colour in [0, 1] per channel at opacity α in [0, 1], as a fluor α + darkness pixel: darkness is premultiplied by α (an empty pixel is 0, opaque black is all ones).
#[inline]
fn pack(rgb: [f32; 3], alpha: f32) -> u32 {
    let a = (alpha * 255.) as u32;
    let d = |c: f32| ((1. - c) * alpha * 255.) as u32;
    (a << 24) | (d(rgb[0]) << 16) | (d(rgb[1]) << 8) | d(rgb[2])
}

/// Render the mark into `buf` (`w × h`, α + darkness), rows across the cores, in f32.
pub fn render(buf: &mut [u32], w: usize, h: usize, p: &Params) {
    let k = scale(p.half, w, h);
    let v = p.view();
    buf.par_chunks_mut(w).enumerate().for_each(|(py, row)| {
        for px in 0..w {
            let (x, y) = plane(&v, px, py, w, h, k);
            row[px] = pack(shade(x, y, &v, k, &Look::default(), (0., 0., 1.), 1.).0, 1.);
        }
    });
}

/// The most samples one pixel takes.
pub const MAX_SAMPLES: usize = 1 << 6;

/// A cheap integer hash (a 32-bit avalanche: xor-shift and multiply rounds, the constants are the mix's design) of a pixel and a sample index, as a point in the unit square.
#[inline]
fn jitter(px: u32, py: u32, m: u32) -> (f32, f32) {
    let mut h = px.wrapping_mul(0x9E37_79B1) ^ py.wrapping_mul(0x85EB_CA77) ^ m.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    let scale = 1. / (1u32 << 16) as f32;
    ((h & 0xFFFF) as f32 * scale, (h >> 16) as f32 * scale)
}

/// Render the mark spherized into a disc of radius `r` pixels centred at (`cx`, `cy`) in `buf` (rows of `w`, α + darkness); pixels outside the disc are left empty (α = 0), so it can go over any background.
/// A disc point p (in units of r) shows z = eye + R·rot·p/√(1 − |p|²): the plane wrapped onto a hemisphere seen from above, its infinity at the rim.
/// Everything is multiplied by √(1 − |p|^k), k the look's `vignette`, so the ball tapers to black at its rim, where the warp packs the field tightest.
/// The disc's edge is the ordinary anti-aliased circle (as Photon's avatars): full inside r − ½, empty outside r + ½, and between them (r + ½)² − d² over (r + ½)² − (r − ½)².
/// Sampling is adaptive and jittered: each sample sits at a hashed point inside its pixel (`jitter`, deterministic, so the same mark renders the same everywhere). Every pixel takes two; while the newest sample differs from the mean of the ones before it by more than m levels (8-bit, in the display's gamma, any channel) after m samples, the pixel takes another, up to `MAX_SAMPLES`. Samples are averaged in linear light.
pub fn render_disc(buf: &mut [u32], w: usize, cx: f32, cy: f32, r: f32, p: &Params, look: &Look) {
    render_disc_counted(buf, w, cx, cy, r, p, look);
}

/// As `render_disc`, returning how many pixels took each number of samples (index = samples taken; pixels outside the disc are not counted).
pub fn render_disc_counted(buf: &mut [u32], w: usize, cx: f32, cy: f32, r: f32, p: &Params, look: &Look) -> Vec<u64> {
    if look.button {
        return render_button(buf, w, cx, cy, r, p, look);
    }
    let v = mirrored(p.view(), look);
    // The engraving's hairline: 2⁻⁷ of the disc's radius.
    let hair = r * (1. / (1 << 7) as f32);
    let rad = p.radius as f32;
    let (r_in, r_out) = ((r - 0.5).max(0.), r + 0.5);
    let (r_in2, r_out2) = (r_in * r_in, r_out * r_out);
    // One sample at disc offset (dx, dy) pixels from the centre: its colour (display gamma), already tapered.
    let sample = |dx: f32, dy: f32| -> [f32; 3] {
        let (qx, qy) = (dx / r, dy / r);
        let q2 = qx * qx + qy * qy;
        // At and past the rim the taper is 0: black.
        if q2 >= 1. {
            return [0.; 3];
        }
        let g = rad / (1. - q2).sqrt();
        let (a, b) = (qx * g, qy * g);
        let (x, y) = (v.centre.0 + a * v.rot.0 - b * v.rot.1, v.centre.1 + a * v.rot.1 + b * v.rot.0);
        // Plane units per pixel: the map's area scale, √(radial · tangential stretch) = R/(1 − |p|²) per disc unit.
        let k = rad / (r * (1. - q2));
        // |p|^k as (|p|²)^(k/2): k is even in every use, so it stays a whole power.
        let taper = (1. - q2.powi(look.vignette / 2)).sqrt();
        shade(x, y, &v, k, look, (qx, qy, (1. - q2).sqrt()), hair).0.map(|c| c * taper)
    };
    buf.par_chunks_mut(w)
        .enumerate()
        .map(|(py, row)| {
            let mut hist = vec![0u64; MAX_SAMPLES + 1];
            for px in 0..w {
                let (dx, dy) = (px as f32 + 0.5 - cx, py as f32 + 0.5 - cy);
                let d2 = dx * dx + dy * dy;
                if d2 >= r_out2 {
                    row[px] = 0;
                    continue;
                }
                let cover = if d2 <= r_in2 { 1. } else { (r_out2 - d2) / (r_out2 - r_in2) };
                // Linear-light sum of the samples so far.
                let mut sum = [0f32; 3];
                let mut m = 0usize;
                loop {
                    let (jx, jy) = jitter(px as u32, py as u32, m as u32);
                    let s = sample(px as f32 + jx - cx, py as f32 + jy - cy);
                    // The newest sample against the mean of the ones before it, in 8-bit display levels.
                    let spread = if m == 0 {
                        f32::MAX
                    } else {
                        let mean = sum.map(|c| (c / m as f32).sqrt());
                        (0..3).map(|i| (s[i] - mean[i]).abs() * 255.).fold(0., f32::max)
                    };
                    for i in 0..3 {
                        sum[i] += s[i] * s[i];
                    }
                    m += 1;
                    // Two samples always; after that another while the last one moved by more than m − 1 levels (2 after the second, 3 after the third, …).
                    let threshold = m as f32;
                    if m >= MAX_SAMPLES || (m >= 2 && spread <= threshold) {
                        break;
                    }
                }
                hist[m] += 1;
                row[px] = pack(sum.map(|c| (c / m as f32).sqrt()), cover);
            }
            hist
        })
        .reduce(|| vec![0u64; MAX_SAMPLES + 1], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect())
}

/// How far the button's glow reaches past the ball's edge, as a fraction of its radius: 1/12, as ferros's circles glow (payloads/display, `GLOW_FRACTION`, owner 2026-10-24); and its fade, 2^GLOW_OCTAVES_LOG2 octaves of squared distance from full light to none (four, as `GLOW_OCT_LOG2`).
const GLOW_FRACTION: f32 = 12.;
const GLOW_OCTAVES_LOG2: i32 = 2;

/// The button's ball, as a fraction of the disc's radius: small enough that its glow ends exactly at the disc's edge, BALL·(1 + 1/GLOW_FRACTION) = 1.
const BALL: f32 = GLOW_FRACTION / (GLOW_FRACTION + 1.);

/// The mark as a round button (attest), in a disc of radius `r` pixels centred at (`cx`, `cy`): the picture as a ball, spherized as `render_disc` does, its rim tapering (by the look's vignette) into the device's own colour instead of to black, and that colour glowing on past the edge as ferros's circles glow (`draw::glow_log`): full at the edge, down by the same step every octave of q = s² − BALL² (the squared distance past it), gone 1/GLOW_FRACTION of the radius out, which is the disc's edge. The glow's ends are soft (`positive_part`), not clipped, and it is held to nothing at the disc's edge by 1 − s¹⁶. No rim, no glass: what shine there is comes from the set's own slope (see `shade`).
/// The colour is the theme's hue at lightness 0.72, as saturated as the soft scheme allows.
/// The ball is opaque and the glow is light over the background (coverage = its brightness), so samples are averaged with their coverage (premultiplied, in linear light); pixels across the ball's edge take at least 2³ samples, since a sharp circle can fall between two jittered samples on one side.
fn render_button(buf: &mut [u32], w: usize, cx: f32, cy: f32, r: f32, p: &Params, look: &Look) -> Vec<u64> {
    let v = mirrored(p.view(), look);
    let r_pic = r * BALL;
    let hair = r_pic * (1. / (1 << 7) as f32);
    let rad = p.radius as f32;
    let aura = soft(&v.theme, 0.72, 0.15, 0., (0., 0.));
    let b2 = BALL * BALL;
    // Where the glow ends (the disc's edge) in q, and how many octaves of q it fades over.
    let q_end = 1. - b2;
    let octaves = (1 << GLOW_OCTAVES_LOG2) as f32;
    // One sample at (dx, dy) pixels from the centre: its colour (display gamma) and its coverage.
    let sample = |dx: f32, dy: f32| -> ([f32; 3], f32) {
        let (sx, sy) = (dx / r, dy / r);
        let s2 = sx * sx + sy * sy;
        if s2 >= 1. {
            return ([0.; 3], 0.);
        }
        // Where the sample falls against the ball, as one ratio (|p|² in the ball's units), so the glow and the ball split on the same number.
        // WHY: the edge itself must belong to neither side's singularity. Tested apart (s2 ≥ b2 for the glow, then (sx/BALL)² + (sy/BALL)² for the ball), a sample exactly on the edge took log₂(0) = −∞ in the glow (its soft clamp then gave ∞ − ∞) and one a rounding inside it took 1/√(1 − q2) with q2 rounded to 1; either NaN poisoned its pixel's whole sum (measured: about one pixel in 2¹⁰ on the rim at 2¹⁰ samples, 2026-10-29).
        let q2 = s2 / b2;
        if q2 > 1. {
            // q2 > 1 means s2 > b2 exactly, so the log is finite. u: 0 at q_end/2⁴ (full light holds inside it), 1 at q_end; the light 1 − u between, softly 1 before and 0 after.
            let u = 1. + (s2 - b2).log2() / octaves - q_end.log2() / octaves;
            let light = (positive_part(1. - u) - positive_part(-u)) * (1. - s2.powi(1 << 3));
            return (aura, light);
        }
        // On the rim itself the warp is infinite and the taper is 0: only the device's colour shows.
        if q2 == 1. {
            return (aura, 1.);
        }
        let (qx, qy) = (sx / BALL, sy / BALL);
        let g = rad / (1. - q2).sqrt();
        let (a, b) = (qx * g, qy * g);
        let (x, y) = (v.centre.0 + a * v.rot.0 - b * v.rot.1, v.centre.1 + a * v.rot.1 + b * v.rot.0);
        let k = rad / (r_pic * (1. - q2));
        let taper = (1. - q2.powi(look.vignette / 2)).sqrt();
        let c = shade(x, y, &v, k, look, (qx, qy, (1. - q2).sqrt()), hair).0;
        ([0, 1, 2].map(|i| aura[i] + (c[i] - aura[i]) * taper), 1.)
    };
    let e_out = r * BALL;
    // The fewest samples a pixel takes before it may stop (doubling passes).
    let floor = if let Aa::Doubling { min_log2, .. } = look.aa { 1usize << min_log2 } else { 0 };
    // The most samples a pixel can take under this look.
    let cap = match look.aa {
        Aa::Linear => MAX_SAMPLES,
        Aa::Doubling { max_log2, .. } => 1 << max_log2,
        Aa::Fixed(n) => n as usize,
        Aa::Stratified(k) => (k * k) as usize,
    };
    buf.par_chunks_mut(w)
        .enumerate()
        .map(|(py, row)| {
            let mut hist = vec![0u64; cap + 1];
            for px in 0..w {
                let (dx, dy) = (px as f32 + 0.5 - cx, py as f32 + 0.5 - cy);
                let d = (dx * dx + dy * dy).sqrt();
                if d >= r + 1. {
                    row[px] = 0;
                    continue;
                }
                // Across the ball's edge: at least 2³ samples.
                let least = if (d - e_out).abs() < 1. { 1 << 3 } else { 2 };
                // Linear-light sums, premultiplied by coverage, and the coverage's sum; for doubling passes, the same sums as they stood before the current pass, and where it ends.
                let mut sum = [0f32; 3];
                let mut cover = 0f32;
                let mut m = 0usize;
                let (mut before, mut before_cover, mut pass_end) = ([0f32; 3], 0f32, 1usize);
                loop {
                    let (jx, jy) = jitter(px as u32, py as u32, m as u32);
                    // Stratified: the m-th sample in cell (m mod k, m div k) of the k × k grid.
                    let (jx, jy) = if let Aa::Stratified(k) = look.aa { let (kf, mu) = (k as f32, m as u32); (((mu % k) as f32 + jx) / kf, ((mu / k) as f32 + jy) / kf) } else { (jx, jy) };
                    let (s, a) = sample(px as f32 + jx - cx, py as f32 + jy - cy);
                    // Linear: the newest sample against the mean of the ones before it, premultiplied, in 8-bit display levels (coverage counted as a fourth channel).
                    let spread = if look.aa == Aa::Linear && m > 0 {
                        let ma = cover / m as f32;
                        let mean = sum.map(|c| (c / m as f32).sqrt());
                        let now = s.map(|c| c * a.sqrt());
                        (0..3).map(|i| (now[i] - mean[i]).abs() * 255.).fold((a - ma).abs() * 255., f32::max)
                    } else {
                        f32::MAX
                    };
                    for i in 0..3 {
                        sum[i] += s[i] * s[i] * a;
                    }
                    cover += a;
                    m += 1;
                    let done = match look.aa {
                        Aa::Linear => m >= cap || (m >= least && spread <= m as f32),
                        Aa::Fixed(_) | Aa::Stratified(_) => m >= cap,
                        Aa::Doubling { threshold, .. } => {
                            if m < pass_end {
                                false
                            } else {
                                // A pass has just ended: its mean against the mean of all before it (the first sample alone has nothing to compare).
                                let old = pass_end / 2;
                                let agree = old > 0 && {
                                    let (n_new, n_old) = ((m - old) as f32, old as f32);
                                    let level = |s: f32, n: f32| (s / n).sqrt() * 255.;
                                    let alpha = ((cover - before_cover) / n_new - before_cover / n_old).abs() * 255.;
                                    (0..3).map(|i| (level(sum[i] - before[i], n_new) - level(before[i], n_old)).abs()).fold(alpha, f32::max) <= threshold
                                };
                                before = sum;
                                before_cover = cover;
                                pass_end *= 2;
                                m >= cap || (m >= least.max(floor) && agree)
                            }
                        }
                    };
                    if done {
                        break;
                    }
                }
                hist[m] += 1;
                let rgb = if cover > 0. { sum.map(|c| (c / cover).sqrt()) } else { [0.; 3] };
                row[px] = pack(rgb, cover / m as f32);
            }
            hist
        })
        .reduce(|| vec![0u64; cap + 1], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect())
}

/// The view as the look shows it: mirrored (attest) is the conjugate picture, J(c̄) about the eye's conjugate turned the other way, which is the plain one flipped top to bottom with its whorl turning the other way; the lamps stay where they are.
fn mirrored(mut v: View, look: &Look) -> View {
    if look.mirror.of(v.attest.mirror as u8) == 1 {
        v.c.1 = -v.c.1;
        v.centre.1 = -v.centre.1;
        v.rot.1 = -v.rot.1;
    }
    v
}

/// Where this picture came from in the Mandelbrot set: the Julia set's c.
pub fn locus(p: &Params) -> (f64, f64) {
    p.c
}

/// The whole-set locator: the Mandelbrot set from Re −2.25 to 0.75 and Im −1.3 to 2.1 (high enough to show where the descent starts), fitted to the pane on both axes.
pub fn render_whole(buf: &mut [u32], w: usize, h: usize, p: &Params) {
    let (ext_w, ext_h) = (3., 3.4);
    // The larger of the two scales is the one at which both extents fit the pane; this picks a fit, it does not clamp anything.
    let k = (ext_w / w as f64).max(ext_h / h as f64);
    render_locator(buf, w, h, p, (-0.75, 0.4), k);
}

/// The near locator: a 2⁵× zoom around the picture's c, 2⁻⁴ across the pane's span.
pub fn render_near(buf: &mut [u32], w: usize, h: usize, p: &Params) {
    let span = 2. * (w * h) as f64 / (w + h) as f64;
    render_locator(buf, w, h, p, locus(p), (1. / (1 << 5) as f64) / (span * 0.5));
}

/// A locator pane at `k` plane units per pixel about `centre`, imaginary axis up, in this device's colouring: every point the descent checked as a pixel in its own random colour, and a crosshair on c.
fn render_locator(buf: &mut [u32], w: usize, h: usize, p: &Params, centre: (f64, f64), k: f64) {
    let span = 2. * (w * h) as f64 / (w + h) as f64;
    let q = Params { kind: Kind::Plane, c: (0., 0.), centre, half: k * span * 0.5, rot: (1., 0.), ..*p };
    render(buf, w, h, &q);
    // Rows run downward and the imaginary axis runs up: flip.
    let (top, bottom) = buf.split_at_mut(w * (h / 2));
    for row in 0..h / 2 {
        let b = (h - h / 2 - 1 - row) * w;
        top[row * w..(row + 1) * w].swap_with_slice(&mut bottom[b..b + w]);
    }
    let at = locus(p);
    let to_px = |c: (f64, f64)| ((c.0 - centre.0) / k + w as f64 * 0.5 - 0.5, h as f64 * 0.5 - 0.5 - (c.1 - centre.1) / k);
    // WHY/PROOF: the points come from the plane and can lie off this pane or on its edge; a write outside it would land in the neighbouring pane's rows or past the buffer.
    let mut put = |x: f64, y: f64, colour: u32| {
        if x >= 0. && y >= 0. && (x as usize) < w && (y as usize) < h {
            buf[y as usize * w + x as usize] = colour;
        }
    };
    let (_, _, trail) = descend(at.0);
    for (i, &pt) in trail.iter().enumerate() {
        let hsh = blake3::hash(&(i as u32).to_le_bytes());
        let b = hsh.as_bytes();
        let (x, y) = to_px(pt);
        put(x, y, pack([b[0] | (1 << 6), b[1] | (1 << 6), b[2] | (1 << 6)].map(|c| c as f32 / 255.), 1.));
    }
    let (mx, my) = to_px(at);
    let white = pack([1.; 3], 1.);
    // Crosshair arms from 2² to 2⁴ pixels out, leaving the point itself clear.
    for i in (1 << 2)..(1 << 4) {
        let f = i as f64;
        put(mx + f, my, white);
        put(mx - f, my, white);
        put(mx, my + f, white);
        put(mx, my - f, white);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every stand-in device draws a picture with structure, and a change in the text changes it.
    #[test]
    fn pictures_have_structure_and_follow_the_text() {
        let (w, h) = (1 << 7, 1 << 8);
        for device in 0u32..(1 << 3) {
            let salt = *blake3::hash(&device.to_le_bytes()).as_bytes();
            let mut a = vec![0u32; w * h];
            let mut b = vec![0u32; w * h];
            render(&mut a, w, h, &Params::new(&salt, "ab"));
            render(&mut b, w, h, &Params::new(&salt, "abc"));
            let mut seen: Vec<u32> = a.clone();
            seen.sort_unstable();
            seen.dedup();
            assert!(seen.len() > 1 << 6, "device {device}: only {} distinct pixels", seen.len());
            assert!(a != b, "device {device}: the text did not change the picture");
        }
    }

    /// Every colour the shading can ask for stays inside the display's gamut at every hue: no channel below 0 or above 1 (a hair of rounding allowed), for lightness from 0.05 to past white (the shoulder brings it in) at the most saturation and tint any layer gets, and beyond.
    #[test]
    fn soft_colours_stay_in_gamut() {
        let mut th = Theme::from_bytes(&[0; 32]);
        for i in 0..(1 << 8) {
            th.hue = i as f32 / (1 << 8) as f32 * TAU as f32;
            for l in [0.05f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1., 1.2] {
                for sat in [0f32, 0.07, 0.124, 0.2] {
                    let rgb = soft(&th, l, sat, 0., (0.04 * th.hue.cos(), 0.04 * th.hue.sin()));
                    assert!(rgb.iter().all(|x| x.is_finite() && *x >= 0. && *x <= 1. + 1. / (1 << 10) as f32), "hue {i}, L {l}, sat {sat}: {rgb:?}");
                }
            }
        }
    }

    /// Photon's avatars never move: eight stand-in devices' 2⁶-pixel avatars hash to what they drew once the soft flame joined every colouring method and the escape moved to 2⁶² (2026-10-29) (any change to the avatar path, including a new draw from the salt that shifts an old one, fails here).
    #[test]
    fn avatars_hold_still() {
        let mut h = blake3::Hasher::new();
        for device in 0u32..(1 << 3) {
            let salt = *blake3::hash(&device.to_le_bytes()).as_bytes();
            let d = 1usize << 6;
            let mut px = vec![0u32; d * d];
            let r = d as f32 * 0.5;
            render_disc(&mut px, d, r, r, r, &Params::new(&salt, ""), &Look::avatar());
            for p in &px {
                h.update(&p.to_le_bytes());
            }
        }
        assert_eq!(h.finalize().to_hex().as_str(), "897e69498fc4ad607fef075cdc7adf3d5f46fb661c1bdf588a0f978e0567602a");
    }

    /// A spiral's eye really is periodic: f^p carries it back onto itself.
    #[test]
    fn a_spiral_eye_is_periodic() {
        let c = (-0.8, 0.2);
        let bits = *blake3::hash(b"eye").as_bytes();
        for p in 1..=PERIODS {
            let (eye, _, _) = cycle_point(c, &bits, p).expect("Newton closes a cycle from the repeated address");
            let mut z = eye;
            for _ in 0..p {
                let zz = cmul(z, z);
                z = (zz.0 + c.0, zz.1 + c.1);
            }
            assert!((z.0 - eye.0).abs() + (z.1 - eye.1).abs() < 1. / (1 << 16) as f64, "period {p}: f^p(eye) − eye = {:?}", (z.0 - eye.0, z.1 - eye.1));
        }
    }


}
