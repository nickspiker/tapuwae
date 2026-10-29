# tapuwae

*Tapuwae* (Māori): a footprint, a track left by someone passing.

A mark drawn from a hash, recognisably one source's every time: the root spiral of a quadratic Julia set, placed by a descent onto the Mandelbrot set's skyline and spherized into a softly lit disc in one soft colour scheme.

- **Placement:** a straight-down descent onto the Mandelbrot set by proven distance-bound steps, then the Julia set just above the landing point.
- **Framing:** dead centre on the root whorl (the repelling cycle closest to neutral), never a sub-spiral.
- **Look:** OKLab pastels kept inside sRGB by construction, banded per source into families; lit from the escape potential's own slope; flames, distance contours and a near-infinity escape after mandelbrot-exploder.

Used by ferros's attest screen and by Photon for default avatars and dev-build orbs.

```rust
let p = tapuwae::Params::new(&salt, "");
let mut px = vec![0u32; 256 * 256];
tapuwae::render_disc(&mut px, 256, 128., 128., 128., &p, &tapuwae::Look::default());
```

Pixels are fluor's α + darkness (premultiplied); outside the disc α is 0.

`render_fresh` draws the mark of a device no identity has attested on yet: the Mandelbrot set in the device's colour, each component glowing toward its rim, a distance-estimate glow outside.

Without the default `std` feature the crate is `no_std` (ferros's processes): rows render one after another and the float maths comes from libm.

## Speed: NEON now, the GPU later

The render is f32 per sample, a few dozen iterations then the shading, and the shading is most of the cost (measured on the desktop: about 70%, half of it libm). The plan: samples on a stratified grid (`Aa::Stratified`), four samples of a pixel at a time in NEON lanes, the shading's transcendentals as our own polynomials (no libm, no data-dependent branches).

**This must move to the GPU once ferros drives the Mali** (it needs the CSF firmware and a shader compiler: ferros KERNEL.md). A pixel's 4 × 4 grid is exactly one 16-lane Valhall warp, and the shading written branch-free for NEON is already a shader's shape; keep it that way.

License: MIT OR Apache-2.0.
