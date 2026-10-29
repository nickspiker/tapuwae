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

License: MIT OR Apache-2.0.
