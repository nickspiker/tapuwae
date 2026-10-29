//! The float maths std gives f32 and f64 as methods, for a build without std (ferros's no_std processes, ahau's attest screen): the same names as methods, from libm, so the renderer reads the same either way.
//! With std these are not compiled and the inherent methods are used; results can differ in the last bits between the two (libm against the platform's), so the avatars' pinned hash is a std build's.

pub trait F32Math {
    fn sqrt(self) -> f32;
    fn exp2(self) -> f32;
    fn sin(self) -> f32;
    fn cos(self) -> f32;
    fn floor(self) -> f32;
    fn powi(self, n: i32) -> f32;
}

impl F32Math for f32 {
    fn sqrt(self) -> f32 {
        libm::sqrtf(self)
    }
    fn exp2(self) -> f32 {
        libm::exp2f(self)
    }
    fn sin(self) -> f32 {
        libm::sinf(self)
    }
    fn cos(self) -> f32 {
        libm::cosf(self)
    }
    fn floor(self) -> f32 {
        libm::floorf(self)
    }
    fn powi(self, n: i32) -> f32 {
        libm::powf(self, n as f32)
    }
}

pub trait F64Math {
    fn sqrt(self) -> f64;
    fn ln(self) -> f64;
    fn exp2(self) -> f64;
    fn sin(self) -> f64;
    fn cos(self) -> f64;
    fn atan2(self, x: f64) -> f64;
    fn hypot(self, y: f64) -> f64;
    fn powi(self, n: i32) -> f64;
    fn rem_euclid(self, m: f64) -> f64;
    fn exp_m1(self) -> f64;
}

impl F64Math for f64 {
    fn sqrt(self) -> f64 {
        libm::sqrt(self)
    }
    fn ln(self) -> f64 {
        libm::log(self)
    }
    fn exp2(self) -> f64 {
        libm::exp2(self)
    }
    fn sin(self) -> f64 {
        libm::sin(self)
    }
    fn cos(self) -> f64 {
        libm::cos(self)
    }
    fn atan2(self, x: f64) -> f64 {
        libm::atan2(self, x)
    }
    fn hypot(self, y: f64) -> f64 {
        libm::hypot(self, y)
    }
    fn powi(self, n: i32) -> f64 {
        libm::pow(self, n as f64)
    }
    fn exp_m1(self) -> f64 {
        libm::expm1(self)
    }
    fn rem_euclid(self, m: f64) -> f64 {
        let r = self - libm::floor(self / m) * m;
        r
    }
}
