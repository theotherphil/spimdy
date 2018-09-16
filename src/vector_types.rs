
use std::{
    arch::x86_64::*,
    ops::*
};

use super::*;
use ::maths::*;

#[derive(Copy, Clone)]
pub struct Vi32 { pub value: __m256i }

macro_rules! impl_from {
    ($type:ident, $from_type:ident) => {
        impl From<$from_type> for $type {
            fn from(x: $from_type) -> $type {
                $type::constant(x)
            }
        }

        impl From<[$from_type; 8]> for $type {
            fn from (xs: [$from_type; 8]) -> $type {
                $type::new(xs[7], xs[6], xs[5], xs[4], xs[3], xs[2], xs[1], xs[0])
            }
        }
    }
}

impl Vi32 {
    pub fn zeroes() -> Vi32 { unsafe { Vi32 { value: _mm256_setzero_si256() } } }
    pub fn constant(v: i32) -> Vi32 { unsafe { Vi32 { value: _mm256_set1_epi32(v) } } }
    pub fn new(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> Vi32 { 
        unsafe { Vi32 { value: _mm256_set_epi32(a, b, c, d, e, f, g, h) } }
    }

    // Can't use PartialOrd, as that has the wrong return type
    pub fn less_than<I: Into<Self>>(&self, other: I) -> Vbool { 
        unsafe {
            let other = other.into();
            let c = _mm256_cmpgt_epi32(other.value, self.value);
            Vbool { value: c }
        }
    }

    // Can't use PartialOrd, as that has the wrong return type
    pub fn greater_than<I: Into<Self>>(&self, other: I) -> Vbool { 
        unsafe {
            let other = other.into();
            let c = _mm256_cmpgt_epi32(self.value, other.value);
            Vbool { value: c }
        }
    }

    pub fn array(&self) -> [i32; 8] {
        i32_array_from_m256i(self.value)
    }
}

macro_rules! impl_vi32_op {
    ($trait:ident, $method:ident, $intrinsic:ident) => {
        impl_bin_op!(Vi32, value, $trait, $method, $intrinsic);
        impl_bin_op_with_primitive_rhs!(Vi32, i32, value, $trait, $method, $intrinsic, _mm256_set1_epi32);
        impl_bin_op_with_primitive_lhs!(Vi32, i32, value, $trait, $method, $intrinsic, _mm256_set1_epi32);
    };
}

impl_vi32_op!(Add, add, _mm256_add_epi32);
impl_vi32_op!(Sub, sub, _mm256_sub_epi32);
impl_vi32_op!(Mul, mul, _mm256_mul_epi32);

impl_from!(Vi32, i32);
impl_debug_for_m256i_wrapper!(Vi32, value, |a| a);

// C++ version isn't assignable, as masking happens on assignment
// What's the Rust equivalent?
#[derive(Copy, Clone, Debug)]
pub struct Vf32 { pub value: __m256 }

impl Vf32 {
    pub fn zeroes() -> Vf32 { unsafe { Vf32 { value: _mm256_setzero_ps() } } }
    pub fn constant(v: f32) -> Vf32 { unsafe { Vf32 { value: _mm256_set1_ps(v) } } }
    pub fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Vf32 { 
        unsafe { Vf32 { value: _mm256_set_ps(a, b, c, d, e, f, g, h) } }
    }

    // Can't use PartialOrd, as that has the wrong return type
    pub fn less_than<I: Into<Self>>(&self, other: I) -> Vbool { 
        unsafe {
            let other = other.into();
            let c = _mm256_cmp_ps(self.value, other.value, _CMP_LT_OQ);
            Vbool { value: _mm256_castps_si256(c) }
        }
    }

    // Can't use PartialOrd, as that has the wrong return type
    pub fn greater_than<I: Into<Self>>(&self, other: I) -> Vbool { 
        unsafe {
            let other = other.into();
            let c = _mm256_cmp_ps(self.value, other.value, _CMP_GT_OQ);
            Vbool { value: _mm256_castps_si256(c) }
        }
    }

    pub fn array(&self) -> [f32; 8] {
        f32_array_from_m256(self.value)
    }

    pub fn sqrt(&self) -> Vf32 {
        unsafe { Vf32 { value: _mm256_sqrt_ps(self.value) } }
    }
    
    pub fn abs(&self) -> Vf32 {
        unsafe {
            Vf32 { value: _mm256_andnot_ps(_mm256_set1_ps(-0.0), self.value) }
        }
    }

    pub fn exp(&self) -> Vf32 {
        Vf32 { value: exp256_ps(self.value) }
    }

    pub fn log(&self) -> Vf32 {
        Vf32 { value: log256_ps(self.value) }
    }
}

impl_from!(Vf32, f32);

macro_rules! impl_vf32_op {
    ($trait:ident, $method:ident, $intrinsic:ident) => {
        impl_bin_op!(Vf32, value, $trait, $method, $intrinsic);
        impl_bin_op_with_primitive_rhs!(Vf32, f32, value, $trait, $method, $intrinsic, _mm256_set1_ps);
        impl_bin_op_with_primitive_lhs!(Vf32, f32, value, $trait, $method, $intrinsic, _mm256_set1_ps);
    };
}

impl_vf32_op!(Add, add, _mm256_add_ps);
impl_vf32_op!(Sub, sub, _mm256_sub_ps);
impl_vf32_op!(Mul, mul, _mm256_mul_ps);
impl_vf32_op!(Div, div, _mm256_div_ps);

impl MulAssign for Vf32 {
    fn mul_assign(&mut self, rhs: Self) {
        self.value = self.mul(rhs).value;
    }
}

impl Neg for Vf32 {
    type Output = Self;

    fn neg(self) -> Self {
        unsafe {
            Vf32 { value: _mm256_sub_ps(_mm256_xor_ps(self.value, self.value), self.value) }
        }
    }
}

#[derive(Copy, Clone)]
pub struct Vbool { pub value: __m256i }

impl_debug_for_m256i_wrapper!(Vbool, value, |a| a != 0);

impl PartialEq for Vbool {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let neq = _mm256_xor_si256(self.value, other.value);
            _mm256_testz_si256(neq, neq) == 1
        }
    }
}

impl Eq for Vbool {}

/// Representation of bools used when storing them in vectors of i32s
fn i32_repr(b: bool) -> i32 {
    if b { !0 } else { 0 }
}

impl Vbool {
    pub fn constant(v: bool) -> Vbool { unsafe { Vbool { value: _mm256_set1_epi32(i32_repr(v)) } } }
    pub fn new(a: bool, b: bool, c: bool, d: bool, e: bool, f: bool, g: bool, h: bool) -> Vbool { 
        unsafe { 
            Vbool {
                value: _mm256_set_epi32(i32_repr(a), i32_repr(b), i32_repr(c), i32_repr(d), i32_repr(e), i32_repr(f), i32_repr(g), i32_repr(h))
            }
        } 
    }
    pub fn array(&self) -> [bool; 8] {
        bool_array_from_m256i(self.value)
    }
}

impl_from!(Vbool, bool);

impl Not for Vbool {
    type Output = Self;

    fn not(self) -> Self {
        unsafe { Vbool { value: _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_setzero_si256(), _mm256_setzero_si256()), self.value) } }
    }
}


#[derive(Copy, Clone)]
pub struct LinearInt {
    pub value: __m256i
}

impl_debug_for_m256i_wrapper!(LinearInt, value, |a| a);

impl LinearInt {
    pub fn to_vf32(self) -> Vf32 {
        unsafe { Vf32 { value:  _mm256_cvtepi32_ps(self.value) } }
    }

    pub fn array(&self) -> [i32; 8] {
        i32_array_from_m256i(self.value)
    }

    // Totally unsafe - take a pointer to who knows where, and hang onto a reference to it with no lifetimes
    pub fn at(&self, ptr: &[f32]) -> Vf32LinearRef {
        unsafe {
            Vf32LinearRef { 
                value: ptr.as_ptr().offset(_mm256_cvtsi256_si32(self.value) as isize) as *mut f32
            }
        }
    }

    pub fn at_vi32(&self, ptr: &[i32]) -> Vi32LinearRef {
        unsafe {
            Vi32LinearRef { 
                value: ptr.as_ptr().offset(_mm256_cvtsi256_si32(self.value) as isize) as *mut i32
            }
        }
    }
}

impl Add<i32> for LinearInt {
    type Output = Self;

    fn add(self, rhs: i32) -> Self {
        unsafe { LinearInt { value: _mm256_add_epi32(self.value, _mm256_set1_epi32(rhs)) } }
    }
}

impl Add<LinearInt> for i32 {
    type Output = LinearInt;

    fn add(self, rhs: LinearInt) -> LinearInt {
        unsafe { LinearInt { value: _mm256_add_epi32( _mm256_set1_epi32(self), rhs.value) } }
    }
}

// Same as Vbool. Remove duplication
impl PartialEq for LinearInt {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            let neq = _mm256_xor_si256(self.value, other.value);
            _mm256_testz_si256(neq, neq) == 1
        }
    }
}

/// reference to a Vf32 stored linearly in memory
pub struct Vf32LinearRef {
    pub value: *mut f32
}

/// reference to a Vi32 stored linearly in memory
pub struct Vi32LinearRef {
    pub value: *mut i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use kernel::program_index;

    #[test]
    fn test_vbool_debug() {
        assert_eq!(
            format!("{:?}", Vbool::constant(false)),
            "Vbool { value: [false, false, false, false, false, false, false, false] }");
        assert_eq!(
            format!("{:?}", Vbool::constant(true)),
            "Vbool { value: [true, true, true, true, true, true, true, true] }");
        assert_eq!(
            format!("{:?}", Vbool::new(true, false, false, true, false, true, true, false)),
            "Vbool { value: [false, true, true, false, true, false, false, true] }");
    }

    #[test]
    fn test_linear_int_debug() {
        assert_eq!(
            format!("{:?}", program_index()),
            "LinearInt { value: [0, 1, 2, 3, 4, 5, 6, 7] }");
    }

    #[test]
    fn test_linear_int_add() {
        unsafe {
            let actual_l = program_index() + 1;
            let actual_r = 1 + program_index();
            let expected = LinearInt { value: _mm256_set_epi32(8, 7, 6, 5, 4, 3, 2, 1) };
            assert_eq!(actual_l, expected);
            assert_eq!(actual_r, expected);
        }
    }


    #[test]
    fn test_less_than() {
        let u: Vf32 = 0.0.into();
        let v: Vf32 = [2.0, 1.0, 0.0, -1.0, 7.0, -2.0, 3.0, 4.0].into();
        let l = v.less_than(u);
        println!("{:?}", l);
        assert_eq!(l, Vbool::new(false, false, true, false, true, false, false, false));
    }

    #[test]
    fn test_vbool_eq() {
        assert!(Vbool::constant(false) != Vbool::constant(true));
        assert_eq!(Vbool::constant(true), Vbool::constant(true));
        assert_eq!(Vbool::constant(false), Vbool::constant(false));
    }
}

