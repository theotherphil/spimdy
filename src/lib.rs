#![feature(test)]
extern crate test;
extern crate rand;
extern crate rayon;
extern crate packed_simd;

pub use kernel::*;
pub use vector_types::*;
use std::arch::x86_64::*;
use packed_simd::*;
//use rayon::prelude::*;

fn bool_array_from_m256i(x: __m256i) -> [bool; 8] {
    let arr = i32_array_from_m256i(x);
    [arr[0] != 0, arr[1] != 0, arr[2] != 0, arr[3] != 0, arr[4] != 0, arr[5] != 0, arr[6] != 0, arr[7] != 0]
}

fn i32_array_from_m256i(x: __m256i) -> [i32; 8] {
    // PHIL: there's no reason to assume that this is aligned...
    let mut arr = [0i32; 8];
    let p = arr.as_mut_ptr() as *mut __m256i;
    unsafe {
        _mm256_storeu_si256(p, x);
    }
    arr
}

fn f32_array_from_m256(x: __m256) -> [f32; 8] {
    let mut a = [0.0; 8];
    unsafe {
        _mm256_storeu_ps(a.as_mut_ptr(), x);
    }
    a
}

/// Generates a debug implementation for a type that contains a single __m256i field.
/// The compiler-derived implementation treats ___m256i as four 64 bit ints, but
/// we want to treat it as eight values.
macro_rules! impl_debug_for_m256i_wrapper {
    ($type:ident, $field:ident, $convert_element_from_i32:expr) => {
        impl std::fmt::Debug for $type {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                let a = ::i32_array_from_m256i(self.$field);
                let c = $convert_element_from_i32;
                let t = [c(a[0]), c(a[1]), c(a[2]), c(a[3]), c(a[4]), c(a[5]), c(a[6]), c(a[7])];
                write!(f, stringify!($type))?;
                write!(f, " {{ value: {:?} }}", t)
            }
        }
    }
}

/// Generates an implementation of a binary operator for a type that contains a single
/// vector register field
macro_rules! impl_bin_op {
    ($type:ident, $field:ident, $trait:ident, $method:ident, $intrinsic:ident) => {
        impl $trait for $type {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                unsafe { $type { $field: $intrinsic(self.$field, other.$field) } }
            }
        }
    };
}

macro_rules! impl_bin_op_with_primitive_rhs {
    ($type:ident, $primitive_type:ident, $field:ident, $trait:ident, $method:ident, $intrinsic:ident, $set:ident) => {
        impl $trait<$primitive_type> for $type {
            type Output = Self;

            fn $method(self, other: $primitive_type) -> Self {
                unsafe { $type { $field: $intrinsic(self.$field, $set(other)) } }
            }
        }
    };
}

macro_rules! impl_bin_op_with_primitive_lhs {
    ($type:ident, $primitive_type:ident, $field:ident, $trait:ident, $method:ident, $intrinsic:ident, $set:ident) => {
        impl $trait<$type> for $primitive_type {
            type Output = $type;

            fn $method(self, other: $type) -> $type {
                unsafe { $type { $field: $intrinsic($set(self), other.$field) } }
            }
        }
    };
}

pub mod vector_types;
#[macro_use]
pub mod kernel;
pub mod examples;
mod maths;

#[cfg(test)]
mod tests {
    #[test]
    fn test_packed() {
        use packed_simd::*;
        let a = i32x4::new(1, 2, 3, 4);
        let b = i32x4::new(5, 6, 7, 8);
        assert_eq!(a + b, i32x4::new(6, 8, 10, 12));
    }
}