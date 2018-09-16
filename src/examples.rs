
// Shouldn't be in the library, but will worry about code organisation
// when I have something that's actually usable

use ::kernel::*;
use ::vector_types::*;
use std::arch::x86_64::*;

//--------------------------------------------------------------------
// Conditional
//--------------------------------------------------------------------

/// if-else taking a single float
 pub fn conditional_single(a: f32) -> f32 {
        if a < 0.0 {
            1.0
        } else {
            a + 1.0
        }
}

/// Manual implementation of vector if-else via intrinsics
pub fn conditional_intrinsics(mut a: __m256) -> __m256 {
    unsafe {
        let zero = _mm256_setzero_ps();
        let ones = _mm256_set1_ps(1.0);
        let all_bits = _mm256_set1_ps(f32::from_bits(0xffffffff));
        let mut mask = _mm256_cmp_ps(a, zero, _CMP_LT_OQ);
        a = _mm256_blendv_ps(a, ones, mask);
        mask = _mm256_xor_ps(mask, all_bits);
        let a1 = _mm256_add_ps(a, ones);
        a = _mm256_blendv_ps(a, a1, mask);
        a
    }
}

// if-else using this library's vector types
pub fn conditional_spmd(k: &mut Kernel, a: Vf32) -> Vf32 {
    let mut ret: Vf32 = 0.0.into();
    if_else!(k,
        a.less_than(0.0),
        |c| {
            c.store(&mut ret, 1.0.into());
        }, 
        |c| {
            c.store(&mut ret, a + 1.0);
        }
    );
    ret
}

//--------------------------------------------------------------------
// Function call
//--------------------------------------------------------------------

pub fn function_call_single(a: f32) -> f32 {
    if a > 0.0 {
        2.0 * conditional_single(a)
    } else {
        1.0 - conditional_single(a)
    }
}

pub fn function_call_spmd(k: &mut Kernel, a: Vf32) -> Vf32 {
    let mut ret: Vf32 = 0.0.into();
    if_else!(k,
        a.greater_than(0.0),
        |c| {
            // Borrow checker woes - need to compute v outside of assign call
            let v = conditional_spmd(&mut c.clone_exec_mask(), a) * 2.0;
            c.store(&mut ret, v);
        },
        |c| {
            let v = 1.0 - conditional_spmd(&mut c.clone_exec_mask(), a);
            c.store(&mut ret, v);
        }
    );
    ret
}

//--------------------------------------------------------------------
// foreach loop
//--------------------------------------------------------------------

pub fn foreach_single(vin: &[f32], vout: &mut [f32]) {
    assert!(vin.len() == vout.len());
    for index in 0..vin.len() {
        let mut v = vin[index];
        if v < 3.0 {
            v = v * v;
        } else {
            v = v.sqrt();
        }
        vout[index] = v;
    }
}

pub fn foreach_spmd(vin: &[f32], vout: &mut [f32]) {
    assert!(vin.len() == vout.len());
    let mut kernel = Kernel::new();
    kernel.foreach(0, vin.len() as i32, |c, index| {
        let v = c.load_ref(&index.at(vin));
        let mut u = 0.0.into();
        c.vif(v.less_than(3.0), |d| {
            d.store(&mut u, v * v);
        });
        c.vif(!v.less_than(3.0), |d| {
            d.store(&mut u, v.sqrt());
        });
        c.store_ref(&mut index.at(vout), &u);
    });
}

//--------------------------------------------------------------------
// Mandelbrot - taken from the ispc example
//--------------------------------------------------------------------

// Returns the number of iterations run
fn mandel_single(c_re: f32, c_im: f32, count: i32) -> i32 {
    let (mut z_re, mut z_im) = (c_re, c_im);
    let mut i: i32 = 0;

    while i < count {
        if z_re * z_re + z_im * z_im > 4.0 {
            break;
        }
        let new_re = z_re * z_re - z_im * z_im;
        let new_im = 2.0 * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
        i += 1;
    }
        
    i
}

pub fn mandelbrot_single(
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: i32,
    height: i32,
    max_iterations: i32,
    output: &mut [i32]
) {
    let dx = (x1 - x0) / width as f32;
    let dy = (y1 - y0) / height as f32;

    for j in 0..height {
        for i in 0..width {
            let x = x0 + i as f32 * dx;
            let y = y0 + j as f32 * dy;

            let index = j * width + i;
            output[index as usize] = mandel_single(x, y, max_iterations);
        }
    }
}

pub fn mandel_spmd(k: &mut Kernel, c_re: Vf32, c_im: Vf32, count: i32) -> Vi32 {
    let (mut z_re, mut z_im) = (c_re, c_im);
    let mut i: Vi32 = 0.into();

    vfor!(k,
        init: |c| {
            c.store(&mut i, 0.into());
        },
        cond: |_| {
            i.less_than(Vi32::constant(count))
        },
        incr: |c| {
            let incr = i + 1;
            c.store(&mut i, incr);
        },
        body: |c| {
            c.vif(
                (z_re * z_re + z_im * z_im).greater_than(4.0),
                |d| { d.vbreak(); }
            );
            let new_re = z_re * z_re - z_im * z_im;
            let new_im = 2.0 * z_re * z_im;

            c.unmasked(|d| {
                d.store(&mut z_re, c_re + new_re);
                d.store(&mut z_im, c_im + new_im);
            });
        }
    );

    i
}

pub fn mandelbrot_spmd(
    k: &mut Kernel,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    width: i32,
    height: i32,
    max_iterations: i32,
    output: &mut [i32]
) {
    let dx = (x1 - x0) / width as f32;
    let dy = (y1 - y0) / height as f32;

    for j in 0..height {
        //println!("j: {}", j);

        // Note that we'll be doing programCount computations in parallel,
        // so increment i by that much.  This assumes that width evenly
        // divides programCount.
        k.foreach(0, width, |c, i| {
            //println!("i: {:?}", i);

            // Figure out the position on the complex plane to compute the
            // number of iterations at.  Note that the x values are
            // different across different program instances, since its
            // initializer incorporates the value of the programIndex
            // variable.
            let x: Vf32 = x0 + i.to_vf32() * dx;
            let y: Vf32 = (y0 + j as f32 * dy).into();
            let index = j * width + i;

            let v = mandel_spmd(&mut c.clone_exec_mask(), x, y, max_iterations);
            c.store_ref_vi32(&mut index.at_vi32(output), &v);
        });
    }
}

//--------------------------------------------------------------------
// Options - taken from the ispc example
//--------------------------------------------------------------------

// Cumulative normal distribution function
#[inline]
fn cnd_single(x: f32) -> f32 {
    let l = x.abs();
    let k = 1.0 / (1.0 + 0.2316419 * l);
    let k2 = k * k;
    let k3 = k2 * k;
    let k4 = k2 * k2;
    let k5 = k3 * k2;

    let inv_sqrt_2pi = 0.39894228040;
    let mut w = 0.31938153 * k
              - 0.356563782 * k2
              + 1.781477937 * k3
              - 1.821255978 * k4
              + 1.330274429 * k5;

    w *= inv_sqrt_2pi * (-l * l * 0.5).exp();

    if x > 0.0 { 1.0 - w } else { w }
}

pub fn black_scholes_single(
    sa: &[f32],
    xa: &[f32],
    ta: &[f32],
    ra: &[f32],
    va: &[f32],
    result: &mut [f32],
    count: i32
) {
    for i in 0..count as usize {
        let s = sa[i];
        let x = xa[i];
        let t = ta[i];
        let r = ra[i];
        let v = va[i];
        let d1 = ((s/x).ln() + (r + v * v * 0.5) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();
        result[i] = s * cnd_single(d1) - x * (-r * t).exp() * cnd_single(d2);
    }
}

fn cnd_spmd(ker: &mut Kernel, x: Vf32) -> Vf32 {
    let l = x.abs();
    let k = 1.0 / (1.0 + 0.2316419 * l);
    let k2 = k * k;
    let k3 = k2 * k;
    let k4 = k2 * k2;
    let k5 = k3 * k2;

    let inv_sqrt_2pi = 0.39894228040;
    let mut w = 0.31938153 * k
                - 0.356563782 * k2
                + 1.781477937 * k3
                - 1.821255978 * k4
                + 1.330274429 * k5;

    w *= inv_sqrt_2pi * (-l * l * 0.5).exp();

    ker.vif(
        x.greater_than(0.0),
        |c| {
            let t = 1.0 - w;
            c.store(&mut w, t);
        }
    );

    w
}

pub fn black_scholes_spmd(
    k: &mut Kernel,
    sa: &[f32],
    xa: &[f32],
    ta: &[f32],
    ra: &[f32],
    va: &[f32],
    result: &mut [f32],
    count: i32
) {
    k.foreach(0, count, |c, i| {
        let s = c.load_ref(&i.at(sa));
        let x = c.load_ref(&i.at(xa));
        let t = c.load_ref(&i.at(ta));
        let r = c.load_ref(&i.at(ra));
        let v = c.load_ref(&i.at(va));

        let d1 = ((s/x).log() + (r + v * v * 0.5) * t) / (v * t.sqrt());
        let d2 = d1 - v * t.sqrt();

        let c1 = cnd_spmd(&mut c.clone_exec_mask(), d1);
        let c2 = cnd_spmd(&mut c.clone_exec_mask(), d2);
        let value = s * c1 - x * (-r * t).exp() * c2;

        c.store_ref(&mut i.at(result), &value);
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::*;
  
    fn assert_equal(lhs: &[f32], rhs: &[f32]) {
        let eq = lhs.iter().zip(rhs).all(|(x, y)| x.partial_cmp(y) == Some(std::cmp::Ordering::Equal));
        assert!(eq, "lhs = {:?}, rhs = {:?}", lhs, rhs);
    }

    fn eight_f32s() -> [f32; 8] {
        [4.0, 3.0, -2.0, 7.0, -1.0, 0.0, 1.0, 2.0]
    }

    fn map_eight_f32s(mut f: impl FnMut(f32) -> f32) -> Vec<f32> {
        eight_f32s().iter().map(|x| f(*x)).collect()
    }

//--------------------------------------------------------------------
// Conditional
//--------------------------------------------------------------------
    #[test]
    fn test_conditional_intrinsics() {
        unsafe {
            let scalar = map_eight_f32s(conditional_single);
            // The values from eight_f32s in reverse order - the _mm*set* methods take their arguments in
            // decreasing order of significance but the .array() methods on vector types return values in
            // memory order. As x86 is little-endian these are opposite directions.
            let manual = conditional_intrinsics(_mm256_set_ps(2.0, 1.0, 0.0, -1.0, 7.0, -2.0, 3.0, 4.0));
            assert_equal(&scalar, &::f32_array_from_m256(manual));
        }
    }

    #[test]
    fn test_conditional_spmd() {
        let scalar = map_eight_f32s(conditional_single);
        let mut k = Kernel::new();
        let vector = conditional_spmd(&mut k, eight_f32s().into());
        assert_equal(&scalar, &vector.array());
    }

    #[bench]
    fn bench_conditional_single(b: &mut Bencher) {
        let f = conditional_single;
        b.iter(|| {
            for _ in 0..1000 {
                black_box([f(4.0), f(3.0), f(-2.0), f(7.0), f(-1.0), f(0.0), f(1.0), f(2.0)]);
            }
        });
    }

    #[bench]
    fn bench_conditional_intrinsics(b: &mut Bencher) {
        unsafe {
            b.iter(|| {
                for _ in 0..1000 {
                    black_box(conditional_intrinsics(_mm256_set_ps(4.0, 3.0, -2.0, 7.0, -1.0, 0.0, 1.0, 2.0)));
                }
            });
        }
    }

    #[bench]
    fn bench_conditional_spmd(b: &mut Bencher) {
        b.iter(|| {
            for _ in 0..1000 {
                let mut k = Kernel::new();
                black_box(conditional_spmd(&mut k, eight_f32s().into()));
            }
        });
    }

//--------------------------------------------------------------------
// Function call
//--------------------------------------------------------------------
    #[test]
    fn test_function_call_spmd() {
        let scalar = map_eight_f32s(function_call_single);
        let mut k = Kernel::new();
        let vector = function_call_spmd(&mut k, eight_f32s().into());
        assert_equal(&scalar, &vector.array());
    }

//--------------------------------------------------------------------
// foreach loop
//--------------------------------------------------------------------
    #[test]
    fn test_foreach_spmd() {
        let vin: Vec<f32> = (0..17).map(|x| x as f32).collect();
        let mut vout1: Vec<f32> = (0..17).map(|x| x as f32).collect();
        let mut vout2: Vec<f32> = (0..17).map(|x| x as f32).collect();

        foreach_single(&vin, &mut vout1);
        foreach_spmd(&vin, &mut vout2);
        assert_equal(&vout1, &vout2);
    }

    #[bench]
    fn bench_foreach_single(b: &mut Bencher) {
        let vin: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let mut vout: Vec<f32> = (0..100).map(|x| x as f32).collect();

        b.iter(|| {
            foreach_single(&vin, &mut vout);
        });
    }

    #[bench]
    fn bench_foreach_spmd(b: &mut Bencher) {
        let vin: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let mut vout: Vec<f32> = (0..100).map(|x| x as f32).collect();

        b.iter(|| {
            foreach_spmd(&vin, &mut vout);
        });
    }

//--------------------------------------------------------------------
// Mandelbrot
//--------------------------------------------------------------------
    #[test]
    fn test_mandelbrot_spmd() {
        // width needs to be evenly divisible by 8
        let width = 80;
        let height = 100;
        let x0 = -2.0;
        let x1 = 1.0;
        let y0 = -1.0;
        let y1 = 1.0;
        let max_iterations = 10;
        let mut out1 = vec![0; width * height];
        let mut out2 = vec![0; width * height];

        mandelbrot_single(x0, y0, x1, y1, width as i32, height as i32, max_iterations, &mut out1);
        spmd!(|k| {
            mandelbrot_spmd(k, x0, y0, x1, y1, width as i32, height as i32, max_iterations, &mut out2)
        });
        assert_eq!(out1, out2);
    }

    #[bench]
    fn bench_mandelbrot_single(b: &mut Bencher) {
        // width needs to be evenly divisible by 8
        let width = 32;
        let height = 20;
        let x0 = -2.0;
        let x1 = 1.0;
        let y0 = -1.0;
        let y1 = 1.0;
        let max_iterations = 10;
        let mut out = vec![0; width * height];

        b.iter(|| {
            mandelbrot_single(x0, y0, x1, y1, width as i32, height as i32, max_iterations, &mut out);
            black_box(&mut out);
        });
    }

    #[bench]
    fn bench_mandelbrot_spmd(b: &mut Bencher) {
        // width needs to be evenly divisible by 8
        let width = 32;
        let height = 20;
        let x0 = -2.0;
        let x1 = 1.0;
        let y0 = -1.0;
        let y1 = 1.0;
        let max_iterations = 10;
        let mut out = vec![0; width * height];

        b.iter(|| {
            let mut k = Kernel::new();
            mandelbrot_spmd(&mut k, x0, y0, x1, y1, width as i32, height as i32, max_iterations, &mut out);
            black_box(&mut out);
        });
    }

    //--------------------------------------------------------------------
    // Options
    //--------------------------------------------------------------------

    #[test]
    fn test_black_scholes() {
        let num_options = 128 * 1024;
        let s = vec![100.0; num_options]; // stock price
        let x = vec![98.0; num_options]; // option strike price
        let t = vec![2.0; num_options]; // time (years)
        let r = vec![0.02; num_options]; // risk-free interest rate
        let v = vec![5.0; num_options]; // volatility

        let mut result1 = vec![0.0; num_options];
        let mut result2 = vec![0.0; num_options];

        black_scholes_single(&s, &x, &t, &r, &v, &mut result1, num_options as i32);
        spmd!(|k| {
            black_scholes_spmd(k, &s, &x, &t, &r, &v, &mut result2, num_options as i32);
        });

        assert_equal(&result1, &result2);
    }

    #[bench]
    fn bench_black_scholes_single(b: &mut Bencher) {
        let num_options = 128 * 1024;
        let s = vec![100.0; num_options]; // stock price
        let x = vec![98.0; num_options]; // option strike price
        let t = vec![2.0; num_options]; // time (years)
        let r = vec![0.02; num_options]; // risk-free interest rate
        let v = vec![5.0; num_options]; // volatility

        let mut result = vec![0.0; num_options];

        b.iter(|| {
            black_scholes_single(&s, &x, &t, &r, &v, &mut result, num_options as i32);
            black_box(&mut result);
        });  
    }

    #[bench]
    fn bench_black_scholes_spmd(b: &mut Bencher) {
        let num_options = 128 * 1024;
        let s = vec![100.0; num_options]; // stock price
        let x = vec![98.0; num_options]; // option strike price
        let t = vec![2.0; num_options]; // time (years)
        let r = vec![0.02; num_options]; // risk-free interest rate
        let v = vec![5.0; num_options]; // volatility

        let mut result = vec![0.0; num_options];
 
        b.iter(|| {
            let mut k = Kernel::new();
            //println!("***** ITER");
            black_scholes_spmd(&mut k, &s, &x, &t, &r, &v, &mut result, num_options as i32);
            black_box(&mut result);
        });  
    }
}
