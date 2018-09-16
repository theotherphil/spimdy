
use std::arch::x86_64::*;

//**********
// This code is a modified version of the original (most obviously it's been transliterated into Rust)
//**********
/* 
   AVX implementation of sin, cos, sincos, exp and log

   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/

   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#[allow(non_snake_case)]
pub fn exp256_ps(mut x: __m256) -> __m256 {
unsafe {
    let exp_hi = _mm256_set1_ps(88.3762626647949);
    let exp_lo = _mm256_set1_ps(-88.3762626647949);

    let cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
    let cephes_exp_C1 = _mm256_set1_ps(0.693359375);
    let cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

    let cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
    let cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
    let cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
    let cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
    let cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
    let cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
    let mut tmp: __m256;
    let mut fx: __m256;
    let mut imm0: __m256i;
    let one = _mm256_set1_ps(1.0);

    x     = _mm256_min_ps(x, exp_hi);
    x     = _mm256_max_ps(x, exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx    = _mm256_mul_ps(x, cephes_LOG2EF);
    fx    = _mm256_add_ps(fx, _mm256_set1_ps(0.5));
    tmp   = _mm256_floor_ps(fx);
    let mut mask  = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
    mask  = _mm256_and_ps(mask, one);
    fx    = _mm256_sub_ps(tmp, mask);
    tmp   = _mm256_mul_ps(fx, cephes_exp_C1);
    let mut z     = _mm256_mul_ps(fx, cephes_exp_C2);
    x     = _mm256_sub_ps(x, tmp);
    x     = _mm256_sub_ps(x, z);
    z     = _mm256_mul_ps(x,x);

    let mut y     = cephes_exp_p0;
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p1);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p2);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p3);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p4);
    y     = _mm256_mul_ps(y, x);
    y     = _mm256_add_ps(y, cephes_exp_p5);
    y     = _mm256_mul_ps(y, z);
    y     = _mm256_add_ps(y, x);
    y     = _mm256_add_ps(y, one);

    /* build 2^n */
    imm0  = _mm256_cvttps_epi32(fx);
    imm0  = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0  = _mm256_slli_epi32(imm0, 23);
    let pow2n = _mm256_castsi256_ps(imm0);
    y     = _mm256_mul_ps(y, pow2n);
    y
}
}

// /* declare some AVX constants -- why can't I figure a better way to do that? */
// #define _PS256_CONST(Name, Val)                                            \
//   static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
//#define _PI32_CONST256(Name, Val)                                            \
//static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

// /* the smallest non denormalized float number */
// _PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
//_PS256_CONST(1  , 1.0f);
//_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);
//_PS256_CONST(0p5, 0.5f);
//_PI32_CONST256(0x7f, 0x7f);

/* natural logarithm computed for 8 simultaneous float 
   return NaN for x <= 0
*/
#[allow(non_snake_case)]
pub fn log256_ps(mut x: __m256) -> __m256 {
unsafe {
    let one = _mm256_set1_ps(1.0);
    let min_norm_pos = _mm256_set1_ps(f32::from_bits(0x00800000));
    let inv_mant_mask = _mm256_set1_ps(f32::from_bits(!0x7f800000));
    let half = _mm256_set1_ps(0.5);
    let p0x7f = _mm256_set1_epi32(0x7f);
    let cephes_SQRTHF = _mm256_set1_ps(0.707106781186547524);
    let cephes_log_p0 = _mm256_set1_ps(7.0376836292E-2);
    let cephes_log_p1 = _mm256_set1_ps(- 1.1514610310E-1);
    let cephes_log_p2 = _mm256_set1_ps(1.1676998740E-1);
    let cephes_log_p3 = _mm256_set1_ps(- 1.2420140846E-1);
    let cephes_log_p4 = _mm256_set1_ps(1.4249322787E-1);
    let cephes_log_p5 = _mm256_set1_ps(-1.6668057665E-1);
    let cephes_log_p6 = _mm256_set1_ps(2.0000714765E-1);
    let cephes_log_p7 = _mm256_set1_ps(-2.4999993993E-1);
    let cephes_log_p8 = _mm256_set1_ps(3.3333331174E-1);
    let cephes_log_q1 = _mm256_set1_ps(-2.12194440e-4);
    let cephes_log_q2 = _mm256_set1_ps(0.693359375);

    let mut imm0: __m256i;

    let invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);

    x = _mm256_max_ps(x, min_norm_pos);  /* cut off denormalized stuff */

    // can be done with AVX2
    imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);

    /* keep only the fractional part */
    x = _mm256_and_ps(x, inv_mant_mask);
    x = _mm256_or_ps(x, half);

    // this is again another AVX2 instruction
    imm0 = _mm256_sub_epi32(imm0, p0x7f);
    let mut e = _mm256_cvtepi32_ps(imm0);

    e = _mm256_add_ps(e, one);

    /* part2: 
        if( x < SQRTHF ) {
        e -= 1;
        x = x + x - 1.0;
        } else { x = x - 1.0; }
    */
    let mask = _mm256_cmp_ps(x, cephes_SQRTHF, _CMP_LT_OS);
    let mut tmp = _mm256_and_ps(x, mask);
    x = _mm256_sub_ps(x, one);
    e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
    x = _mm256_add_ps(x, tmp);

    let z = _mm256_mul_ps(x, x);

    let mut y = cephes_log_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p5);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p6);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p7);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_log_p8);
    y = _mm256_mul_ps(y, x);

    y = _mm256_mul_ps(y, z);

    tmp = _mm256_mul_ps(e, cephes_log_q1);
    y = _mm256_add_ps(y, tmp);


    tmp = _mm256_mul_ps(z, half);
    y = _mm256_sub_ps(y, tmp);

    tmp = _mm256_mul_ps(e, cephes_log_q2);
    x = _mm256_add_ps(x, y);
    x = _mm256_add_ps(x, tmp);
    x = _mm256_or_ps(x, invalid_mask); // negative arg will be NAN
    x
}
}