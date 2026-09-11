/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::r2c::rfft_raders::{RadersRfftCombiner, combine_scalar};
use num_complex::Complex;
use std::arch::x86_64::*;

/// Reconstruct complex bins from the two halves of the real convolution.
pub(crate) struct AvxRadersRfftCombiner {}

impl RadersRfftCombiner<f32> for AvxRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f32>], lo: &[f32], hi: &[f32], signs: &[f32], first: f32) {
        unsafe { combine_f32(y, lo, hi, signs, first) }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn combine_f32(y: &mut [Complex<f32>], lo: &[f32], hi: &[f32], signs: &[f32], first: f32) {
    let (y_chunks, y_rem) = y.as_chunks_mut::<4>();
    let (lo_chunks, lo_rem) = lo.as_chunks::<4>();
    let (hi_chunks, hi_rem) = hi.as_chunks::<4>();
    let (sign_chunks, sign_rem) = signs.as_chunks::<4>();
    unsafe {
        let first_v = _mm_set1_ps(first);
        for (((y, lo), hi), signs) in y_chunks
            .iter_mut()
            .zip(lo_chunks)
            .zip(hi_chunks)
            .zip(sign_chunks)
        {
            let lo = _mm_loadu_ps(lo.as_ptr());
            let hi = _mm_loadu_ps(hi.as_ptr());
            let signs = _mm_loadu_ps(signs.as_ptr());
            let re = _mm_add_ps(first_v, _mm_add_ps(lo, hi));
            let im = _mm_mul_ps(_mm_sub_ps(lo, hi), signs);
            let result = _mm256_setr_m128(_mm_unpacklo_ps(re, im), _mm_unpackhi_ps(re, im));
            _mm256_storeu_ps(y.as_mut_ptr().cast(), result);
        }
    }
    combine_scalar(y_rem, lo_rem, hi_rem, sign_rem, first);
}

impl RadersRfftCombiner<f64> for AvxRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f64>], lo: &[f64], hi: &[f64], signs: &[f64], first: f64) {
        unsafe { combine_f64(y, lo, hi, signs, first) }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn combine_f64(y: &mut [Complex<f64>], lo: &[f64], hi: &[f64], signs: &[f64], first: f64) {
    let (y_chunks, y_rem) = y.as_chunks_mut::<2>();
    let (lo_chunks, lo_rem) = lo.as_chunks::<2>();
    let (hi_chunks, hi_rem) = hi.as_chunks::<2>();
    let (sign_chunks, sign_rem) = signs.as_chunks::<2>();
    unsafe {
        let first_v = _mm_set1_pd(first);
        for (((y, lo), hi), signs) in y_chunks
            .iter_mut()
            .zip(lo_chunks)
            .zip(hi_chunks)
            .zip(sign_chunks)
        {
            let lo = _mm_loadu_pd(lo.as_ptr());
            let hi = _mm_loadu_pd(hi.as_ptr());
            let signs = _mm_loadu_pd(signs.as_ptr());
            let re = _mm_add_pd(first_v, _mm_add_pd(lo, hi));
            let im = _mm_mul_pd(_mm_sub_pd(lo, hi), signs);
            let result = _mm256_setr_m128d(_mm_unpacklo_pd(re, im), _mm_unpackhi_pd(re, im));
            _mm256_storeu_pd(y.as_mut_ptr().cast(), result);
        }
    }
    combine_scalar(y_rem, lo_rem, hi_rem, sign_rem, first);
}
