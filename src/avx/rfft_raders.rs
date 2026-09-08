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

/// `y[p] = (first + re[p], Re(y[p] * twiddles[p]))` on AVX2 + FMA.
pub(crate) struct AvxRadersRfftCombiner {}

impl RadersRfftCombiner<f32> for AvxRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f32>], twiddles: &[Complex<f32>], re: &[f32], first: f32) {
        unsafe { combine_f32(y, twiddles, re, first) }
    }
}

impl RadersRfftCombiner<f64> for AvxRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f64>], twiddles: &[Complex<f64>], re: &[f64], first: f64) {
        unsafe { combine_f64(y, twiddles, re, first) }
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
fn combine_f32(y: &mut [Complex<f32>], twiddles: &[Complex<f32>], re: &[f32], first: f32) {
    let (y_chunks, y_rem) = y.as_chunks_mut::<4>();
    let (twiddle_chunks, twiddle_rem) = twiddles.as_chunks::<4>();
    let (re_chunks, re_rem) = re.as_chunks::<4>();
    unsafe {
        let first_v = _mm256_set1_ps(first);
        let expand = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
        for ((y, twiddle), re) in y_chunks
            .iter_mut()
            .zip(twiddle_chunks.iter())
            .zip(re_chunks.iter())
        {
            let y_ptr = y.as_mut_ptr().cast::<f32>();
            let v = _mm256_loadu_ps(y_ptr);
            let t = _mm256_loadu_ps(twiddle.as_ptr().cast());
            // [re*tre, im*tim, ...] -> odd lanes hold re*tre - im*tim
            let product = _mm256_mul_ps(v, t);
            let swapped = _mm256_permute_ps::<0b10_11_00_01>(product);
            let im = _mm256_sub_ps(swapped, product);
            let re =
                _mm256_permutevar8x32_ps(_mm256_castps128_ps256(_mm_loadu_ps(re.as_ptr())), expand);
            let re = _mm256_add_ps(re, first_v);
            _mm256_storeu_ps(y_ptr, _mm256_blend_ps::<0b1010_1010>(re, im));
        }
    }
    combine_scalar(y_rem, twiddle_rem, re_rem, first);
}

#[target_feature(enable = "avx2", enable = "fma")]
fn combine_f64(y: &mut [Complex<f64>], twiddles: &[Complex<f64>], re: &[f64], first: f64) {
    let (y_chunks, y_rem) = y.as_chunks_mut::<2>();
    let (twiddle_chunks, twiddle_rem) = twiddles.as_chunks::<2>();
    let (re_chunks, re_rem) = re.as_chunks::<2>();
    unsafe {
        let first_v = _mm256_set1_pd(first);
        for ((y, twiddle), re) in y_chunks
            .iter_mut()
            .zip(twiddle_chunks.iter())
            .zip(re_chunks.iter())
        {
            let y_ptr = y.as_mut_ptr().cast::<f64>();
            let v = _mm256_loadu_pd(y_ptr);
            let t = _mm256_loadu_pd(twiddle.as_ptr().cast());
            let product = _mm256_mul_pd(v, t);
            let swapped = _mm256_permute_pd::<0b0101>(product);
            let im = _mm256_sub_pd(swapped, product);
            let re = _mm256_permute4x64_pd::<0b01_01_00_00>(_mm256_castpd128_pd256(_mm_loadu_pd(
                re.as_ptr(),
            )));
            let re = _mm256_add_pd(re, first_v);
            _mm256_storeu_pd(y_ptr, _mm256_blend_pd::<0b1010>(re, im));
        }
    }
    combine_scalar(y_rem, twiddle_rem, re_rem, first);
}
