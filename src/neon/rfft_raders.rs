/*
 * // Copyright (c) Radzivon Bartoshyk 09/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //2028
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
use std::arch::aarch64::*;

/// Reconstruct complex bins from the two halves of the real convolution.
pub(crate) struct NeonRadersRfftCombiner {}

impl RadersRfftCombiner<f32> for NeonRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f32>], lo: &[f32], hi: &[f32], signs: &[f32], first: f32) {
        let (y_chunks, y_rem) = y.as_chunks_mut::<4>();
        let (lo_chunks, lo_rem) = lo.as_chunks::<4>();
        let (hi_chunks, hi_rem) = hi.as_chunks::<4>();
        let (sign_chunks, sign_rem) = signs.as_chunks::<4>();
        unsafe {
            let first_v = vdupq_n_f32(first);
            for (((y, lo), hi), signs) in y_chunks
                .iter_mut()
                .zip(lo_chunks)
                .zip(hi_chunks)
                .zip(sign_chunks)
            {
                let lo = vld1q_f32(lo.as_ptr());
                let hi = vld1q_f32(hi.as_ptr());
                let signs = vld1q_f32(signs.as_ptr());
                let re = vaddq_f32(first_v, vaddq_f32(lo, hi));
                let im = vmulq_f32(vsubq_f32(lo, hi), signs);
                vst2q_f32(y.as_mut_ptr().cast(), float32x4x2_t(re, im));
            }
        }
        combine_scalar(y_rem, lo_rem, hi_rem, sign_rem, first);
    }
}

impl RadersRfftCombiner<f64> for NeonRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f64>], lo: &[f64], hi: &[f64], signs: &[f64], first: f64) {
        let (y_chunks, y_rem) = y.as_chunks_mut::<2>();
        let (lo_chunks, lo_rem) = lo.as_chunks::<2>();
        let (hi_chunks, hi_rem) = hi.as_chunks::<2>();
        let (sign_chunks, sign_rem) = signs.as_chunks::<2>();
        unsafe {
            let first_v = vdupq_n_f64(first);
            for (((y, lo), hi), signs) in y_chunks
                .iter_mut()
                .zip(lo_chunks)
                .zip(hi_chunks)
                .zip(sign_chunks)
            {
                let lo = vld1q_f64(lo.as_ptr());
                let hi = vld1q_f64(hi.as_ptr());
                let signs = vld1q_f64(signs.as_ptr());
                let re = vaddq_f64(first_v, vaddq_f64(lo, hi));
                let im = vmulq_f64(vsubq_f64(lo, hi), signs);
                vst2q_f64(y.as_mut_ptr().cast(), float64x2x2_t(re, im));
            }
        }
        combine_scalar(y_rem, lo_rem, hi_rem, sign_rem, first);
    }
}
