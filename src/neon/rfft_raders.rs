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

/// `y[p] = (first + re[p], Re(y[p] * twiddles[p]))` with de-interleaved NEON loads.
pub(crate) struct NeonRadersRfftCombiner {}

impl RadersRfftCombiner<f32> for NeonRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f32>], twiddles: &[Complex<f32>], re: &[f32], first: f32) {
        let (y_chunks, y_rem) = y.as_chunks_mut::<4>();
        let (twiddle_chunks, twiddle_rem) = twiddles.as_chunks::<4>();
        let (re_chunks, re_rem) = re.as_chunks::<4>();
        unsafe {
            let first_v = vdupq_n_f32(first);
            for ((y, twiddle), re) in y_chunks
                .iter_mut()
                .zip(twiddle_chunks.iter())
                .zip(re_chunks.iter())
            {
                let y_ptr = y.as_mut_ptr().cast::<f32>();
                let float32x4x2_t(y_re, y_im) = vld2q_f32(y_ptr);
                let float32x4x2_t(twiddle_re, twiddle_im) = vld2q_f32(twiddle.as_ptr().cast());
                let re = vld1q_f32(re.as_ptr());
                let im = vfmsq_f32(vmulq_f32(y_re, twiddle_re), y_im, twiddle_im);
                vst2q_f32(y_ptr, float32x4x2_t(vaddq_f32(re, first_v), im));
            }
        }
        combine_scalar(y_rem, twiddle_rem, re_rem, first);
    }
}

impl RadersRfftCombiner<f64> for NeonRadersRfftCombiner {
    fn combine(&self, y: &mut [Complex<f64>], twiddles: &[Complex<f64>], re: &[f64], first: f64) {
        let (y_chunks, y_rem) = y.as_chunks_mut::<2>();
        let (twiddle_chunks, twiddle_rem) = twiddles.as_chunks::<2>();
        let (re_chunks, re_rem) = re.as_chunks::<2>();
        unsafe {
            let first_v = vdupq_n_f64(first);
            for ((y, twiddle), re) in y_chunks
                .iter_mut()
                .zip(twiddle_chunks.iter())
                .zip(re_chunks.iter())
            {
                let y_ptr = y.as_mut_ptr().cast::<f64>();
                let float64x2x2_t(y_re, y_im) = vld2q_f64(y_ptr);
                let float64x2x2_t(twiddle_re, twiddle_im) = vld2q_f64(twiddle.as_ptr().cast());
                let re = vld1q_f64(re.as_ptr());
                let im = vfmsq_f64(vmulq_f64(y_re, twiddle_re), y_im, twiddle_im);
                vst2q_f64(y_ptr, float64x2x2_t(vaddq_f64(re, first_v), im));
            }
        }
        combine_scalar(y_rem, twiddle_rem, re_rem, first);
    }
}
