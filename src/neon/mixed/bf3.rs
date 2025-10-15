/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use crate::FftDirection;
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly3d {
    tw_re: float64x2_t,
    tw_im: float64x2_t,
}

pub(crate) struct ColumnButterfly3f {
    tw_re: float32x4_t,
    tw_im: float32x4_t,
}

impl ColumnButterfly3d {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f64>(1, 3, direction);
        unsafe {
            Self {
                tw_re: vdupq_n_f64(twiddle.re),
                tw_im: vld1q_f64([-twiddle.im, twiddle.im].as_ptr().cast()),
            }
        }
    }

    #[inline]
    pub(crate) fn exec(&self, store: [NeonStoreD; 3]) -> [NeonStoreD; 3] {
        unsafe {
            let xp = vaddq_f64(store[1].v, store[2].v);
            let xn = vsubq_f64(store[1].v, store[2].v);
            let sum = vaddq_f64(store[0].v, xp);

            let w_1 = vfmaq_f64(store[0].v, self.tw_re, xp);

            let xn_rot = vextq_f64::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfmaq_f64(w_1, self.tw_im, xn_rot);
            let y2 = vfmsq_f64(w_1, self.tw_im, xn_rot);
            [
                NeonStoreD::raw(y0),
                NeonStoreD::raw(y1),
                NeonStoreD::raw(y2),
            ]
        }
    }
}

impl ColumnButterfly3f {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f32>(1, 3, direction);
        unsafe {
            Self {
                tw_re: vdupq_n_f32(twiddle.re),
                tw_im: vld1q_f32(
                    [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im]
                        .as_ptr()
                        .cast(),
                ),
            }
        }
    }

    #[inline]
    pub(crate) fn exec(&self, store: [NeonStoreF; 3]) -> [NeonStoreF; 3] {
        unsafe {
            let xp = vaddq_f32(store[1].v, store[2].v);
            let xn = vsubq_f32(store[1].v, store[2].v);
            let sum = vaddq_f32(store[0].v, xp);

            let w_1 = vfmaq_f32(store[0].v, self.tw_re, xp);

            let xn_rot = vrev64q_f32(xn);

            let y0 = sum;
            let y1 = vfmaq_f32(w_1, self.tw_im, xn_rot);
            let y2 = vfmsq_f32(w_1, self.tw_im, xn_rot);
            [
                NeonStoreF::raw(y0),
                NeonStoreF::raw(y1),
                NeonStoreF::raw(y2),
            ]
        }
    }

    #[inline]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 3]) -> [NeonStoreFh; 3] {
        unsafe {
            let xp = vadd_f32(store[1].v, store[2].v);
            let xn = vsub_f32(store[1].v, store[2].v);
            let sum = vadd_f32(store[0].v, xp);

            let w_1 = vfma_f32(store[0].v, vget_low_f32(self.tw_re), xp);

            let xn_rot = vext_f32::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfma_f32(w_1, vget_low_f32(self.tw_im), xn_rot);
            let y2 = vfms_f32(w_1, vget_low_f32(self.tw_im), xn_rot);
            [
                NeonStoreFh::raw(y0),
                NeonStoreFh::raw(y1),
                NeonStoreFh::raw(y2),
            ]
        }
    }
}
