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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly6d {
    tw_re: float64x2_t,
    tw_im: float64x2_t,
}

impl ColumnButterfly6d {
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

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 6]) -> [NeonStoreD; 6] {
        let (t0, t2, t4) = NeonButterfly::butterfly3_f64(
            store[0].v, store[2].v, store[4].v, self.tw_re, self.tw_im,
        );
        let (t1, t3, t5) = NeonButterfly::butterfly3_f64(
            store[3].v, store[5].v, store[1].v, self.tw_re, self.tw_im,
        );
        let (y0, y3) = NeonButterfly::butterfly2_f64(t0, t1);
        let (y4, y1) = NeonButterfly::butterfly2_f64(t2, t3);
        let (y2, y5) = NeonButterfly::butterfly2_f64(t4, t5);
        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y4),
            NeonStoreD::raw(y5),
        ]
    }
}

pub(crate) struct ColumnButterfly6f {
    tw_re: float32x4_t,
    tw_im: float32x4_t,
}

impl ColumnButterfly6f {
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

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 6]) -> [NeonStoreF; 6] {
        let (t0, t2, t4) = NeonButterfly::butterfly3_f32(
            store[0].v, store[2].v, store[4].v, self.tw_re, self.tw_im,
        );
        let (t1, t3, t5) = NeonButterfly::butterfly3_f32(
            store[3].v, store[5].v, store[1].v, self.tw_re, self.tw_im,
        );
        let (y0, y3) = NeonButterfly::butterfly2_f32(t0, t1);
        let (y4, y1) = NeonButterfly::butterfly2_f32(t2, t3);
        let (y2, y5) = NeonButterfly::butterfly2_f32(t4, t5);
        [
            NeonStoreF::raw(y0),
            NeonStoreF::raw(y1),
            NeonStoreF::raw(y2),
            NeonStoreF::raw(y3),
            NeonStoreF::raw(y4),
            NeonStoreF::raw(y5),
        ]
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 6]) -> [NeonStoreFh; 6] {
        unsafe {
            let (t0, t2, t4) = NeonButterfly::butterfly3h_f32(
                store[0].v,
                store[2].v,
                store[4].v,
                vget_low_f32(self.tw_re),
                vget_low_f32(self.tw_im),
            );
            let (t1, t3, t5) = NeonButterfly::butterfly3h_f32(
                store[3].v,
                store[5].v,
                store[1].v,
                vget_low_f32(self.tw_re),
                vget_low_f32(self.tw_im),
            );
            let (y0, y3) = NeonButterfly::butterfly2h_f32(t0, t1);
            let (y4, y1) = NeonButterfly::butterfly2h_f32(t2, t3);
            let (y2, y5) = NeonButterfly::butterfly2h_f32(t4, t5);
            [
                NeonStoreFh::raw(y0),
                NeonStoreFh::raw(y1),
                NeonStoreFh::raw(y2),
                NeonStoreFh::raw(y3),
                NeonStoreFh::raw(y4),
                NeonStoreFh::raw(y5),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly6f {
    tw_re: float32x4_t,
    tw_im: float32x4_t,
    n_tw_im: float32x4_t,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly6f {
    pub(crate) fn new(direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f32>(1, 3, direction);
        unsafe {
            let q = vld1q_f32(
                [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im]
                    .as_ptr()
                    .cast(),
            );
            Self {
                tw_re: vdupq_n_f32(twiddle.re),
                tw_im: q,
                n_tw_im: vnegq_f32(q),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 6]) -> [NeonStoreF; 6] {
        let (t0, t2, t4) = NeonButterfly::butterfly3_f32_fcma(
            store[0].v,
            store[2].v,
            store[4].v,
            self.tw_re,
            self.tw_im,
            self.n_tw_im,
        );
        let (t1, t3, t5) = NeonButterfly::butterfly3_f32_fcma(
            store[3].v,
            store[5].v,
            store[1].v,
            self.tw_re,
            self.tw_im,
            self.n_tw_im,
        );
        let (y0, y3) = NeonButterfly::butterfly2_f32(t0, t1);
        let (y4, y1) = NeonButterfly::butterfly2_f32(t2, t3);
        let (y2, y5) = NeonButterfly::butterfly2_f32(t4, t5);
        [
            NeonStoreF::raw(y0),
            NeonStoreF::raw(y1),
            NeonStoreF::raw(y2),
            NeonStoreF::raw(y3),
            NeonStoreF::raw(y4),
            NeonStoreF::raw(y5),
        ]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 6]) -> [NeonStoreFh; 6] {
        let (t0, t2, t4) = NeonButterfly::butterfly3h_f32_fcma(
            store[0].v,
            store[2].v,
            store[4].v,
            vget_low_f32(self.tw_re),
            vget_low_f32(self.tw_im),
            vget_low_f32(self.n_tw_im),
        );
        let (t1, t3, t5) = NeonButterfly::butterfly3h_f32_fcma(
            store[3].v,
            store[5].v,
            store[1].v,
            vget_low_f32(self.tw_re),
            vget_low_f32(self.tw_im),
            vget_low_f32(self.n_tw_im),
        );
        let (y0, y3) = NeonButterfly::butterfly2h_f32(t0, t1);
        let (y4, y1) = NeonButterfly::butterfly2h_f32(t2, t3);
        let (y2, y5) = NeonButterfly::butterfly2h_f32(t4, t5);
        [
            NeonStoreFh::raw(y0),
            NeonStoreFh::raw(y1),
            NeonStoreFh::raw(y2),
            NeonStoreFh::raw(y3),
            NeonStoreFh::raw(y4),
            NeonStoreFh::raw(y5),
        ]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly6d {
    tw_re: float64x2_t,
    tw_im: float64x2_t,
    n_tw_im: float64x2_t,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly6d {
    #[inline]
    pub(crate) fn new(direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f64>(1, 3, direction);
        unsafe {
            let q = vld1q_f64([-twiddle.im, twiddle.im].as_ptr().cast());
            Self {
                tw_re: vdupq_n_f64(twiddle.re),
                tw_im: q,
                n_tw_im: vnegq_f64(q),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 6]) -> [NeonStoreD; 6] {
        let (t0, t2, t4) = NeonButterfly::butterfly3_f64_fcma(
            store[0].v,
            store[2].v,
            store[4].v,
            self.tw_re,
            self.tw_im,
            self.n_tw_im,
        );
        let (t1, t3, t5) = NeonButterfly::butterfly3_f64_fcma(
            store[3].v,
            store[5].v,
            store[1].v,
            self.tw_re,
            self.tw_im,
            self.n_tw_im,
        );
        let (y0, y3) = NeonButterfly::butterfly2_f64(t0, t1);
        let (y4, y1) = NeonButterfly::butterfly2_f64(t2, t3);
        let (y2, y5) = NeonButterfly::butterfly2_f64(t4, t5);
        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y4),
            NeonStoreD::raw(y5),
        ]
    }
}
