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
#[cfg(feature = "fcma")]
use crate::neon::butterflies::{FastFcmaBf4d, FastFcmaBf4f};
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly12d {
    tw_re: float64x2_t,
    tw_im: float64x2_t,
    rotate: float64x2_t,
}

impl ColumnButterfly12d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f64>(1, 3, fft_direction);
        unsafe {
            static INVERSE_ROT: [f64; 2] = [-0.0, 0.0];
            static FORWARD_ROT: [f64; 2] = [0.0, -0.0];
            Self {
                rotate: vld1q_f64(match fft_direction {
                    FftDirection::Inverse => INVERSE_ROT.as_ptr(),
                    FftDirection::Forward => FORWARD_ROT.as_ptr(),
                }),
                tw_re: vdupq_n_f64(twiddle.re),
                tw_im: vld1q_f64([-twiddle.im, twiddle.im].as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 12]) -> [NeonStoreD; 12] {
        let (u0, u1, u2, u3) = NeonButterfly::butterfly4_f64(
            store[0].v,
            store[3].v,
            store[6].v,
            store[9].v,
            self.rotate,
        );
        let (u4, u5, u6, u7) = NeonButterfly::butterfly4_f64(
            store[4].v,
            store[7].v,
            store[10].v,
            store[1].v,
            self.rotate,
        );
        let (u8, u9, u10, u11) = NeonButterfly::butterfly4_f64(
            store[8].v,
            store[11].v,
            store[2].v,
            store[5].v,
            self.rotate,
        );

        let (y0, y4, y8) = NeonButterfly::butterfly3_f64(u0, u4, u8, self.tw_re, self.tw_im);
        let (y9, y1, y5) = NeonButterfly::butterfly3_f64(u1, u5, u9, self.tw_re, self.tw_im);
        let (y6, y10, y2) = NeonButterfly::butterfly3_f64(u2, u6, u10, self.tw_re, self.tw_im);
        let (y3, y7, y11) = NeonButterfly::butterfly3_f64(u3, u7, u11, self.tw_re, self.tw_im);

        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y4),
            NeonStoreD::raw(y5),
            NeonStoreD::raw(y6),
            NeonStoreD::raw(y7),
            NeonStoreD::raw(y8),
            NeonStoreD::raw(y9),
            NeonStoreD::raw(y10),
            NeonStoreD::raw(y11),
        ]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly12d {
    tw_re: float64x2_t,
    tw_im: float64x2_t,
    bf4: FastFcmaBf4d,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly12d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f64>(1, 3, fft_direction);
        unsafe {
            let q = vld1q_f64([-twiddle.im, twiddle.im].as_ptr().cast());
            Self {
                tw_re: vdupq_n_f64(twiddle.re),
                tw_im: q,
                bf4: FastFcmaBf4d::new(fft_direction),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 12]) -> [NeonStoreD; 12] {
        let (u0, u1, u2, u3) = self
            .bf4
            .exec(store[0].v, store[3].v, store[6].v, store[9].v);
        let (u4, u5, u6, u7) = self
            .bf4
            .exec(store[4].v, store[7].v, store[10].v, store[1].v);
        let (u8, u9, u10, u11) = self
            .bf4
            .exec(store[8].v, store[11].v, store[2].v, store[5].v);

        let (y0, y4, y8) = NeonButterfly::butterfly3_f64_fcma(u0, u4, u8, self.tw_re, self.tw_im);
        let (y9, y1, y5) = NeonButterfly::butterfly3_f64_fcma(u1, u5, u9, self.tw_re, self.tw_im);
        let (y6, y10, y2) = NeonButterfly::butterfly3_f64_fcma(u2, u6, u10, self.tw_re, self.tw_im);
        let (y3, y7, y11) = NeonButterfly::butterfly3_f64_fcma(u3, u7, u11, self.tw_re, self.tw_im);

        [
            NeonStoreD::raw(y0),
            NeonStoreD::raw(y1),
            NeonStoreD::raw(y2),
            NeonStoreD::raw(y3),
            NeonStoreD::raw(y4),
            NeonStoreD::raw(y5),
            NeonStoreD::raw(y6),
            NeonStoreD::raw(y7),
            NeonStoreD::raw(y8),
            NeonStoreD::raw(y9),
            NeonStoreD::raw(y10),
            NeonStoreD::raw(y11),
        ]
    }
}

pub(crate) struct ColumnButterfly12f {
    tw_re: float32x4_t,
    tw_im: float32x4_t,
    rotate: float32x4_t,
}

impl ColumnButterfly12f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f32>(1, 3, fft_direction);
        unsafe {
            static INVERSE_ROT: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            static FORWARD_ROT: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
            Self {
                rotate: vld1q_f32(match fft_direction {
                    FftDirection::Inverse => INVERSE_ROT.as_ptr(),
                    FftDirection::Forward => FORWARD_ROT.as_ptr(),
                }),
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
    pub(crate) fn exec(&self, store: [NeonStoreF; 12]) -> [NeonStoreF; 12] {
        let (u0, u1, u2, u3) = NeonButterfly::butterfly4_f32(
            store[0].v,
            store[3].v,
            store[6].v,
            store[9].v,
            self.rotate,
        );
        let (u4, u5, u6, u7) = NeonButterfly::butterfly4_f32(
            store[4].v,
            store[7].v,
            store[10].v,
            store[1].v,
            self.rotate,
        );
        let (u8, u9, u10, u11) = NeonButterfly::butterfly4_f32(
            store[8].v,
            store[11].v,
            store[2].v,
            store[5].v,
            self.rotate,
        );

        let (y0, y4, y8) = NeonButterfly::butterfly3_f32(u0, u4, u8, self.tw_re, self.tw_im);
        let (y9, y1, y5) = NeonButterfly::butterfly3_f32(u1, u5, u9, self.tw_re, self.tw_im);
        let (y6, y10, y2) = NeonButterfly::butterfly3_f32(u2, u6, u10, self.tw_re, self.tw_im);
        let (y3, y7, y11) = NeonButterfly::butterfly3_f32(u3, u7, u11, self.tw_re, self.tw_im);

        [
            NeonStoreF::raw(y0),
            NeonStoreF::raw(y1),
            NeonStoreF::raw(y2),
            NeonStoreF::raw(y3),
            NeonStoreF::raw(y4),
            NeonStoreF::raw(y5),
            NeonStoreF::raw(y6),
            NeonStoreF::raw(y7),
            NeonStoreF::raw(y8),
            NeonStoreF::raw(y9),
            NeonStoreF::raw(y10),
            NeonStoreF::raw(y11),
        ]
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 12]) -> [NeonStoreFh; 12] {
        unsafe {
            let (u0, u1, u2, u3) = NeonButterfly::butterfly4h_f32(
                store[0].v,
                store[3].v,
                store[6].v,
                store[9].v,
                vget_low_f32(self.rotate),
            );
            let (u4, u5, u6, u7) = NeonButterfly::butterfly4h_f32(
                store[4].v,
                store[7].v,
                store[10].v,
                store[1].v,
                vget_low_f32(self.rotate),
            );
            let (u8, u9, u10, u11) = NeonButterfly::butterfly4h_f32(
                store[8].v,
                store[11].v,
                store[2].v,
                store[5].v,
                vget_low_f32(self.rotate),
            );

            let (y0, y4, y8) = NeonButterfly::butterfly3h_f32(
                u0,
                u4,
                u8,
                vget_low_f32(self.tw_re),
                vget_low_f32(self.tw_im),
            );
            let (y9, y1, y5) = NeonButterfly::butterfly3h_f32(
                u1,
                u5,
                u9,
                vget_low_f32(self.tw_re),
                vget_low_f32(self.tw_im),
            );
            let (y6, y10, y2) = NeonButterfly::butterfly3h_f32(
                u2,
                u6,
                u10,
                vget_low_f32(self.tw_re),
                vget_low_f32(self.tw_im),
            );
            let (y3, y7, y11) = NeonButterfly::butterfly3h_f32(
                u3,
                u7,
                u11,
                vget_low_f32(self.tw_re),
                vget_low_f32(self.tw_im),
            );

            [
                NeonStoreFh::raw(y0),
                NeonStoreFh::raw(y1),
                NeonStoreFh::raw(y2),
                NeonStoreFh::raw(y3),
                NeonStoreFh::raw(y4),
                NeonStoreFh::raw(y5),
                NeonStoreFh::raw(y6),
                NeonStoreFh::raw(y7),
                NeonStoreFh::raw(y8),
                NeonStoreFh::raw(y9),
                NeonStoreFh::raw(y10),
                NeonStoreFh::raw(y11),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly12f {
    tw_re: float32x4_t,
    tw_im: float32x4_t,
    bf4: FastFcmaBf4f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly12f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<f32>(1, 3, fft_direction);
        unsafe {
            let q = vld1q_f32(
                [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im]
                    .as_ptr()
                    .cast(),
            );
            Self {
                tw_re: vdupq_n_f32(twiddle.re),
                tw_im: q,
                bf4: FastFcmaBf4f::new(fft_direction),
            }
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 12]) -> [NeonStoreF; 12] {
        let (u0, u1, u2, u3) = self
            .bf4
            .exec(store[0].v, store[3].v, store[6].v, store[9].v);
        let (u4, u5, u6, u7) = self
            .bf4
            .exec(store[4].v, store[7].v, store[10].v, store[1].v);
        let (u8, u9, u10, u11) = self
            .bf4
            .exec(store[8].v, store[11].v, store[2].v, store[5].v);

        let (y0, y4, y8) = NeonButterfly::butterfly3_f32_fcma(u0, u4, u8, self.tw_re, self.tw_im);
        let (y9, y1, y5) = NeonButterfly::butterfly3_f32_fcma(u1, u5, u9, self.tw_re, self.tw_im);
        let (y6, y10, y2) = NeonButterfly::butterfly3_f32_fcma(u2, u6, u10, self.tw_re, self.tw_im);
        let (y3, y7, y11) = NeonButterfly::butterfly3_f32_fcma(u3, u7, u11, self.tw_re, self.tw_im);

        [
            NeonStoreF::raw(y0),
            NeonStoreF::raw(y1),
            NeonStoreF::raw(y2),
            NeonStoreF::raw(y3),
            NeonStoreF::raw(y4),
            NeonStoreF::raw(y5),
            NeonStoreF::raw(y6),
            NeonStoreF::raw(y7),
            NeonStoreF::raw(y8),
            NeonStoreF::raw(y9),
            NeonStoreF::raw(y10),
            NeonStoreF::raw(y11),
        ]
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 12]) -> [NeonStoreFh; 12] {
        let (u0, u1, u2, u3) = self
            .bf4
            .exech(store[0].v, store[3].v, store[6].v, store[9].v);
        let (u4, u5, u6, u7) = self
            .bf4
            .exech(store[4].v, store[7].v, store[10].v, store[1].v);
        let (u8, u9, u10, u11) = self
            .bf4
            .exech(store[8].v, store[11].v, store[2].v, store[5].v);

        let (y0, y4, y8) = NeonButterfly::butterfly3h_f32_fcma(
            u0,
            u4,
            u8,
            vget_low_f32(self.tw_re),
            vget_low_f32(self.tw_im),
        );
        let (y9, y1, y5) = NeonButterfly::butterfly3h_f32_fcma(
            u1,
            u5,
            u9,
            vget_low_f32(self.tw_re),
            vget_low_f32(self.tw_im),
        );
        let (y6, y10, y2) = NeonButterfly::butterfly3h_f32_fcma(
            u2,
            u6,
            u10,
            vget_low_f32(self.tw_re),
            vget_low_f32(self.tw_im),
        );
        let (y3, y7, y11) = NeonButterfly::butterfly3h_f32_fcma(
            u3,
            u7,
            u11,
            vget_low_f32(self.tw_re),
            vget_low_f32(self.tw_im),
        );

        [
            NeonStoreFh::raw(y0),
            NeonStoreFh::raw(y1),
            NeonStoreFh::raw(y2),
            NeonStoreFh::raw(y3),
            NeonStoreFh::raw(y4),
            NeonStoreFh::raw(y5),
            NeonStoreFh::raw(y6),
            NeonStoreFh::raw(y7),
            NeonStoreFh::raw(y8),
            NeonStoreFh::raw(y9),
            NeonStoreFh::raw(y10),
            NeonStoreFh::raw(y11),
        ]
    }
}
