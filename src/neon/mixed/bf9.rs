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
use crate::neon::util::*;
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly9d {
    tw1: float64x2_t,
    tw2: float64x2_t,
    tw3_re: float64x2_t,
    tw3_im: float64x2_t,
    tw4: float64x2_t,
}

impl ColumnButterfly9d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let twiddle3 = compute_twiddle::<f64>(1, 3, fft_direction);
            let tw1 = compute_twiddle::<f64>(1, 9, fft_direction);
            let tw2 = compute_twiddle::<f64>(2, 9, fft_direction);
            let tw4 = compute_twiddle::<f64>(4, 9, fft_direction);
            Self {
                tw3_re: vdupq_n_f64(twiddle3.re),
                tw3_im: vld1q_f64(
                    [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                        .as_ptr()
                        .cast(),
                ),
                tw1: vld1q_f64([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                tw2: vld1q_f64([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                tw4: vld1q_f64([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 9]) -> [NeonStoreD; 9] {
        unsafe {
            let (u0, u3, u6) = NeonButterfly::butterfly3_f64(
                store[0].v,
                store[3].v,
                store[6].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u1, mut u4, mut u7) = NeonButterfly::butterfly3_f64(
                store[1].v,
                store[4].v,
                store[7].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u2, mut u5, mut u8) = NeonButterfly::butterfly3_f64(
                store[2].v,
                store[5].v,
                store[8].v,
                self.tw3_re,
                self.tw3_im,
            );

            u4 = mul_complex_f64(u4, self.tw1);
            u7 = mul_complex_f64(u7, self.tw2);
            u5 = mul_complex_f64(u5, self.tw2);
            u8 = mul_complex_f64(u8, self.tw4);

            let (y0, y3, y6) = NeonButterfly::butterfly3_f64(u0, u1, u2, self.tw3_re, self.tw3_im);
            let (y1, y4, y7) = NeonButterfly::butterfly3_f64(u3, u4, u5, self.tw3_re, self.tw3_im);
            let (y2, y5, y8) = NeonButterfly::butterfly3_f64(u6, u7, u8, self.tw3_re, self.tw3_im);
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
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly9d {
    tw1: float64x2_t,
    tw2: float64x2_t,
    tw3_re: float64x2_t,
    tw3_im: float64x2_t,
    tw4: float64x2_t,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly9d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let twiddle3 = compute_twiddle::<f64>(1, 3, fft_direction);
            let tw1 = compute_twiddle::<f64>(1, 9, fft_direction);
            let tw2 = compute_twiddle::<f64>(2, 9, fft_direction);
            let tw4 = compute_twiddle::<f64>(4, 9, fft_direction);
            Self {
                tw3_re: vdupq_n_f64(twiddle3.re),
                tw3_im: vld1q_f64(
                    [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                        .as_ptr()
                        .cast(),
                ),
                tw1: vld1q_f64([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                tw2: vld1q_f64([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                tw4: vld1q_f64([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 9]) -> [NeonStoreD; 9] {
        unsafe {
            let (u0, u3, u6) = NeonButterfly::butterfly3_f64(
                store[0].v,
                store[3].v,
                store[6].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u1, mut u4, mut u7) = NeonButterfly::butterfly3_f64(
                store[1].v,
                store[4].v,
                store[7].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u2, mut u5, mut u8) = NeonButterfly::butterfly3_f64(
                store[2].v,
                store[5].v,
                store[8].v,
                self.tw3_re,
                self.tw3_im,
            );

            u4 = fcma_complex_f64(u4, self.tw1);
            u7 = fcma_complex_f64(u7, self.tw2);
            u5 = fcma_complex_f64(u5, self.tw2);
            u8 = fcma_complex_f64(u8, self.tw4);

            let (y0, y3, y6) = NeonButterfly::butterfly3_f64(u0, u1, u2, self.tw3_re, self.tw3_im);
            let (y1, y4, y7) = NeonButterfly::butterfly3_f64(u3, u4, u5, self.tw3_re, self.tw3_im);
            let (y2, y5, y8) = NeonButterfly::butterfly3_f64(u6, u7, u8, self.tw3_re, self.tw3_im);
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
            ]
        }
    }
}

pub(crate) struct ColumnButterfly9f {
    tw1: float32x4_t,
    tw2: float32x4_t,
    tw3_re: float32x4_t,
    tw3_im: float32x4_t,
    tw4: float32x4_t,
}

impl ColumnButterfly9f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let twiddle3 = compute_twiddle::<f32>(1, 3, fft_direction);
            let tw1 = compute_twiddle::<f32>(1, 9, fft_direction);
            let tw2 = compute_twiddle::<f32>(2, 9, fft_direction);
            let tw4 = compute_twiddle::<f32>(4, 9, fft_direction);
            Self {
                tw3_re: vdupq_n_f32(twiddle3.re),
                tw3_im: vld1q_f32(
                    [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                        .as_ptr()
                        .cast(),
                ),
                tw1: vld1q_f32([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                tw2: vld1q_f32([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                tw4: vld1q_f32([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 9]) -> [NeonStoreF; 9] {
        unsafe {
            let (u0, u3, u6) = NeonButterfly::butterfly3_f32(
                store[0].v,
                store[3].v,
                store[6].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u1, mut u4, mut u7) = NeonButterfly::butterfly3_f32(
                store[1].v,
                store[4].v,
                store[7].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u2, mut u5, mut u8) = NeonButterfly::butterfly3_f32(
                store[2].v,
                store[5].v,
                store[8].v,
                self.tw3_re,
                self.tw3_im,
            );

            u4 = mul_complex_f32(u4, self.tw1);
            u7 = mul_complex_f32(u7, self.tw2);
            u5 = mul_complex_f32(u5, self.tw2);
            u8 = mul_complex_f32(u8, self.tw4);

            let (y0, y3, y6) = NeonButterfly::butterfly3_f32(u0, u1, u2, self.tw3_re, self.tw3_im);
            let (y1, y4, y7) = NeonButterfly::butterfly3_f32(u3, u4, u5, self.tw3_re, self.tw3_im);
            let (y2, y5, y8) = NeonButterfly::butterfly3_f32(u6, u7, u8, self.tw3_re, self.tw3_im);
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
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 9]) -> [NeonStoreFh; 9] {
        unsafe {
            let (u0, u3, u6) = NeonButterfly::butterfly3h_f32(
                store[0].v,
                store[3].v,
                store[6].v,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (u1, mut u4, mut u7) = NeonButterfly::butterfly3h_f32(
                store[1].v,
                store[4].v,
                store[7].v,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (u2, mut u5, mut u8) = NeonButterfly::butterfly3h_f32(
                store[2].v,
                store[5].v,
                store[8].v,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );

            let hu4u7 = mul_complex_f32(
                vcombine_f32(u4, u7),
                vcombine_f32(vget_low_f32(self.tw1), vget_low_f32(self.tw2)),
            );
            let hu5u8 = mul_complex_f32(
                vcombine_f32(u5, u8),
                vcombine_f32(vget_low_f32(self.tw2), vget_low_f32(self.tw4)),
            );
            u4 = vget_low_f32(hu4u7);
            u7 = vget_high_f32(hu4u7);
            u5 = vget_low_f32(hu5u8);
            u8 = vget_high_f32(hu5u8);

            let (y0, y3, y6) = NeonButterfly::butterfly3h_f32(
                u0,
                u1,
                u2,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (y1, y4, y7) = NeonButterfly::butterfly3h_f32(
                u3,
                u4,
                u5,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (y2, y5, y8) = NeonButterfly::butterfly3h_f32(
                u6,
                u7,
                u8,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
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
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly9f {
    tw1: float32x4_t,
    tw2: float32x4_t,
    tw3_re: float32x4_t,
    tw3_im: float32x4_t,
    tw4: float32x4_t,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly9f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let twiddle3 = compute_twiddle::<f32>(1, 3, fft_direction);
            let tw1 = compute_twiddle::<f32>(1, 9, fft_direction);
            let tw2 = compute_twiddle::<f32>(2, 9, fft_direction);
            let tw4 = compute_twiddle::<f32>(4, 9, fft_direction);
            Self {
                tw3_re: vdupq_n_f32(twiddle3.re),
                tw3_im: vld1q_f32(
                    [-twiddle3.im, twiddle3.im, -twiddle3.im, twiddle3.im]
                        .as_ptr()
                        .cast(),
                ),
                tw1: vld1q_f32([tw1.re, tw1.im, tw1.re, tw1.im].as_ptr().cast()),
                tw2: vld1q_f32([tw2.re, tw2.im, tw2.re, tw2.im].as_ptr().cast()),
                tw4: vld1q_f32([tw4.re, tw4.im, tw4.re, tw4.im].as_ptr().cast()),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 9]) -> [NeonStoreF; 9] {
        unsafe {
            let (u0, u3, u6) = NeonButterfly::butterfly3_f32(
                store[0].v,
                store[3].v,
                store[6].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u1, mut u4, mut u7) = NeonButterfly::butterfly3_f32(
                store[1].v,
                store[4].v,
                store[7].v,
                self.tw3_re,
                self.tw3_im,
            );
            let (u2, mut u5, mut u8) = NeonButterfly::butterfly3_f32(
                store[2].v,
                store[5].v,
                store[8].v,
                self.tw3_re,
                self.tw3_im,
            );

            u4 = fcma_complex_f32(u4, self.tw1);
            u7 = fcma_complex_f32(u7, self.tw2);
            u5 = fcma_complex_f32(u5, self.tw2);
            u8 = fcma_complex_f32(u8, self.tw4);

            let (y0, y3, y6) = NeonButterfly::butterfly3_f32(u0, u1, u2, self.tw3_re, self.tw3_im);
            let (y1, y4, y7) = NeonButterfly::butterfly3_f32(u3, u4, u5, self.tw3_re, self.tw3_im);
            let (y2, y5, y8) = NeonButterfly::butterfly3_f32(u6, u7, u8, self.tw3_re, self.tw3_im);
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
            ]
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 9]) -> [NeonStoreFh; 9] {
        unsafe {
            let (u0, u3, u6) = NeonButterfly::butterfly3h_f32(
                store[0].v,
                store[3].v,
                store[6].v,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (u1, mut u4, mut u7) = NeonButterfly::butterfly3h_f32(
                store[1].v,
                store[4].v,
                store[7].v,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (u2, mut u5, mut u8) = NeonButterfly::butterfly3h_f32(
                store[2].v,
                store[5].v,
                store[8].v,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );

            let hu4u7 = fcma_complex_f32(
                vcombine_f32(u4, u7),
                vcombine_f32(vget_low_f32(self.tw1), vget_low_f32(self.tw2)),
            );
            let hu5u8 = fcma_complex_f32(
                vcombine_f32(u5, u8),
                vcombine_f32(vget_low_f32(self.tw2), vget_low_f32(self.tw4)),
            );
            u4 = vget_low_f32(hu4u7);
            u7 = vget_high_f32(hu4u7);
            u5 = vget_low_f32(hu5u8);
            u8 = vget_high_f32(hu5u8);

            let (y0, y3, y6) = NeonButterfly::butterfly3h_f32(
                u0,
                u1,
                u2,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (y1, y4, y7) = NeonButterfly::butterfly3h_f32(
                u3,
                u4,
                u5,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
            );
            let (y2, y5, y8) = NeonButterfly::butterfly3h_f32(
                u6,
                u7,
                u8,
                vget_low_f32(self.tw3_re),
                vget_low_f32(self.tw3_im),
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
            ]
        }
    }
}
