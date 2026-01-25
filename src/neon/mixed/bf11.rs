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
use crate::neon::util::{v_rotate90_f32, v_rotate90_f64, vh_rotate90_f32};
use crate::util::compute_twiddle;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly11d {
    rotate: float64x2_t,
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
    twiddle4: Complex<f64>,
    twiddle5: Complex<f64>,
}

impl ColumnButterfly11d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: vld1q_f64([-0.0f64, 0.0, -0.0f64, 0.0].as_ptr().cast()),
                twiddle1: compute_twiddle(1, 11, fft_direction),
                twiddle2: compute_twiddle(2, 11, fft_direction),
                twiddle3: compute_twiddle(3, 11, fft_direction),
                twiddle4: compute_twiddle(4, 11, fft_direction),
                twiddle5: compute_twiddle(5, 11, fft_direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 11]) -> [NeonStoreD; 11] {
        unsafe {
            let y00 = store[0].v;
            let (x1p10, x1m10) = NeonButterfly::butterfly2_f64(store[1].v, store[10].v);
            let x1m10 = v_rotate90_f64(x1m10, self.rotate);
            let y00 = vaddq_f64(y00, x1p10);
            let (x2p9, x2m9) = NeonButterfly::butterfly2_f64(store[2].v, store[9].v);
            let x2m9 = v_rotate90_f64(x2m9, self.rotate);
            let y00 = vaddq_f64(y00, x2p9);
            let (x3p8, x3m8) = NeonButterfly::butterfly2_f64(store[3].v, store[8].v);
            let x3m8 = v_rotate90_f64(x3m8, self.rotate);
            let y00 = vaddq_f64(y00, x3p8);
            let (x4p7, x4m7) = NeonButterfly::butterfly2_f64(store[4].v, store[7].v);
            let x4m7 = v_rotate90_f64(x4m7, self.rotate);
            let y00 = vaddq_f64(y00, x4p7);
            let (x5p6, x5m6) = NeonButterfly::butterfly2_f64(store[5].v, store[6].v);
            let x5m6 = v_rotate90_f64(x5m6, self.rotate);
            let y00 = vaddq_f64(y00, x5p6);

            let m0110a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle1.re);
            let m0110a = vfmaq_n_f64(m0110a, x2p9, self.twiddle2.re);
            let m0110a = vfmaq_n_f64(m0110a, x3p8, self.twiddle3.re);
            let m0110a = vfmaq_n_f64(m0110a, x4p7, self.twiddle4.re);
            let m0110a = vfmaq_n_f64(m0110a, x5p6, self.twiddle5.re);
            let m0110b = vmulq_n_f64(x1m10, self.twiddle1.im);
            let m0110b = vfmaq_n_f64(m0110b, x2m9, self.twiddle2.im);
            let m0110b = vfmaq_n_f64(m0110b, x3m8, self.twiddle3.im);
            let m0110b = vfmaq_n_f64(m0110b, x4m7, self.twiddle4.im);
            let m0110b = vfmaq_n_f64(m0110b, x5m6, self.twiddle5.im);
            let (y01, y10) = NeonButterfly::butterfly2_f64(m0110a, m0110b);

            let m0209a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle2.re);
            let m0209a = vfmaq_n_f64(m0209a, x2p9, self.twiddle4.re);
            let m0209a = vfmaq_n_f64(m0209a, x3p8, self.twiddle5.re);
            let m0209a = vfmaq_n_f64(m0209a, x4p7, self.twiddle3.re);
            let m0209a = vfmaq_n_f64(m0209a, x5p6, self.twiddle1.re);
            let m0209b = vmulq_n_f64(x1m10, self.twiddle2.im);
            let m0209b = vfmaq_n_f64(m0209b, x2m9, self.twiddle4.im);
            let m0209b = vfmsq_n_f64(m0209b, x3m8, self.twiddle5.im);
            let m0209b = vfmsq_n_f64(m0209b, x4m7, self.twiddle3.im);
            let m0209b = vfmsq_n_f64(m0209b, x5m6, self.twiddle1.im);
            let (y02, y09) = NeonButterfly::butterfly2_f64(m0209a, m0209b);

            let m0308a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle3.re);
            let m0308a = vfmaq_n_f64(m0308a, x2p9, self.twiddle5.re);
            let m0308a = vfmaq_n_f64(m0308a, x3p8, self.twiddle2.re);
            let m0308a = vfmaq_n_f64(m0308a, x4p7, self.twiddle1.re);
            let m0308a = vfmaq_n_f64(m0308a, x5p6, self.twiddle4.re);
            let m0308b = vmulq_n_f64(x1m10, self.twiddle3.im);
            let m0308b = vfmsq_n_f64(m0308b, x2m9, self.twiddle5.im);
            let m0308b = vfmsq_n_f64(m0308b, x3m8, self.twiddle2.im);
            let m0308b = vfmaq_n_f64(m0308b, x4m7, self.twiddle1.im);
            let m0308b = vfmaq_n_f64(m0308b, x5m6, self.twiddle4.im);
            let (y03, y08) = NeonButterfly::butterfly2_f64(m0308a, m0308b);

            let m0407a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle4.re);
            let m0407a = vfmaq_n_f64(m0407a, x2p9, self.twiddle3.re);
            let m0407a = vfmaq_n_f64(m0407a, x3p8, self.twiddle1.re);
            let m0407a = vfmaq_n_f64(m0407a, x4p7, self.twiddle5.re);
            let m0407a = vfmaq_n_f64(m0407a, x5p6, self.twiddle2.re);
            let m0407b = vmulq_n_f64(x1m10, self.twiddle4.im);
            let m0407b = vfmsq_n_f64(m0407b, x2m9, self.twiddle3.im);
            let m0407b = vfmaq_n_f64(m0407b, x3m8, self.twiddle1.im);
            let m0407b = vfmaq_n_f64(m0407b, x4m7, self.twiddle5.im);
            let m0407b = vfmsq_n_f64(m0407b, x5m6, self.twiddle2.im);
            let (y04, y07) = NeonButterfly::butterfly2_f64(m0407a, m0407b);

            let m0506a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle5.re);
            let m0506a = vfmaq_n_f64(m0506a, x2p9, self.twiddle1.re);
            let m0506a = vfmaq_n_f64(m0506a, x3p8, self.twiddle4.re);
            let m0506a = vfmaq_n_f64(m0506a, x4p7, self.twiddle2.re);
            let m0506a = vfmaq_n_f64(m0506a, x5p6, self.twiddle3.re);
            let m0506b = vmulq_n_f64(x1m10, self.twiddle5.im);
            let m0506b = vfmsq_n_f64(m0506b, x2m9, self.twiddle1.im);
            let m0506b = vfmaq_n_f64(m0506b, x3m8, self.twiddle4.im);
            let m0506b = vfmsq_n_f64(m0506b, x4m7, self.twiddle2.im);
            let m0506b = vfmaq_n_f64(m0506b, x5m6, self.twiddle3.im);
            let (y05, y06) = NeonButterfly::butterfly2_f64(m0506a, m0506b);

            [
                NeonStoreD::raw(y00),
                NeonStoreD::raw(y01),
                NeonStoreD::raw(y02),
                NeonStoreD::raw(y03),
                NeonStoreD::raw(y04),
                NeonStoreD::raw(y05),
                NeonStoreD::raw(y06),
                NeonStoreD::raw(y07),
                NeonStoreD::raw(y08),
                NeonStoreD::raw(y09),
                NeonStoreD::raw(y10),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly11d {
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
    twiddle4: Complex<f64>,
    twiddle5: Complex<f64>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly11d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 11]) -> [NeonStoreD; 11] {
        let y00 = store[0].v;
        let (x1p10, x1m10) = NeonButterfly::butterfly2_f64(store[1].v, store[10].v);
        let x1m10 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m10);
        let y00 = vaddq_f64(y00, x1p10);
        let (x2p9, x2m9) = NeonButterfly::butterfly2_f64(store[2].v, store[9].v);
        let x2m9 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m9);
        let y00 = vaddq_f64(y00, x2p9);
        let (x3p8, x3m8) = NeonButterfly::butterfly2_f64(store[3].v, store[8].v);
        let x3m8 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m8);
        let y00 = vaddq_f64(y00, x3p8);
        let (x4p7, x4m7) = NeonButterfly::butterfly2_f64(store[4].v, store[7].v);
        let x4m7 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m7);
        let y00 = vaddq_f64(y00, x4p7);
        let (x5p6, x5m6) = NeonButterfly::butterfly2_f64(store[5].v, store[6].v);
        let x5m6 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m6);
        let y00 = vaddq_f64(y00, x5p6);

        let m0110a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle1.re);
        let m0110a = vfmaq_n_f64(m0110a, x2p9, self.twiddle2.re);
        let m0110a = vfmaq_n_f64(m0110a, x3p8, self.twiddle3.re);
        let m0110a = vfmaq_n_f64(m0110a, x4p7, self.twiddle4.re);
        let m0110a = vfmaq_n_f64(m0110a, x5p6, self.twiddle5.re);
        let m0110b = vmulq_n_f64(x1m10, self.twiddle1.im);
        let m0110b = vfmaq_n_f64(m0110b, x2m9, self.twiddle2.im);
        let m0110b = vfmaq_n_f64(m0110b, x3m8, self.twiddle3.im);
        let m0110b = vfmaq_n_f64(m0110b, x4m7, self.twiddle4.im);
        let m0110b = vfmaq_n_f64(m0110b, x5m6, self.twiddle5.im);
        let (y01, y10) = NeonButterfly::butterfly2_f64(m0110a, m0110b);

        let m0209a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle2.re);
        let m0209a = vfmaq_n_f64(m0209a, x2p9, self.twiddle4.re);
        let m0209a = vfmaq_n_f64(m0209a, x3p8, self.twiddle5.re);
        let m0209a = vfmaq_n_f64(m0209a, x4p7, self.twiddle3.re);
        let m0209a = vfmaq_n_f64(m0209a, x5p6, self.twiddle1.re);
        let m0209b = vmulq_n_f64(x1m10, self.twiddle2.im);
        let m0209b = vfmaq_n_f64(m0209b, x2m9, self.twiddle4.im);
        let m0209b = vfmsq_n_f64(m0209b, x3m8, self.twiddle5.im);
        let m0209b = vfmsq_n_f64(m0209b, x4m7, self.twiddle3.im);
        let m0209b = vfmsq_n_f64(m0209b, x5m6, self.twiddle1.im);
        let (y02, y09) = NeonButterfly::butterfly2_f64(m0209a, m0209b);

        let m0308a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle3.re);
        let m0308a = vfmaq_n_f64(m0308a, x2p9, self.twiddle5.re);
        let m0308a = vfmaq_n_f64(m0308a, x3p8, self.twiddle2.re);
        let m0308a = vfmaq_n_f64(m0308a, x4p7, self.twiddle1.re);
        let m0308a = vfmaq_n_f64(m0308a, x5p6, self.twiddle4.re);
        let m0308b = vmulq_n_f64(x1m10, self.twiddle3.im);
        let m0308b = vfmsq_n_f64(m0308b, x2m9, self.twiddle5.im);
        let m0308b = vfmsq_n_f64(m0308b, x3m8, self.twiddle2.im);
        let m0308b = vfmaq_n_f64(m0308b, x4m7, self.twiddle1.im);
        let m0308b = vfmaq_n_f64(m0308b, x5m6, self.twiddle4.im);
        let (y03, y08) = NeonButterfly::butterfly2_f64(m0308a, m0308b);

        let m0407a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle4.re);
        let m0407a = vfmaq_n_f64(m0407a, x2p9, self.twiddle3.re);
        let m0407a = vfmaq_n_f64(m0407a, x3p8, self.twiddle1.re);
        let m0407a = vfmaq_n_f64(m0407a, x4p7, self.twiddle5.re);
        let m0407a = vfmaq_n_f64(m0407a, x5p6, self.twiddle2.re);
        let m0407b = vmulq_n_f64(x1m10, self.twiddle4.im);
        let m0407b = vfmsq_n_f64(m0407b, x2m9, self.twiddle3.im);
        let m0407b = vfmaq_n_f64(m0407b, x3m8, self.twiddle1.im);
        let m0407b = vfmaq_n_f64(m0407b, x4m7, self.twiddle5.im);
        let m0407b = vfmsq_n_f64(m0407b, x5m6, self.twiddle2.im);
        let (y04, y07) = NeonButterfly::butterfly2_f64(m0407a, m0407b);

        let m0506a = vfmaq_n_f64(store[0].v, x1p10, self.twiddle5.re);
        let m0506a = vfmaq_n_f64(m0506a, x2p9, self.twiddle1.re);
        let m0506a = vfmaq_n_f64(m0506a, x3p8, self.twiddle4.re);
        let m0506a = vfmaq_n_f64(m0506a, x4p7, self.twiddle2.re);
        let m0506a = vfmaq_n_f64(m0506a, x5p6, self.twiddle3.re);
        let m0506b = vmulq_n_f64(x1m10, self.twiddle5.im);
        let m0506b = vfmsq_n_f64(m0506b, x2m9, self.twiddle1.im);
        let m0506b = vfmaq_n_f64(m0506b, x3m8, self.twiddle4.im);
        let m0506b = vfmsq_n_f64(m0506b, x4m7, self.twiddle2.im);
        let m0506b = vfmaq_n_f64(m0506b, x5m6, self.twiddle3.im);
        let (y05, y06) = NeonButterfly::butterfly2_f64(m0506a, m0506b);

        [
            NeonStoreD::raw(y00),
            NeonStoreD::raw(y01),
            NeonStoreD::raw(y02),
            NeonStoreD::raw(y03),
            NeonStoreD::raw(y04),
            NeonStoreD::raw(y05),
            NeonStoreD::raw(y06),
            NeonStoreD::raw(y07),
            NeonStoreD::raw(y08),
            NeonStoreD::raw(y09),
            NeonStoreD::raw(y10),
        ]
    }
}

pub(crate) struct ColumnButterfly11f {
    rotate: float32x4_t,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
    twiddle4: Complex<f32>,
    twiddle5: Complex<f32>,
}

impl ColumnButterfly11f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            Self {
                rotate: vld1q_f32([-0.0f32, 0.0, -0.0f32, 0.0].as_ptr().cast()),
                twiddle1: compute_twiddle(1, 11, fft_direction),
                twiddle2: compute_twiddle(2, 11, fft_direction),
                twiddle3: compute_twiddle(3, 11, fft_direction),
                twiddle4: compute_twiddle(4, 11, fft_direction),
                twiddle5: compute_twiddle(5, 11, fft_direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 11]) -> [NeonStoreF; 11] {
        unsafe {
            let y00 = store[0].v;
            let (x1p10, x1m10) = NeonButterfly::butterfly2_f32(store[1].v, store[10].v);
            let x1m10 = v_rotate90_f32(x1m10, self.rotate);
            let y00 = vaddq_f32(y00, x1p10);
            let (x2p9, x2m9) = NeonButterfly::butterfly2_f32(store[2].v, store[9].v);
            let x2m9 = v_rotate90_f32(x2m9, self.rotate);
            let y00 = vaddq_f32(y00, x2p9);
            let (x3p8, x3m8) = NeonButterfly::butterfly2_f32(store[3].v, store[8].v);
            let x3m8 = v_rotate90_f32(x3m8, self.rotate);
            let y00 = vaddq_f32(y00, x3p8);
            let (x4p7, x4m7) = NeonButterfly::butterfly2_f32(store[4].v, store[7].v);
            let x4m7 = v_rotate90_f32(x4m7, self.rotate);
            let y00 = vaddq_f32(y00, x4p7);
            let (x5p6, x5m6) = NeonButterfly::butterfly2_f32(store[5].v, store[6].v);
            let x5m6 = v_rotate90_f32(x5m6, self.rotate);
            let y00 = vaddq_f32(y00, x5p6);

            let m0110a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle1.re);
            let m0110a = vfmaq_n_f32(m0110a, x2p9, self.twiddle2.re);
            let m0110a = vfmaq_n_f32(m0110a, x3p8, self.twiddle3.re);
            let m0110a = vfmaq_n_f32(m0110a, x4p7, self.twiddle4.re);
            let m0110a = vfmaq_n_f32(m0110a, x5p6, self.twiddle5.re);
            let m0110b = vmulq_n_f32(x1m10, self.twiddle1.im);
            let m0110b = vfmaq_n_f32(m0110b, x2m9, self.twiddle2.im);
            let m0110b = vfmaq_n_f32(m0110b, x3m8, self.twiddle3.im);
            let m0110b = vfmaq_n_f32(m0110b, x4m7, self.twiddle4.im);
            let m0110b = vfmaq_n_f32(m0110b, x5m6, self.twiddle5.im);
            let (y01, y10) = NeonButterfly::butterfly2_f32(m0110a, m0110b);

            let m0209a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle2.re);
            let m0209a = vfmaq_n_f32(m0209a, x2p9, self.twiddle4.re);
            let m0209a = vfmaq_n_f32(m0209a, x3p8, self.twiddle5.re);
            let m0209a = vfmaq_n_f32(m0209a, x4p7, self.twiddle3.re);
            let m0209a = vfmaq_n_f32(m0209a, x5p6, self.twiddle1.re);
            let m0209b = vmulq_n_f32(x1m10, self.twiddle2.im);
            let m0209b = vfmaq_n_f32(m0209b, x2m9, self.twiddle4.im);
            let m0209b = vfmsq_n_f32(m0209b, x3m8, self.twiddle5.im);
            let m0209b = vfmsq_n_f32(m0209b, x4m7, self.twiddle3.im);
            let m0209b = vfmsq_n_f32(m0209b, x5m6, self.twiddle1.im);
            let (y02, y09) = NeonButterfly::butterfly2_f32(m0209a, m0209b);

            let m0308a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle3.re);
            let m0308a = vfmaq_n_f32(m0308a, x2p9, self.twiddle5.re);
            let m0308a = vfmaq_n_f32(m0308a, x3p8, self.twiddle2.re);
            let m0308a = vfmaq_n_f32(m0308a, x4p7, self.twiddle1.re);
            let m0308a = vfmaq_n_f32(m0308a, x5p6, self.twiddle4.re);
            let m0308b = vmulq_n_f32(x1m10, self.twiddle3.im);
            let m0308b = vfmsq_n_f32(m0308b, x2m9, self.twiddle5.im);
            let m0308b = vfmsq_n_f32(m0308b, x3m8, self.twiddle2.im);
            let m0308b = vfmaq_n_f32(m0308b, x4m7, self.twiddle1.im);
            let m0308b = vfmaq_n_f32(m0308b, x5m6, self.twiddle4.im);
            let (y03, y08) = NeonButterfly::butterfly2_f32(m0308a, m0308b);

            let m0407a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle4.re);
            let m0407a = vfmaq_n_f32(m0407a, x2p9, self.twiddle3.re);
            let m0407a = vfmaq_n_f32(m0407a, x3p8, self.twiddle1.re);
            let m0407a = vfmaq_n_f32(m0407a, x4p7, self.twiddle5.re);
            let m0407a = vfmaq_n_f32(m0407a, x5p6, self.twiddle2.re);
            let m0407b = vmulq_n_f32(x1m10, self.twiddle4.im);
            let m0407b = vfmsq_n_f32(m0407b, x2m9, self.twiddle3.im);
            let m0407b = vfmaq_n_f32(m0407b, x3m8, self.twiddle1.im);
            let m0407b = vfmaq_n_f32(m0407b, x4m7, self.twiddle5.im);
            let m0407b = vfmsq_n_f32(m0407b, x5m6, self.twiddle2.im);
            let (y04, y07) = NeonButterfly::butterfly2_f32(m0407a, m0407b);

            let m0506a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle5.re);
            let m0506a = vfmaq_n_f32(m0506a, x2p9, self.twiddle1.re);
            let m0506a = vfmaq_n_f32(m0506a, x3p8, self.twiddle4.re);
            let m0506a = vfmaq_n_f32(m0506a, x4p7, self.twiddle2.re);
            let m0506a = vfmaq_n_f32(m0506a, x5p6, self.twiddle3.re);
            let m0506b = vmulq_n_f32(x1m10, self.twiddle5.im);
            let m0506b = vfmsq_n_f32(m0506b, x2m9, self.twiddle1.im);
            let m0506b = vfmaq_n_f32(m0506b, x3m8, self.twiddle4.im);
            let m0506b = vfmsq_n_f32(m0506b, x4m7, self.twiddle2.im);
            let m0506b = vfmaq_n_f32(m0506b, x5m6, self.twiddle3.im);
            let (y05, y06) = NeonButterfly::butterfly2_f32(m0506a, m0506b);

            [
                NeonStoreF::raw(y00),
                NeonStoreF::raw(y01),
                NeonStoreF::raw(y02),
                NeonStoreF::raw(y03),
                NeonStoreF::raw(y04),
                NeonStoreF::raw(y05),
                NeonStoreF::raw(y06),
                NeonStoreF::raw(y07),
                NeonStoreF::raw(y08),
                NeonStoreF::raw(y09),
                NeonStoreF::raw(y10),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 11]) -> [NeonStoreFh; 11] {
        unsafe {
            let y00 = store[0].v;
            let (x1p10, x1m10) = NeonButterfly::butterfly2h_f32(store[1].v, store[10].v);
            let x1m10 = vh_rotate90_f32(x1m10, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x1p10);
            let (x2p9, x2m9) = NeonButterfly::butterfly2h_f32(store[2].v, store[9].v);
            let x2m9 = vh_rotate90_f32(x2m9, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x2p9);
            let (x3p8, x3m8) = NeonButterfly::butterfly2h_f32(store[3].v, store[8].v);
            let x3m8 = vh_rotate90_f32(x3m8, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x3p8);
            let (x4p7, x4m7) = NeonButterfly::butterfly2h_f32(store[4].v, store[7].v);
            let x4m7 = vh_rotate90_f32(x4m7, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x4p7);
            let (x5p6, x5m6) = NeonButterfly::butterfly2h_f32(store[5].v, store[6].v);
            let x5m6 = vh_rotate90_f32(x5m6, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x5p6);

            let m0110a = vfma_n_f32(store[0].v, x1p10, self.twiddle1.re);
            let m0110a = vfma_n_f32(m0110a, x2p9, self.twiddle2.re);
            let m0110a = vfma_n_f32(m0110a, x3p8, self.twiddle3.re);
            let m0110a = vfma_n_f32(m0110a, x4p7, self.twiddle4.re);
            let m0110a = vfma_n_f32(m0110a, x5p6, self.twiddle5.re);
            let m0110b = vmul_n_f32(x1m10, self.twiddle1.im);
            let m0110b = vfma_n_f32(m0110b, x2m9, self.twiddle2.im);
            let m0110b = vfma_n_f32(m0110b, x3m8, self.twiddle3.im);
            let m0110b = vfma_n_f32(m0110b, x4m7, self.twiddle4.im);
            let m0110b = vfma_n_f32(m0110b, x5m6, self.twiddle5.im);
            let (y01, y10) = NeonButterfly::butterfly2h_f32(m0110a, m0110b);

            let m0209a = vfma_n_f32(store[0].v, x1p10, self.twiddle2.re);
            let m0209a = vfma_n_f32(m0209a, x2p9, self.twiddle4.re);
            let m0209a = vfma_n_f32(m0209a, x3p8, self.twiddle5.re);
            let m0209a = vfma_n_f32(m0209a, x4p7, self.twiddle3.re);
            let m0209a = vfma_n_f32(m0209a, x5p6, self.twiddle1.re);
            let m0209b = vmul_n_f32(x1m10, self.twiddle2.im);
            let m0209b = vfma_n_f32(m0209b, x2m9, self.twiddle4.im);
            let m0209b = vfms_n_f32(m0209b, x3m8, self.twiddle5.im);
            let m0209b = vfms_n_f32(m0209b, x4m7, self.twiddle3.im);
            let m0209b = vfms_n_f32(m0209b, x5m6, self.twiddle1.im);
            let (y02, y09) = NeonButterfly::butterfly2h_f32(m0209a, m0209b);

            let m0308a = vfma_n_f32(store[0].v, x1p10, self.twiddle3.re);
            let m0308a = vfma_n_f32(m0308a, x2p9, self.twiddle5.re);
            let m0308a = vfma_n_f32(m0308a, x3p8, self.twiddle2.re);
            let m0308a = vfma_n_f32(m0308a, x4p7, self.twiddle1.re);
            let m0308a = vfma_n_f32(m0308a, x5p6, self.twiddle4.re);
            let m0308b = vmul_n_f32(x1m10, self.twiddle3.im);
            let m0308b = vfms_n_f32(m0308b, x2m9, self.twiddle5.im);
            let m0308b = vfms_n_f32(m0308b, x3m8, self.twiddle2.im);
            let m0308b = vfma_n_f32(m0308b, x4m7, self.twiddle1.im);
            let m0308b = vfma_n_f32(m0308b, x5m6, self.twiddle4.im);
            let (y03, y08) = NeonButterfly::butterfly2h_f32(m0308a, m0308b);

            let m0407a = vfma_n_f32(store[0].v, x1p10, self.twiddle4.re);
            let m0407a = vfma_n_f32(m0407a, x2p9, self.twiddle3.re);
            let m0407a = vfma_n_f32(m0407a, x3p8, self.twiddle1.re);
            let m0407a = vfma_n_f32(m0407a, x4p7, self.twiddle5.re);
            let m0407a = vfma_n_f32(m0407a, x5p6, self.twiddle2.re);
            let m0407b = vmul_n_f32(x1m10, self.twiddle4.im);
            let m0407b = vfms_n_f32(m0407b, x2m9, self.twiddle3.im);
            let m0407b = vfma_n_f32(m0407b, x3m8, self.twiddle1.im);
            let m0407b = vfma_n_f32(m0407b, x4m7, self.twiddle5.im);
            let m0407b = vfms_n_f32(m0407b, x5m6, self.twiddle2.im);
            let (y04, y07) = NeonButterfly::butterfly2h_f32(m0407a, m0407b);

            let m0506a = vfma_n_f32(store[0].v, x1p10, self.twiddle5.re);
            let m0506a = vfma_n_f32(m0506a, x2p9, self.twiddle1.re);
            let m0506a = vfma_n_f32(m0506a, x3p8, self.twiddle4.re);
            let m0506a = vfma_n_f32(m0506a, x4p7, self.twiddle2.re);
            let m0506a = vfma_n_f32(m0506a, x5p6, self.twiddle3.re);
            let m0506b = vmul_n_f32(x1m10, self.twiddle5.im);
            let m0506b = vfms_n_f32(m0506b, x2m9, self.twiddle1.im);
            let m0506b = vfma_n_f32(m0506b, x3m8, self.twiddle4.im);
            let m0506b = vfms_n_f32(m0506b, x4m7, self.twiddle2.im);
            let m0506b = vfma_n_f32(m0506b, x5m6, self.twiddle3.im);
            let (y05, y06) = NeonButterfly::butterfly2h_f32(m0506a, m0506b);

            [
                NeonStoreFh::raw(y00),
                NeonStoreFh::raw(y01),
                NeonStoreFh::raw(y02),
                NeonStoreFh::raw(y03),
                NeonStoreFh::raw(y04),
                NeonStoreFh::raw(y05),
                NeonStoreFh::raw(y06),
                NeonStoreFh::raw(y07),
                NeonStoreFh::raw(y08),
                NeonStoreFh::raw(y09),
                NeonStoreFh::raw(y10),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly11f {
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
    twiddle4: Complex<f32>,
    twiddle5: Complex<f32>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly11f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 11]) -> [NeonStoreF; 11] {
        let y00 = store[0].v;
        let (x1p10, x1m10) = NeonButterfly::butterfly2_f32(store[1].v, store[10].v);
        let x1m10 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m10);
        let y00 = vaddq_f32(y00, x1p10);
        let (x2p9, x2m9) = NeonButterfly::butterfly2_f32(store[2].v, store[9].v);
        let x2m9 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m9);
        let y00 = vaddq_f32(y00, x2p9);
        let (x3p8, x3m8) = NeonButterfly::butterfly2_f32(store[3].v, store[8].v);
        let x3m8 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m8);
        let y00 = vaddq_f32(y00, x3p8);
        let (x4p7, x4m7) = NeonButterfly::butterfly2_f32(store[4].v, store[7].v);
        let x4m7 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m7);
        let y00 = vaddq_f32(y00, x4p7);
        let (x5p6, x5m6) = NeonButterfly::butterfly2_f32(store[5].v, store[6].v);
        let x5m6 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m6);
        let y00 = vaddq_f32(y00, x5p6);

        let m0110a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle1.re);
        let m0110a = vfmaq_n_f32(m0110a, x2p9, self.twiddle2.re);
        let m0110a = vfmaq_n_f32(m0110a, x3p8, self.twiddle3.re);
        let m0110a = vfmaq_n_f32(m0110a, x4p7, self.twiddle4.re);
        let m0110a = vfmaq_n_f32(m0110a, x5p6, self.twiddle5.re);
        let m0110b = vmulq_n_f32(x1m10, self.twiddle1.im);
        let m0110b = vfmaq_n_f32(m0110b, x2m9, self.twiddle2.im);
        let m0110b = vfmaq_n_f32(m0110b, x3m8, self.twiddle3.im);
        let m0110b = vfmaq_n_f32(m0110b, x4m7, self.twiddle4.im);
        let m0110b = vfmaq_n_f32(m0110b, x5m6, self.twiddle5.im);
        let (y01, y10) = NeonButterfly::butterfly2_f32(m0110a, m0110b);

        let m0209a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle2.re);
        let m0209a = vfmaq_n_f32(m0209a, x2p9, self.twiddle4.re);
        let m0209a = vfmaq_n_f32(m0209a, x3p8, self.twiddle5.re);
        let m0209a = vfmaq_n_f32(m0209a, x4p7, self.twiddle3.re);
        let m0209a = vfmaq_n_f32(m0209a, x5p6, self.twiddle1.re);
        let m0209b = vmulq_n_f32(x1m10, self.twiddle2.im);
        let m0209b = vfmaq_n_f32(m0209b, x2m9, self.twiddle4.im);
        let m0209b = vfmsq_n_f32(m0209b, x3m8, self.twiddle5.im);
        let m0209b = vfmsq_n_f32(m0209b, x4m7, self.twiddle3.im);
        let m0209b = vfmsq_n_f32(m0209b, x5m6, self.twiddle1.im);
        let (y02, y09) = NeonButterfly::butterfly2_f32(m0209a, m0209b);

        let m0308a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle3.re);
        let m0308a = vfmaq_n_f32(m0308a, x2p9, self.twiddle5.re);
        let m0308a = vfmaq_n_f32(m0308a, x3p8, self.twiddle2.re);
        let m0308a = vfmaq_n_f32(m0308a, x4p7, self.twiddle1.re);
        let m0308a = vfmaq_n_f32(m0308a, x5p6, self.twiddle4.re);
        let m0308b = vmulq_n_f32(x1m10, self.twiddle3.im);
        let m0308b = vfmsq_n_f32(m0308b, x2m9, self.twiddle5.im);
        let m0308b = vfmsq_n_f32(m0308b, x3m8, self.twiddle2.im);
        let m0308b = vfmaq_n_f32(m0308b, x4m7, self.twiddle1.im);
        let m0308b = vfmaq_n_f32(m0308b, x5m6, self.twiddle4.im);
        let (y03, y08) = NeonButterfly::butterfly2_f32(m0308a, m0308b);

        let m0407a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle4.re);
        let m0407a = vfmaq_n_f32(m0407a, x2p9, self.twiddle3.re);
        let m0407a = vfmaq_n_f32(m0407a, x3p8, self.twiddle1.re);
        let m0407a = vfmaq_n_f32(m0407a, x4p7, self.twiddle5.re);
        let m0407a = vfmaq_n_f32(m0407a, x5p6, self.twiddle2.re);
        let m0407b = vmulq_n_f32(x1m10, self.twiddle4.im);
        let m0407b = vfmsq_n_f32(m0407b, x2m9, self.twiddle3.im);
        let m0407b = vfmaq_n_f32(m0407b, x3m8, self.twiddle1.im);
        let m0407b = vfmaq_n_f32(m0407b, x4m7, self.twiddle5.im);
        let m0407b = vfmsq_n_f32(m0407b, x5m6, self.twiddle2.im);
        let (y04, y07) = NeonButterfly::butterfly2_f32(m0407a, m0407b);

        let m0506a = vfmaq_n_f32(store[0].v, x1p10, self.twiddle5.re);
        let m0506a = vfmaq_n_f32(m0506a, x2p9, self.twiddle1.re);
        let m0506a = vfmaq_n_f32(m0506a, x3p8, self.twiddle4.re);
        let m0506a = vfmaq_n_f32(m0506a, x4p7, self.twiddle2.re);
        let m0506a = vfmaq_n_f32(m0506a, x5p6, self.twiddle3.re);
        let m0506b = vmulq_n_f32(x1m10, self.twiddle5.im);
        let m0506b = vfmsq_n_f32(m0506b, x2m9, self.twiddle1.im);
        let m0506b = vfmaq_n_f32(m0506b, x3m8, self.twiddle4.im);
        let m0506b = vfmsq_n_f32(m0506b, x4m7, self.twiddle2.im);
        let m0506b = vfmaq_n_f32(m0506b, x5m6, self.twiddle3.im);
        let (y05, y06) = NeonButterfly::butterfly2_f32(m0506a, m0506b);

        [
            NeonStoreF::raw(y00),
            NeonStoreF::raw(y01),
            NeonStoreF::raw(y02),
            NeonStoreF::raw(y03),
            NeonStoreF::raw(y04),
            NeonStoreF::raw(y05),
            NeonStoreF::raw(y06),
            NeonStoreF::raw(y07),
            NeonStoreF::raw(y08),
            NeonStoreF::raw(y09),
            NeonStoreF::raw(y10),
        ]
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 11]) -> [NeonStoreFh; 11] {
        let y00 = store[0].v;
        let (x1p10, x1m10) = NeonButterfly::butterfly2h_f32(store[1].v, store[10].v);
        let x1m10 = vcadd_rot90_f32(vdup_n_f32(0.), x1m10);
        let y00 = vadd_f32(y00, x1p10);
        let (x2p9, x2m9) = NeonButterfly::butterfly2h_f32(store[2].v, store[9].v);
        let x2m9 = vcadd_rot90_f32(vdup_n_f32(0.), x2m9);
        let y00 = vadd_f32(y00, x2p9);
        let (x3p8, x3m8) = NeonButterfly::butterfly2h_f32(store[3].v, store[8].v);
        let x3m8 = vcadd_rot90_f32(vdup_n_f32(0.), x3m8);
        let y00 = vadd_f32(y00, x3p8);
        let (x4p7, x4m7) = NeonButterfly::butterfly2h_f32(store[4].v, store[7].v);
        let x4m7 = vcadd_rot90_f32(vdup_n_f32(0.), x4m7);
        let y00 = vadd_f32(y00, x4p7);
        let (x5p6, x5m6) = NeonButterfly::butterfly2h_f32(store[5].v, store[6].v);
        let x5m6 = vcadd_rot90_f32(vdup_n_f32(0.), x5m6);
        let y00 = vadd_f32(y00, x5p6);

        let m0110a = vfma_n_f32(store[0].v, x1p10, self.twiddle1.re);
        let m0110a = vfma_n_f32(m0110a, x2p9, self.twiddle2.re);
        let m0110a = vfma_n_f32(m0110a, x3p8, self.twiddle3.re);
        let m0110a = vfma_n_f32(m0110a, x4p7, self.twiddle4.re);
        let m0110a = vfma_n_f32(m0110a, x5p6, self.twiddle5.re);
        let m0110b = vmul_n_f32(x1m10, self.twiddle1.im);
        let m0110b = vfma_n_f32(m0110b, x2m9, self.twiddle2.im);
        let m0110b = vfma_n_f32(m0110b, x3m8, self.twiddle3.im);
        let m0110b = vfma_n_f32(m0110b, x4m7, self.twiddle4.im);
        let m0110b = vfma_n_f32(m0110b, x5m6, self.twiddle5.im);
        let (y01, y10) = NeonButterfly::butterfly2h_f32(m0110a, m0110b);

        let m0209a = vfma_n_f32(store[0].v, x1p10, self.twiddle2.re);
        let m0209a = vfma_n_f32(m0209a, x2p9, self.twiddle4.re);
        let m0209a = vfma_n_f32(m0209a, x3p8, self.twiddle5.re);
        let m0209a = vfma_n_f32(m0209a, x4p7, self.twiddle3.re);
        let m0209a = vfma_n_f32(m0209a, x5p6, self.twiddle1.re);
        let m0209b = vmul_n_f32(x1m10, self.twiddle2.im);
        let m0209b = vfma_n_f32(m0209b, x2m9, self.twiddle4.im);
        let m0209b = vfms_n_f32(m0209b, x3m8, self.twiddle5.im);
        let m0209b = vfms_n_f32(m0209b, x4m7, self.twiddle3.im);
        let m0209b = vfms_n_f32(m0209b, x5m6, self.twiddle1.im);
        let (y02, y09) = NeonButterfly::butterfly2h_f32(m0209a, m0209b);

        let m0308a = vfma_n_f32(store[0].v, x1p10, self.twiddle3.re);
        let m0308a = vfma_n_f32(m0308a, x2p9, self.twiddle5.re);
        let m0308a = vfma_n_f32(m0308a, x3p8, self.twiddle2.re);
        let m0308a = vfma_n_f32(m0308a, x4p7, self.twiddle1.re);
        let m0308a = vfma_n_f32(m0308a, x5p6, self.twiddle4.re);
        let m0308b = vmul_n_f32(x1m10, self.twiddle3.im);
        let m0308b = vfms_n_f32(m0308b, x2m9, self.twiddle5.im);
        let m0308b = vfms_n_f32(m0308b, x3m8, self.twiddle2.im);
        let m0308b = vfma_n_f32(m0308b, x4m7, self.twiddle1.im);
        let m0308b = vfma_n_f32(m0308b, x5m6, self.twiddle4.im);
        let (y03, y08) = NeonButterfly::butterfly2h_f32(m0308a, m0308b);

        let m0407a = vfma_n_f32(store[0].v, x1p10, self.twiddle4.re);
        let m0407a = vfma_n_f32(m0407a, x2p9, self.twiddle3.re);
        let m0407a = vfma_n_f32(m0407a, x3p8, self.twiddle1.re);
        let m0407a = vfma_n_f32(m0407a, x4p7, self.twiddle5.re);
        let m0407a = vfma_n_f32(m0407a, x5p6, self.twiddle2.re);
        let m0407b = vmul_n_f32(x1m10, self.twiddle4.im);
        let m0407b = vfms_n_f32(m0407b, x2m9, self.twiddle3.im);
        let m0407b = vfma_n_f32(m0407b, x3m8, self.twiddle1.im);
        let m0407b = vfma_n_f32(m0407b, x4m7, self.twiddle5.im);
        let m0407b = vfms_n_f32(m0407b, x5m6, self.twiddle2.im);
        let (y04, y07) = NeonButterfly::butterfly2h_f32(m0407a, m0407b);

        let m0506a = vfma_n_f32(store[0].v, x1p10, self.twiddle5.re);
        let m0506a = vfma_n_f32(m0506a, x2p9, self.twiddle1.re);
        let m0506a = vfma_n_f32(m0506a, x3p8, self.twiddle4.re);
        let m0506a = vfma_n_f32(m0506a, x4p7, self.twiddle2.re);
        let m0506a = vfma_n_f32(m0506a, x5p6, self.twiddle3.re);
        let m0506b = vmul_n_f32(x1m10, self.twiddle5.im);
        let m0506b = vfms_n_f32(m0506b, x2m9, self.twiddle1.im);
        let m0506b = vfma_n_f32(m0506b, x3m8, self.twiddle4.im);
        let m0506b = vfms_n_f32(m0506b, x4m7, self.twiddle2.im);
        let m0506b = vfma_n_f32(m0506b, x5m6, self.twiddle3.im);
        let (y05, y06) = NeonButterfly::butterfly2h_f32(m0506a, m0506b);

        [
            NeonStoreFh::raw(y00),
            NeonStoreFh::raw(y01),
            NeonStoreFh::raw(y02),
            NeonStoreFh::raw(y03),
            NeonStoreFh::raw(y04),
            NeonStoreFh::raw(y05),
            NeonStoreFh::raw(y06),
            NeonStoreFh::raw(y07),
            NeonStoreFh::raw(y08),
            NeonStoreFh::raw(y09),
            NeonStoreFh::raw(y10),
        ]
    }
}
