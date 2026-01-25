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

pub(crate) struct ColumnButterfly13d {
    rotate: float64x2_t,
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
    twiddle4: Complex<f64>,
    twiddle5: Complex<f64>,
    twiddle6: Complex<f64>,
}

impl ColumnButterfly13d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            static ROT_90: [f64; 2] = [-0.0, 0.0];
            Self {
                rotate: vld1q_f64(ROT_90.as_ptr()),
                twiddle1: compute_twiddle(1, 13, fft_direction),
                twiddle2: compute_twiddle(2, 13, fft_direction),
                twiddle3: compute_twiddle(3, 13, fft_direction),
                twiddle4: compute_twiddle(4, 13, fft_direction),
                twiddle5: compute_twiddle(5, 13, fft_direction),
                twiddle6: compute_twiddle(6, 13, fft_direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 13]) -> [NeonStoreD; 13] {
        unsafe {
            let y00 = store[0].v;
            let (x1p12, x1m12) = NeonButterfly::butterfly2_f64(store[1].v, store[12].v);
            let x1m12 = v_rotate90_f64(x1m12, self.rotate);
            let y00 = vaddq_f64(y00, x1p12);
            let (x2p11, x2m11) = NeonButterfly::butterfly2_f64(store[2].v, store[11].v);
            let x2m11 = v_rotate90_f64(x2m11, self.rotate);
            let y00 = vaddq_f64(y00, x2p11);
            let (x3p10, x3m10) = NeonButterfly::butterfly2_f64(store[3].v, store[10].v);
            let x3m10 = v_rotate90_f64(x3m10, self.rotate);
            let y00 = vaddq_f64(y00, x3p10);
            let (x4p9, x4m9) = NeonButterfly::butterfly2_f64(store[4].v, store[9].v);
            let x4m9 = v_rotate90_f64(x4m9, self.rotate);
            let y00 = vaddq_f64(y00, x4p9);
            let (x5p8, x5m8) = NeonButterfly::butterfly2_f64(store[5].v, store[8].v);
            let x5m8 = v_rotate90_f64(x5m8, self.rotate);
            let y00 = vaddq_f64(y00, x5p8);
            let (x6p7, x6m7) = NeonButterfly::butterfly2_f64(store[6].v, store[7].v);
            let x6m7 = v_rotate90_f64(x6m7, self.rotate);
            let y00 = vaddq_f64(y00, x6p7);

            let m0112a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle1.re);
            let m0112a = vfmaq_n_f64(m0112a, x2p11, self.twiddle2.re);
            let m0112a = vfmaq_n_f64(m0112a, x3p10, self.twiddle3.re);
            let m0112a = vfmaq_n_f64(m0112a, x4p9, self.twiddle4.re);
            let m0112a = vfmaq_n_f64(m0112a, x5p8, self.twiddle5.re);
            let m0112a = vfmaq_n_f64(m0112a, x6p7, self.twiddle6.re);
            let m0112b = vmulq_n_f64(x1m12, self.twiddle1.im);
            let m0112b = vfmaq_n_f64(m0112b, x2m11, self.twiddle2.im);
            let m0112b = vfmaq_n_f64(m0112b, x3m10, self.twiddle3.im);
            let m0112b = vfmaq_n_f64(m0112b, x4m9, self.twiddle4.im);
            let m0112b = vfmaq_n_f64(m0112b, x5m8, self.twiddle5.im);
            let m0112b = vfmaq_n_f64(m0112b, x6m7, self.twiddle6.im);
            let (y01, y12) = NeonButterfly::butterfly2_f64(m0112a, m0112b);

            let m0211a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle2.re);
            let m0211a = vfmaq_n_f64(m0211a, x2p11, self.twiddle4.re);
            let m0211a = vfmaq_n_f64(m0211a, x3p10, self.twiddle6.re);
            let m0211a = vfmaq_n_f64(m0211a, x4p9, self.twiddle5.re);
            let m0211a = vfmaq_n_f64(m0211a, x5p8, self.twiddle3.re);
            let m0211a = vfmaq_n_f64(m0211a, x6p7, self.twiddle1.re);
            let m0211b = vmulq_n_f64(x1m12, self.twiddle2.im);
            let m0211b = vfmaq_n_f64(m0211b, x2m11, self.twiddle4.im);
            let m0211b = vfmaq_n_f64(m0211b, x3m10, self.twiddle6.im);
            let m0211b = vfmsq_n_f64(m0211b, x4m9, self.twiddle5.im);
            let m0211b = vfmsq_n_f64(m0211b, x5m8, self.twiddle3.im);
            let m0211b = vfmsq_n_f64(m0211b, x6m7, self.twiddle1.im);
            let (y02, y11) = NeonButterfly::butterfly2_f64(m0211a, m0211b);

            let m0310a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle3.re);
            let m0310a = vfmaq_n_f64(m0310a, x2p11, self.twiddle6.re);
            let m0310a = vfmaq_n_f64(m0310a, x3p10, self.twiddle4.re);
            let m0310a = vfmaq_n_f64(m0310a, x4p9, self.twiddle1.re);
            let m0310a = vfmaq_n_f64(m0310a, x5p8, self.twiddle2.re);
            let m0310a = vfmaq_n_f64(m0310a, x6p7, self.twiddle5.re);
            let m0310b = vmulq_n_f64(x1m12, self.twiddle3.im);
            let m0310b = vfmaq_n_f64(m0310b, x2m11, self.twiddle6.im);
            let m0310b = vfmsq_n_f64(m0310b, x3m10, self.twiddle4.im);
            let m0310b = vfmsq_n_f64(m0310b, x4m9, self.twiddle1.im);
            let m0310b = vfmaq_n_f64(m0310b, x5m8, self.twiddle2.im);
            let m0310b = vfmaq_n_f64(m0310b, x6m7, self.twiddle5.im);
            let (y03, y10) = NeonButterfly::butterfly2_f64(m0310a, m0310b);

            let m0409a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle4.re);
            let m0409a = vfmaq_n_f64(m0409a, x2p11, self.twiddle5.re);
            let m0409a = vfmaq_n_f64(m0409a, x3p10, self.twiddle1.re);
            let m0409a = vfmaq_n_f64(m0409a, x4p9, self.twiddle3.re);
            let m0409a = vfmaq_n_f64(m0409a, x5p8, self.twiddle6.re);
            let m0409a = vfmaq_n_f64(m0409a, x6p7, self.twiddle2.re);
            let m0409b = vmulq_n_f64(x1m12, self.twiddle4.im);
            let m0409b = vfmsq_n_f64(m0409b, x2m11, self.twiddle5.im);
            let m0409b = vfmsq_n_f64(m0409b, x3m10, self.twiddle1.im);
            let m0409b = vfmaq_n_f64(m0409b, x4m9, self.twiddle3.im);
            let m0409b = vfmsq_n_f64(m0409b, x5m8, self.twiddle6.im);
            let m0409b = vfmsq_n_f64(m0409b, x6m7, self.twiddle2.im);
            let (y04, y09) = NeonButterfly::butterfly2_f64(m0409a, m0409b);

            let m0508a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle5.re);
            let m0508a = vfmaq_n_f64(m0508a, x2p11, self.twiddle3.re);
            let m0508a = vfmaq_n_f64(m0508a, x3p10, self.twiddle2.re);
            let m0508a = vfmaq_n_f64(m0508a, x4p9, self.twiddle6.re);
            let m0508a = vfmaq_n_f64(m0508a, x5p8, self.twiddle1.re);
            let m0508a = vfmaq_n_f64(m0508a, x6p7, self.twiddle4.re);
            let m0508b = vmulq_n_f64(x1m12, self.twiddle5.im);
            let m0508b = vfmsq_n_f64(m0508b, x2m11, self.twiddle3.im);
            let m0508b = vfmaq_n_f64(m0508b, x3m10, self.twiddle2.im);
            let m0508b = vfmsq_n_f64(m0508b, x4m9, self.twiddle6.im);
            let m0508b = vfmsq_n_f64(m0508b, x5m8, self.twiddle1.im);
            let m0508b = vfmaq_n_f64(m0508b, x6m7, self.twiddle4.im);
            let (y05, y08) = NeonButterfly::butterfly2_f64(m0508a, m0508b);

            let m0607a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle6.re);
            let m0607a = vfmaq_n_f64(m0607a, x2p11, self.twiddle1.re);
            let m0607a = vfmaq_n_f64(m0607a, x3p10, self.twiddle5.re);
            let m0607a = vfmaq_n_f64(m0607a, x4p9, self.twiddle2.re);
            let m0607a = vfmaq_n_f64(m0607a, x5p8, self.twiddle4.re);
            let m0607a = vfmaq_n_f64(m0607a, x6p7, self.twiddle3.re);
            let m0607b = vmulq_n_f64(x1m12, self.twiddle6.im);
            let m0607b = vfmsq_n_f64(m0607b, x2m11, self.twiddle1.im);
            let m0607b = vfmaq_n_f64(m0607b, x3m10, self.twiddle5.im);
            let m0607b = vfmsq_n_f64(m0607b, x4m9, self.twiddle2.im);
            let m0607b = vfmaq_n_f64(m0607b, x5m8, self.twiddle4.im);
            let m0607b = vfmsq_n_f64(m0607b, x6m7, self.twiddle3.im);
            let (y06, y07) = NeonButterfly::butterfly2_f64(m0607a, m0607b);

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
                NeonStoreD::raw(y11),
                NeonStoreD::raw(y12),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly13d {
    twiddle1: Complex<f64>,
    twiddle2: Complex<f64>,
    twiddle3: Complex<f64>,
    twiddle4: Complex<f64>,
    twiddle5: Complex<f64>,
    twiddle6: Complex<f64>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly13d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) unsafe fn exec(&self, store: [NeonStoreD; 13]) -> [NeonStoreD; 13] {
        let y00 = store[0].v;
        let (x1p12, x1m12) = NeonButterfly::butterfly2_f64(store[1].v, store[12].v);
        let x1m12 = vcaddq_rot90_f64(vdupq_n_f64(0.), x1m12);
        let y00 = vaddq_f64(y00, x1p12);
        let (x2p11, x2m11) = NeonButterfly::butterfly2_f64(store[2].v, store[11].v);
        let x2m11 = vcaddq_rot90_f64(vdupq_n_f64(0.), x2m11);
        let y00 = vaddq_f64(y00, x2p11);
        let (x3p10, x3m10) = NeonButterfly::butterfly2_f64(store[3].v, store[10].v);
        let x3m10 = vcaddq_rot90_f64(vdupq_n_f64(0.), x3m10);
        let y00 = vaddq_f64(y00, x3p10);
        let (x4p9, x4m9) = NeonButterfly::butterfly2_f64(store[4].v, store[9].v);
        let x4m9 = vcaddq_rot90_f64(vdupq_n_f64(0.), x4m9);
        let y00 = vaddq_f64(y00, x4p9);
        let (x5p8, x5m8) = NeonButterfly::butterfly2_f64(store[5].v, store[8].v);
        let x5m8 = vcaddq_rot90_f64(vdupq_n_f64(0.), x5m8);
        let y00 = vaddq_f64(y00, x5p8);
        let (x6p7, x6m7) = NeonButterfly::butterfly2_f64(store[6].v, store[7].v);
        let x6m7 = vcaddq_rot90_f64(vdupq_n_f64(0.), x6m7);
        let y00 = vaddq_f64(y00, x6p7);

        let m0112a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle1.re);
        let m0112a = vfmaq_n_f64(m0112a, x2p11, self.twiddle2.re);
        let m0112a = vfmaq_n_f64(m0112a, x3p10, self.twiddle3.re);
        let m0112a = vfmaq_n_f64(m0112a, x4p9, self.twiddle4.re);
        let m0112a = vfmaq_n_f64(m0112a, x5p8, self.twiddle5.re);
        let m0112a = vfmaq_n_f64(m0112a, x6p7, self.twiddle6.re);
        let m0112b = vmulq_n_f64(x1m12, self.twiddle1.im);
        let m0112b = vfmaq_n_f64(m0112b, x2m11, self.twiddle2.im);
        let m0112b = vfmaq_n_f64(m0112b, x3m10, self.twiddle3.im);
        let m0112b = vfmaq_n_f64(m0112b, x4m9, self.twiddle4.im);
        let m0112b = vfmaq_n_f64(m0112b, x5m8, self.twiddle5.im);
        let m0112b = vfmaq_n_f64(m0112b, x6m7, self.twiddle6.im);
        let (y01, y12) = NeonButterfly::butterfly2_f64(m0112a, m0112b);

        let m0211a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle2.re);
        let m0211a = vfmaq_n_f64(m0211a, x2p11, self.twiddle4.re);
        let m0211a = vfmaq_n_f64(m0211a, x3p10, self.twiddle6.re);
        let m0211a = vfmaq_n_f64(m0211a, x4p9, self.twiddle5.re);
        let m0211a = vfmaq_n_f64(m0211a, x5p8, self.twiddle3.re);
        let m0211a = vfmaq_n_f64(m0211a, x6p7, self.twiddle1.re);
        let m0211b = vmulq_n_f64(x1m12, self.twiddle2.im);
        let m0211b = vfmaq_n_f64(m0211b, x2m11, self.twiddle4.im);
        let m0211b = vfmaq_n_f64(m0211b, x3m10, self.twiddle6.im);
        let m0211b = vfmsq_n_f64(m0211b, x4m9, self.twiddle5.im);
        let m0211b = vfmsq_n_f64(m0211b, x5m8, self.twiddle3.im);
        let m0211b = vfmsq_n_f64(m0211b, x6m7, self.twiddle1.im);
        let (y02, y11) = NeonButterfly::butterfly2_f64(m0211a, m0211b);

        let m0310a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle3.re);
        let m0310a = vfmaq_n_f64(m0310a, x2p11, self.twiddle6.re);
        let m0310a = vfmaq_n_f64(m0310a, x3p10, self.twiddle4.re);
        let m0310a = vfmaq_n_f64(m0310a, x4p9, self.twiddle1.re);
        let m0310a = vfmaq_n_f64(m0310a, x5p8, self.twiddle2.re);
        let m0310a = vfmaq_n_f64(m0310a, x6p7, self.twiddle5.re);
        let m0310b = vmulq_n_f64(x1m12, self.twiddle3.im);
        let m0310b = vfmaq_n_f64(m0310b, x2m11, self.twiddle6.im);
        let m0310b = vfmsq_n_f64(m0310b, x3m10, self.twiddle4.im);
        let m0310b = vfmsq_n_f64(m0310b, x4m9, self.twiddle1.im);
        let m0310b = vfmaq_n_f64(m0310b, x5m8, self.twiddle2.im);
        let m0310b = vfmaq_n_f64(m0310b, x6m7, self.twiddle5.im);
        let (y03, y10) = NeonButterfly::butterfly2_f64(m0310a, m0310b);

        let m0409a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle4.re);
        let m0409a = vfmaq_n_f64(m0409a, x2p11, self.twiddle5.re);
        let m0409a = vfmaq_n_f64(m0409a, x3p10, self.twiddle1.re);
        let m0409a = vfmaq_n_f64(m0409a, x4p9, self.twiddle3.re);
        let m0409a = vfmaq_n_f64(m0409a, x5p8, self.twiddle6.re);
        let m0409a = vfmaq_n_f64(m0409a, x6p7, self.twiddle2.re);
        let m0409b = vmulq_n_f64(x1m12, self.twiddle4.im);
        let m0409b = vfmsq_n_f64(m0409b, x2m11, self.twiddle5.im);
        let m0409b = vfmsq_n_f64(m0409b, x3m10, self.twiddle1.im);
        let m0409b = vfmaq_n_f64(m0409b, x4m9, self.twiddle3.im);
        let m0409b = vfmsq_n_f64(m0409b, x5m8, self.twiddle6.im);
        let m0409b = vfmsq_n_f64(m0409b, x6m7, self.twiddle2.im);
        let (y04, y09) = NeonButterfly::butterfly2_f64(m0409a, m0409b);

        let m0508a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle5.re);
        let m0508a = vfmaq_n_f64(m0508a, x2p11, self.twiddle3.re);
        let m0508a = vfmaq_n_f64(m0508a, x3p10, self.twiddle2.re);
        let m0508a = vfmaq_n_f64(m0508a, x4p9, self.twiddle6.re);
        let m0508a = vfmaq_n_f64(m0508a, x5p8, self.twiddle1.re);
        let m0508a = vfmaq_n_f64(m0508a, x6p7, self.twiddle4.re);
        let m0508b = vmulq_n_f64(x1m12, self.twiddle5.im);
        let m0508b = vfmsq_n_f64(m0508b, x2m11, self.twiddle3.im);
        let m0508b = vfmaq_n_f64(m0508b, x3m10, self.twiddle2.im);
        let m0508b = vfmsq_n_f64(m0508b, x4m9, self.twiddle6.im);
        let m0508b = vfmsq_n_f64(m0508b, x5m8, self.twiddle1.im);
        let m0508b = vfmaq_n_f64(m0508b, x6m7, self.twiddle4.im);
        let (y05, y08) = NeonButterfly::butterfly2_f64(m0508a, m0508b);

        let m0607a = vfmaq_n_f64(store[0].v, x1p12, self.twiddle6.re);
        let m0607a = vfmaq_n_f64(m0607a, x2p11, self.twiddle1.re);
        let m0607a = vfmaq_n_f64(m0607a, x3p10, self.twiddle5.re);
        let m0607a = vfmaq_n_f64(m0607a, x4p9, self.twiddle2.re);
        let m0607a = vfmaq_n_f64(m0607a, x5p8, self.twiddle4.re);
        let m0607a = vfmaq_n_f64(m0607a, x6p7, self.twiddle3.re);
        let m0607b = vmulq_n_f64(x1m12, self.twiddle6.im);
        let m0607b = vfmsq_n_f64(m0607b, x2m11, self.twiddle1.im);
        let m0607b = vfmaq_n_f64(m0607b, x3m10, self.twiddle5.im);
        let m0607b = vfmsq_n_f64(m0607b, x4m9, self.twiddle2.im);
        let m0607b = vfmaq_n_f64(m0607b, x5m8, self.twiddle4.im);
        let m0607b = vfmsq_n_f64(m0607b, x6m7, self.twiddle3.im);
        let (y06, y07) = NeonButterfly::butterfly2_f64(m0607a, m0607b);

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
            NeonStoreD::raw(y11),
            NeonStoreD::raw(y12),
        ]
    }
}

pub(crate) struct ColumnButterfly13f {
    rotate: float32x4_t,
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
    twiddle4: Complex<f32>,
    twiddle5: Complex<f32>,
    twiddle6: Complex<f32>,
}

impl ColumnButterfly13f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            static ROT_90: [f32; 4] = [-0.0, 0.0, -0.0, 0.0];
            Self {
                rotate: vld1q_f32(ROT_90.as_ptr()),
                twiddle1: compute_twiddle(1, 13, fft_direction),
                twiddle2: compute_twiddle(2, 13, fft_direction),
                twiddle3: compute_twiddle(3, 13, fft_direction),
                twiddle4: compute_twiddle(4, 13, fft_direction),
                twiddle5: compute_twiddle(5, 13, fft_direction),
                twiddle6: compute_twiddle(6, 13, fft_direction),
            }
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 13]) -> [NeonStoreF; 13] {
        unsafe {
            let y00 = store[0].v;
            let (x1p12, x1m12) = NeonButterfly::butterfly2_f32(store[1].v, store[12].v);
            let x1m12 = v_rotate90_f32(x1m12, self.rotate);
            let y00 = vaddq_f32(y00, x1p12);
            let (x2p11, x2m11) = NeonButterfly::butterfly2_f32(store[2].v, store[11].v);
            let x2m11 = v_rotate90_f32(x2m11, self.rotate);
            let y00 = vaddq_f32(y00, x2p11);
            let (x3p10, x3m10) = NeonButterfly::butterfly2_f32(store[3].v, store[10].v);
            let x3m10 = v_rotate90_f32(x3m10, self.rotate);
            let y00 = vaddq_f32(y00, x3p10);
            let (x4p9, x4m9) = NeonButterfly::butterfly2_f32(store[4].v, store[9].v);
            let x4m9 = v_rotate90_f32(x4m9, self.rotate);
            let y00 = vaddq_f32(y00, x4p9);
            let (x5p8, x5m8) = NeonButterfly::butterfly2_f32(store[5].v, store[8].v);
            let x5m8 = v_rotate90_f32(x5m8, self.rotate);
            let y00 = vaddq_f32(y00, x5p8);
            let (x6p7, x6m7) = NeonButterfly::butterfly2_f32(store[6].v, store[7].v);
            let x6m7 = v_rotate90_f32(x6m7, self.rotate);
            let y00 = vaddq_f32(y00, x6p7);

            let m0112a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle1.re);
            let m0112a = vfmaq_n_f32(m0112a, x2p11, self.twiddle2.re);
            let m0112a = vfmaq_n_f32(m0112a, x3p10, self.twiddle3.re);
            let m0112a = vfmaq_n_f32(m0112a, x4p9, self.twiddle4.re);
            let m0112a = vfmaq_n_f32(m0112a, x5p8, self.twiddle5.re);
            let m0112a = vfmaq_n_f32(m0112a, x6p7, self.twiddle6.re);
            let m0112b = vmulq_n_f32(x1m12, self.twiddle1.im);
            let m0112b = vfmaq_n_f32(m0112b, x2m11, self.twiddle2.im);
            let m0112b = vfmaq_n_f32(m0112b, x3m10, self.twiddle3.im);
            let m0112b = vfmaq_n_f32(m0112b, x4m9, self.twiddle4.im);
            let m0112b = vfmaq_n_f32(m0112b, x5m8, self.twiddle5.im);
            let m0112b = vfmaq_n_f32(m0112b, x6m7, self.twiddle6.im);
            let (y01, y12) = NeonButterfly::butterfly2_f32(m0112a, m0112b);

            let m0211a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle2.re);
            let m0211a = vfmaq_n_f32(m0211a, x2p11, self.twiddle4.re);
            let m0211a = vfmaq_n_f32(m0211a, x3p10, self.twiddle6.re);
            let m0211a = vfmaq_n_f32(m0211a, x4p9, self.twiddle5.re);
            let m0211a = vfmaq_n_f32(m0211a, x5p8, self.twiddle3.re);
            let m0211a = vfmaq_n_f32(m0211a, x6p7, self.twiddle1.re);
            let m0211b = vmulq_n_f32(x1m12, self.twiddle2.im);
            let m0211b = vfmaq_n_f32(m0211b, x2m11, self.twiddle4.im);
            let m0211b = vfmaq_n_f32(m0211b, x3m10, self.twiddle6.im);
            let m0211b = vfmsq_n_f32(m0211b, x4m9, self.twiddle5.im);
            let m0211b = vfmsq_n_f32(m0211b, x5m8, self.twiddle3.im);
            let m0211b = vfmsq_n_f32(m0211b, x6m7, self.twiddle1.im);
            let (y02, y11) = NeonButterfly::butterfly2_f32(m0211a, m0211b);

            let m0310a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle3.re);
            let m0310a = vfmaq_n_f32(m0310a, x2p11, self.twiddle6.re);
            let m0310a = vfmaq_n_f32(m0310a, x3p10, self.twiddle4.re);
            let m0310a = vfmaq_n_f32(m0310a, x4p9, self.twiddle1.re);
            let m0310a = vfmaq_n_f32(m0310a, x5p8, self.twiddle2.re);
            let m0310a = vfmaq_n_f32(m0310a, x6p7, self.twiddle5.re);
            let m0310b = vmulq_n_f32(x1m12, self.twiddle3.im);
            let m0310b = vfmaq_n_f32(m0310b, x2m11, self.twiddle6.im);
            let m0310b = vfmsq_n_f32(m0310b, x3m10, self.twiddle4.im);
            let m0310b = vfmsq_n_f32(m0310b, x4m9, self.twiddle1.im);
            let m0310b = vfmaq_n_f32(m0310b, x5m8, self.twiddle2.im);
            let m0310b = vfmaq_n_f32(m0310b, x6m7, self.twiddle5.im);
            let (y03, y10) = NeonButterfly::butterfly2_f32(m0310a, m0310b);

            let m0409a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle4.re);
            let m0409a = vfmaq_n_f32(m0409a, x2p11, self.twiddle5.re);
            let m0409a = vfmaq_n_f32(m0409a, x3p10, self.twiddle1.re);
            let m0409a = vfmaq_n_f32(m0409a, x4p9, self.twiddle3.re);
            let m0409a = vfmaq_n_f32(m0409a, x5p8, self.twiddle6.re);
            let m0409a = vfmaq_n_f32(m0409a, x6p7, self.twiddle2.re);
            let m0409b = vmulq_n_f32(x1m12, self.twiddle4.im);
            let m0409b = vfmsq_n_f32(m0409b, x2m11, self.twiddle5.im);
            let m0409b = vfmsq_n_f32(m0409b, x3m10, self.twiddle1.im);
            let m0409b = vfmaq_n_f32(m0409b, x4m9, self.twiddle3.im);
            let m0409b = vfmsq_n_f32(m0409b, x5m8, self.twiddle6.im);
            let m0409b = vfmsq_n_f32(m0409b, x6m7, self.twiddle2.im);
            let (y04, y09) = NeonButterfly::butterfly2_f32(m0409a, m0409b);

            let m0508a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle5.re);
            let m0508a = vfmaq_n_f32(m0508a, x2p11, self.twiddle3.re);
            let m0508a = vfmaq_n_f32(m0508a, x3p10, self.twiddle2.re);
            let m0508a = vfmaq_n_f32(m0508a, x4p9, self.twiddle6.re);
            let m0508a = vfmaq_n_f32(m0508a, x5p8, self.twiddle1.re);
            let m0508a = vfmaq_n_f32(m0508a, x6p7, self.twiddle4.re);
            let m0508b = vmulq_n_f32(x1m12, self.twiddle5.im);
            let m0508b = vfmsq_n_f32(m0508b, x2m11, self.twiddle3.im);
            let m0508b = vfmaq_n_f32(m0508b, x3m10, self.twiddle2.im);
            let m0508b = vfmsq_n_f32(m0508b, x4m9, self.twiddle6.im);
            let m0508b = vfmsq_n_f32(m0508b, x5m8, self.twiddle1.im);
            let m0508b = vfmaq_n_f32(m0508b, x6m7, self.twiddle4.im);
            let (y05, y08) = NeonButterfly::butterfly2_f32(m0508a, m0508b);

            let m0607a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle6.re);
            let m0607a = vfmaq_n_f32(m0607a, x2p11, self.twiddle1.re);
            let m0607a = vfmaq_n_f32(m0607a, x3p10, self.twiddle5.re);
            let m0607a = vfmaq_n_f32(m0607a, x4p9, self.twiddle2.re);
            let m0607a = vfmaq_n_f32(m0607a, x5p8, self.twiddle4.re);
            let m0607a = vfmaq_n_f32(m0607a, x6p7, self.twiddle3.re);
            let m0607b = vmulq_n_f32(x1m12, self.twiddle6.im);
            let m0607b = vfmsq_n_f32(m0607b, x2m11, self.twiddle1.im);
            let m0607b = vfmaq_n_f32(m0607b, x3m10, self.twiddle5.im);
            let m0607b = vfmsq_n_f32(m0607b, x4m9, self.twiddle2.im);
            let m0607b = vfmaq_n_f32(m0607b, x5m8, self.twiddle4.im);
            let m0607b = vfmsq_n_f32(m0607b, x6m7, self.twiddle3.im);
            let (y06, y07) = NeonButterfly::butterfly2_f32(m0607a, m0607b);

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
                NeonStoreF::raw(y11),
                NeonStoreF::raw(y12),
            ]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 13]) -> [NeonStoreFh; 13] {
        unsafe {
            let y00 = store[0].v;
            let (x1p12, x1m12) = NeonButterfly::butterfly2h_f32(store[1].v, store[12].v);
            let x1m12 = vh_rotate90_f32(x1m12, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x1p12);
            let (x2p11, x2m11) = NeonButterfly::butterfly2h_f32(store[2].v, store[11].v);
            let x2m11 = vh_rotate90_f32(x2m11, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x2p11);
            let (x3p10, x3m10) = NeonButterfly::butterfly2h_f32(store[3].v, store[10].v);
            let x3m10 = vh_rotate90_f32(x3m10, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x3p10);
            let (x4p9, x4m9) = NeonButterfly::butterfly2h_f32(store[4].v, store[9].v);
            let x4m9 = vh_rotate90_f32(x4m9, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x4p9);
            let (x5p8, x5m8) = NeonButterfly::butterfly2h_f32(store[5].v, store[8].v);
            let x5m8 = vh_rotate90_f32(x5m8, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x5p8);
            let (x6p7, x6m7) = NeonButterfly::butterfly2h_f32(store[6].v, store[7].v);
            let x6m7 = vh_rotate90_f32(x6m7, vget_low_f32(self.rotate));
            let y00 = vadd_f32(y00, x6p7);

            let m0112a = vfma_n_f32(store[0].v, x1p12, self.twiddle1.re);
            let m0112a = vfma_n_f32(m0112a, x2p11, self.twiddle2.re);
            let m0112a = vfma_n_f32(m0112a, x3p10, self.twiddle3.re);
            let m0112a = vfma_n_f32(m0112a, x4p9, self.twiddle4.re);
            let m0112a = vfma_n_f32(m0112a, x5p8, self.twiddle5.re);
            let m0112a = vfma_n_f32(m0112a, x6p7, self.twiddle6.re);
            let m0112b = vmul_n_f32(x1m12, self.twiddle1.im);
            let m0112b = vfma_n_f32(m0112b, x2m11, self.twiddle2.im);
            let m0112b = vfma_n_f32(m0112b, x3m10, self.twiddle3.im);
            let m0112b = vfma_n_f32(m0112b, x4m9, self.twiddle4.im);
            let m0112b = vfma_n_f32(m0112b, x5m8, self.twiddle5.im);
            let m0112b = vfma_n_f32(m0112b, x6m7, self.twiddle6.im);
            let (y01, y12) = NeonButterfly::butterfly2h_f32(m0112a, m0112b);

            let m0211a = vfma_n_f32(store[0].v, x1p12, self.twiddle2.re);
            let m0211a = vfma_n_f32(m0211a, x2p11, self.twiddle4.re);
            let m0211a = vfma_n_f32(m0211a, x3p10, self.twiddle6.re);
            let m0211a = vfma_n_f32(m0211a, x4p9, self.twiddle5.re);
            let m0211a = vfma_n_f32(m0211a, x5p8, self.twiddle3.re);
            let m0211a = vfma_n_f32(m0211a, x6p7, self.twiddle1.re);
            let m0211b = vmul_n_f32(x1m12, self.twiddle2.im);
            let m0211b = vfma_n_f32(m0211b, x2m11, self.twiddle4.im);
            let m0211b = vfma_n_f32(m0211b, x3m10, self.twiddle6.im);
            let m0211b = vfms_n_f32(m0211b, x4m9, self.twiddle5.im);
            let m0211b = vfms_n_f32(m0211b, x5m8, self.twiddle3.im);
            let m0211b = vfms_n_f32(m0211b, x6m7, self.twiddle1.im);
            let (y02, y11) = NeonButterfly::butterfly2h_f32(m0211a, m0211b);

            let m0310a = vfma_n_f32(store[0].v, x1p12, self.twiddle3.re);
            let m0310a = vfma_n_f32(m0310a, x2p11, self.twiddle6.re);
            let m0310a = vfma_n_f32(m0310a, x3p10, self.twiddle4.re);
            let m0310a = vfma_n_f32(m0310a, x4p9, self.twiddle1.re);
            let m0310a = vfma_n_f32(m0310a, x5p8, self.twiddle2.re);
            let m0310a = vfma_n_f32(m0310a, x6p7, self.twiddle5.re);
            let m0310b = vmul_n_f32(x1m12, self.twiddle3.im);
            let m0310b = vfma_n_f32(m0310b, x2m11, self.twiddle6.im);
            let m0310b = vfms_n_f32(m0310b, x3m10, self.twiddle4.im);
            let m0310b = vfms_n_f32(m0310b, x4m9, self.twiddle1.im);
            let m0310b = vfma_n_f32(m0310b, x5m8, self.twiddle2.im);
            let m0310b = vfma_n_f32(m0310b, x6m7, self.twiddle5.im);
            let (y03, y10) = NeonButterfly::butterfly2h_f32(m0310a, m0310b);

            let m0409a = vfma_n_f32(store[0].v, x1p12, self.twiddle4.re);
            let m0409a = vfma_n_f32(m0409a, x2p11, self.twiddle5.re);
            let m0409a = vfma_n_f32(m0409a, x3p10, self.twiddle1.re);
            let m0409a = vfma_n_f32(m0409a, x4p9, self.twiddle3.re);
            let m0409a = vfma_n_f32(m0409a, x5p8, self.twiddle6.re);
            let m0409a = vfma_n_f32(m0409a, x6p7, self.twiddle2.re);
            let m0409b = vmul_n_f32(x1m12, self.twiddle4.im);
            let m0409b = vfms_n_f32(m0409b, x2m11, self.twiddle5.im);
            let m0409b = vfms_n_f32(m0409b, x3m10, self.twiddle1.im);
            let m0409b = vfma_n_f32(m0409b, x4m9, self.twiddle3.im);
            let m0409b = vfms_n_f32(m0409b, x5m8, self.twiddle6.im);
            let m0409b = vfms_n_f32(m0409b, x6m7, self.twiddle2.im);
            let (y04, y09) = NeonButterfly::butterfly2h_f32(m0409a, m0409b);

            let m0508a = vfma_n_f32(store[0].v, x1p12, self.twiddle5.re);
            let m0508a = vfma_n_f32(m0508a, x2p11, self.twiddle3.re);
            let m0508a = vfma_n_f32(m0508a, x3p10, self.twiddle2.re);
            let m0508a = vfma_n_f32(m0508a, x4p9, self.twiddle6.re);
            let m0508a = vfma_n_f32(m0508a, x5p8, self.twiddle1.re);
            let m0508a = vfma_n_f32(m0508a, x6p7, self.twiddle4.re);
            let m0508b = vmul_n_f32(x1m12, self.twiddle5.im);
            let m0508b = vfms_n_f32(m0508b, x2m11, self.twiddle3.im);
            let m0508b = vfma_n_f32(m0508b, x3m10, self.twiddle2.im);
            let m0508b = vfms_n_f32(m0508b, x4m9, self.twiddle6.im);
            let m0508b = vfms_n_f32(m0508b, x5m8, self.twiddle1.im);
            let m0508b = vfma_n_f32(m0508b, x6m7, self.twiddle4.im);
            let (y05, y08) = NeonButterfly::butterfly2h_f32(m0508a, m0508b);

            let m0607a = vfma_n_f32(store[0].v, x1p12, self.twiddle6.re);
            let m0607a = vfma_n_f32(m0607a, x2p11, self.twiddle1.re);
            let m0607a = vfma_n_f32(m0607a, x3p10, self.twiddle5.re);
            let m0607a = vfma_n_f32(m0607a, x4p9, self.twiddle2.re);
            let m0607a = vfma_n_f32(m0607a, x5p8, self.twiddle4.re);
            let m0607a = vfma_n_f32(m0607a, x6p7, self.twiddle3.re);
            let m0607b = vmul_n_f32(x1m12, self.twiddle6.im);
            let m0607b = vfms_n_f32(m0607b, x2m11, self.twiddle1.im);
            let m0607b = vfma_n_f32(m0607b, x3m10, self.twiddle5.im);
            let m0607b = vfms_n_f32(m0607b, x4m9, self.twiddle2.im);
            let m0607b = vfma_n_f32(m0607b, x5m8, self.twiddle4.im);
            let m0607b = vfms_n_f32(m0607b, x6m7, self.twiddle3.im);
            let (y06, y07) = NeonButterfly::butterfly2h_f32(m0607a, m0607b);

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
                NeonStoreFh::raw(y11),
                NeonStoreFh::raw(y12),
            ]
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly13f {
    twiddle1: Complex<f32>,
    twiddle2: Complex<f32>,
    twiddle3: Complex<f32>,
    twiddle4: Complex<f32>,
    twiddle5: Complex<f32>,
    twiddle6: Complex<f32>,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly13f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) unsafe fn exec(&self, store: [NeonStoreF; 13]) -> [NeonStoreF; 13] {
        let y00 = store[0].v;
        let (x1p12, x1m12) = NeonButterfly::butterfly2_f32(store[1].v, store[12].v);
        let x1m12 = vcaddq_rot90_f32(vdupq_n_f32(0.), x1m12);
        let y00 = vaddq_f32(y00, x1p12);
        let (x2p11, x2m11) = NeonButterfly::butterfly2_f32(store[2].v, store[11].v);
        let x2m11 = vcaddq_rot90_f32(vdupq_n_f32(0.), x2m11);
        let y00 = vaddq_f32(y00, x2p11);
        let (x3p10, x3m10) = NeonButterfly::butterfly2_f32(store[3].v, store[10].v);
        let x3m10 = vcaddq_rot90_f32(vdupq_n_f32(0.), x3m10);
        let y00 = vaddq_f32(y00, x3p10);
        let (x4p9, x4m9) = NeonButterfly::butterfly2_f32(store[4].v, store[9].v);
        let x4m9 = vcaddq_rot90_f32(vdupq_n_f32(0.), x4m9);
        let y00 = vaddq_f32(y00, x4p9);
        let (x5p8, x5m8) = NeonButterfly::butterfly2_f32(store[5].v, store[8].v);
        let x5m8 = vcaddq_rot90_f32(vdupq_n_f32(0.), x5m8);
        let y00 = vaddq_f32(y00, x5p8);
        let (x6p7, x6m7) = NeonButterfly::butterfly2_f32(store[6].v, store[7].v);
        let x6m7 = vcaddq_rot90_f32(vdupq_n_f32(0.), x6m7);
        let y00 = vaddq_f32(y00, x6p7);

        let m0112a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle1.re);
        let m0112a = vfmaq_n_f32(m0112a, x2p11, self.twiddle2.re);
        let m0112a = vfmaq_n_f32(m0112a, x3p10, self.twiddle3.re);
        let m0112a = vfmaq_n_f32(m0112a, x4p9, self.twiddle4.re);
        let m0112a = vfmaq_n_f32(m0112a, x5p8, self.twiddle5.re);
        let m0112a = vfmaq_n_f32(m0112a, x6p7, self.twiddle6.re);
        let m0112b = vmulq_n_f32(x1m12, self.twiddle1.im);
        let m0112b = vfmaq_n_f32(m0112b, x2m11, self.twiddle2.im);
        let m0112b = vfmaq_n_f32(m0112b, x3m10, self.twiddle3.im);
        let m0112b = vfmaq_n_f32(m0112b, x4m9, self.twiddle4.im);
        let m0112b = vfmaq_n_f32(m0112b, x5m8, self.twiddle5.im);
        let m0112b = vfmaq_n_f32(m0112b, x6m7, self.twiddle6.im);
        let (y01, y12) = NeonButterfly::butterfly2_f32(m0112a, m0112b);

        let m0211a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle2.re);
        let m0211a = vfmaq_n_f32(m0211a, x2p11, self.twiddle4.re);
        let m0211a = vfmaq_n_f32(m0211a, x3p10, self.twiddle6.re);
        let m0211a = vfmaq_n_f32(m0211a, x4p9, self.twiddle5.re);
        let m0211a = vfmaq_n_f32(m0211a, x5p8, self.twiddle3.re);
        let m0211a = vfmaq_n_f32(m0211a, x6p7, self.twiddle1.re);
        let m0211b = vmulq_n_f32(x1m12, self.twiddle2.im);
        let m0211b = vfmaq_n_f32(m0211b, x2m11, self.twiddle4.im);
        let m0211b = vfmaq_n_f32(m0211b, x3m10, self.twiddle6.im);
        let m0211b = vfmsq_n_f32(m0211b, x4m9, self.twiddle5.im);
        let m0211b = vfmsq_n_f32(m0211b, x5m8, self.twiddle3.im);
        let m0211b = vfmsq_n_f32(m0211b, x6m7, self.twiddle1.im);
        let (y02, y11) = NeonButterfly::butterfly2_f32(m0211a, m0211b);

        let m0310a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle3.re);
        let m0310a = vfmaq_n_f32(m0310a, x2p11, self.twiddle6.re);
        let m0310a = vfmaq_n_f32(m0310a, x3p10, self.twiddle4.re);
        let m0310a = vfmaq_n_f32(m0310a, x4p9, self.twiddle1.re);
        let m0310a = vfmaq_n_f32(m0310a, x5p8, self.twiddle2.re);
        let m0310a = vfmaq_n_f32(m0310a, x6p7, self.twiddle5.re);
        let m0310b = vmulq_n_f32(x1m12, self.twiddle3.im);
        let m0310b = vfmaq_n_f32(m0310b, x2m11, self.twiddle6.im);
        let m0310b = vfmsq_n_f32(m0310b, x3m10, self.twiddle4.im);
        let m0310b = vfmsq_n_f32(m0310b, x4m9, self.twiddle1.im);
        let m0310b = vfmaq_n_f32(m0310b, x5m8, self.twiddle2.im);
        let m0310b = vfmaq_n_f32(m0310b, x6m7, self.twiddle5.im);
        let (y03, y10) = NeonButterfly::butterfly2_f32(m0310a, m0310b);

        let m0409a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle4.re);
        let m0409a = vfmaq_n_f32(m0409a, x2p11, self.twiddle5.re);
        let m0409a = vfmaq_n_f32(m0409a, x3p10, self.twiddle1.re);
        let m0409a = vfmaq_n_f32(m0409a, x4p9, self.twiddle3.re);
        let m0409a = vfmaq_n_f32(m0409a, x5p8, self.twiddle6.re);
        let m0409a = vfmaq_n_f32(m0409a, x6p7, self.twiddle2.re);
        let m0409b = vmulq_n_f32(x1m12, self.twiddle4.im);
        let m0409b = vfmsq_n_f32(m0409b, x2m11, self.twiddle5.im);
        let m0409b = vfmsq_n_f32(m0409b, x3m10, self.twiddle1.im);
        let m0409b = vfmaq_n_f32(m0409b, x4m9, self.twiddle3.im);
        let m0409b = vfmsq_n_f32(m0409b, x5m8, self.twiddle6.im);
        let m0409b = vfmsq_n_f32(m0409b, x6m7, self.twiddle2.im);
        let (y04, y09) = NeonButterfly::butterfly2_f32(m0409a, m0409b);

        let m0508a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle5.re);
        let m0508a = vfmaq_n_f32(m0508a, x2p11, self.twiddle3.re);
        let m0508a = vfmaq_n_f32(m0508a, x3p10, self.twiddle2.re);
        let m0508a = vfmaq_n_f32(m0508a, x4p9, self.twiddle6.re);
        let m0508a = vfmaq_n_f32(m0508a, x5p8, self.twiddle1.re);
        let m0508a = vfmaq_n_f32(m0508a, x6p7, self.twiddle4.re);
        let m0508b = vmulq_n_f32(x1m12, self.twiddle5.im);
        let m0508b = vfmsq_n_f32(m0508b, x2m11, self.twiddle3.im);
        let m0508b = vfmaq_n_f32(m0508b, x3m10, self.twiddle2.im);
        let m0508b = vfmsq_n_f32(m0508b, x4m9, self.twiddle6.im);
        let m0508b = vfmsq_n_f32(m0508b, x5m8, self.twiddle1.im);
        let m0508b = vfmaq_n_f32(m0508b, x6m7, self.twiddle4.im);
        let (y05, y08) = NeonButterfly::butterfly2_f32(m0508a, m0508b);

        let m0607a = vfmaq_n_f32(store[0].v, x1p12, self.twiddle6.re);
        let m0607a = vfmaq_n_f32(m0607a, x2p11, self.twiddle1.re);
        let m0607a = vfmaq_n_f32(m0607a, x3p10, self.twiddle5.re);
        let m0607a = vfmaq_n_f32(m0607a, x4p9, self.twiddle2.re);
        let m0607a = vfmaq_n_f32(m0607a, x5p8, self.twiddle4.re);
        let m0607a = vfmaq_n_f32(m0607a, x6p7, self.twiddle3.re);
        let m0607b = vmulq_n_f32(x1m12, self.twiddle6.im);
        let m0607b = vfmsq_n_f32(m0607b, x2m11, self.twiddle1.im);
        let m0607b = vfmaq_n_f32(m0607b, x3m10, self.twiddle5.im);
        let m0607b = vfmsq_n_f32(m0607b, x4m9, self.twiddle2.im);
        let m0607b = vfmaq_n_f32(m0607b, x5m8, self.twiddle4.im);
        let m0607b = vfmsq_n_f32(m0607b, x6m7, self.twiddle3.im);
        let (y06, y07) = NeonButterfly::butterfly2_f32(m0607a, m0607b);

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
            NeonStoreF::raw(y11),
            NeonStoreF::raw(y12),
        ]
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) unsafe fn exech(&self, store: [NeonStoreFh; 13]) -> [NeonStoreFh; 13] {
        let y00 = store[0].v;
        let (x1p12, x1m12) = NeonButterfly::butterfly2h_f32(store[1].v, store[12].v);
        let x1m12 = vcadd_rot90_f32(vdup_n_f32(0.), x1m12);
        let y00 = vadd_f32(y00, x1p12);
        let (x2p11, x2m11) = NeonButterfly::butterfly2h_f32(store[2].v, store[11].v);
        let x2m11 = vcadd_rot90_f32(vdup_n_f32(0.), x2m11);
        let y00 = vadd_f32(y00, x2p11);
        let (x3p10, x3m10) = NeonButterfly::butterfly2h_f32(store[3].v, store[10].v);
        let x3m10 = vcadd_rot90_f32(vdup_n_f32(0.), x3m10);
        let y00 = vadd_f32(y00, x3p10);
        let (x4p9, x4m9) = NeonButterfly::butterfly2h_f32(store[4].v, store[9].v);
        let x4m9 = vcadd_rot90_f32(vdup_n_f32(0.), x4m9);
        let y00 = vadd_f32(y00, x4p9);
        let (x5p8, x5m8) = NeonButterfly::butterfly2h_f32(store[5].v, store[8].v);
        let x5m8 = vcadd_rot90_f32(vdup_n_f32(0.), x5m8);
        let y00 = vadd_f32(y00, x5p8);
        let (x6p7, x6m7) = NeonButterfly::butterfly2h_f32(store[6].v, store[7].v);
        let x6m7 = vcadd_rot90_f32(vdup_n_f32(0.), x6m7);
        let y00 = vadd_f32(y00, x6p7);

        let m0112a = vfma_n_f32(store[0].v, x1p12, self.twiddle1.re);
        let m0112a = vfma_n_f32(m0112a, x2p11, self.twiddle2.re);
        let m0112a = vfma_n_f32(m0112a, x3p10, self.twiddle3.re);
        let m0112a = vfma_n_f32(m0112a, x4p9, self.twiddle4.re);
        let m0112a = vfma_n_f32(m0112a, x5p8, self.twiddle5.re);
        let m0112a = vfma_n_f32(m0112a, x6p7, self.twiddle6.re);
        let m0112b = vmul_n_f32(x1m12, self.twiddle1.im);
        let m0112b = vfma_n_f32(m0112b, x2m11, self.twiddle2.im);
        let m0112b = vfma_n_f32(m0112b, x3m10, self.twiddle3.im);
        let m0112b = vfma_n_f32(m0112b, x4m9, self.twiddle4.im);
        let m0112b = vfma_n_f32(m0112b, x5m8, self.twiddle5.im);
        let m0112b = vfma_n_f32(m0112b, x6m7, self.twiddle6.im);
        let (y01, y12) = NeonButterfly::butterfly2h_f32(m0112a, m0112b);

        let m0211a = vfma_n_f32(store[0].v, x1p12, self.twiddle2.re);
        let m0211a = vfma_n_f32(m0211a, x2p11, self.twiddle4.re);
        let m0211a = vfma_n_f32(m0211a, x3p10, self.twiddle6.re);
        let m0211a = vfma_n_f32(m0211a, x4p9, self.twiddle5.re);
        let m0211a = vfma_n_f32(m0211a, x5p8, self.twiddle3.re);
        let m0211a = vfma_n_f32(m0211a, x6p7, self.twiddle1.re);
        let m0211b = vmul_n_f32(x1m12, self.twiddle2.im);
        let m0211b = vfma_n_f32(m0211b, x2m11, self.twiddle4.im);
        let m0211b = vfma_n_f32(m0211b, x3m10, self.twiddle6.im);
        let m0211b = vfms_n_f32(m0211b, x4m9, self.twiddle5.im);
        let m0211b = vfms_n_f32(m0211b, x5m8, self.twiddle3.im);
        let m0211b = vfms_n_f32(m0211b, x6m7, self.twiddle1.im);
        let (y02, y11) = NeonButterfly::butterfly2h_f32(m0211a, m0211b);

        let m0310a = vfma_n_f32(store[0].v, x1p12, self.twiddle3.re);
        let m0310a = vfma_n_f32(m0310a, x2p11, self.twiddle6.re);
        let m0310a = vfma_n_f32(m0310a, x3p10, self.twiddle4.re);
        let m0310a = vfma_n_f32(m0310a, x4p9, self.twiddle1.re);
        let m0310a = vfma_n_f32(m0310a, x5p8, self.twiddle2.re);
        let m0310a = vfma_n_f32(m0310a, x6p7, self.twiddle5.re);
        let m0310b = vmul_n_f32(x1m12, self.twiddle3.im);
        let m0310b = vfma_n_f32(m0310b, x2m11, self.twiddle6.im);
        let m0310b = vfms_n_f32(m0310b, x3m10, self.twiddle4.im);
        let m0310b = vfms_n_f32(m0310b, x4m9, self.twiddle1.im);
        let m0310b = vfma_n_f32(m0310b, x5m8, self.twiddle2.im);
        let m0310b = vfma_n_f32(m0310b, x6m7, self.twiddle5.im);
        let (y03, y10) = NeonButterfly::butterfly2h_f32(m0310a, m0310b);

        let m0409a = vfma_n_f32(store[0].v, x1p12, self.twiddle4.re);
        let m0409a = vfma_n_f32(m0409a, x2p11, self.twiddle5.re);
        let m0409a = vfma_n_f32(m0409a, x3p10, self.twiddle1.re);
        let m0409a = vfma_n_f32(m0409a, x4p9, self.twiddle3.re);
        let m0409a = vfma_n_f32(m0409a, x5p8, self.twiddle6.re);
        let m0409a = vfma_n_f32(m0409a, x6p7, self.twiddle2.re);
        let m0409b = vmul_n_f32(x1m12, self.twiddle4.im);
        let m0409b = vfms_n_f32(m0409b, x2m11, self.twiddle5.im);
        let m0409b = vfms_n_f32(m0409b, x3m10, self.twiddle1.im);
        let m0409b = vfma_n_f32(m0409b, x4m9, self.twiddle3.im);
        let m0409b = vfms_n_f32(m0409b, x5m8, self.twiddle6.im);
        let m0409b = vfms_n_f32(m0409b, x6m7, self.twiddle2.im);
        let (y04, y09) = NeonButterfly::butterfly2h_f32(m0409a, m0409b);

        let m0508a = vfma_n_f32(store[0].v, x1p12, self.twiddle5.re);
        let m0508a = vfma_n_f32(m0508a, x2p11, self.twiddle3.re);
        let m0508a = vfma_n_f32(m0508a, x3p10, self.twiddle2.re);
        let m0508a = vfma_n_f32(m0508a, x4p9, self.twiddle6.re);
        let m0508a = vfma_n_f32(m0508a, x5p8, self.twiddle1.re);
        let m0508a = vfma_n_f32(m0508a, x6p7, self.twiddle4.re);
        let m0508b = vmul_n_f32(x1m12, self.twiddle5.im);
        let m0508b = vfms_n_f32(m0508b, x2m11, self.twiddle3.im);
        let m0508b = vfma_n_f32(m0508b, x3m10, self.twiddle2.im);
        let m0508b = vfms_n_f32(m0508b, x4m9, self.twiddle6.im);
        let m0508b = vfms_n_f32(m0508b, x5m8, self.twiddle1.im);
        let m0508b = vfma_n_f32(m0508b, x6m7, self.twiddle4.im);
        let (y05, y08) = NeonButterfly::butterfly2h_f32(m0508a, m0508b);

        let m0607a = vfma_n_f32(store[0].v, x1p12, self.twiddle6.re);
        let m0607a = vfma_n_f32(m0607a, x2p11, self.twiddle1.re);
        let m0607a = vfma_n_f32(m0607a, x3p10, self.twiddle5.re);
        let m0607a = vfma_n_f32(m0607a, x4p9, self.twiddle2.re);
        let m0607a = vfma_n_f32(m0607a, x5p8, self.twiddle4.re);
        let m0607a = vfma_n_f32(m0607a, x6p7, self.twiddle3.re);
        let m0607b = vmul_n_f32(x1m12, self.twiddle6.im);
        let m0607b = vfms_n_f32(m0607b, x2m11, self.twiddle1.im);
        let m0607b = vfma_n_f32(m0607b, x3m10, self.twiddle5.im);
        let m0607b = vfms_n_f32(m0607b, x4m9, self.twiddle2.im);
        let m0607b = vfma_n_f32(m0607b, x5m8, self.twiddle4.im);
        let m0607b = vfms_n_f32(m0607b, x6m7, self.twiddle3.im);
        let (y06, y07) = NeonButterfly::butterfly2h_f32(m0607a, m0607b);

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
            NeonStoreFh::raw(y11),
            NeonStoreFh::raw(y12),
        ]
    }
}
