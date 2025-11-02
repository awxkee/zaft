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

use crate::avx::butterflies::AvxButterfly;
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_ps,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly29<T> {
    direction: FftDirection,
    rotate: AvxRotate<T>,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    twiddle7: Complex<T>,
    twiddle8: Complex<T>,
    twiddle9: Complex<T>,
    twiddle10: Complex<T>,
    twiddle11: Complex<T>,
    twiddle12: Complex<T>,
    twiddle13: Complex<T>,
    twiddle14: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly29<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            rotate: unsafe { AvxRotate::new(FftDirection::Inverse) },
            twiddle1: compute_twiddle(1, 29, fft_direction),
            twiddle2: compute_twiddle(2, 29, fft_direction),
            twiddle3: compute_twiddle(3, 29, fft_direction),
            twiddle4: compute_twiddle(4, 29, fft_direction),
            twiddle5: compute_twiddle(5, 29, fft_direction),
            twiddle6: compute_twiddle(6, 29, fft_direction),
            twiddle7: compute_twiddle(7, 29, fft_direction),
            twiddle8: compute_twiddle(8, 29, fft_direction),
            twiddle9: compute_twiddle(9, 29, fft_direction),
            twiddle10: compute_twiddle(10, 29, fft_direction),
            twiddle11: compute_twiddle(11, 29, fft_direction),
            twiddle12: compute_twiddle(12, 29, fft_direction),
            twiddle13: compute_twiddle(13, 29, fft_direction),
            twiddle14: compute_twiddle(14, 29, fft_direction),
        }
    }
}

impl AvxButterfly29<f64> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f64(&self, v: [__m256d; 29]) -> [__m256d; 29] {
        unsafe {
            let y00 = v[0];
            let (x1p28, x1m28) = AvxButterfly::butterfly2_f64(v[1], v[28]);
            let x1m28 = self.rotate.rotate_m256d(x1m28);
            let y00 = _mm256_add_pd(y00, x1p28);
            let (x2p27, x2m27) = AvxButterfly::butterfly2_f64(v[2], v[27]);
            let x2m27 = self.rotate.rotate_m256d(x2m27);
            let y00 = _mm256_add_pd(y00, x2p27);
            let (x3p26, x3m26) = AvxButterfly::butterfly2_f64(v[3], v[26]);
            let x3m26 = self.rotate.rotate_m256d(x3m26);
            let y00 = _mm256_add_pd(y00, x3p26);
            let (x4p25, x4m25) = AvxButterfly::butterfly2_f64(v[4], v[25]);
            let x4m25 = self.rotate.rotate_m256d(x4m25);
            let y00 = _mm256_add_pd(y00, x4p25);
            let (x5p24, x5m24) = AvxButterfly::butterfly2_f64(v[5], v[24]);
            let x5m24 = self.rotate.rotate_m256d(x5m24);
            let y00 = _mm256_add_pd(y00, x5p24);
            let (x6p23, x6m23) = AvxButterfly::butterfly2_f64(v[6], v[23]);
            let x6m23 = self.rotate.rotate_m256d(x6m23);
            let y00 = _mm256_add_pd(y00, x6p23);
            let (x7p22, x7m22) = AvxButterfly::butterfly2_f64(v[7], v[22]);
            let x7m22 = self.rotate.rotate_m256d(x7m22);
            let y00 = _mm256_add_pd(y00, x7p22);
            let (x8p21, x8m21) = AvxButterfly::butterfly2_f64(v[8], v[21]);
            let x8m21 = self.rotate.rotate_m256d(x8m21);
            let y00 = _mm256_add_pd(y00, x8p21);
            let (x9p20, x9m20) = AvxButterfly::butterfly2_f64(v[9], v[20]);
            let x9m20 = self.rotate.rotate_m256d(x9m20);
            let y00 = _mm256_add_pd(y00, x9p20);
            let (x10p19, x10m19) = AvxButterfly::butterfly2_f64(v[10], v[19]);
            let x10m19 = self.rotate.rotate_m256d(x10m19);
            let y00 = _mm256_add_pd(y00, x10p19);
            let (x11p18, x11m18) = AvxButterfly::butterfly2_f64(v[11], v[18]);
            let x11m18 = self.rotate.rotate_m256d(x11m18);
            let y00 = _mm256_add_pd(y00, x11p18);
            let (x12p17, x12m17) = AvxButterfly::butterfly2_f64(v[12], v[17]);
            let x12m17 = self.rotate.rotate_m256d(x12m17);
            let y00 = _mm256_add_pd(y00, x12p17);
            let (x13p16, x13m16) = AvxButterfly::butterfly2_f64(v[13], v[16]);
            let x13m16 = self.rotate.rotate_m256d(x13m16);
            let y00 = _mm256_add_pd(y00, x13p16);
            let (x14p15, x14m15) = AvxButterfly::butterfly2_f64(v[14], v[15]);
            let x14m15 = self.rotate.rotate_m256d(x14m15);
            let y00 = _mm256_add_pd(y00, x14p15);

            let m0128a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle1.re), v[0]);
            let m0128a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle2.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle3.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle4.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle5.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle6.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle7.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle8.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle9.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle10.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle11.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle12.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle13.re), m0128a);
            let m0128a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle14.re), m0128a);
            let m0128b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle1.im));
            let m0128b = _mm256_fmadd_pd(x2m27, _mm256_set1_pd(self.twiddle2.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle3.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x4m25, _mm256_set1_pd(self.twiddle4.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle5.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle6.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle7.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x8m21, _mm256_set1_pd(self.twiddle8.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle9.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle10.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x11m18, _mm256_set1_pd(self.twiddle11.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x12m17, _mm256_set1_pd(self.twiddle12.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle13.im), m0128b);
            let m0128b = _mm256_fmadd_pd(x14m15, _mm256_set1_pd(self.twiddle14.im), m0128b);
            let (y01, y28) = AvxButterfly::butterfly2_f64(m0128a, m0128b);

            let m0227a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle2.re), v[0]);
            let m0227a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle4.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle6.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle8.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle10.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle12.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle14.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle13.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle11.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle9.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle7.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle5.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle3.re), m0227a);
            let m0227a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle1.re), m0227a);
            let m0227b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle2.im));
            let m0227b = _mm256_fmadd_pd(x2m27, _mm256_set1_pd(self.twiddle4.im), m0227b);
            let m0227b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle6.im), m0227b);
            let m0227b = _mm256_fmadd_pd(x4m25, _mm256_set1_pd(self.twiddle8.im), m0227b);
            let m0227b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle10.im), m0227b);
            let m0227b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle12.im), m0227b);
            let m0227b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle14.im), m0227b);
            let m0227b = _mm256_fnmadd_pd(x8m21, _mm256_set1_pd(self.twiddle13.im), m0227b);
            let m0227b = _mm256_fnmadd_pd(x9m20, _mm256_set1_pd(self.twiddle11.im), m0227b);
            let m0227b = _mm256_fnmadd_pd(x10m19, _mm256_set1_pd(self.twiddle9.im), m0227b);
            let m0227b = _mm256_fnmadd_pd(x11m18, _mm256_set1_pd(self.twiddle7.im), m0227b);
            let m0227b = _mm256_fnmadd_pd(x12m17, _mm256_set1_pd(self.twiddle5.im), m0227b);
            let m0227b = _mm256_fnmadd_pd(x13m16, _mm256_set1_pd(self.twiddle3.im), m0227b);
            let m0227b = _mm256_fnmadd_pd(x14m15, _mm256_set1_pd(self.twiddle1.im), m0227b);
            let (y02, y27) = AvxButterfly::butterfly2_f64(m0227a, m0227b);

            let m0326a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle3.re), v[0]);
            let m0326a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle6.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle9.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle12.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle14.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle11.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle8.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle5.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle2.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle1.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle4.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle7.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle10.re), m0326a);
            let m0326a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle13.re), m0326a);
            let m0326b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle3.im));
            let m0326b = _mm256_fmadd_pd(x2m27, _mm256_set1_pd(self.twiddle6.im), m0326b);
            let m0326b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle9.im), m0326b);
            let m0326b = _mm256_fmadd_pd(x4m25, _mm256_set1_pd(self.twiddle12.im), m0326b);
            let m0326b = _mm256_fnmadd_pd(x5m24, _mm256_set1_pd(self.twiddle14.im), m0326b);
            let m0326b = _mm256_fnmadd_pd(x6m23, _mm256_set1_pd(self.twiddle11.im), m0326b);
            let m0326b = _mm256_fnmadd_pd(x7m22, _mm256_set1_pd(self.twiddle8.im), m0326b);
            let m0326b = _mm256_fnmadd_pd(x8m21, _mm256_set1_pd(self.twiddle5.im), m0326b);
            let m0326b = _mm256_fnmadd_pd(x9m20, _mm256_set1_pd(self.twiddle2.im), m0326b);
            let m0326b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle1.im), m0326b);
            let m0326b = _mm256_fmadd_pd(x11m18, _mm256_set1_pd(self.twiddle4.im), m0326b);
            let m0326b = _mm256_fmadd_pd(x12m17, _mm256_set1_pd(self.twiddle7.im), m0326b);
            let m0326b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle10.im), m0326b);
            let m0326b = _mm256_fmadd_pd(x14m15, _mm256_set1_pd(self.twiddle13.im), m0326b);
            let (y03, y26) = AvxButterfly::butterfly2_f64(m0326a, m0326b);

            let m0425a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle4.re), v[0]);
            let m0425a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle8.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle12.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle13.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle9.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle5.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle1.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle3.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle7.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle11.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle14.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle10.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle6.re), m0425a);
            let m0425a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle2.re), m0425a);
            let m0425b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle4.im));
            let m0425b = _mm256_fmadd_pd(x2m27, _mm256_set1_pd(self.twiddle8.im), m0425b);
            let m0425b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle12.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle13.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x5m24, _mm256_set1_pd(self.twiddle9.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x6m23, _mm256_set1_pd(self.twiddle5.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x7m22, _mm256_set1_pd(self.twiddle1.im), m0425b);
            let m0425b = _mm256_fmadd_pd(x8m21, _mm256_set1_pd(self.twiddle3.im), m0425b);
            let m0425b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle7.im), m0425b);
            let m0425b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle11.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x11m18, _mm256_set1_pd(self.twiddle14.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x12m17, _mm256_set1_pd(self.twiddle10.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x13m16, _mm256_set1_pd(self.twiddle6.im), m0425b);
            let m0425b = _mm256_fnmadd_pd(x14m15, _mm256_set1_pd(self.twiddle2.im), m0425b);
            let (y04, y25) = AvxButterfly::butterfly2_f64(m0425a, m0425b);

            let m0524a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle5.re), v[0]);
            let m0524a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle10.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle14.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle9.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle4.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle1.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle6.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle11.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle13.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle8.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle3.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle2.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle7.re), m0524a);
            let m0524a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle12.re), m0524a);
            let m0524b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle5.im));
            let m0524b = _mm256_fmadd_pd(x2m27, _mm256_set1_pd(self.twiddle10.im), m0524b);
            let m0524b = _mm256_fnmadd_pd(x3m26, _mm256_set1_pd(self.twiddle14.im), m0524b);
            let m0524b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle9.im), m0524b);
            let m0524b = _mm256_fnmadd_pd(x5m24, _mm256_set1_pd(self.twiddle4.im), m0524b);
            let m0524b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle1.im), m0524b);
            let m0524b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle6.im), m0524b);
            let m0524b = _mm256_fmadd_pd(x8m21, _mm256_set1_pd(self.twiddle11.im), m0524b);
            let m0524b = _mm256_fnmadd_pd(x9m20, _mm256_set1_pd(self.twiddle13.im), m0524b);
            let m0524b = _mm256_fnmadd_pd(x10m19, _mm256_set1_pd(self.twiddle8.im), m0524b);
            let m0524b = _mm256_fnmadd_pd(x11m18, _mm256_set1_pd(self.twiddle3.im), m0524b);
            let m0524b = _mm256_fmadd_pd(x12m17, _mm256_set1_pd(self.twiddle2.im), m0524b);
            let m0524b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle7.im), m0524b);
            let m0524b = _mm256_fmadd_pd(x14m15, _mm256_set1_pd(self.twiddle12.im), m0524b);
            let (y05, y24) = AvxButterfly::butterfly2_f64(m0524a, m0524b);

            let m0623a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle6.re), v[0]);
            let m0623a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle12.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle11.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle5.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle1.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle7.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle13.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle10.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle4.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle2.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle8.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle14.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle9.re), m0623a);
            let m0623a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle3.re), m0623a);
            let m0623b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle6.im));
            let m0623b = _mm256_fmadd_pd(x2m27, _mm256_set1_pd(self.twiddle12.im), m0623b);
            let m0623b = _mm256_fnmadd_pd(x3m26, _mm256_set1_pd(self.twiddle11.im), m0623b);
            let m0623b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle5.im), m0623b);
            let m0623b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle1.im), m0623b);
            let m0623b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle7.im), m0623b);
            let m0623b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle13.im), m0623b);
            let m0623b = _mm256_fnmadd_pd(x8m21, _mm256_set1_pd(self.twiddle10.im), m0623b);
            let m0623b = _mm256_fnmadd_pd(x9m20, _mm256_set1_pd(self.twiddle4.im), m0623b);
            let m0623b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle2.im), m0623b);
            let m0623b = _mm256_fmadd_pd(x11m18, _mm256_set1_pd(self.twiddle8.im), m0623b);
            let m0623b = _mm256_fmadd_pd(x12m17, _mm256_set1_pd(self.twiddle14.im), m0623b);
            let m0623b = _mm256_fnmadd_pd(x13m16, _mm256_set1_pd(self.twiddle9.im), m0623b);
            let m0623b = _mm256_fnmadd_pd(x14m15, _mm256_set1_pd(self.twiddle3.im), m0623b);
            let (y06, y23) = AvxButterfly::butterfly2_f64(m0623a, m0623b);

            let m0722a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle7.re), v[0]);
            let m0722a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle14.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle8.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle1.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle6.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle13.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle9.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle2.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle5.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle12.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle10.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle3.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle4.re), m0722a);
            let m0722a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle11.re), m0722a);
            let m0722b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle7.im));
            let m0722b = _mm256_fmadd_pd(x2m27, _mm256_set1_pd(self.twiddle14.im), m0722b);
            let m0722b = _mm256_fnmadd_pd(x3m26, _mm256_set1_pd(self.twiddle8.im), m0722b);
            let m0722b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle1.im), m0722b);
            let m0722b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle6.im), m0722b);
            let m0722b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle13.im), m0722b);
            let m0722b = _mm256_fnmadd_pd(x7m22, _mm256_set1_pd(self.twiddle9.im), m0722b);
            let m0722b = _mm256_fnmadd_pd(x8m21, _mm256_set1_pd(self.twiddle2.im), m0722b);
            let m0722b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle5.im), m0722b);
            let m0722b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle12.im), m0722b);
            let m0722b = _mm256_fnmadd_pd(x11m18, _mm256_set1_pd(self.twiddle10.im), m0722b);
            let m0722b = _mm256_fnmadd_pd(x12m17, _mm256_set1_pd(self.twiddle3.im), m0722b);
            let m0722b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle4.im), m0722b);
            let m0722b = _mm256_fmadd_pd(x14m15, _mm256_set1_pd(self.twiddle11.im), m0722b);
            let (y07, y22) = AvxButterfly::butterfly2_f64(m0722a, m0722b);

            let m0821a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle8.re), v[0]);
            let m0821a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle13.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle5.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle3.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle11.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle10.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle2.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle6.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle14.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle7.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle1.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle9.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle12.re), m0821a);
            let m0821a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle4.re), m0821a);
            let m0821b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle8.im));
            let m0821b = _mm256_fnmadd_pd(x2m27, _mm256_set1_pd(self.twiddle13.im), m0821b);
            let m0821b = _mm256_fnmadd_pd(x3m26, _mm256_set1_pd(self.twiddle5.im), m0821b);
            let m0821b = _mm256_fmadd_pd(x4m25, _mm256_set1_pd(self.twiddle3.im), m0821b);
            let m0821b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle11.im), m0821b);
            let m0821b = _mm256_fnmadd_pd(x6m23, _mm256_set1_pd(self.twiddle10.im), m0821b);
            let m0821b = _mm256_fnmadd_pd(x7m22, _mm256_set1_pd(self.twiddle2.im), m0821b);
            let m0821b = _mm256_fmadd_pd(x8m21, _mm256_set1_pd(self.twiddle6.im), m0821b);
            let m0821b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle14.im), m0821b);
            let m0821b = _mm256_fnmadd_pd(x10m19, _mm256_set1_pd(self.twiddle7.im), m0821b);
            let m0821b = _mm256_fmadd_pd(x11m18, _mm256_set1_pd(self.twiddle1.im), m0821b);
            let m0821b = _mm256_fmadd_pd(x12m17, _mm256_set1_pd(self.twiddle9.im), m0821b);
            let m0821b = _mm256_fnmadd_pd(x13m16, _mm256_set1_pd(self.twiddle12.im), m0821b);
            let m0821b = _mm256_fnmadd_pd(x14m15, _mm256_set1_pd(self.twiddle4.im), m0821b);
            let (y08, y21) = AvxButterfly::butterfly2_f64(m0821a, m0821b);

            let m0920a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle9.re), v[0]);
            let m0920a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle11.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle2.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle7.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle13.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle4.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle5.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle14.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle6.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle3.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle12.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle8.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle1.re), m0920a);
            let m0920a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle10.re), m0920a);
            let m0920b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle9.im));
            let m0920b = _mm256_fnmadd_pd(x2m27, _mm256_set1_pd(self.twiddle11.im), m0920b);
            let m0920b = _mm256_fnmadd_pd(x3m26, _mm256_set1_pd(self.twiddle2.im), m0920b);
            let m0920b = _mm256_fmadd_pd(x4m25, _mm256_set1_pd(self.twiddle7.im), m0920b);
            let m0920b = _mm256_fnmadd_pd(x5m24, _mm256_set1_pd(self.twiddle13.im), m0920b);
            let m0920b = _mm256_fnmadd_pd(x6m23, _mm256_set1_pd(self.twiddle4.im), m0920b);
            let m0920b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle5.im), m0920b);
            let m0920b = _mm256_fmadd_pd(x8m21, _mm256_set1_pd(self.twiddle14.im), m0920b);
            let m0920b = _mm256_fnmadd_pd(x9m20, _mm256_set1_pd(self.twiddle6.im), m0920b);
            let m0920b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle3.im), m0920b);
            let m0920b = _mm256_fmadd_pd(x11m18, _mm256_set1_pd(self.twiddle12.im), m0920b);
            let m0920b = _mm256_fnmadd_pd(x12m17, _mm256_set1_pd(self.twiddle8.im), m0920b);
            let m0920b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle1.im), m0920b);
            let m0920b = _mm256_fmadd_pd(x14m15, _mm256_set1_pd(self.twiddle10.im), m0920b);
            let (y09, y20) = AvxButterfly::butterfly2_f64(m0920a, m0920b);

            let m1019a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle10.re), v[0]);
            let m1019a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle9.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle1.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle11.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle8.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle2.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle12.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle7.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle3.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle13.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle6.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle4.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle14.re), m1019a);
            let m1019a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle5.re), m1019a);
            let m1019b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle10.im));
            let m1019b = _mm256_fnmadd_pd(x2m27, _mm256_set1_pd(self.twiddle9.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle1.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x4m25, _mm256_set1_pd(self.twiddle11.im), m1019b);
            let m1019b = _mm256_fnmadd_pd(x5m24, _mm256_set1_pd(self.twiddle8.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle2.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle12.im), m1019b);
            let m1019b = _mm256_fnmadd_pd(x8m21, _mm256_set1_pd(self.twiddle7.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle3.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle13.im), m1019b);
            let m1019b = _mm256_fnmadd_pd(x11m18, _mm256_set1_pd(self.twiddle6.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x12m17, _mm256_set1_pd(self.twiddle4.im), m1019b);
            let m1019b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle14.im), m1019b);
            let m1019b = _mm256_fnmadd_pd(x14m15, _mm256_set1_pd(self.twiddle5.im), m1019b);
            let (y10, y19) = AvxButterfly::butterfly2_f64(m1019a, m1019b);

            let m1118a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle11.re), v[0]);
            let m1118a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle7.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle4.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle14.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle3.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle8.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle10.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle1.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle12.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle6.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle5.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle13.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle2.re), m1118a);
            let m1118a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle9.re), m1118a);
            let m1118b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle11.im));
            let m1118b = _mm256_fnmadd_pd(x2m27, _mm256_set1_pd(self.twiddle7.im), m1118b);
            let m1118b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle4.im), m1118b);
            let m1118b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle14.im), m1118b);
            let m1118b = _mm256_fnmadd_pd(x5m24, _mm256_set1_pd(self.twiddle3.im), m1118b);
            let m1118b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle8.im), m1118b);
            let m1118b = _mm256_fnmadd_pd(x7m22, _mm256_set1_pd(self.twiddle10.im), m1118b);
            let m1118b = _mm256_fmadd_pd(x8m21, _mm256_set1_pd(self.twiddle1.im), m1118b);
            let m1118b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle12.im), m1118b);
            let m1118b = _mm256_fnmadd_pd(x10m19, _mm256_set1_pd(self.twiddle6.im), m1118b);
            let m1118b = _mm256_fmadd_pd(x11m18, _mm256_set1_pd(self.twiddle5.im), m1118b);
            let m1118b = _mm256_fnmadd_pd(x12m17, _mm256_set1_pd(self.twiddle13.im), m1118b);
            let m1118b = _mm256_fnmadd_pd(x13m16, _mm256_set1_pd(self.twiddle2.im), m1118b);
            let m1118b = _mm256_fmadd_pd(x14m15, _mm256_set1_pd(self.twiddle9.im), m1118b);
            let (y11, y18) = AvxButterfly::butterfly2_f64(m1118a, m1118b);

            let m1217a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle12.re), v[0]);
            let m1217a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle5.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle7.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle10.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle2.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle14.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle3.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle9.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle8.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle4.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle13.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle1.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle11.re), m1217a);
            let m1217a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle6.re), m1217a);
            let m1217b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle12.im));
            let m1217b = _mm256_fnmadd_pd(x2m27, _mm256_set1_pd(self.twiddle5.im), m1217b);
            let m1217b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle7.im), m1217b);
            let m1217b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle10.im), m1217b);
            let m1217b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle2.im), m1217b);
            let m1217b = _mm256_fmadd_pd(x6m23, _mm256_set1_pd(self.twiddle14.im), m1217b);
            let m1217b = _mm256_fnmadd_pd(x7m22, _mm256_set1_pd(self.twiddle3.im), m1217b);
            let m1217b = _mm256_fmadd_pd(x8m21, _mm256_set1_pd(self.twiddle9.im), m1217b);
            let m1217b = _mm256_fnmadd_pd(x9m20, _mm256_set1_pd(self.twiddle8.im), m1217b);
            let m1217b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle4.im), m1217b);
            let m1217b = _mm256_fnmadd_pd(x11m18, _mm256_set1_pd(self.twiddle13.im), m1217b);
            let m1217b = _mm256_fnmadd_pd(x12m17, _mm256_set1_pd(self.twiddle1.im), m1217b);
            let m1217b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle11.im), m1217b);
            let m1217b = _mm256_fnmadd_pd(x14m15, _mm256_set1_pd(self.twiddle6.im), m1217b);
            let (y12, y17) = AvxButterfly::butterfly2_f64(m1217a, m1217b);

            let m1316a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle13.re), v[0]);
            let m1316a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle3.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle10.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle6.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle7.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle9.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle4.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle12.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle1.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle14.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle2.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle11.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle5.re), m1316a);
            let m1316a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle8.re), m1316a);
            let m1316b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle13.im));
            let m1316b = _mm256_fnmadd_pd(x2m27, _mm256_set1_pd(self.twiddle3.im), m1316b);
            let m1316b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle10.im), m1316b);
            let m1316b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle6.im), m1316b);
            let m1316b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle7.im), m1316b);
            let m1316b = _mm256_fnmadd_pd(x6m23, _mm256_set1_pd(self.twiddle9.im), m1316b);
            let m1316b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle4.im), m1316b);
            let m1316b = _mm256_fnmadd_pd(x8m21, _mm256_set1_pd(self.twiddle12.im), m1316b);
            let m1316b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle1.im), m1316b);
            let m1316b = _mm256_fmadd_pd(x10m19, _mm256_set1_pd(self.twiddle14.im), m1316b);
            let m1316b = _mm256_fnmadd_pd(x11m18, _mm256_set1_pd(self.twiddle2.im), m1316b);
            let m1316b = _mm256_fmadd_pd(x12m17, _mm256_set1_pd(self.twiddle11.im), m1316b);
            let m1316b = _mm256_fnmadd_pd(x13m16, _mm256_set1_pd(self.twiddle5.im), m1316b);
            let m1316b = _mm256_fmadd_pd(x14m15, _mm256_set1_pd(self.twiddle8.im), m1316b);
            let (y13, y16) = AvxButterfly::butterfly2_f64(m1316a, m1316b);

            let m1415a = _mm256_fmadd_pd(x1p28, _mm256_set1_pd(self.twiddle14.re), v[0]);
            let m1415a = _mm256_fmadd_pd(x2p27, _mm256_set1_pd(self.twiddle1.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x3p26, _mm256_set1_pd(self.twiddle13.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x4p25, _mm256_set1_pd(self.twiddle2.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x5p24, _mm256_set1_pd(self.twiddle12.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x6p23, _mm256_set1_pd(self.twiddle3.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x7p22, _mm256_set1_pd(self.twiddle11.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x8p21, _mm256_set1_pd(self.twiddle4.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x9p20, _mm256_set1_pd(self.twiddle10.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x10p19, _mm256_set1_pd(self.twiddle5.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x11p18, _mm256_set1_pd(self.twiddle9.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x12p17, _mm256_set1_pd(self.twiddle6.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x13p16, _mm256_set1_pd(self.twiddle8.re), m1415a);
            let m1415a = _mm256_fmadd_pd(x14p15, _mm256_set1_pd(self.twiddle7.re), m1415a);
            let m1415b = _mm256_mul_pd(x1m28, _mm256_set1_pd(self.twiddle14.im));
            let m1415b = _mm256_fnmadd_pd(x2m27, _mm256_set1_pd(self.twiddle1.im), m1415b);
            let m1415b = _mm256_fmadd_pd(x3m26, _mm256_set1_pd(self.twiddle13.im), m1415b);
            let m1415b = _mm256_fnmadd_pd(x4m25, _mm256_set1_pd(self.twiddle2.im), m1415b);
            let m1415b = _mm256_fmadd_pd(x5m24, _mm256_set1_pd(self.twiddle12.im), m1415b);
            let m1415b = _mm256_fnmadd_pd(x6m23, _mm256_set1_pd(self.twiddle3.im), m1415b);
            let m1415b = _mm256_fmadd_pd(x7m22, _mm256_set1_pd(self.twiddle11.im), m1415b);
            let m1415b = _mm256_fnmadd_pd(x8m21, _mm256_set1_pd(self.twiddle4.im), m1415b);
            let m1415b = _mm256_fmadd_pd(x9m20, _mm256_set1_pd(self.twiddle10.im), m1415b);
            let m1415b = _mm256_fnmadd_pd(x10m19, _mm256_set1_pd(self.twiddle5.im), m1415b);
            let m1415b = _mm256_fmadd_pd(x11m18, _mm256_set1_pd(self.twiddle9.im), m1415b);
            let m1415b = _mm256_fnmadd_pd(x12m17, _mm256_set1_pd(self.twiddle6.im), m1415b);
            let m1415b = _mm256_fmadd_pd(x13m16, _mm256_set1_pd(self.twiddle8.im), m1415b);
            let m1415b = _mm256_fnmadd_pd(x14m15, _mm256_set1_pd(self.twiddle7.im), m1415b);
            let (y14, y15) = AvxButterfly::butterfly2_f64(m1415a, m1415b);
            [
                y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
                y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28,
            ]
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 29 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(29) {
                let u0u1 = _mm256_loadu_pd(chunk.as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = _mm256_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());
                let u18u19 = _mm256_loadu_pd(chunk.get_unchecked(18..).as_ptr().cast());
                let u20u21 = _mm256_loadu_pd(chunk.get_unchecked(20..).as_ptr().cast());
                let u22u23 = _mm256_loadu_pd(chunk.get_unchecked(22..).as_ptr().cast());
                let u24u25 = _mm256_loadu_pd(chunk.get_unchecked(24..).as_ptr().cast());
                let u26u27 = _mm256_loadu_pd(chunk.get_unchecked(26..).as_ptr().cast());
                let u28 = _mm_loadu_pd(chunk.get_unchecked(28..).as_ptr().cast());

                const HI_LO: i32 = 0b0010_0001;
                const LO_LO: i32 = 0b0010_0000;

                let q = self.kernel_f64([
                    u0u1,
                    _mm256_permute2f128_pd::<HI_LO>(u0u1, u0u1),
                    u2u3,
                    _mm256_permute2f128_pd::<HI_LO>(u2u3, u2u3),
                    u4u5,
                    _mm256_permute2f128_pd::<HI_LO>(u4u5, u4u5),
                    u6u7,
                    _mm256_permute2f128_pd::<HI_LO>(u6u7, u6u7),
                    u8u9,
                    _mm256_permute2f128_pd::<HI_LO>(u8u9, u8u9),
                    u10u11,
                    _mm256_permute2f128_pd::<HI_LO>(u10u11, u10u11),
                    u12u13,
                    _mm256_permute2f128_pd::<HI_LO>(u12u13, u12u13),
                    u14u15,
                    _mm256_permute2f128_pd::<HI_LO>(u14u15, u14u15),
                    u16u17,
                    _mm256_permute2f128_pd::<HI_LO>(u16u17, u16u17),
                    u18u19,
                    _mm256_permute2f128_pd::<HI_LO>(u18u19, u18u19),
                    u20u21,
                    _mm256_permute2f128_pd::<HI_LO>(u20u21, u20u21),
                    u22u23,
                    _mm256_permute2f128_pd::<HI_LO>(u22u23, u22u23),
                    u24u25,
                    _mm256_permute2f128_pd::<HI_LO>(u24u25, u24u25),
                    u26u27,
                    _mm256_permute2f128_pd::<HI_LO>(u26u27, u26u27),
                    _mm256_castpd128_pd256(u28),
                ]);

                _mm256_storeu_pd(
                    chunk.as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[0], q[1]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[2], q[3]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[4], q[5]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[6], q[7]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[8], q[9]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[10], q[11]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[12], q[13]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[14], q[15]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[16], q[17]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[18], q[19]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[20], q[21]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[22], q[23]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[24], q[25]),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[26], q[27]),
                );
                _mm_storeu_pd(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    _mm256_castpd256_pd128(q[28]),
                );
            }
            Ok(())
        }
    }
}

impl FftExecutor<f64> for AvxButterfly29<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        29
    }
}

impl AvxButterfly29<f32> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f32(&self, v: [__m128; 29]) -> [__m128; 29] {
        unsafe {
            let y00 = v[0];
            let (x1p28, x1m28) = AvxButterfly::butterfly2_f32_m128(v[1], v[28]);
            let x1m28 = self.rotate.rotate_m128(x1m28);
            let y00 = _mm_add_ps(y00, x1p28);
            let (x2p27, x2m27) = AvxButterfly::butterfly2_f32_m128(v[2], v[27]);
            let x2m27 = self.rotate.rotate_m128(x2m27);
            let y00 = _mm_add_ps(y00, x2p27);
            let (x3p26, x3m26) = AvxButterfly::butterfly2_f32_m128(v[3], v[26]);
            let x3m26 = self.rotate.rotate_m128(x3m26);
            let y00 = _mm_add_ps(y00, x3p26);
            let (x4p25, x4m25) = AvxButterfly::butterfly2_f32_m128(v[4], v[25]);
            let x4m25 = self.rotate.rotate_m128(x4m25);
            let y00 = _mm_add_ps(y00, x4p25);
            let (x5p24, x5m24) = AvxButterfly::butterfly2_f32_m128(v[5], v[24]);
            let x5m24 = self.rotate.rotate_m128(x5m24);
            let y00 = _mm_add_ps(y00, x5p24);
            let (x6p23, x6m23) = AvxButterfly::butterfly2_f32_m128(v[6], v[23]);
            let x6m23 = self.rotate.rotate_m128(x6m23);
            let y00 = _mm_add_ps(y00, x6p23);
            let (x7p22, x7m22) = AvxButterfly::butterfly2_f32_m128(v[7], v[22]);
            let x7m22 = self.rotate.rotate_m128(x7m22);
            let y00 = _mm_add_ps(y00, x7p22);
            let (x8p21, x8m21) = AvxButterfly::butterfly2_f32_m128(v[8], v[21]);
            let x8m21 = self.rotate.rotate_m128(x8m21);
            let y00 = _mm_add_ps(y00, x8p21);
            let (x9p20, x9m20) = AvxButterfly::butterfly2_f32_m128(v[9], v[20]);
            let x9m20 = self.rotate.rotate_m128(x9m20);
            let y00 = _mm_add_ps(y00, x9p20);
            let (x10p19, x10m19) = AvxButterfly::butterfly2_f32_m128(v[10], v[19]);
            let x10m19 = self.rotate.rotate_m128(x10m19);
            let y00 = _mm_add_ps(y00, x10p19);
            let (x11p18, x11m18) = AvxButterfly::butterfly2_f32_m128(v[11], v[18]);
            let x11m18 = self.rotate.rotate_m128(x11m18);
            let y00 = _mm_add_ps(y00, x11p18);
            let (x12p17, x12m17) = AvxButterfly::butterfly2_f32_m128(v[12], v[17]);
            let x12m17 = self.rotate.rotate_m128(x12m17);
            let y00 = _mm_add_ps(y00, x12p17);
            let (x13p16, x13m16) = AvxButterfly::butterfly2_f32_m128(v[13], v[16]);
            let x13m16 = self.rotate.rotate_m128(x13m16);
            let y00 = _mm_add_ps(y00, x13p16);
            let (x14p15, x14m15) = AvxButterfly::butterfly2_f32_m128(v[14], v[15]);
            let x14m15 = self.rotate.rotate_m128(x14m15);
            let y00 = _mm_add_ps(y00, x14p15);

            let m0128a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle1.re), v[0]);
            let m0128a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle2.re), m0128a);
            let m0128a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle3.re), m0128a);
            let m0128a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle4.re), m0128a);
            let m0128a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle5.re), m0128a);
            let m0128a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle6.re), m0128a);
            let m0128a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle7.re), m0128a);
            let m0128a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle8.re), m0128a);
            let m0128a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle9.re), m0128a);
            let m0128a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle10.re), m0128a);
            let m0128a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle11.re), m0128a);
            let m0128a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle12.re), m0128a);
            let m0128a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle13.re), m0128a);
            let m0128a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle14.re), m0128a);
            let m0128b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle1.im));
            let m0128b = _mm_fmadd_ps(x2m27, _mm_set1_ps(self.twiddle2.im), m0128b);
            let m0128b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle3.im), m0128b);
            let m0128b = _mm_fmadd_ps(x4m25, _mm_set1_ps(self.twiddle4.im), m0128b);
            let m0128b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle5.im), m0128b);
            let m0128b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle6.im), m0128b);
            let m0128b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle7.im), m0128b);
            let m0128b = _mm_fmadd_ps(x8m21, _mm_set1_ps(self.twiddle8.im), m0128b);
            let m0128b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle9.im), m0128b);
            let m0128b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle10.im), m0128b);
            let m0128b = _mm_fmadd_ps(x11m18, _mm_set1_ps(self.twiddle11.im), m0128b);
            let m0128b = _mm_fmadd_ps(x12m17, _mm_set1_ps(self.twiddle12.im), m0128b);
            let m0128b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle13.im), m0128b);
            let m0128b = _mm_fmadd_ps(x14m15, _mm_set1_ps(self.twiddle14.im), m0128b);
            let (y01, y28) = AvxButterfly::butterfly2_f32_m128(m0128a, m0128b);

            let m0227a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle2.re), v[0]);
            let m0227a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle4.re), m0227a);
            let m0227a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle6.re), m0227a);
            let m0227a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle8.re), m0227a);
            let m0227a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle10.re), m0227a);
            let m0227a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle12.re), m0227a);
            let m0227a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle14.re), m0227a);
            let m0227a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle13.re), m0227a);
            let m0227a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle11.re), m0227a);
            let m0227a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle9.re), m0227a);
            let m0227a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle7.re), m0227a);
            let m0227a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle5.re), m0227a);
            let m0227a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle3.re), m0227a);
            let m0227a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle1.re), m0227a);
            let m0227b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle2.im));
            let m0227b = _mm_fmadd_ps(x2m27, _mm_set1_ps(self.twiddle4.im), m0227b);
            let m0227b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle6.im), m0227b);
            let m0227b = _mm_fmadd_ps(x4m25, _mm_set1_ps(self.twiddle8.im), m0227b);
            let m0227b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle10.im), m0227b);
            let m0227b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle12.im), m0227b);
            let m0227b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle14.im), m0227b);
            let m0227b = _mm_fnmadd_ps(x8m21, _mm_set1_ps(self.twiddle13.im), m0227b);
            let m0227b = _mm_fnmadd_ps(x9m20, _mm_set1_ps(self.twiddle11.im), m0227b);
            let m0227b = _mm_fnmadd_ps(x10m19, _mm_set1_ps(self.twiddle9.im), m0227b);
            let m0227b = _mm_fnmadd_ps(x11m18, _mm_set1_ps(self.twiddle7.im), m0227b);
            let m0227b = _mm_fnmadd_ps(x12m17, _mm_set1_ps(self.twiddle5.im), m0227b);
            let m0227b = _mm_fnmadd_ps(x13m16, _mm_set1_ps(self.twiddle3.im), m0227b);
            let m0227b = _mm_fnmadd_ps(x14m15, _mm_set1_ps(self.twiddle1.im), m0227b);
            let (y02, y27) = AvxButterfly::butterfly2_f32_m128(m0227a, m0227b);

            let m0326a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle3.re), v[0]);
            let m0326a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle6.re), m0326a);
            let m0326a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle9.re), m0326a);
            let m0326a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle12.re), m0326a);
            let m0326a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle14.re), m0326a);
            let m0326a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle11.re), m0326a);
            let m0326a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle8.re), m0326a);
            let m0326a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle5.re), m0326a);
            let m0326a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle2.re), m0326a);
            let m0326a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle1.re), m0326a);
            let m0326a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle4.re), m0326a);
            let m0326a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle7.re), m0326a);
            let m0326a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle10.re), m0326a);
            let m0326a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle13.re), m0326a);
            let m0326b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle3.im));
            let m0326b = _mm_fmadd_ps(x2m27, _mm_set1_ps(self.twiddle6.im), m0326b);
            let m0326b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle9.im), m0326b);
            let m0326b = _mm_fmadd_ps(x4m25, _mm_set1_ps(self.twiddle12.im), m0326b);
            let m0326b = _mm_fnmadd_ps(x5m24, _mm_set1_ps(self.twiddle14.im), m0326b);
            let m0326b = _mm_fnmadd_ps(x6m23, _mm_set1_ps(self.twiddle11.im), m0326b);
            let m0326b = _mm_fnmadd_ps(x7m22, _mm_set1_ps(self.twiddle8.im), m0326b);
            let m0326b = _mm_fnmadd_ps(x8m21, _mm_set1_ps(self.twiddle5.im), m0326b);
            let m0326b = _mm_fnmadd_ps(x9m20, _mm_set1_ps(self.twiddle2.im), m0326b);
            let m0326b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle1.im), m0326b);
            let m0326b = _mm_fmadd_ps(x11m18, _mm_set1_ps(self.twiddle4.im), m0326b);
            let m0326b = _mm_fmadd_ps(x12m17, _mm_set1_ps(self.twiddle7.im), m0326b);
            let m0326b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle10.im), m0326b);
            let m0326b = _mm_fmadd_ps(x14m15, _mm_set1_ps(self.twiddle13.im), m0326b);
            let (y03, y26) = AvxButterfly::butterfly2_f32_m128(m0326a, m0326b);

            let m0425a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle4.re), v[0]);
            let m0425a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle8.re), m0425a);
            let m0425a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle12.re), m0425a);
            let m0425a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle13.re), m0425a);
            let m0425a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle9.re), m0425a);
            let m0425a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle5.re), m0425a);
            let m0425a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle1.re), m0425a);
            let m0425a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle3.re), m0425a);
            let m0425a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle7.re), m0425a);
            let m0425a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle11.re), m0425a);
            let m0425a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle14.re), m0425a);
            let m0425a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle10.re), m0425a);
            let m0425a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle6.re), m0425a);
            let m0425a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle2.re), m0425a);
            let m0425b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle4.im));
            let m0425b = _mm_fmadd_ps(x2m27, _mm_set1_ps(self.twiddle8.im), m0425b);
            let m0425b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle12.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle13.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x5m24, _mm_set1_ps(self.twiddle9.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x6m23, _mm_set1_ps(self.twiddle5.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x7m22, _mm_set1_ps(self.twiddle1.im), m0425b);
            let m0425b = _mm_fmadd_ps(x8m21, _mm_set1_ps(self.twiddle3.im), m0425b);
            let m0425b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle7.im), m0425b);
            let m0425b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle11.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x11m18, _mm_set1_ps(self.twiddle14.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x12m17, _mm_set1_ps(self.twiddle10.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x13m16, _mm_set1_ps(self.twiddle6.im), m0425b);
            let m0425b = _mm_fnmadd_ps(x14m15, _mm_set1_ps(self.twiddle2.im), m0425b);
            let (y04, y25) = AvxButterfly::butterfly2_f32_m128(m0425a, m0425b);

            let m0524a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle5.re), v[0]);
            let m0524a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle10.re), m0524a);
            let m0524a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle14.re), m0524a);
            let m0524a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle9.re), m0524a);
            let m0524a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle4.re), m0524a);
            let m0524a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle1.re), m0524a);
            let m0524a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle6.re), m0524a);
            let m0524a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle11.re), m0524a);
            let m0524a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle13.re), m0524a);
            let m0524a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle8.re), m0524a);
            let m0524a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle3.re), m0524a);
            let m0524a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle2.re), m0524a);
            let m0524a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle7.re), m0524a);
            let m0524a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle12.re), m0524a);
            let m0524b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle5.im));
            let m0524b = _mm_fmadd_ps(x2m27, _mm_set1_ps(self.twiddle10.im), m0524b);
            let m0524b = _mm_fnmadd_ps(x3m26, _mm_set1_ps(self.twiddle14.im), m0524b);
            let m0524b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle9.im), m0524b);
            let m0524b = _mm_fnmadd_ps(x5m24, _mm_set1_ps(self.twiddle4.im), m0524b);
            let m0524b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle1.im), m0524b);
            let m0524b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle6.im), m0524b);
            let m0524b = _mm_fmadd_ps(x8m21, _mm_set1_ps(self.twiddle11.im), m0524b);
            let m0524b = _mm_fnmadd_ps(x9m20, _mm_set1_ps(self.twiddle13.im), m0524b);
            let m0524b = _mm_fnmadd_ps(x10m19, _mm_set1_ps(self.twiddle8.im), m0524b);
            let m0524b = _mm_fnmadd_ps(x11m18, _mm_set1_ps(self.twiddle3.im), m0524b);
            let m0524b = _mm_fmadd_ps(x12m17, _mm_set1_ps(self.twiddle2.im), m0524b);
            let m0524b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle7.im), m0524b);
            let m0524b = _mm_fmadd_ps(x14m15, _mm_set1_ps(self.twiddle12.im), m0524b);
            let (y05, y24) = AvxButterfly::butterfly2_f32_m128(m0524a, m0524b);

            let m0623a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle6.re), v[0]);
            let m0623a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle12.re), m0623a);
            let m0623a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle11.re), m0623a);
            let m0623a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle5.re), m0623a);
            let m0623a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle1.re), m0623a);
            let m0623a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle7.re), m0623a);
            let m0623a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle13.re), m0623a);
            let m0623a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle10.re), m0623a);
            let m0623a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle4.re), m0623a);
            let m0623a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle2.re), m0623a);
            let m0623a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle8.re), m0623a);
            let m0623a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle14.re), m0623a);
            let m0623a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle9.re), m0623a);
            let m0623a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle3.re), m0623a);
            let m0623b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle6.im));
            let m0623b = _mm_fmadd_ps(x2m27, _mm_set1_ps(self.twiddle12.im), m0623b);
            let m0623b = _mm_fnmadd_ps(x3m26, _mm_set1_ps(self.twiddle11.im), m0623b);
            let m0623b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle5.im), m0623b);
            let m0623b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle1.im), m0623b);
            let m0623b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle7.im), m0623b);
            let m0623b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle13.im), m0623b);
            let m0623b = _mm_fnmadd_ps(x8m21, _mm_set1_ps(self.twiddle10.im), m0623b);
            let m0623b = _mm_fnmadd_ps(x9m20, _mm_set1_ps(self.twiddle4.im), m0623b);
            let m0623b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle2.im), m0623b);
            let m0623b = _mm_fmadd_ps(x11m18, _mm_set1_ps(self.twiddle8.im), m0623b);
            let m0623b = _mm_fmadd_ps(x12m17, _mm_set1_ps(self.twiddle14.im), m0623b);
            let m0623b = _mm_fnmadd_ps(x13m16, _mm_set1_ps(self.twiddle9.im), m0623b);
            let m0623b = _mm_fnmadd_ps(x14m15, _mm_set1_ps(self.twiddle3.im), m0623b);
            let (y06, y23) = AvxButterfly::butterfly2_f32_m128(m0623a, m0623b);

            let m0722a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle7.re), v[0]);
            let m0722a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle14.re), m0722a);
            let m0722a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle8.re), m0722a);
            let m0722a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle1.re), m0722a);
            let m0722a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle6.re), m0722a);
            let m0722a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle13.re), m0722a);
            let m0722a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle9.re), m0722a);
            let m0722a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle2.re), m0722a);
            let m0722a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle5.re), m0722a);
            let m0722a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle12.re), m0722a);
            let m0722a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle10.re), m0722a);
            let m0722a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle3.re), m0722a);
            let m0722a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle4.re), m0722a);
            let m0722a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle11.re), m0722a);
            let m0722b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle7.im));
            let m0722b = _mm_fmadd_ps(x2m27, _mm_set1_ps(self.twiddle14.im), m0722b);
            let m0722b = _mm_fnmadd_ps(x3m26, _mm_set1_ps(self.twiddle8.im), m0722b);
            let m0722b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle1.im), m0722b);
            let m0722b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle6.im), m0722b);
            let m0722b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle13.im), m0722b);
            let m0722b = _mm_fnmadd_ps(x7m22, _mm_set1_ps(self.twiddle9.im), m0722b);
            let m0722b = _mm_fnmadd_ps(x8m21, _mm_set1_ps(self.twiddle2.im), m0722b);
            let m0722b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle5.im), m0722b);
            let m0722b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle12.im), m0722b);
            let m0722b = _mm_fnmadd_ps(x11m18, _mm_set1_ps(self.twiddle10.im), m0722b);
            let m0722b = _mm_fnmadd_ps(x12m17, _mm_set1_ps(self.twiddle3.im), m0722b);
            let m0722b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle4.im), m0722b);
            let m0722b = _mm_fmadd_ps(x14m15, _mm_set1_ps(self.twiddle11.im), m0722b);
            let (y07, y22) = AvxButterfly::butterfly2_f32_m128(m0722a, m0722b);

            let m0821a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle8.re), v[0]);
            let m0821a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle13.re), m0821a);
            let m0821a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle5.re), m0821a);
            let m0821a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle3.re), m0821a);
            let m0821a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle11.re), m0821a);
            let m0821a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle10.re), m0821a);
            let m0821a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle2.re), m0821a);
            let m0821a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle6.re), m0821a);
            let m0821a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle14.re), m0821a);
            let m0821a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle7.re), m0821a);
            let m0821a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle1.re), m0821a);
            let m0821a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle9.re), m0821a);
            let m0821a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle12.re), m0821a);
            let m0821a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle4.re), m0821a);
            let m0821b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle8.im));
            let m0821b = _mm_fnmadd_ps(x2m27, _mm_set1_ps(self.twiddle13.im), m0821b);
            let m0821b = _mm_fnmadd_ps(x3m26, _mm_set1_ps(self.twiddle5.im), m0821b);
            let m0821b = _mm_fmadd_ps(x4m25, _mm_set1_ps(self.twiddle3.im), m0821b);
            let m0821b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle11.im), m0821b);
            let m0821b = _mm_fnmadd_ps(x6m23, _mm_set1_ps(self.twiddle10.im), m0821b);
            let m0821b = _mm_fnmadd_ps(x7m22, _mm_set1_ps(self.twiddle2.im), m0821b);
            let m0821b = _mm_fmadd_ps(x8m21, _mm_set1_ps(self.twiddle6.im), m0821b);
            let m0821b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle14.im), m0821b);
            let m0821b = _mm_fnmadd_ps(x10m19, _mm_set1_ps(self.twiddle7.im), m0821b);
            let m0821b = _mm_fmadd_ps(x11m18, _mm_set1_ps(self.twiddle1.im), m0821b);
            let m0821b = _mm_fmadd_ps(x12m17, _mm_set1_ps(self.twiddle9.im), m0821b);
            let m0821b = _mm_fnmadd_ps(x13m16, _mm_set1_ps(self.twiddle12.im), m0821b);
            let m0821b = _mm_fnmadd_ps(x14m15, _mm_set1_ps(self.twiddle4.im), m0821b);
            let (y08, y21) = AvxButterfly::butterfly2_f32_m128(m0821a, m0821b);

            let m0920a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle9.re), v[0]);
            let m0920a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle11.re), m0920a);
            let m0920a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle2.re), m0920a);
            let m0920a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle7.re), m0920a);
            let m0920a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle13.re), m0920a);
            let m0920a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle4.re), m0920a);
            let m0920a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle5.re), m0920a);
            let m0920a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle14.re), m0920a);
            let m0920a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle6.re), m0920a);
            let m0920a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle3.re), m0920a);
            let m0920a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle12.re), m0920a);
            let m0920a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle8.re), m0920a);
            let m0920a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle1.re), m0920a);
            let m0920a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle10.re), m0920a);
            let m0920b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle9.im));
            let m0920b = _mm_fnmadd_ps(x2m27, _mm_set1_ps(self.twiddle11.im), m0920b);
            let m0920b = _mm_fnmadd_ps(x3m26, _mm_set1_ps(self.twiddle2.im), m0920b);
            let m0920b = _mm_fmadd_ps(x4m25, _mm_set1_ps(self.twiddle7.im), m0920b);
            let m0920b = _mm_fnmadd_ps(x5m24, _mm_set1_ps(self.twiddle13.im), m0920b);
            let m0920b = _mm_fnmadd_ps(x6m23, _mm_set1_ps(self.twiddle4.im), m0920b);
            let m0920b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle5.im), m0920b);
            let m0920b = _mm_fmadd_ps(x8m21, _mm_set1_ps(self.twiddle14.im), m0920b);
            let m0920b = _mm_fnmadd_ps(x9m20, _mm_set1_ps(self.twiddle6.im), m0920b);
            let m0920b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle3.im), m0920b);
            let m0920b = _mm_fmadd_ps(x11m18, _mm_set1_ps(self.twiddle12.im), m0920b);
            let m0920b = _mm_fnmadd_ps(x12m17, _mm_set1_ps(self.twiddle8.im), m0920b);
            let m0920b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle1.im), m0920b);
            let m0920b = _mm_fmadd_ps(x14m15, _mm_set1_ps(self.twiddle10.im), m0920b);
            let (y09, y20) = AvxButterfly::butterfly2_f32_m128(m0920a, m0920b);

            let m1019a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle10.re), v[0]);
            let m1019a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle9.re), m1019a);
            let m1019a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle1.re), m1019a);
            let m1019a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle11.re), m1019a);
            let m1019a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle8.re), m1019a);
            let m1019a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle2.re), m1019a);
            let m1019a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle12.re), m1019a);
            let m1019a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle7.re), m1019a);
            let m1019a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle3.re), m1019a);
            let m1019a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle13.re), m1019a);
            let m1019a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle6.re), m1019a);
            let m1019a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle4.re), m1019a);
            let m1019a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle14.re), m1019a);
            let m1019a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle5.re), m1019a);
            let m1019b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle10.im));
            let m1019b = _mm_fnmadd_ps(x2m27, _mm_set1_ps(self.twiddle9.im), m1019b);
            let m1019b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle1.im), m1019b);
            let m1019b = _mm_fmadd_ps(x4m25, _mm_set1_ps(self.twiddle11.im), m1019b);
            let m1019b = _mm_fnmadd_ps(x5m24, _mm_set1_ps(self.twiddle8.im), m1019b);
            let m1019b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle2.im), m1019b);
            let m1019b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle12.im), m1019b);
            let m1019b = _mm_fnmadd_ps(x8m21, _mm_set1_ps(self.twiddle7.im), m1019b);
            let m1019b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle3.im), m1019b);
            let m1019b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle13.im), m1019b);
            let m1019b = _mm_fnmadd_ps(x11m18, _mm_set1_ps(self.twiddle6.im), m1019b);
            let m1019b = _mm_fmadd_ps(x12m17, _mm_set1_ps(self.twiddle4.im), m1019b);
            let m1019b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle14.im), m1019b);
            let m1019b = _mm_fnmadd_ps(x14m15, _mm_set1_ps(self.twiddle5.im), m1019b);
            let (y10, y19) = AvxButterfly::butterfly2_f32_m128(m1019a, m1019b);

            let m1118a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle11.re), v[0]);
            let m1118a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle7.re), m1118a);
            let m1118a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle4.re), m1118a);
            let m1118a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle14.re), m1118a);
            let m1118a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle3.re), m1118a);
            let m1118a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle8.re), m1118a);
            let m1118a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle10.re), m1118a);
            let m1118a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle1.re), m1118a);
            let m1118a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle12.re), m1118a);
            let m1118a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle6.re), m1118a);
            let m1118a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle5.re), m1118a);
            let m1118a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle13.re), m1118a);
            let m1118a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle2.re), m1118a);
            let m1118a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle9.re), m1118a);
            let m1118b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle11.im));
            let m1118b = _mm_fnmadd_ps(x2m27, _mm_set1_ps(self.twiddle7.im), m1118b);
            let m1118b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle4.im), m1118b);
            let m1118b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle14.im), m1118b);
            let m1118b = _mm_fnmadd_ps(x5m24, _mm_set1_ps(self.twiddle3.im), m1118b);
            let m1118b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle8.im), m1118b);
            let m1118b = _mm_fnmadd_ps(x7m22, _mm_set1_ps(self.twiddle10.im), m1118b);
            let m1118b = _mm_fmadd_ps(x8m21, _mm_set1_ps(self.twiddle1.im), m1118b);
            let m1118b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle12.im), m1118b);
            let m1118b = _mm_fnmadd_ps(x10m19, _mm_set1_ps(self.twiddle6.im), m1118b);
            let m1118b = _mm_fmadd_ps(x11m18, _mm_set1_ps(self.twiddle5.im), m1118b);
            let m1118b = _mm_fnmadd_ps(x12m17, _mm_set1_ps(self.twiddle13.im), m1118b);
            let m1118b = _mm_fnmadd_ps(x13m16, _mm_set1_ps(self.twiddle2.im), m1118b);
            let m1118b = _mm_fmadd_ps(x14m15, _mm_set1_ps(self.twiddle9.im), m1118b);
            let (y11, y18) = AvxButterfly::butterfly2_f32_m128(m1118a, m1118b);

            let m1217a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle12.re), v[0]);
            let m1217a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle5.re), m1217a);
            let m1217a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle7.re), m1217a);
            let m1217a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle10.re), m1217a);
            let m1217a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle2.re), m1217a);
            let m1217a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle14.re), m1217a);
            let m1217a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle3.re), m1217a);
            let m1217a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle9.re), m1217a);
            let m1217a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle8.re), m1217a);
            let m1217a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle4.re), m1217a);
            let m1217a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle13.re), m1217a);
            let m1217a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle1.re), m1217a);
            let m1217a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle11.re), m1217a);
            let m1217a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle6.re), m1217a);
            let m1217b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle12.im));
            let m1217b = _mm_fnmadd_ps(x2m27, _mm_set1_ps(self.twiddle5.im), m1217b);
            let m1217b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle7.im), m1217b);
            let m1217b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle10.im), m1217b);
            let m1217b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle2.im), m1217b);
            let m1217b = _mm_fmadd_ps(x6m23, _mm_set1_ps(self.twiddle14.im), m1217b);
            let m1217b = _mm_fnmadd_ps(x7m22, _mm_set1_ps(self.twiddle3.im), m1217b);
            let m1217b = _mm_fmadd_ps(x8m21, _mm_set1_ps(self.twiddle9.im), m1217b);
            let m1217b = _mm_fnmadd_ps(x9m20, _mm_set1_ps(self.twiddle8.im), m1217b);
            let m1217b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle4.im), m1217b);
            let m1217b = _mm_fnmadd_ps(x11m18, _mm_set1_ps(self.twiddle13.im), m1217b);
            let m1217b = _mm_fnmadd_ps(x12m17, _mm_set1_ps(self.twiddle1.im), m1217b);
            let m1217b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle11.im), m1217b);
            let m1217b = _mm_fnmadd_ps(x14m15, _mm_set1_ps(self.twiddle6.im), m1217b);
            let (y12, y17) = AvxButterfly::butterfly2_f32_m128(m1217a, m1217b);

            let m1316a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle13.re), v[0]);
            let m1316a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle3.re), m1316a);
            let m1316a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle10.re), m1316a);
            let m1316a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle6.re), m1316a);
            let m1316a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle7.re), m1316a);
            let m1316a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle9.re), m1316a);
            let m1316a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle4.re), m1316a);
            let m1316a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle12.re), m1316a);
            let m1316a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle1.re), m1316a);
            let m1316a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle14.re), m1316a);
            let m1316a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle2.re), m1316a);
            let m1316a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle11.re), m1316a);
            let m1316a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle5.re), m1316a);
            let m1316a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle8.re), m1316a);
            let m1316b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle13.im));
            let m1316b = _mm_fnmadd_ps(x2m27, _mm_set1_ps(self.twiddle3.im), m1316b);
            let m1316b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle10.im), m1316b);
            let m1316b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle6.im), m1316b);
            let m1316b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle7.im), m1316b);
            let m1316b = _mm_fnmadd_ps(x6m23, _mm_set1_ps(self.twiddle9.im), m1316b);
            let m1316b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle4.im), m1316b);
            let m1316b = _mm_fnmadd_ps(x8m21, _mm_set1_ps(self.twiddle12.im), m1316b);
            let m1316b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle1.im), m1316b);
            let m1316b = _mm_fmadd_ps(x10m19, _mm_set1_ps(self.twiddle14.im), m1316b);
            let m1316b = _mm_fnmadd_ps(x11m18, _mm_set1_ps(self.twiddle2.im), m1316b);
            let m1316b = _mm_fmadd_ps(x12m17, _mm_set1_ps(self.twiddle11.im), m1316b);
            let m1316b = _mm_fnmadd_ps(x13m16, _mm_set1_ps(self.twiddle5.im), m1316b);
            let m1316b = _mm_fmadd_ps(x14m15, _mm_set1_ps(self.twiddle8.im), m1316b);
            let (y13, y16) = AvxButterfly::butterfly2_f32_m128(m1316a, m1316b);

            let m1415a = _mm_fmadd_ps(x1p28, _mm_set1_ps(self.twiddle14.re), v[0]);
            let m1415a = _mm_fmadd_ps(x2p27, _mm_set1_ps(self.twiddle1.re), m1415a);
            let m1415a = _mm_fmadd_ps(x3p26, _mm_set1_ps(self.twiddle13.re), m1415a);
            let m1415a = _mm_fmadd_ps(x4p25, _mm_set1_ps(self.twiddle2.re), m1415a);
            let m1415a = _mm_fmadd_ps(x5p24, _mm_set1_ps(self.twiddle12.re), m1415a);
            let m1415a = _mm_fmadd_ps(x6p23, _mm_set1_ps(self.twiddle3.re), m1415a);
            let m1415a = _mm_fmadd_ps(x7p22, _mm_set1_ps(self.twiddle11.re), m1415a);
            let m1415a = _mm_fmadd_ps(x8p21, _mm_set1_ps(self.twiddle4.re), m1415a);
            let m1415a = _mm_fmadd_ps(x9p20, _mm_set1_ps(self.twiddle10.re), m1415a);
            let m1415a = _mm_fmadd_ps(x10p19, _mm_set1_ps(self.twiddle5.re), m1415a);
            let m1415a = _mm_fmadd_ps(x11p18, _mm_set1_ps(self.twiddle9.re), m1415a);
            let m1415a = _mm_fmadd_ps(x12p17, _mm_set1_ps(self.twiddle6.re), m1415a);
            let m1415a = _mm_fmadd_ps(x13p16, _mm_set1_ps(self.twiddle8.re), m1415a);
            let m1415a = _mm_fmadd_ps(x14p15, _mm_set1_ps(self.twiddle7.re), m1415a);
            let m1415b = _mm_mul_ps(x1m28, _mm_set1_ps(self.twiddle14.im));
            let m1415b = _mm_fnmadd_ps(x2m27, _mm_set1_ps(self.twiddle1.im), m1415b);
            let m1415b = _mm_fmadd_ps(x3m26, _mm_set1_ps(self.twiddle13.im), m1415b);
            let m1415b = _mm_fnmadd_ps(x4m25, _mm_set1_ps(self.twiddle2.im), m1415b);
            let m1415b = _mm_fmadd_ps(x5m24, _mm_set1_ps(self.twiddle12.im), m1415b);
            let m1415b = _mm_fnmadd_ps(x6m23, _mm_set1_ps(self.twiddle3.im), m1415b);
            let m1415b = _mm_fmadd_ps(x7m22, _mm_set1_ps(self.twiddle11.im), m1415b);
            let m1415b = _mm_fnmadd_ps(x8m21, _mm_set1_ps(self.twiddle4.im), m1415b);
            let m1415b = _mm_fmadd_ps(x9m20, _mm_set1_ps(self.twiddle10.im), m1415b);
            let m1415b = _mm_fnmadd_ps(x10m19, _mm_set1_ps(self.twiddle5.im), m1415b);
            let m1415b = _mm_fmadd_ps(x11m18, _mm_set1_ps(self.twiddle9.im), m1415b);
            let m1415b = _mm_fnmadd_ps(x12m17, _mm_set1_ps(self.twiddle6.im), m1415b);
            let m1415b = _mm_fmadd_ps(x13m16, _mm_set1_ps(self.twiddle8.im), m1415b);
            let m1415b = _mm_fnmadd_ps(x14m15, _mm_set1_ps(self.twiddle7.im), m1415b);
            let (y14, y15) = AvxButterfly::butterfly2_f32_m128(m1415a, m1415b);
            [
                y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
                y16, y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28,
            ]
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 29 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(29) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16u17u18u19 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());
                let u20u21u22u23 = _mm256_loadu_ps(chunk.get_unchecked(20..).as_ptr().cast());
                let u24u25u26u27 = _mm256_loadu_ps(chunk.get_unchecked(24..).as_ptr().cast());
                let u28 = _m128s_load_f32x2(chunk.get_unchecked(28..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u8u9 = _mm256_castps256_ps128(u8u9u10u11);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);
                let u12u13 = _mm256_castps256_ps128(u12u13u14u15);
                let u14u15 = _mm256_extractf128_ps::<1>(u12u13u14u15);
                let u16u17 = _mm256_castps256_ps128(u16u17u18u19);
                let u18u19 = _mm256_extractf128_ps::<1>(u16u17u18u19);
                let u20u21 = _mm256_castps256_ps128(u20u21u22u23);
                let u22u23 = _mm256_extractf128_ps::<1>(u20u21u22u23);
                let u24u25 = _mm256_castps256_ps128(u24u25u26u27);
                let u26u27 = _mm256_extractf128_ps::<1>(u24u25u26u27);

                let q = self.kernel_f32([
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    u2u3,
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    u4u5,
                    _mm_unpackhi_ps64(u4u5, u4u5),
                    u6u7,
                    _mm_unpackhi_ps64(u6u7, u6u7),
                    u8u9,
                    _mm_unpackhi_ps64(u8u9, u8u9),
                    u10u11,
                    _mm_unpackhi_ps64(u10u11, u10u11),
                    u12u13,
                    _mm_unpackhi_ps64(u12u13, u12u13),
                    u14u15,
                    _mm_unpackhi_ps64(u14u15, u14u15),
                    u16u17,
                    _mm_unpackhi_ps64(u16u17, u16u17),
                    u18u19,
                    _mm_unpackhi_ps64(u18u19, u18u19),
                    u20u21,
                    _mm_unpackhi_ps64(u20u21, u20u21),
                    u22u23,
                    _mm_unpackhi_ps64(u22u23, u22u23),
                    u24u25,
                    _mm_unpackhi_ps64(u24u25, u24u25),
                    u26u27,
                    _mm_unpackhi_ps64(u26u27, u26u27),
                    u28,
                ]);

                _mm256_storeu_ps(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(q[0], q[1]), _mm_unpacklo_ps64(q[2], q[3])),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(q[4], q[5]), _mm_unpacklo_ps64(q[6], q[7])),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[8], q[9]),
                        _mm_unpacklo_ps64(q[10], q[11]),
                    ),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[12], q[13]),
                        _mm_unpacklo_ps64(q[14], q[15]),
                    ),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[16], q[17]),
                        _mm_unpacklo_ps64(q[18], q[19]),
                    ),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[20], q[21]),
                        _mm_unpacklo_ps64(q[22], q[23]),
                    ),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[24], q[25]),
                        _mm_unpacklo_ps64(q[26], q[27]),
                    ),
                );

                _m128s_store_f32x2(chunk.get_unchecked_mut(28..).as_mut_ptr().cast(), q[28]);
            }
        }

        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly29<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        29
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use crate::util::has_valid_avx;
    use rand::Rng;

    #[test]
    fn test_butterfly29_f32() {
        if !has_valid_avx() {
            return;
        }
        for i in 1..4 {
            let size = 29usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly29::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly29::new(FftDirection::Inverse);

            let radix_forward_ref = Dft::new(29, FftDirection::Forward).unwrap();

            radix_forward.execute(&mut input).unwrap();
            radix_forward_ref.execute(&mut ref0).unwrap();

            input
                .iter()
                .zip(ref0.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "forward at {idx} a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "forward at {idx} a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 29f32)).collect();

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-4,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-4,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }

    #[test]
    fn test_butterfly29_f64() {
        if !has_valid_avx() {
            return;
        }
        for i in 1..4 {
            let size = 29usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly29::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly29::new(FftDirection::Inverse);

            let radix_forward_ref = Dft::new(29, FftDirection::Forward).unwrap();

            radix_forward.execute(&mut input).unwrap();
            radix_forward_ref.execute(&mut ref0).unwrap();

            input
                .iter()
                .zip(ref0.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "forward at {idx} a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "forward at {idx} a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 29f64)).collect();

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }
}
