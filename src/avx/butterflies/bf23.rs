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
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_ps};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly23<T> {
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
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly23<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            rotate: unsafe { AvxRotate::new(FftDirection::Inverse) },
            twiddle1: compute_twiddle(1, 23, fft_direction),
            twiddle2: compute_twiddle(2, 23, fft_direction),
            twiddle3: compute_twiddle(3, 23, fft_direction),
            twiddle4: compute_twiddle(4, 23, fft_direction),
            twiddle5: compute_twiddle(5, 23, fft_direction),
            twiddle6: compute_twiddle(6, 23, fft_direction),
            twiddle7: compute_twiddle(7, 23, fft_direction),
            twiddle8: compute_twiddle(8, 23, fft_direction),
            twiddle9: compute_twiddle(9, 23, fft_direction),
            twiddle10: compute_twiddle(10, 23, fft_direction),
            twiddle11: compute_twiddle(11, 23, fft_direction),
        }
    }
}

impl AvxButterfly23<f64> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f64(&self, v: [__m256d; 23]) -> [__m256d; 23] {
        unsafe {
            let y00 = v[0];
            let (x1p22, x1m22) = AvxButterfly::butterfly2_f64(v[1], v[22]);
            let x1m22 = self.rotate.rotate_m256d(x1m22);
            let y00 = _mm256_add_pd(y00, x1p22);
            let (x2p21, x2m21) = AvxButterfly::butterfly2_f64(v[2], v[21]);
            let x2m21 = self.rotate.rotate_m256d(x2m21);
            let y00 = _mm256_add_pd(y00, x2p21);
            let (x3p20, x3m20) = AvxButterfly::butterfly2_f64(v[3], v[20]);
            let x3m20 = self.rotate.rotate_m256d(x3m20);
            let y00 = _mm256_add_pd(y00, x3p20);
            let (x4p19, x4m19) = AvxButterfly::butterfly2_f64(v[4], v[19]);
            let x4m19 = self.rotate.rotate_m256d(x4m19);
            let y00 = _mm256_add_pd(y00, x4p19);
            let (x5p18, x5m18) = AvxButterfly::butterfly2_f64(v[5], v[18]);
            let x5m18 = self.rotate.rotate_m256d(x5m18);
            let y00 = _mm256_add_pd(y00, x5p18);
            let (x6p17, x6m17) = AvxButterfly::butterfly2_f64(v[6], v[17]);
            let x6m17 = self.rotate.rotate_m256d(x6m17);
            let y00 = _mm256_add_pd(y00, x6p17);
            let (x7p16, x7m16) = AvxButterfly::butterfly2_f64(v[7], v[16]);
            let x7m16 = self.rotate.rotate_m256d(x7m16);
            let y00 = _mm256_add_pd(y00, x7p16);
            let (x8p15, x8m15) = AvxButterfly::butterfly2_f64(v[8], v[15]);
            let x8m15 = self.rotate.rotate_m256d(x8m15);
            let y00 = _mm256_add_pd(y00, x8p15);
            let (x9p14, x9m14) = AvxButterfly::butterfly2_f64(v[9], v[14]);
            let x9m14 = self.rotate.rotate_m256d(x9m14);
            let y00 = _mm256_add_pd(y00, x9p14);
            let (x10p13, x10m13) = AvxButterfly::butterfly2_f64(v[10], v[13]);
            let x10m13 = self.rotate.rotate_m256d(x10m13);
            let y00 = _mm256_add_pd(y00, x10p13);
            let (x11p12, x11m12) = AvxButterfly::butterfly2_f64(v[11], v[12]);
            let x11m12 = self.rotate.rotate_m256d(x11m12);
            let y00 = _mm256_add_pd(y00, x11p12);

            let m0122a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle1.re), v[0]);
            let m0122a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle2.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle3.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle4.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle5.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle6.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle7.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle8.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle9.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle10.re), m0122a);
            let m0122a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle11.re), m0122a);
            let m0122b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle1.im));
            let m0122b = _mm256_fmadd_pd(x2m21, _mm256_set1_pd(self.twiddle2.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x3m20, _mm256_set1_pd(self.twiddle3.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x4m19, _mm256_set1_pd(self.twiddle4.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x5m18, _mm256_set1_pd(self.twiddle5.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x6m17, _mm256_set1_pd(self.twiddle6.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x7m16, _mm256_set1_pd(self.twiddle7.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x8m15, _mm256_set1_pd(self.twiddle8.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x9m14, _mm256_set1_pd(self.twiddle9.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x10m13, _mm256_set1_pd(self.twiddle10.im), m0122b);
            let m0122b = _mm256_fmadd_pd(x11m12, _mm256_set1_pd(self.twiddle11.im), m0122b);
            let (y01, y22) = AvxButterfly::butterfly2_f64(m0122a, m0122b);

            let m0221a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle2.re), v[0]);
            let m0221a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle4.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle6.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle8.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle10.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle11.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle9.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle7.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle5.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle3.re), m0221a);
            let m0221a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle1.re), m0221a);
            let m0221b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle2.im));
            let m0221b = _mm256_fmadd_pd(x2m21, _mm256_set1_pd(self.twiddle4.im), m0221b);
            let m0221b = _mm256_fmadd_pd(x3m20, _mm256_set1_pd(self.twiddle6.im), m0221b);
            let m0221b = _mm256_fmadd_pd(x4m19, _mm256_set1_pd(self.twiddle8.im), m0221b);
            let m0221b = _mm256_fmadd_pd(x5m18, _mm256_set1_pd(self.twiddle10.im), m0221b);
            let m0221b = _mm256_fnmadd_pd(x6m17, _mm256_set1_pd(self.twiddle11.im), m0221b);
            let m0221b = _mm256_fnmadd_pd(x7m16, _mm256_set1_pd(self.twiddle9.im), m0221b);
            let m0221b = _mm256_fnmadd_pd(x8m15, _mm256_set1_pd(self.twiddle7.im), m0221b);
            let m0221b = _mm256_fnmadd_pd(x9m14, _mm256_set1_pd(self.twiddle5.im), m0221b);
            let m0221b = _mm256_fnmadd_pd(x10m13, _mm256_set1_pd(self.twiddle3.im), m0221b);
            let m0221b = _mm256_fnmadd_pd(x11m12, _mm256_set1_pd(self.twiddle1.im), m0221b);
            let (y02, y21) = AvxButterfly::butterfly2_f64(m0221a, m0221b);

            let m0320a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle3.re), v[0]);
            let m0320a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle6.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle9.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle11.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle8.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle5.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle2.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle1.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle4.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle7.re), m0320a);
            let m0320a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle10.re), m0320a);
            let m0320b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle3.im));
            let m0320b = _mm256_fmadd_pd(x2m21, _mm256_set1_pd(self.twiddle6.im), m0320b);
            let m0320b = _mm256_fmadd_pd(x3m20, _mm256_set1_pd(self.twiddle9.im), m0320b);
            let m0320b = _mm256_fnmadd_pd(x4m19, _mm256_set1_pd(self.twiddle11.im), m0320b);
            let m0320b = _mm256_fnmadd_pd(x5m18, _mm256_set1_pd(self.twiddle8.im), m0320b);
            let m0320b = _mm256_fnmadd_pd(x6m17, _mm256_set1_pd(self.twiddle5.im), m0320b);
            let m0320b = _mm256_fnmadd_pd(x7m16, _mm256_set1_pd(self.twiddle2.im), m0320b);
            let m0320b = _mm256_fmadd_pd(x8m15, _mm256_set1_pd(self.twiddle1.im), m0320b);
            let m0320b = _mm256_fmadd_pd(x9m14, _mm256_set1_pd(self.twiddle4.im), m0320b);
            let m0320b = _mm256_fmadd_pd(x10m13, _mm256_set1_pd(self.twiddle7.im), m0320b);
            let m0320b = _mm256_fmadd_pd(x11m12, _mm256_set1_pd(self.twiddle10.im), m0320b);
            let (y03, y20) = AvxButterfly::butterfly2_f64(m0320a, m0320b);

            let m0419a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle4.re), v[0]);
            let m0419a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle8.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle11.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle7.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle3.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle1.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle5.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle9.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle10.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle6.re), m0419a);
            let m0419a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle2.re), m0419a);
            let m0419b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle4.im));
            let m0419b = _mm256_fmadd_pd(x2m21, _mm256_set1_pd(self.twiddle8.im), m0419b);
            let m0419b = _mm256_fnmadd_pd(x3m20, _mm256_set1_pd(self.twiddle11.im), m0419b);
            let m0419b = _mm256_fnmadd_pd(x4m19, _mm256_set1_pd(self.twiddle7.im), m0419b);
            let m0419b = _mm256_fnmadd_pd(x5m18, _mm256_set1_pd(self.twiddle3.im), m0419b);
            let m0419b = _mm256_fmadd_pd(x6m17, _mm256_set1_pd(self.twiddle1.im), m0419b);
            let m0419b = _mm256_fmadd_pd(x7m16, _mm256_set1_pd(self.twiddle5.im), m0419b);
            let m0419b = _mm256_fmadd_pd(x8m15, _mm256_set1_pd(self.twiddle9.im), m0419b);
            let m0419b = _mm256_fnmadd_pd(x9m14, _mm256_set1_pd(self.twiddle10.im), m0419b);
            let m0419b = _mm256_fnmadd_pd(x10m13, _mm256_set1_pd(self.twiddle6.im), m0419b);
            let m0419b = _mm256_fnmadd_pd(x11m12, _mm256_set1_pd(self.twiddle2.im), m0419b);
            let (y04, y19) = AvxButterfly::butterfly2_f64(m0419a, m0419b);

            let m0518a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle5.re), v[0]);
            let m0518a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle10.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle8.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle3.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle2.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle7.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle11.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle6.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle1.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle4.re), m0518a);
            let m0518a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle9.re), m0518a);
            let m0518b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle5.im));
            let m0518b = _mm256_fmadd_pd(x2m21, _mm256_set1_pd(self.twiddle10.im), m0518b);
            let m0518b = _mm256_fnmadd_pd(x3m20, _mm256_set1_pd(self.twiddle8.im), m0518b);
            let m0518b = _mm256_fnmadd_pd(x4m19, _mm256_set1_pd(self.twiddle3.im), m0518b);
            let m0518b = _mm256_fmadd_pd(x5m18, _mm256_set1_pd(self.twiddle2.im), m0518b);
            let m0518b = _mm256_fmadd_pd(x6m17, _mm256_set1_pd(self.twiddle7.im), m0518b);
            let m0518b = _mm256_fnmadd_pd(x7m16, _mm256_set1_pd(self.twiddle11.im), m0518b);
            let m0518b = _mm256_fnmadd_pd(x8m15, _mm256_set1_pd(self.twiddle6.im), m0518b);
            let m0518b = _mm256_fnmadd_pd(x9m14, _mm256_set1_pd(self.twiddle1.im), m0518b);
            let m0518b = _mm256_fmadd_pd(x10m13, _mm256_set1_pd(self.twiddle4.im), m0518b);
            let m0518b = _mm256_fmadd_pd(x11m12, _mm256_set1_pd(self.twiddle9.im), m0518b);
            let (y05, y18) = AvxButterfly::butterfly2_f64(m0518a, m0518b);

            let m0617a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle6.re), v[0]);
            let m0617a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle11.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle5.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle1.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle7.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle10.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle4.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle2.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle8.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle9.re), m0617a);
            let m0617a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle3.re), m0617a);
            let m0617b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle6.im));
            let m0617b = _mm256_fnmadd_pd(x2m21, _mm256_set1_pd(self.twiddle11.im), m0617b);
            let m0617b = _mm256_fnmadd_pd(x3m20, _mm256_set1_pd(self.twiddle5.im), m0617b);
            let m0617b = _mm256_fmadd_pd(x4m19, _mm256_set1_pd(self.twiddle1.im), m0617b);
            let m0617b = _mm256_fmadd_pd(x5m18, _mm256_set1_pd(self.twiddle7.im), m0617b);
            let m0617b = _mm256_fnmadd_pd(x6m17, _mm256_set1_pd(self.twiddle10.im), m0617b);
            let m0617b = _mm256_fnmadd_pd(x7m16, _mm256_set1_pd(self.twiddle4.im), m0617b);
            let m0617b = _mm256_fmadd_pd(x8m15, _mm256_set1_pd(self.twiddle2.im), m0617b);
            let m0617b = _mm256_fmadd_pd(x9m14, _mm256_set1_pd(self.twiddle8.im), m0617b);
            let m0617b = _mm256_fnmadd_pd(x10m13, _mm256_set1_pd(self.twiddle9.im), m0617b);
            let m0617b = _mm256_fnmadd_pd(x11m12, _mm256_set1_pd(self.twiddle3.im), m0617b);
            let (y06, y17) = AvxButterfly::butterfly2_f64(m0617a, m0617b);

            let m0716a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle7.re), v[0]);
            let m0716a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle9.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle2.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle5.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle11.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle4.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle3.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle10.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle6.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle1.re), m0716a);
            let m0716a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle8.re), m0716a);
            let m0716b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle7.im));
            let m0716b = _mm256_fnmadd_pd(x2m21, _mm256_set1_pd(self.twiddle9.im), m0716b);
            let m0716b = _mm256_fnmadd_pd(x3m20, _mm256_set1_pd(self.twiddle2.im), m0716b);
            let m0716b = _mm256_fmadd_pd(x4m19, _mm256_set1_pd(self.twiddle5.im), m0716b);
            let m0716b = _mm256_fnmadd_pd(x5m18, _mm256_set1_pd(self.twiddle11.im), m0716b);
            let m0716b = _mm256_fnmadd_pd(x6m17, _mm256_set1_pd(self.twiddle4.im), m0716b);
            let m0716b = _mm256_fmadd_pd(x7m16, _mm256_set1_pd(self.twiddle3.im), m0716b);
            let m0716b = _mm256_fmadd_pd(x8m15, _mm256_set1_pd(self.twiddle10.im), m0716b);
            let m0716b = _mm256_fnmadd_pd(x9m14, _mm256_set1_pd(self.twiddle6.im), m0716b);
            let m0716b = _mm256_fmadd_pd(x10m13, _mm256_set1_pd(self.twiddle1.im), m0716b);
            let m0716b = _mm256_fmadd_pd(x11m12, _mm256_set1_pd(self.twiddle8.im), m0716b);
            let (y07, y16) = AvxButterfly::butterfly2_f64(m0716a, m0716b);

            let m0815a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle8.re), v[0]);
            let m0815a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle7.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle1.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle9.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle6.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle2.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle10.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle5.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle3.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle11.re), m0815a);
            let m0815a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle4.re), m0815a);
            let m0815b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle8.im));
            let m0815b = _mm256_fnmadd_pd(x2m21, _mm256_set1_pd(self.twiddle7.im), m0815b);
            let m0815b = _mm256_fmadd_pd(x3m20, _mm256_set1_pd(self.twiddle1.im), m0815b);
            let m0815b = _mm256_fmadd_pd(x4m19, _mm256_set1_pd(self.twiddle9.im), m0815b);
            let m0815b = _mm256_fnmadd_pd(x5m18, _mm256_set1_pd(self.twiddle6.im), m0815b);
            let m0815b = _mm256_fmadd_pd(x6m17, _mm256_set1_pd(self.twiddle2.im), m0815b);
            let m0815b = _mm256_fmadd_pd(x7m16, _mm256_set1_pd(self.twiddle10.im), m0815b);
            let m0815b = _mm256_fnmadd_pd(x8m15, _mm256_set1_pd(self.twiddle5.im), m0815b);
            let m0815b = _mm256_fmadd_pd(x9m14, _mm256_set1_pd(self.twiddle3.im), m0815b);
            let m0815b = _mm256_fmadd_pd(x10m13, _mm256_set1_pd(self.twiddle11.im), m0815b);
            let m0815b = _mm256_fnmadd_pd(x11m12, _mm256_set1_pd(self.twiddle4.im), m0815b);
            let (y08, y15) = AvxButterfly::butterfly2_f64(m0815a, m0815b);

            let m0914a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle9.re), v[0]);
            let m0914a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle5.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle4.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle10.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle1.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle8.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle6.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle3.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle11.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle2.re), m0914a);
            let m0914a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle7.re), m0914a);
            let m0914b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle9.im));
            let m0914b = _mm256_fnmadd_pd(x2m21, _mm256_set1_pd(self.twiddle5.im), m0914b);
            let m0914b = _mm256_fmadd_pd(x3m20, _mm256_set1_pd(self.twiddle4.im), m0914b);
            let m0914b = _mm256_fnmadd_pd(x4m19, _mm256_set1_pd(self.twiddle10.im), m0914b);
            let m0914b = _mm256_fnmadd_pd(x5m18, _mm256_set1_pd(self.twiddle1.im), m0914b);
            let m0914b = _mm256_fmadd_pd(x6m17, _mm256_set1_pd(self.twiddle8.im), m0914b);
            let m0914b = _mm256_fnmadd_pd(x7m16, _mm256_set1_pd(self.twiddle6.im), m0914b);
            let m0914b = _mm256_fmadd_pd(x8m15, _mm256_set1_pd(self.twiddle3.im), m0914b);
            let m0914b = _mm256_fnmadd_pd(x9m14, _mm256_set1_pd(self.twiddle11.im), m0914b);
            let m0914b = _mm256_fnmadd_pd(x10m13, _mm256_set1_pd(self.twiddle2.im), m0914b);
            let m0914b = _mm256_fmadd_pd(x11m12, _mm256_set1_pd(self.twiddle7.im), m0914b);
            let (y09, y14) = AvxButterfly::butterfly2_f64(m0914a, m0914b);

            let m1013a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle10.re), v[0]);
            let m1013a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle3.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle7.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle6.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle4.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle9.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle1.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle11.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle2.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle8.re), m1013a);
            let m1013a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle5.re), m1013a);
            let m1013b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle10.im));
            let m1013b = _mm256_fnmadd_pd(x2m21, _mm256_set1_pd(self.twiddle3.im), m1013b);
            let m1013b = _mm256_fmadd_pd(x3m20, _mm256_set1_pd(self.twiddle7.im), m1013b);
            let m1013b = _mm256_fnmadd_pd(x4m19, _mm256_set1_pd(self.twiddle6.im), m1013b);
            let m1013b = _mm256_fmadd_pd(x5m18, _mm256_set1_pd(self.twiddle4.im), m1013b);
            let m1013b = _mm256_fnmadd_pd(x6m17, _mm256_set1_pd(self.twiddle9.im), m1013b);
            let m1013b = _mm256_fmadd_pd(x7m16, _mm256_set1_pd(self.twiddle1.im), m1013b);
            let m1013b = _mm256_fmadd_pd(x8m15, _mm256_set1_pd(self.twiddle11.im), m1013b);
            let m1013b = _mm256_fnmadd_pd(x9m14, _mm256_set1_pd(self.twiddle2.im), m1013b);
            let m1013b = _mm256_fmadd_pd(x10m13, _mm256_set1_pd(self.twiddle8.im), m1013b);
            let m1013b = _mm256_fnmadd_pd(x11m12, _mm256_set1_pd(self.twiddle5.im), m1013b);
            let (y10, y13) = AvxButterfly::butterfly2_f64(m1013a, m1013b);

            let m1112a = _mm256_fmadd_pd(x1p22, _mm256_set1_pd(self.twiddle11.re), v[0]);
            let m1112a = _mm256_fmadd_pd(x2p21, _mm256_set1_pd(self.twiddle1.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x3p20, _mm256_set1_pd(self.twiddle10.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x4p19, _mm256_set1_pd(self.twiddle2.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x5p18, _mm256_set1_pd(self.twiddle9.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x6p17, _mm256_set1_pd(self.twiddle3.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x7p16, _mm256_set1_pd(self.twiddle8.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x8p15, _mm256_set1_pd(self.twiddle4.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x9p14, _mm256_set1_pd(self.twiddle7.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x10p13, _mm256_set1_pd(self.twiddle5.re), m1112a);
            let m1112a = _mm256_fmadd_pd(x11p12, _mm256_set1_pd(self.twiddle6.re), m1112a);
            let m1112b = _mm256_mul_pd(x1m22, _mm256_set1_pd(self.twiddle11.im));
            let m1112b = _mm256_fnmadd_pd(x2m21, _mm256_set1_pd(self.twiddle1.im), m1112b);
            let m1112b = _mm256_fmadd_pd(x3m20, _mm256_set1_pd(self.twiddle10.im), m1112b);
            let m1112b = _mm256_fnmadd_pd(x4m19, _mm256_set1_pd(self.twiddle2.im), m1112b);
            let m1112b = _mm256_fmadd_pd(x5m18, _mm256_set1_pd(self.twiddle9.im), m1112b);
            let m1112b = _mm256_fnmadd_pd(x6m17, _mm256_set1_pd(self.twiddle3.im), m1112b);
            let m1112b = _mm256_fmadd_pd(x7m16, _mm256_set1_pd(self.twiddle8.im), m1112b);
            let m1112b = _mm256_fnmadd_pd(x8m15, _mm256_set1_pd(self.twiddle4.im), m1112b);
            let m1112b = _mm256_fmadd_pd(x9m14, _mm256_set1_pd(self.twiddle7.im), m1112b);
            let m1112b = _mm256_fnmadd_pd(x10m13, _mm256_set1_pd(self.twiddle5.im), m1112b);
            let m1112b = _mm256_fmadd_pd(x11m12, _mm256_set1_pd(self.twiddle6.im), m1112b);
            let (y11, y12) = AvxButterfly::butterfly2_f64(m1112a, m1112b);

            [
                y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
                y16, y17, y18, y19, y20, y21, y22,
            ]
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 23 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(23) {
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
                let u22u = _mm_loadu_pd(chunk.get_unchecked(22..).as_ptr().cast());

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
                    _mm256_castpd128_pd256(u22u),
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
                _mm_storeu_pd(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    _mm256_castpd256_pd128(q[22]),
                );
            }
            Ok(())
        }
    }
}

impl FftExecutor<f64> for AvxButterfly23<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        23
    }
}

impl AvxButterfly23<f32> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f32(&self, v: [__m128; 23]) -> [__m128; 23] {
        unsafe {
            let y00 = v[0];
            let (x1p22, x1m22) = AvxButterfly::butterfly2_f32_m128(v[1], v[22]);
            let x1m22 = self.rotate.rotate_m128(x1m22);
            let y00 = _mm_add_ps(y00, x1p22);
            let (x2p21, x2m21) = AvxButterfly::butterfly2_f32_m128(v[2], v[21]);
            let x2m21 = self.rotate.rotate_m128(x2m21);
            let y00 = _mm_add_ps(y00, x2p21);
            let (x3p20, x3m20) = AvxButterfly::butterfly2_f32_m128(v[3], v[20]);
            let x3m20 = self.rotate.rotate_m128(x3m20);
            let y00 = _mm_add_ps(y00, x3p20);
            let (x4p19, x4m19) = AvxButterfly::butterfly2_f32_m128(v[4], v[19]);
            let x4m19 = self.rotate.rotate_m128(x4m19);
            let y00 = _mm_add_ps(y00, x4p19);
            let (x5p18, x5m18) = AvxButterfly::butterfly2_f32_m128(v[5], v[18]);
            let x5m18 = self.rotate.rotate_m128(x5m18);
            let y00 = _mm_add_ps(y00, x5p18);
            let (x6p17, x6m17) = AvxButterfly::butterfly2_f32_m128(v[6], v[17]);
            let x6m17 = self.rotate.rotate_m128(x6m17);
            let y00 = _mm_add_ps(y00, x6p17);
            let (x7p16, x7m16) = AvxButterfly::butterfly2_f32_m128(v[7], v[16]);
            let x7m16 = self.rotate.rotate_m128(x7m16);
            let y00 = _mm_add_ps(y00, x7p16);
            let (x8p15, x8m15) = AvxButterfly::butterfly2_f32_m128(v[8], v[15]);
            let x8m15 = self.rotate.rotate_m128(x8m15);
            let y00 = _mm_add_ps(y00, x8p15);
            let (x9p14, x9m14) = AvxButterfly::butterfly2_f32_m128(v[9], v[14]);
            let x9m14 = self.rotate.rotate_m128(x9m14);
            let y00 = _mm_add_ps(y00, x9p14);
            let (x10p13, x10m13) = AvxButterfly::butterfly2_f32_m128(v[10], v[13]);
            let x10m13 = self.rotate.rotate_m128(x10m13);
            let y00 = _mm_add_ps(y00, x10p13);
            let (x11p12, x11m12) = AvxButterfly::butterfly2_f32_m128(v[11], v[12]);
            let x11m12 = self.rotate.rotate_m128(x11m12);
            let y00 = _mm_add_ps(y00, x11p12);

            let m0122a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle1.re), v[0]);
            let m0122a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle2.re), m0122a);
            let m0122a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle3.re), m0122a);
            let m0122a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle4.re), m0122a);
            let m0122a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle5.re), m0122a);
            let m0122a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle6.re), m0122a);
            let m0122a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle7.re), m0122a);
            let m0122a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle8.re), m0122a);
            let m0122a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle9.re), m0122a);
            let m0122a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle10.re), m0122a);
            let m0122a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle11.re), m0122a);
            let m0122b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle1.im));
            let m0122b = _mm_fmadd_ps(x2m21, _mm_set1_ps(self.twiddle2.im), m0122b);
            let m0122b = _mm_fmadd_ps(x3m20, _mm_set1_ps(self.twiddle3.im), m0122b);
            let m0122b = _mm_fmadd_ps(x4m19, _mm_set1_ps(self.twiddle4.im), m0122b);
            let m0122b = _mm_fmadd_ps(x5m18, _mm_set1_ps(self.twiddle5.im), m0122b);
            let m0122b = _mm_fmadd_ps(x6m17, _mm_set1_ps(self.twiddle6.im), m0122b);
            let m0122b = _mm_fmadd_ps(x7m16, _mm_set1_ps(self.twiddle7.im), m0122b);
            let m0122b = _mm_fmadd_ps(x8m15, _mm_set1_ps(self.twiddle8.im), m0122b);
            let m0122b = _mm_fmadd_ps(x9m14, _mm_set1_ps(self.twiddle9.im), m0122b);
            let m0122b = _mm_fmadd_ps(x10m13, _mm_set1_ps(self.twiddle10.im), m0122b);
            let m0122b = _mm_fmadd_ps(x11m12, _mm_set1_ps(self.twiddle11.im), m0122b);
            let (y01, y22) = AvxButterfly::butterfly2_f32_m128(m0122a, m0122b);

            let m0221a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle2.re), v[0]);
            let m0221a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle4.re), m0221a);
            let m0221a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle6.re), m0221a);
            let m0221a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle8.re), m0221a);
            let m0221a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle10.re), m0221a);
            let m0221a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle11.re), m0221a);
            let m0221a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle9.re), m0221a);
            let m0221a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle7.re), m0221a);
            let m0221a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle5.re), m0221a);
            let m0221a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle3.re), m0221a);
            let m0221a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle1.re), m0221a);
            let m0221b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle2.im));
            let m0221b = _mm_fmadd_ps(x2m21, _mm_set1_ps(self.twiddle4.im), m0221b);
            let m0221b = _mm_fmadd_ps(x3m20, _mm_set1_ps(self.twiddle6.im), m0221b);
            let m0221b = _mm_fmadd_ps(x4m19, _mm_set1_ps(self.twiddle8.im), m0221b);
            let m0221b = _mm_fmadd_ps(x5m18, _mm_set1_ps(self.twiddle10.im), m0221b);
            let m0221b = _mm_fnmadd_ps(x6m17, _mm_set1_ps(self.twiddle11.im), m0221b);
            let m0221b = _mm_fnmadd_ps(x7m16, _mm_set1_ps(self.twiddle9.im), m0221b);
            let m0221b = _mm_fnmadd_ps(x8m15, _mm_set1_ps(self.twiddle7.im), m0221b);
            let m0221b = _mm_fnmadd_ps(x9m14, _mm_set1_ps(self.twiddle5.im), m0221b);
            let m0221b = _mm_fnmadd_ps(x10m13, _mm_set1_ps(self.twiddle3.im), m0221b);
            let m0221b = _mm_fnmadd_ps(x11m12, _mm_set1_ps(self.twiddle1.im), m0221b);
            let (y02, y21) = AvxButterfly::butterfly2_f32_m128(m0221a, m0221b);

            let m0320a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle3.re), v[0]);
            let m0320a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle6.re), m0320a);
            let m0320a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle9.re), m0320a);
            let m0320a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle11.re), m0320a);
            let m0320a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle8.re), m0320a);
            let m0320a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle5.re), m0320a);
            let m0320a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle2.re), m0320a);
            let m0320a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle1.re), m0320a);
            let m0320a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle4.re), m0320a);
            let m0320a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle7.re), m0320a);
            let m0320a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle10.re), m0320a);
            let m0320b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle3.im));
            let m0320b = _mm_fmadd_ps(x2m21, _mm_set1_ps(self.twiddle6.im), m0320b);
            let m0320b = _mm_fmadd_ps(x3m20, _mm_set1_ps(self.twiddle9.im), m0320b);
            let m0320b = _mm_fnmadd_ps(x4m19, _mm_set1_ps(self.twiddle11.im), m0320b);
            let m0320b = _mm_fnmadd_ps(x5m18, _mm_set1_ps(self.twiddle8.im), m0320b);
            let m0320b = _mm_fnmadd_ps(x6m17, _mm_set1_ps(self.twiddle5.im), m0320b);
            let m0320b = _mm_fnmadd_ps(x7m16, _mm_set1_ps(self.twiddle2.im), m0320b);
            let m0320b = _mm_fmadd_ps(x8m15, _mm_set1_ps(self.twiddle1.im), m0320b);
            let m0320b = _mm_fmadd_ps(x9m14, _mm_set1_ps(self.twiddle4.im), m0320b);
            let m0320b = _mm_fmadd_ps(x10m13, _mm_set1_ps(self.twiddle7.im), m0320b);
            let m0320b = _mm_fmadd_ps(x11m12, _mm_set1_ps(self.twiddle10.im), m0320b);
            let (y03, y20) = AvxButterfly::butterfly2_f32_m128(m0320a, m0320b);

            let m0419a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle4.re), v[0]);
            let m0419a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle8.re), m0419a);
            let m0419a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle11.re), m0419a);
            let m0419a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle7.re), m0419a);
            let m0419a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle3.re), m0419a);
            let m0419a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle1.re), m0419a);
            let m0419a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle5.re), m0419a);
            let m0419a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle9.re), m0419a);
            let m0419a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle10.re), m0419a);
            let m0419a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle6.re), m0419a);
            let m0419a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle2.re), m0419a);
            let m0419b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle4.im));
            let m0419b = _mm_fmadd_ps(x2m21, _mm_set1_ps(self.twiddle8.im), m0419b);
            let m0419b = _mm_fnmadd_ps(x3m20, _mm_set1_ps(self.twiddle11.im), m0419b);
            let m0419b = _mm_fnmadd_ps(x4m19, _mm_set1_ps(self.twiddle7.im), m0419b);
            let m0419b = _mm_fnmadd_ps(x5m18, _mm_set1_ps(self.twiddle3.im), m0419b);
            let m0419b = _mm_fmadd_ps(x6m17, _mm_set1_ps(self.twiddle1.im), m0419b);
            let m0419b = _mm_fmadd_ps(x7m16, _mm_set1_ps(self.twiddle5.im), m0419b);
            let m0419b = _mm_fmadd_ps(x8m15, _mm_set1_ps(self.twiddle9.im), m0419b);
            let m0419b = _mm_fnmadd_ps(x9m14, _mm_set1_ps(self.twiddle10.im), m0419b);
            let m0419b = _mm_fnmadd_ps(x10m13, _mm_set1_ps(self.twiddle6.im), m0419b);
            let m0419b = _mm_fnmadd_ps(x11m12, _mm_set1_ps(self.twiddle2.im), m0419b);
            let (y04, y19) = AvxButterfly::butterfly2_f32_m128(m0419a, m0419b);

            let m0518a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle5.re), v[0]);
            let m0518a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle10.re), m0518a);
            let m0518a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle8.re), m0518a);
            let m0518a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle3.re), m0518a);
            let m0518a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle2.re), m0518a);
            let m0518a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle7.re), m0518a);
            let m0518a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle11.re), m0518a);
            let m0518a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle6.re), m0518a);
            let m0518a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle1.re), m0518a);
            let m0518a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle4.re), m0518a);
            let m0518a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle9.re), m0518a);
            let m0518b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle5.im));
            let m0518b = _mm_fmadd_ps(x2m21, _mm_set1_ps(self.twiddle10.im), m0518b);
            let m0518b = _mm_fnmadd_ps(x3m20, _mm_set1_ps(self.twiddle8.im), m0518b);
            let m0518b = _mm_fnmadd_ps(x4m19, _mm_set1_ps(self.twiddle3.im), m0518b);
            let m0518b = _mm_fmadd_ps(x5m18, _mm_set1_ps(self.twiddle2.im), m0518b);
            let m0518b = _mm_fmadd_ps(x6m17, _mm_set1_ps(self.twiddle7.im), m0518b);
            let m0518b = _mm_fnmadd_ps(x7m16, _mm_set1_ps(self.twiddle11.im), m0518b);
            let m0518b = _mm_fnmadd_ps(x8m15, _mm_set1_ps(self.twiddle6.im), m0518b);
            let m0518b = _mm_fnmadd_ps(x9m14, _mm_set1_ps(self.twiddle1.im), m0518b);
            let m0518b = _mm_fmadd_ps(x10m13, _mm_set1_ps(self.twiddle4.im), m0518b);
            let m0518b = _mm_fmadd_ps(x11m12, _mm_set1_ps(self.twiddle9.im), m0518b);
            let (y05, y18) = AvxButterfly::butterfly2_f32_m128(m0518a, m0518b);

            let m0617a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle6.re), v[0]);
            let m0617a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle11.re), m0617a);
            let m0617a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle5.re), m0617a);
            let m0617a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle1.re), m0617a);
            let m0617a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle7.re), m0617a);
            let m0617a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle10.re), m0617a);
            let m0617a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle4.re), m0617a);
            let m0617a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle2.re), m0617a);
            let m0617a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle8.re), m0617a);
            let m0617a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle9.re), m0617a);
            let m0617a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle3.re), m0617a);
            let m0617b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle6.im));
            let m0617b = _mm_fnmadd_ps(x2m21, _mm_set1_ps(self.twiddle11.im), m0617b);
            let m0617b = _mm_fnmadd_ps(x3m20, _mm_set1_ps(self.twiddle5.im), m0617b);
            let m0617b = _mm_fmadd_ps(x4m19, _mm_set1_ps(self.twiddle1.im), m0617b);
            let m0617b = _mm_fmadd_ps(x5m18, _mm_set1_ps(self.twiddle7.im), m0617b);
            let m0617b = _mm_fnmadd_ps(x6m17, _mm_set1_ps(self.twiddle10.im), m0617b);
            let m0617b = _mm_fnmadd_ps(x7m16, _mm_set1_ps(self.twiddle4.im), m0617b);
            let m0617b = _mm_fmadd_ps(x8m15, _mm_set1_ps(self.twiddle2.im), m0617b);
            let m0617b = _mm_fmadd_ps(x9m14, _mm_set1_ps(self.twiddle8.im), m0617b);
            let m0617b = _mm_fnmadd_ps(x10m13, _mm_set1_ps(self.twiddle9.im), m0617b);
            let m0617b = _mm_fnmadd_ps(x11m12, _mm_set1_ps(self.twiddle3.im), m0617b);
            let (y06, y17) = AvxButterfly::butterfly2_f32_m128(m0617a, m0617b);

            let m0716a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle7.re), v[0]);
            let m0716a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle9.re), m0716a);
            let m0716a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle2.re), m0716a);
            let m0716a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle5.re), m0716a);
            let m0716a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle11.re), m0716a);
            let m0716a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle4.re), m0716a);
            let m0716a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle3.re), m0716a);
            let m0716a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle10.re), m0716a);
            let m0716a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle6.re), m0716a);
            let m0716a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle1.re), m0716a);
            let m0716a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle8.re), m0716a);
            let m0716b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle7.im));
            let m0716b = _mm_fnmadd_ps(x2m21, _mm_set1_ps(self.twiddle9.im), m0716b);
            let m0716b = _mm_fnmadd_ps(x3m20, _mm_set1_ps(self.twiddle2.im), m0716b);
            let m0716b = _mm_fmadd_ps(x4m19, _mm_set1_ps(self.twiddle5.im), m0716b);
            let m0716b = _mm_fnmadd_ps(x5m18, _mm_set1_ps(self.twiddle11.im), m0716b);
            let m0716b = _mm_fnmadd_ps(x6m17, _mm_set1_ps(self.twiddle4.im), m0716b);
            let m0716b = _mm_fmadd_ps(x7m16, _mm_set1_ps(self.twiddle3.im), m0716b);
            let m0716b = _mm_fmadd_ps(x8m15, _mm_set1_ps(self.twiddle10.im), m0716b);
            let m0716b = _mm_fnmadd_ps(x9m14, _mm_set1_ps(self.twiddle6.im), m0716b);
            let m0716b = _mm_fmadd_ps(x10m13, _mm_set1_ps(self.twiddle1.im), m0716b);
            let m0716b = _mm_fmadd_ps(x11m12, _mm_set1_ps(self.twiddle8.im), m0716b);
            let (y07, y16) = AvxButterfly::butterfly2_f32_m128(m0716a, m0716b);

            let m0815a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle8.re), v[0]);
            let m0815a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle7.re), m0815a);
            let m0815a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle1.re), m0815a);
            let m0815a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle9.re), m0815a);
            let m0815a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle6.re), m0815a);
            let m0815a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle2.re), m0815a);
            let m0815a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle10.re), m0815a);
            let m0815a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle5.re), m0815a);
            let m0815a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle3.re), m0815a);
            let m0815a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle11.re), m0815a);
            let m0815a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle4.re), m0815a);
            let m0815b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle8.im));
            let m0815b = _mm_fnmadd_ps(x2m21, _mm_set1_ps(self.twiddle7.im), m0815b);
            let m0815b = _mm_fmadd_ps(x3m20, _mm_set1_ps(self.twiddle1.im), m0815b);
            let m0815b = _mm_fmadd_ps(x4m19, _mm_set1_ps(self.twiddle9.im), m0815b);
            let m0815b = _mm_fnmadd_ps(x5m18, _mm_set1_ps(self.twiddle6.im), m0815b);
            let m0815b = _mm_fmadd_ps(x6m17, _mm_set1_ps(self.twiddle2.im), m0815b);
            let m0815b = _mm_fmadd_ps(x7m16, _mm_set1_ps(self.twiddle10.im), m0815b);
            let m0815b = _mm_fnmadd_ps(x8m15, _mm_set1_ps(self.twiddle5.im), m0815b);
            let m0815b = _mm_fmadd_ps(x9m14, _mm_set1_ps(self.twiddle3.im), m0815b);
            let m0815b = _mm_fmadd_ps(x10m13, _mm_set1_ps(self.twiddle11.im), m0815b);
            let m0815b = _mm_fnmadd_ps(x11m12, _mm_set1_ps(self.twiddle4.im), m0815b);
            let (y08, y15) = AvxButterfly::butterfly2_f32_m128(m0815a, m0815b);

            let m0914a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle9.re), v[0]);
            let m0914a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle5.re), m0914a);
            let m0914a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle4.re), m0914a);
            let m0914a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle10.re), m0914a);
            let m0914a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle1.re), m0914a);
            let m0914a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle8.re), m0914a);
            let m0914a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle6.re), m0914a);
            let m0914a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle3.re), m0914a);
            let m0914a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle11.re), m0914a);
            let m0914a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle2.re), m0914a);
            let m0914a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle7.re), m0914a);
            let m0914b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle9.im));
            let m0914b = _mm_fnmadd_ps(x2m21, _mm_set1_ps(self.twiddle5.im), m0914b);
            let m0914b = _mm_fmadd_ps(x3m20, _mm_set1_ps(self.twiddle4.im), m0914b);
            let m0914b = _mm_fnmadd_ps(x4m19, _mm_set1_ps(self.twiddle10.im), m0914b);
            let m0914b = _mm_fnmadd_ps(x5m18, _mm_set1_ps(self.twiddle1.im), m0914b);
            let m0914b = _mm_fmadd_ps(x6m17, _mm_set1_ps(self.twiddle8.im), m0914b);
            let m0914b = _mm_fnmadd_ps(x7m16, _mm_set1_ps(self.twiddle6.im), m0914b);
            let m0914b = _mm_fmadd_ps(x8m15, _mm_set1_ps(self.twiddle3.im), m0914b);
            let m0914b = _mm_fnmadd_ps(x9m14, _mm_set1_ps(self.twiddle11.im), m0914b);
            let m0914b = _mm_fnmadd_ps(x10m13, _mm_set1_ps(self.twiddle2.im), m0914b);
            let m0914b = _mm_fmadd_ps(x11m12, _mm_set1_ps(self.twiddle7.im), m0914b);
            let (y09, y14) = AvxButterfly::butterfly2_f32_m128(m0914a, m0914b);

            let m1013a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle10.re), v[0]);
            let m1013a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle3.re), m1013a);
            let m1013a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle7.re), m1013a);
            let m1013a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle6.re), m1013a);
            let m1013a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle4.re), m1013a);
            let m1013a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle9.re), m1013a);
            let m1013a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle1.re), m1013a);
            let m1013a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle11.re), m1013a);
            let m1013a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle2.re), m1013a);
            let m1013a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle8.re), m1013a);
            let m1013a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle5.re), m1013a);
            let m1013b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle10.im));
            let m1013b = _mm_fnmadd_ps(x2m21, _mm_set1_ps(self.twiddle3.im), m1013b);
            let m1013b = _mm_fmadd_ps(x3m20, _mm_set1_ps(self.twiddle7.im), m1013b);
            let m1013b = _mm_fnmadd_ps(x4m19, _mm_set1_ps(self.twiddle6.im), m1013b);
            let m1013b = _mm_fmadd_ps(x5m18, _mm_set1_ps(self.twiddle4.im), m1013b);
            let m1013b = _mm_fnmadd_ps(x6m17, _mm_set1_ps(self.twiddle9.im), m1013b);
            let m1013b = _mm_fmadd_ps(x7m16, _mm_set1_ps(self.twiddle1.im), m1013b);
            let m1013b = _mm_fmadd_ps(x8m15, _mm_set1_ps(self.twiddle11.im), m1013b);
            let m1013b = _mm_fnmadd_ps(x9m14, _mm_set1_ps(self.twiddle2.im), m1013b);
            let m1013b = _mm_fmadd_ps(x10m13, _mm_set1_ps(self.twiddle8.im), m1013b);
            let m1013b = _mm_fnmadd_ps(x11m12, _mm_set1_ps(self.twiddle5.im), m1013b);
            let (y10, y13) = AvxButterfly::butterfly2_f32_m128(m1013a, m1013b);

            let m1112a = _mm_fmadd_ps(x1p22, _mm_set1_ps(self.twiddle11.re), v[0]);
            let m1112a = _mm_fmadd_ps(x2p21, _mm_set1_ps(self.twiddle1.re), m1112a);
            let m1112a = _mm_fmadd_ps(x3p20, _mm_set1_ps(self.twiddle10.re), m1112a);
            let m1112a = _mm_fmadd_ps(x4p19, _mm_set1_ps(self.twiddle2.re), m1112a);
            let m1112a = _mm_fmadd_ps(x5p18, _mm_set1_ps(self.twiddle9.re), m1112a);
            let m1112a = _mm_fmadd_ps(x6p17, _mm_set1_ps(self.twiddle3.re), m1112a);
            let m1112a = _mm_fmadd_ps(x7p16, _mm_set1_ps(self.twiddle8.re), m1112a);
            let m1112a = _mm_fmadd_ps(x8p15, _mm_set1_ps(self.twiddle4.re), m1112a);
            let m1112a = _mm_fmadd_ps(x9p14, _mm_set1_ps(self.twiddle7.re), m1112a);
            let m1112a = _mm_fmadd_ps(x10p13, _mm_set1_ps(self.twiddle5.re), m1112a);
            let m1112a = _mm_fmadd_ps(x11p12, _mm_set1_ps(self.twiddle6.re), m1112a);
            let m1112b = _mm_mul_ps(x1m22, _mm_set1_ps(self.twiddle11.im));
            let m1112b = _mm_fnmadd_ps(x2m21, _mm_set1_ps(self.twiddle1.im), m1112b);
            let m1112b = _mm_fmadd_ps(x3m20, _mm_set1_ps(self.twiddle10.im), m1112b);
            let m1112b = _mm_fnmadd_ps(x4m19, _mm_set1_ps(self.twiddle2.im), m1112b);
            let m1112b = _mm_fmadd_ps(x5m18, _mm_set1_ps(self.twiddle9.im), m1112b);
            let m1112b = _mm_fnmadd_ps(x6m17, _mm_set1_ps(self.twiddle3.im), m1112b);
            let m1112b = _mm_fmadd_ps(x7m16, _mm_set1_ps(self.twiddle8.im), m1112b);
            let m1112b = _mm_fnmadd_ps(x8m15, _mm_set1_ps(self.twiddle4.im), m1112b);
            let m1112b = _mm_fmadd_ps(x9m14, _mm_set1_ps(self.twiddle7.im), m1112b);
            let m1112b = _mm_fnmadd_ps(x10m13, _mm_set1_ps(self.twiddle5.im), m1112b);
            let m1112b = _mm_fmadd_ps(x11m12, _mm_set1_ps(self.twiddle6.im), m1112b);
            let (y11, y12) = AvxButterfly::butterfly2_f32_m128(m1112a, m1112b);

            [
                y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15,
                y16, y17, y18, y19, y20, y21, y22,
            ]
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 23 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(23) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16u17u18u19 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());
                let u19u20u21u22 = _mm256_loadu_ps(chunk.get_unchecked(19..).as_ptr().cast());

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
                let u19u20 = _mm256_castps256_ps128(u19u20u21u22);
                let u21u22 = _mm256_extractf128_ps::<1>(u19u20u21u22);

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
                    _mm_unpackhi_ps64(u19u20, u19u20),
                    u21u22,
                    _mm_unpackhi_ps64(u21u22, u21u22),
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
                    chunk.get_unchecked_mut(19..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[19], q[20]),
                        _mm_unpacklo_ps64(q[21], q[22]),
                    ),
                );
            }
        }

        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly23<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        23
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use crate::util::has_valid_avx;
    use rand::Rng;

    #[test]
    fn test_butterfly23_f32() {
        if !has_valid_avx() {
            return;
        }
        for i in 1..4 {
            let size = 23usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly23::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly23::new(FftDirection::Inverse);

            let radix_forward_ref = Dft::new(23, FftDirection::Forward).unwrap();

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

            input = input.iter().map(|&x| x * (1.0 / 23f32)).collect();

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
    fn test_butterfly23_f64() {
        if !has_valid_avx() {
            return;
        }
        for i in 1..4 {
            let size = 23usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly23::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly23::new(FftDirection::Inverse);

            let radix_forward_ref = Dft::new(23, FftDirection::Forward).unwrap();

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

            input = input.iter().map(|&x| x * (1.0 / 23f64)).collect();

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
