/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use crate::avx::butterflies::{
    AvxButterfly, shift_load2d, shift_load2dd, shift_load4, shift_store2d, shift_store2dd,
    shift_store4,
};
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_ps};
use crate::traits::FftTrigonometry;
use crate::util::make_twiddles;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly31<T> {
    direction: FftDirection,
    rotate: AvxRotate<T>,
    twiddles_re: [T; 15],
    twiddles_im: [T; 15],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly31<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddles = make_twiddles::<15, T>(31, fft_direction);
        Self {
            direction: fft_direction,
            rotate: unsafe { AvxRotate::<T>::new(FftDirection::Inverse) },
            twiddles_re: std::array::from_fn(|x| twiddles[x].re),
            twiddles_im: std::array::from_fn(|x| twiddles[x].im),
        }
    }
}

impl AvxButterfly31<f64> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn kernel_f64(&self, v: [__m256d; 31]) -> [__m256d; 31] {
        let y00 = v[0];
        let (x1p30, x1m30) = AvxButterfly::butterfly2_f64(v[1], v[30]);
        let x1m30 = self.rotate.rotate_m256d(x1m30);
        let y00 = _mm256_add_pd(y00, x1p30);
        let (x2p29, x2m29) = AvxButterfly::butterfly2_f64(v[2], v[29]);
        let x2m29 = self.rotate.rotate_m256d(x2m29);
        let y00 = _mm256_add_pd(y00, x2p29);
        let (x3p28, x3m28) = AvxButterfly::butterfly2_f64(v[3], v[28]);
        let x3m28 = self.rotate.rotate_m256d(x3m28);
        let y00 = _mm256_add_pd(y00, x3p28);
        let (x4p27, x4m27) = AvxButterfly::butterfly2_f64(v[4], v[27]);
        let x4m27 = self.rotate.rotate_m256d(x4m27);
        let y00 = _mm256_add_pd(y00, x4p27);
        let (x5p26, x5m26) = AvxButterfly::butterfly2_f64(v[5], v[26]);
        let x5m26 = self.rotate.rotate_m256d(x5m26);
        let y00 = _mm256_add_pd(y00, x5p26);
        let (x6p25, x6m25) = AvxButterfly::butterfly2_f64(v[6], v[25]);
        let x6m25 = self.rotate.rotate_m256d(x6m25);
        let y00 = _mm256_add_pd(y00, x6p25);
        let (x7p24, x7m24) = AvxButterfly::butterfly2_f64(v[7], v[24]);
        let x7m24 = self.rotate.rotate_m256d(x7m24);
        let y00 = _mm256_add_pd(y00, x7p24);
        let (x8p23, x8m23) = AvxButterfly::butterfly2_f64(v[8], v[23]);
        let x8m23 = self.rotate.rotate_m256d(x8m23);
        let y00 = _mm256_add_pd(y00, x8p23);
        let (x9p22, x9m22) = AvxButterfly::butterfly2_f64(v[9], v[22]);
        let x9m22 = self.rotate.rotate_m256d(x9m22);
        let y00 = _mm256_add_pd(y00, x9p22);
        let (x10p21, x10m21) = AvxButterfly::butterfly2_f64(v[10], v[21]);
        let x10m21 = self.rotate.rotate_m256d(x10m21);
        let y00 = _mm256_add_pd(y00, x10p21);
        let (x11p20, x11m20) = AvxButterfly::butterfly2_f64(v[11], v[20]);
        let x11m20 = self.rotate.rotate_m256d(x11m20);
        let y00 = _mm256_add_pd(y00, x11p20);
        let (x12p19, x12m19) = AvxButterfly::butterfly2_f64(v[12], v[19]);
        let x12m19 = self.rotate.rotate_m256d(x12m19);
        let y00 = _mm256_add_pd(y00, x12p19);
        let (x13p18, x13m18) = AvxButterfly::butterfly2_f64(v[13], v[18]);
        let x13m18 = self.rotate.rotate_m256d(x13m18);
        let y00 = _mm256_add_pd(y00, x13p18);
        let (x14p17, x14m17) = AvxButterfly::butterfly2_f64(v[14], v[17]);
        let x14m17 = self.rotate.rotate_m256d(x14m17);
        let y00 = _mm256_add_pd(y00, x14p17);
        let (x15p16, x15m16) = AvxButterfly::butterfly2_f64(v[15], v[16]);
        let x15m16 = self.rotate.rotate_m256d(x15m16);
        let y00 = _mm256_add_pd(y00, x15p16);

        let m0130a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[0]), v[0]);
        let m0130a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[1]), m0130a);
        let m0130a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[2]), m0130a);
        let m0130a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[3]), m0130a);
        let m0130a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[4]), m0130a);
        let m0130a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[5]), m0130a);
        let m0130a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[6]), m0130a);
        let m0130a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[7]), m0130a);
        let m0130a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[8]), m0130a);
        let m0130a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[9]), m0130a);
        let m0130a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[10]), m0130a);
        let m0130a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[11]), m0130a);
        let m0130a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[12]), m0130a);
        let m0130a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[13]), m0130a);
        let m0130a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[14]), m0130a);
        let m0130b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[0]));
        let m0130b = _mm256_fmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[1]), m0130b);
        let m0130b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[2]), m0130b);
        let m0130b = _mm256_fmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[3]), m0130b);
        let m0130b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[4]), m0130b);
        let m0130b = _mm256_fmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[5]), m0130b);
        let m0130b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[6]), m0130b);
        let m0130b = _mm256_fmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[7]), m0130b);
        let m0130b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[8]), m0130b);
        let m0130b = _mm256_fmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[9]), m0130b);
        let m0130b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[10]), m0130b);
        let m0130b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[11]), m0130b);
        let m0130b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[12]), m0130b);
        let m0130b = _mm256_fmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[13]), m0130b);
        let m0130b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[14]), m0130b);
        let (y01, y30) = AvxButterfly::butterfly2_f64(m0130a, m0130b);

        let m0229a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[1]), v[0]);
        let m0229a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[3]), m0229a);
        let m0229a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[5]), m0229a);
        let m0229a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[7]), m0229a);
        let m0229a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[9]), m0229a);
        let m0229a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[11]), m0229a);
        let m0229a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[13]), m0229a);
        let m0229a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[14]), m0229a);
        let m0229a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[12]), m0229a);
        let m0229a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[10]), m0229a);
        let m0229a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[8]), m0229a);
        let m0229a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[6]), m0229a);
        let m0229a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[4]), m0229a);
        let m0229a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[2]), m0229a);
        let m0229a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[0]), m0229a);
        let m0229b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[1]));
        let m0229b = _mm256_fmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[3]), m0229b);
        let m0229b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[5]), m0229b);
        let m0229b = _mm256_fmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[7]), m0229b);
        let m0229b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[9]), m0229b);
        let m0229b = _mm256_fmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[11]), m0229b);
        let m0229b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[13]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[14]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[12]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[10]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[8]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[6]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[4]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[2]), m0229b);
        let m0229b = _mm256_fnmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[0]), m0229b);
        let (y02, y29) = AvxButterfly::butterfly2_f64(m0229a, m0229b);

        let m0328a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[2]), v[0]);
        let m0328a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[5]), m0328a);
        let m0328a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[8]), m0328a);
        let m0328a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[11]), m0328a);
        let m0328a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[14]), m0328a);
        let m0328a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[12]), m0328a);
        let m0328a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[9]), m0328a);
        let m0328a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[6]), m0328a);
        let m0328a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[3]), m0328a);
        let m0328a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[0]), m0328a);
        let m0328a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[1]), m0328a);
        let m0328a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[4]), m0328a);
        let m0328a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[7]), m0328a);
        let m0328a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[10]), m0328a);
        let m0328a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[13]), m0328a);
        let m0328b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[2]));
        let m0328b = _mm256_fmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[5]), m0328b);
        let m0328b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[8]), m0328b);
        let m0328b = _mm256_fmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[11]), m0328b);
        let m0328b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[14]), m0328b);
        let m0328b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[12]), m0328b);
        let m0328b = _mm256_fnmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[9]), m0328b);
        let m0328b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[6]), m0328b);
        let m0328b = _mm256_fnmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[3]), m0328b);
        let m0328b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[0]), m0328b);
        let m0328b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[1]), m0328b);
        let m0328b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[4]), m0328b);
        let m0328b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[7]), m0328b);
        let m0328b = _mm256_fmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[10]), m0328b);
        let m0328b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[13]), m0328b);
        let (y03, y28) = AvxButterfly::butterfly2_f64(m0328a, m0328b);

        let m0427a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[3]), v[0]);
        let m0427a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[7]), m0427a);
        let m0427a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[11]), m0427a);
        let m0427a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[14]), m0427a);
        let m0427a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[10]), m0427a);
        let m0427a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[6]), m0427a);
        let m0427a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[2]), m0427a);
        let m0427a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[0]), m0427a);
        let m0427a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[4]), m0427a);
        let m0427a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[8]), m0427a);
        let m0427a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[12]), m0427a);
        let m0427a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[13]), m0427a);
        let m0427a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[9]), m0427a);
        let m0427a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[5]), m0427a);
        let m0427a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[1]), m0427a);
        let m0427b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[3]));
        let m0427b = _mm256_fmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[7]), m0427b);
        let m0427b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[11]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[14]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[10]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[6]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[2]), m0427b);
        let m0427b = _mm256_fmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[0]), m0427b);
        let m0427b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[4]), m0427b);
        let m0427b = _mm256_fmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[8]), m0427b);
        let m0427b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[12]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[13]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[9]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[5]), m0427b);
        let m0427b = _mm256_fnmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[1]), m0427b);
        let (y04, y27) = AvxButterfly::butterfly2_f64(m0427a, m0427b);

        let m0526a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[4]), v[0]);
        let m0526a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[9]), m0526a);
        let m0526a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[14]), m0526a);
        let m0526a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[10]), m0526a);
        let m0526a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[5]), m0526a);
        let m0526a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[0]), m0526a);
        let m0526a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[3]), m0526a);
        let m0526a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[8]), m0526a);
        let m0526a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[13]), m0526a);
        let m0526a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[11]), m0526a);
        let m0526a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[6]), m0526a);
        let m0526a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[1]), m0526a);
        let m0526a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[2]), m0526a);
        let m0526a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[7]), m0526a);
        let m0526a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[12]), m0526a);
        let m0526b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[4]));
        let m0526b = _mm256_fmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[9]), m0526b);
        let m0526b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[14]), m0526b);
        let m0526b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[10]), m0526b);
        let m0526b = _mm256_fnmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[5]), m0526b);
        let m0526b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[0]), m0526b);
        let m0526b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[3]), m0526b);
        let m0526b = _mm256_fmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[8]), m0526b);
        let m0526b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[13]), m0526b);
        let m0526b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[11]), m0526b);
        let m0526b = _mm256_fnmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[6]), m0526b);
        let m0526b = _mm256_fnmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[1]), m0526b);
        let m0526b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[2]), m0526b);
        let m0526b = _mm256_fmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[7]), m0526b);
        let m0526b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[12]), m0526b);
        let (y05, y26) = AvxButterfly::butterfly2_f64(m0526a, m0526b);

        let m0625a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[5]), v[0]);
        let m0625a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[11]), m0625a);
        let m0625a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[12]), m0625a);
        let m0625a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[6]), m0625a);
        let m0625a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[0]), m0625a);
        let m0625a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[4]), m0625a);
        let m0625a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[10]), m0625a);
        let m0625a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[13]), m0625a);
        let m0625a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[7]), m0625a);
        let m0625a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[1]), m0625a);
        let m0625a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[3]), m0625a);
        let m0625a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[9]), m0625a);
        let m0625a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[14]), m0625a);
        let m0625a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[8]), m0625a);
        let m0625a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[2]), m0625a);
        let m0625b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[5]));
        let m0625b = _mm256_fmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[11]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[12]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[6]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[0]), m0625b);
        let m0625b = _mm256_fmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[4]), m0625b);
        let m0625b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[10]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[13]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[7]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[1]), m0625b);
        let m0625b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[3]), m0625b);
        let m0625b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[9]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[14]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[8]), m0625b);
        let m0625b = _mm256_fnmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[2]), m0625b);
        let (y06, y25) = AvxButterfly::butterfly2_f64(m0625a, m0625b);

        let m0724a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[6]), v[0]);
        let m0724a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[13]), m0724a);
        let m0724a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[9]), m0724a);
        let m0724a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[2]), m0724a);
        let m0724a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[3]), m0724a);
        let m0724a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[10]), m0724a);
        let m0724a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[12]), m0724a);
        let m0724a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[5]), m0724a);
        let m0724a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[0]), m0724a);
        let m0724a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[7]), m0724a);
        let m0724a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[14]), m0724a);
        let m0724a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[8]), m0724a);
        let m0724a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[1]), m0724a);
        let m0724a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[4]), m0724a);
        let m0724a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[11]), m0724a);
        let m0724b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[6]));
        let m0724b = _mm256_fmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[13]), m0724b);
        let m0724b = _mm256_fnmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[9]), m0724b);
        let m0724b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[2]), m0724b);
        let m0724b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[3]), m0724b);
        let m0724b = _mm256_fmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[10]), m0724b);
        let m0724b = _mm256_fnmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[12]), m0724b);
        let m0724b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[5]), m0724b);
        let m0724b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[0]), m0724b);
        let m0724b = _mm256_fmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[7]), m0724b);
        let m0724b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[14]), m0724b);
        let m0724b = _mm256_fnmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[8]), m0724b);
        let m0724b = _mm256_fnmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[1]), m0724b);
        let m0724b = _mm256_fmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[4]), m0724b);
        let m0724b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[11]), m0724b);
        let (y07, y24) = AvxButterfly::butterfly2_f64(m0724a, m0724b);

        let m0823a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[7]), v[0]);
        let m0823a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[14]), m0823a);
        let m0823a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[6]), m0823a);
        let m0823a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[0]), m0823a);
        let m0823a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[8]), m0823a);
        let m0823a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[13]), m0823a);
        let m0823a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[5]), m0823a);
        let m0823a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[1]), m0823a);
        let m0823a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[9]), m0823a);
        let m0823a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[12]), m0823a);
        let m0823a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[4]), m0823a);
        let m0823a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[2]), m0823a);
        let m0823a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[10]), m0823a);
        let m0823a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[11]), m0823a);
        let m0823a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[3]), m0823a);
        let m0823b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[7]));
        let m0823b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[14]), m0823b);
        let m0823b = _mm256_fnmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[6]), m0823b);
        let m0823b = _mm256_fmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[0]), m0823b);
        let m0823b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[8]), m0823b);
        let m0823b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[13]), m0823b);
        let m0823b = _mm256_fnmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[5]), m0823b);
        let m0823b = _mm256_fmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[1]), m0823b);
        let m0823b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[9]), m0823b);
        let m0823b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[12]), m0823b);
        let m0823b = _mm256_fnmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[4]), m0823b);
        let m0823b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[2]), m0823b);
        let m0823b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[10]), m0823b);
        let m0823b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[11]), m0823b);
        let m0823b = _mm256_fnmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[3]), m0823b);
        let (y08, y23) = AvxButterfly::butterfly2_f64(m0823a, m0823b);

        let m0922a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[8]), v[0]);
        let m0922a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[12]), m0922a);
        let m0922a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[3]), m0922a);
        let m0922a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[4]), m0922a);
        let m0922a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[13]), m0922a);
        let m0922a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[7]), m0922a);
        let m0922a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[0]), m0922a);
        let m0922a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[9]), m0922a);
        let m0922a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[11]), m0922a);
        let m0922a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[2]), m0922a);
        let m0922a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[5]), m0922a);
        let m0922a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[14]), m0922a);
        let m0922a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[6]), m0922a);
        let m0922a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[1]), m0922a);
        let m0922a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[10]), m0922a);
        let m0922b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[8]));
        let m0922b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[12]), m0922b);
        let m0922b = _mm256_fnmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[3]), m0922b);
        let m0922b = _mm256_fmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[4]), m0922b);
        let m0922b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[13]), m0922b);
        let m0922b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[7]), m0922b);
        let m0922b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[0]), m0922b);
        let m0922b = _mm256_fmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[9]), m0922b);
        let m0922b = _mm256_fnmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[11]), m0922b);
        let m0922b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[2]), m0922b);
        let m0922b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[5]), m0922b);
        let m0922b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[14]), m0922b);
        let m0922b = _mm256_fnmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[6]), m0922b);
        let m0922b = _mm256_fmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[1]), m0922b);
        let m0922b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[10]), m0922b);
        let (y09, y22) = AvxButterfly::butterfly2_f64(m0922a, m0922b);

        let m1021a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[9]), v[0]);
        let m1021a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[10]), m1021a);
        let m1021a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[0]), m1021a);
        let m1021a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[8]), m1021a);
        let m1021a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[11]), m1021a);
        let m1021a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[1]), m1021a);
        let m1021a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[7]), m1021a);
        let m1021a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[12]), m1021a);
        let m1021a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[2]), m1021a);
        let m1021a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[6]), m1021a);
        let m1021a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[13]), m1021a);
        let m1021a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[3]), m1021a);
        let m1021a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[5]), m1021a);
        let m1021a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[14]), m1021a);
        let m1021a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[4]), m1021a);
        let m1021b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[9]));
        let m1021b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[10]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[0]), m1021b);
        let m1021b = _mm256_fmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[8]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[11]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[1]), m1021b);
        let m1021b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[7]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[12]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[2]), m1021b);
        let m1021b = _mm256_fmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[6]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[13]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[3]), m1021b);
        let m1021b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[5]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[14]), m1021b);
        let m1021b = _mm256_fnmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[4]), m1021b);
        let (y10, y21) = AvxButterfly::butterfly2_f64(m1021a, m1021b);

        let m1120a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[10]), v[0]);
        let m1120a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[8]), m1120a);
        let m1120a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[1]), m1120a);
        let m1120a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[12]), m1120a);
        let m1120a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[6]), m1120a);
        let m1120a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[3]), m1120a);
        let m1120a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[14]), m1120a);
        let m1120a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[4]), m1120a);
        let m1120a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[5]), m1120a);
        let m1120a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[13]), m1120a);
        let m1120a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[2]), m1120a);
        let m1120a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[7]), m1120a);
        let m1120a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[11]), m1120a);
        let m1120a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[0]), m1120a);
        let m1120a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[9]), m1120a);
        let m1120b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[10]));
        let m1120b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[8]), m1120b);
        let m1120b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[1]), m1120b);
        let m1120b = _mm256_fmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[12]), m1120b);
        let m1120b = _mm256_fnmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[6]), m1120b);
        let m1120b = _mm256_fmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[3]), m1120b);
        let m1120b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[14]), m1120b);
        let m1120b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[4]), m1120b);
        let m1120b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[5]), m1120b);
        let m1120b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[13]), m1120b);
        let m1120b = _mm256_fnmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[2]), m1120b);
        let m1120b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[7]), m1120b);
        let m1120b = _mm256_fnmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[11]), m1120b);
        let m1120b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[0]), m1120b);
        let m1120b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[9]), m1120b);
        let (y11, y20) = AvxButterfly::butterfly2_f64(m1120a, m1120b);

        let m1219a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[11]), v[0]);
        let m1219a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[6]), m1219a);
        let m1219a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[4]), m1219a);
        let m1219a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[13]), m1219a);
        let m1219a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[1]), m1219a);
        let m1219a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[9]), m1219a);
        let m1219a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[8]), m1219a);
        let m1219a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[2]), m1219a);
        let m1219a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[14]), m1219a);
        let m1219a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[3]), m1219a);
        let m1219a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[7]), m1219a);
        let m1219a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[10]), m1219a);
        let m1219a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[0]), m1219a);
        let m1219a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[12]), m1219a);
        let m1219a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[5]), m1219a);
        let m1219b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[11]));
        let m1219b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[6]), m1219b);
        let m1219b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[4]), m1219b);
        let m1219b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[13]), m1219b);
        let m1219b = _mm256_fnmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[1]), m1219b);
        let m1219b = _mm256_fmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[9]), m1219b);
        let m1219b = _mm256_fnmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[8]), m1219b);
        let m1219b = _mm256_fmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[2]), m1219b);
        let m1219b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[14]), m1219b);
        let m1219b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[3]), m1219b);
        let m1219b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[7]), m1219b);
        let m1219b = _mm256_fnmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[10]), m1219b);
        let m1219b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[0]), m1219b);
        let m1219b = _mm256_fmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[12]), m1219b);
        let m1219b = _mm256_fnmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[5]), m1219b);
        let (y12, y19) = AvxButterfly::butterfly2_f64(m1219a, m1219b);

        let m1318a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[12]), v[0]);
        let m1318a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[4]), m1318a);
        let m1318a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[7]), m1318a);
        let m1318a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[9]), m1318a);
        let m1318a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[2]), m1318a);
        let m1318a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[14]), m1318a);
        let m1318a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[1]), m1318a);
        let m1318a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[10]), m1318a);
        let m1318a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[6]), m1318a);
        let m1318a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[5]), m1318a);
        let m1318a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[11]), m1318a);
        let m1318a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[0]), m1318a);
        let m1318a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[13]), m1318a);
        let m1318a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[3]), m1318a);
        let m1318a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[8]), m1318a);
        let m1318b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[12]));
        let m1318b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[4]), m1318b);
        let m1318b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[7]), m1318b);
        let m1318b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[9]), m1318b);
        let m1318b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[2]), m1318b);
        let m1318b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[14]), m1318b);
        let m1318b = _mm256_fnmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[1]), m1318b);
        let m1318b = _mm256_fmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[10]), m1318b);
        let m1318b = _mm256_fnmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[6]), m1318b);
        let m1318b = _mm256_fmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[5]), m1318b);
        let m1318b = _mm256_fnmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[11]), m1318b);
        let m1318b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[0]), m1318b);
        let m1318b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[13]), m1318b);
        let m1318b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[3]), m1318b);
        let m1318b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[8]), m1318b);
        let (y13, y18) = AvxButterfly::butterfly2_f64(m1318a, m1318b);

        let m1417a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[13]), v[0]);
        let m1417a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[2]), m1417a);
        let m1417a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[10]), m1417a);
        let m1417a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[5]), m1417a);
        let m1417a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[7]), m1417a);
        let m1417a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[8]), m1417a);
        let m1417a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[4]), m1417a);
        let m1417a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[11]), m1417a);
        let m1417a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[1]), m1417a);
        let m1417a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[14]), m1417a);
        let m1417a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[0]), m1417a);
        let m1417a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[12]), m1417a);
        let m1417a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[3]), m1417a);
        let m1417a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[9]), m1417a);
        let m1417a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[6]), m1417a);
        let m1417b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[13]));
        let m1417b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[2]), m1417b);
        let m1417b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[10]), m1417b);
        let m1417b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[5]), m1417b);
        let m1417b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[7]), m1417b);
        let m1417b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[8]), m1417b);
        let m1417b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[4]), m1417b);
        let m1417b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[11]), m1417b);
        let m1417b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[1]), m1417b);
        let m1417b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[14]), m1417b);
        let m1417b = _mm256_fnmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[0]), m1417b);
        let m1417b = _mm256_fmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[12]), m1417b);
        let m1417b = _mm256_fnmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[3]), m1417b);
        let m1417b = _mm256_fmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[9]), m1417b);
        let m1417b = _mm256_fnmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[6]), m1417b);
        let (y14, y17) = AvxButterfly::butterfly2_f64(m1417a, m1417b);

        let m1516a = _mm256_fmadd_pd(x1p30, _mm256_set1_pd(self.twiddles_re[14]), v[0]);
        let m1516a = _mm256_fmadd_pd(x2p29, _mm256_set1_pd(self.twiddles_re[0]), m1516a);
        let m1516a = _mm256_fmadd_pd(x3p28, _mm256_set1_pd(self.twiddles_re[13]), m1516a);
        let m1516a = _mm256_fmadd_pd(x4p27, _mm256_set1_pd(self.twiddles_re[1]), m1516a);
        let m1516a = _mm256_fmadd_pd(x5p26, _mm256_set1_pd(self.twiddles_re[12]), m1516a);
        let m1516a = _mm256_fmadd_pd(x6p25, _mm256_set1_pd(self.twiddles_re[2]), m1516a);
        let m1516a = _mm256_fmadd_pd(x7p24, _mm256_set1_pd(self.twiddles_re[11]), m1516a);
        let m1516a = _mm256_fmadd_pd(x8p23, _mm256_set1_pd(self.twiddles_re[3]), m1516a);
        let m1516a = _mm256_fmadd_pd(x9p22, _mm256_set1_pd(self.twiddles_re[10]), m1516a);
        let m1516a = _mm256_fmadd_pd(x10p21, _mm256_set1_pd(self.twiddles_re[4]), m1516a);
        let m1516a = _mm256_fmadd_pd(x11p20, _mm256_set1_pd(self.twiddles_re[9]), m1516a);
        let m1516a = _mm256_fmadd_pd(x12p19, _mm256_set1_pd(self.twiddles_re[5]), m1516a);
        let m1516a = _mm256_fmadd_pd(x13p18, _mm256_set1_pd(self.twiddles_re[8]), m1516a);
        let m1516a = _mm256_fmadd_pd(x14p17, _mm256_set1_pd(self.twiddles_re[6]), m1516a);
        let m1516a = _mm256_fmadd_pd(x15p16, _mm256_set1_pd(self.twiddles_re[7]), m1516a);
        let m1516b = _mm256_mul_pd(x1m30, _mm256_set1_pd(self.twiddles_im[14]));
        let m1516b = _mm256_fnmadd_pd(x2m29, _mm256_set1_pd(self.twiddles_im[0]), m1516b);
        let m1516b = _mm256_fmadd_pd(x3m28, _mm256_set1_pd(self.twiddles_im[13]), m1516b);
        let m1516b = _mm256_fnmadd_pd(x4m27, _mm256_set1_pd(self.twiddles_im[1]), m1516b);
        let m1516b = _mm256_fmadd_pd(x5m26, _mm256_set1_pd(self.twiddles_im[12]), m1516b);
        let m1516b = _mm256_fnmadd_pd(x6m25, _mm256_set1_pd(self.twiddles_im[2]), m1516b);
        let m1516b = _mm256_fmadd_pd(x7m24, _mm256_set1_pd(self.twiddles_im[11]), m1516b);
        let m1516b = _mm256_fnmadd_pd(x8m23, _mm256_set1_pd(self.twiddles_im[3]), m1516b);
        let m1516b = _mm256_fmadd_pd(x9m22, _mm256_set1_pd(self.twiddles_im[10]), m1516b);
        let m1516b = _mm256_fnmadd_pd(x10m21, _mm256_set1_pd(self.twiddles_im[4]), m1516b);
        let m1516b = _mm256_fmadd_pd(x11m20, _mm256_set1_pd(self.twiddles_im[9]), m1516b);
        let m1516b = _mm256_fnmadd_pd(x12m19, _mm256_set1_pd(self.twiddles_im[5]), m1516b);
        let m1516b = _mm256_fmadd_pd(x13m18, _mm256_set1_pd(self.twiddles_im[8]), m1516b);
        let m1516b = _mm256_fnmadd_pd(x14m17, _mm256_set1_pd(self.twiddles_im[6]), m1516b);
        let m1516b = _mm256_fmadd_pd(x15m16, _mm256_set1_pd(self.twiddles_im[7]), m1516b);
        let (y15, y16) = AvxButterfly::butterfly2_f64(m1516a, m1516b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15, y16,
            y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30,
        ]
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 31 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(62) {
                let (u0, u1) = shift_load2dd!(chunk, 31, 0);
                let (u2, u3) = shift_load2dd!(chunk, 31, 2);
                let (u4, u5) = shift_load2dd!(chunk, 31, 4);
                let (u6, u7) = shift_load2dd!(chunk, 31, 6);
                let (u8, u9) = shift_load2dd!(chunk, 31, 8);
                let (u10, u11) = shift_load2dd!(chunk, 31, 10);
                let (u12, u13) = shift_load2dd!(chunk, 31, 12);
                let (u14, u15) = shift_load2dd!(chunk, 31, 14);
                let (u16, u17) = shift_load2dd!(chunk, 31, 16);
                let (u18, u19) = shift_load2dd!(chunk, 31, 18);
                let (u20, u21) = shift_load2dd!(chunk, 31, 20);
                let (u22, u23) = shift_load2dd!(chunk, 31, 22);
                let (u24, u25) = shift_load2dd!(chunk, 31, 24);
                let (u26, u27) = shift_load2dd!(chunk, 31, 26);
                let (u28, u29) = shift_load2dd!(chunk, 31, 28);
                let u30 = shift_load2d!(chunk, 31, 30);

                let q = self.kernel_f64([
                    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17,
                    u18, u19, u20, u21, u22, u23, u24, u25, u26, u27, u28, u29, u30,
                ]);

                shift_store2dd!(chunk, 31, 0, q[0], q[1]);
                shift_store2dd!(chunk, 31, 2, q[2], q[3]);
                shift_store2dd!(chunk, 31, 4, q[4], q[5]);
                shift_store2dd!(chunk, 31, 6, q[6], q[7]);
                shift_store2dd!(chunk, 31, 8, q[8], q[9]);
                shift_store2dd!(chunk, 31, 10, q[10], q[11]);
                shift_store2dd!(chunk, 31, 12, q[12], q[13]);
                shift_store2dd!(chunk, 31, 14, q[14], q[15]);
                shift_store2dd!(chunk, 31, 16, q[16], q[17]);
                shift_store2dd!(chunk, 31, 18, q[18], q[19]);
                shift_store2dd!(chunk, 31, 20, q[20], q[21]);
                shift_store2dd!(chunk, 31, 22, q[22], q[23]);
                shift_store2dd!(chunk, 31, 24, q[24], q[25]);
                shift_store2dd!(chunk, 31, 26, q[26], q[27]);
                shift_store2dd!(chunk, 31, 28, q[28], q[29]);
                shift_store2d!(chunk, 31, 30, q[30]);
            }

            let rem = in_place.chunks_exact_mut(62).into_remainder();

            for chunk in rem.chunks_exact_mut(31) {
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
                let u28u29 = _mm256_loadu_pd(chunk.get_unchecked(28..).as_ptr().cast());
                let u30 = _mm_loadu_pd(chunk.get_unchecked(30..).as_ptr().cast());

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
                    u28u29,
                    _mm256_permute2f128_pd::<HI_LO>(u28u29, u28u29),
                    _mm256_castpd128_pd256(u30),
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
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(q[28], q[29]),
                );
                _mm_storeu_pd(
                    chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    _mm256_castpd256_pd128(q[30]),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly31<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        31
    }
}

impl AvxButterfly31<f32> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    fn kernel_f32(&self, v: [__m128; 31]) -> [__m128; 31] {
        let y00 = v[0];
        let (x1p30, x1m30) = AvxButterfly::butterfly2_f32_m128(v[1], v[30]);
        let x1m30 = self.rotate.rotate_m128(x1m30);
        let y00 = _mm_add_ps(y00, x1p30);
        let (x2p29, x2m29) = AvxButterfly::butterfly2_f32_m128(v[2], v[29]);
        let x2m29 = self.rotate.rotate_m128(x2m29);
        let y00 = _mm_add_ps(y00, x2p29);
        let (x3p28, x3m28) = AvxButterfly::butterfly2_f32_m128(v[3], v[28]);
        let x3m28 = self.rotate.rotate_m128(x3m28);
        let y00 = _mm_add_ps(y00, x3p28);
        let (x4p27, x4m27) = AvxButterfly::butterfly2_f32_m128(v[4], v[27]);
        let x4m27 = self.rotate.rotate_m128(x4m27);
        let y00 = _mm_add_ps(y00, x4p27);
        let (x5p26, x5m26) = AvxButterfly::butterfly2_f32_m128(v[5], v[26]);
        let x5m26 = self.rotate.rotate_m128(x5m26);
        let y00 = _mm_add_ps(y00, x5p26);
        let (x6p25, x6m25) = AvxButterfly::butterfly2_f32_m128(v[6], v[25]);
        let x6m25 = self.rotate.rotate_m128(x6m25);
        let y00 = _mm_add_ps(y00, x6p25);
        let (x7p24, x7m24) = AvxButterfly::butterfly2_f32_m128(v[7], v[24]);
        let x7m24 = self.rotate.rotate_m128(x7m24);
        let y00 = _mm_add_ps(y00, x7p24);
        let (x8p23, x8m23) = AvxButterfly::butterfly2_f32_m128(v[8], v[23]);
        let x8m23 = self.rotate.rotate_m128(x8m23);
        let y00 = _mm_add_ps(y00, x8p23);
        let (x9p22, x9m22) = AvxButterfly::butterfly2_f32_m128(v[9], v[22]);
        let x9m22 = self.rotate.rotate_m128(x9m22);
        let y00 = _mm_add_ps(y00, x9p22);
        let (x10p21, x10m21) = AvxButterfly::butterfly2_f32_m128(v[10], v[21]);
        let x10m21 = self.rotate.rotate_m128(x10m21);
        let y00 = _mm_add_ps(y00, x10p21);
        let (x11p20, x11m20) = AvxButterfly::butterfly2_f32_m128(v[11], v[20]);
        let x11m20 = self.rotate.rotate_m128(x11m20);
        let y00 = _mm_add_ps(y00, x11p20);
        let (x12p19, x12m19) = AvxButterfly::butterfly2_f32_m128(v[12], v[19]);
        let x12m19 = self.rotate.rotate_m128(x12m19);
        let y00 = _mm_add_ps(y00, x12p19);
        let (x13p18, x13m18) = AvxButterfly::butterfly2_f32_m128(v[13], v[18]);
        let x13m18 = self.rotate.rotate_m128(x13m18);
        let y00 = _mm_add_ps(y00, x13p18);
        let (x14p17, x14m17) = AvxButterfly::butterfly2_f32_m128(v[14], v[17]);
        let x14m17 = self.rotate.rotate_m128(x14m17);
        let y00 = _mm_add_ps(y00, x14p17);
        let (x15p16, x15m16) = AvxButterfly::butterfly2_f32_m128(v[15], v[16]);
        let x15m16 = self.rotate.rotate_m128(x15m16);
        let y00 = _mm_add_ps(y00, x15p16);

        let m0130a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[0]), v[0]);
        let m0130a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[1]), m0130a);
        let m0130a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[2]), m0130a);
        let m0130a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[3]), m0130a);
        let m0130a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[4]), m0130a);
        let m0130a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[5]), m0130a);
        let m0130a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[6]), m0130a);
        let m0130a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[7]), m0130a);
        let m0130a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[8]), m0130a);
        let m0130a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[9]), m0130a);
        let m0130a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[10]), m0130a);
        let m0130a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[11]), m0130a);
        let m0130a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[12]), m0130a);
        let m0130a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[13]), m0130a);
        let m0130a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[14]), m0130a);
        let m0130b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[0]));
        let m0130b = _mm_fmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[1]), m0130b);
        let m0130b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[2]), m0130b);
        let m0130b = _mm_fmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[3]), m0130b);
        let m0130b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[4]), m0130b);
        let m0130b = _mm_fmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[5]), m0130b);
        let m0130b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[6]), m0130b);
        let m0130b = _mm_fmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[7]), m0130b);
        let m0130b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[8]), m0130b);
        let m0130b = _mm_fmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[9]), m0130b);
        let m0130b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[10]), m0130b);
        let m0130b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[11]), m0130b);
        let m0130b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[12]), m0130b);
        let m0130b = _mm_fmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[13]), m0130b);
        let m0130b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[14]), m0130b);
        let (y01, y30) = AvxButterfly::butterfly2_f32_m128(m0130a, m0130b);

        let m0229a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[1]), v[0]);
        let m0229a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[3]), m0229a);
        let m0229a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[5]), m0229a);
        let m0229a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[7]), m0229a);
        let m0229a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[9]), m0229a);
        let m0229a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[11]), m0229a);
        let m0229a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[13]), m0229a);
        let m0229a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[14]), m0229a);
        let m0229a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[12]), m0229a);
        let m0229a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[10]), m0229a);
        let m0229a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[8]), m0229a);
        let m0229a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[6]), m0229a);
        let m0229a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[4]), m0229a);
        let m0229a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[2]), m0229a);
        let m0229a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[0]), m0229a);
        let m0229b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[1]));
        let m0229b = _mm_fmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[3]), m0229b);
        let m0229b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[5]), m0229b);
        let m0229b = _mm_fmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[7]), m0229b);
        let m0229b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[9]), m0229b);
        let m0229b = _mm_fmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[11]), m0229b);
        let m0229b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[13]), m0229b);
        let m0229b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[14]), m0229b);
        let m0229b = _mm_fnmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[12]), m0229b);
        let m0229b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[10]), m0229b);
        let m0229b = _mm_fnmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[8]), m0229b);
        let m0229b = _mm_fnmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[6]), m0229b);
        let m0229b = _mm_fnmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[4]), m0229b);
        let m0229b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[2]), m0229b);
        let m0229b = _mm_fnmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[0]), m0229b);
        let (y02, y29) = AvxButterfly::butterfly2_f32_m128(m0229a, m0229b);

        let m0328a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[2]), v[0]);
        let m0328a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[5]), m0328a);
        let m0328a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[8]), m0328a);
        let m0328a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[11]), m0328a);
        let m0328a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[14]), m0328a);
        let m0328a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[12]), m0328a);
        let m0328a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[9]), m0328a);
        let m0328a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[6]), m0328a);
        let m0328a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[3]), m0328a);
        let m0328a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[0]), m0328a);
        let m0328a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[1]), m0328a);
        let m0328a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[4]), m0328a);
        let m0328a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[7]), m0328a);
        let m0328a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[10]), m0328a);
        let m0328a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[13]), m0328a);
        let m0328b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[2]));
        let m0328b = _mm_fmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[5]), m0328b);
        let m0328b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[8]), m0328b);
        let m0328b = _mm_fmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[11]), m0328b);
        let m0328b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[14]), m0328b);
        let m0328b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[12]), m0328b);
        let m0328b = _mm_fnmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[9]), m0328b);
        let m0328b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[6]), m0328b);
        let m0328b = _mm_fnmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[3]), m0328b);
        let m0328b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[0]), m0328b);
        let m0328b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[1]), m0328b);
        let m0328b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[4]), m0328b);
        let m0328b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[7]), m0328b);
        let m0328b = _mm_fmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[10]), m0328b);
        let m0328b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[13]), m0328b);
        let (y03, y28) = AvxButterfly::butterfly2_f32_m128(m0328a, m0328b);

        let m0427a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[3]), v[0]);
        let m0427a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[7]), m0427a);
        let m0427a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[11]), m0427a);
        let m0427a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[14]), m0427a);
        let m0427a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[10]), m0427a);
        let m0427a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[6]), m0427a);
        let m0427a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[2]), m0427a);
        let m0427a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[0]), m0427a);
        let m0427a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[4]), m0427a);
        let m0427a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[8]), m0427a);
        let m0427a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[12]), m0427a);
        let m0427a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[13]), m0427a);
        let m0427a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[9]), m0427a);
        let m0427a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[5]), m0427a);
        let m0427a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[1]), m0427a);
        let m0427b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[3]));
        let m0427b = _mm_fmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[7]), m0427b);
        let m0427b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[11]), m0427b);
        let m0427b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[14]), m0427b);
        let m0427b = _mm_fnmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[10]), m0427b);
        let m0427b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[6]), m0427b);
        let m0427b = _mm_fnmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[2]), m0427b);
        let m0427b = _mm_fmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[0]), m0427b);
        let m0427b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[4]), m0427b);
        let m0427b = _mm_fmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[8]), m0427b);
        let m0427b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[12]), m0427b);
        let m0427b = _mm_fnmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[13]), m0427b);
        let m0427b = _mm_fnmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[9]), m0427b);
        let m0427b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[5]), m0427b);
        let m0427b = _mm_fnmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[1]), m0427b);
        let (y04, y27) = AvxButterfly::butterfly2_f32_m128(m0427a, m0427b);

        let m0526a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[4]), v[0]);
        let m0526a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[9]), m0526a);
        let m0526a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[14]), m0526a);
        let m0526a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[10]), m0526a);
        let m0526a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[5]), m0526a);
        let m0526a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[0]), m0526a);
        let m0526a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[3]), m0526a);
        let m0526a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[8]), m0526a);
        let m0526a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[13]), m0526a);
        let m0526a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[11]), m0526a);
        let m0526a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[6]), m0526a);
        let m0526a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[1]), m0526a);
        let m0526a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[2]), m0526a);
        let m0526a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[7]), m0526a);
        let m0526a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[12]), m0526a);
        let m0526b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[4]));
        let m0526b = _mm_fmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[9]), m0526b);
        let m0526b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[14]), m0526b);
        let m0526b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[10]), m0526b);
        let m0526b = _mm_fnmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[5]), m0526b);
        let m0526b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[0]), m0526b);
        let m0526b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[3]), m0526b);
        let m0526b = _mm_fmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[8]), m0526b);
        let m0526b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[13]), m0526b);
        let m0526b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[11]), m0526b);
        let m0526b = _mm_fnmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[6]), m0526b);
        let m0526b = _mm_fnmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[1]), m0526b);
        let m0526b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[2]), m0526b);
        let m0526b = _mm_fmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[7]), m0526b);
        let m0526b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[12]), m0526b);
        let (y05, y26) = AvxButterfly::butterfly2_f32_m128(m0526a, m0526b);

        let m0625a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[5]), v[0]);
        let m0625a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[11]), m0625a);
        let m0625a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[12]), m0625a);
        let m0625a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[6]), m0625a);
        let m0625a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[0]), m0625a);
        let m0625a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[4]), m0625a);
        let m0625a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[10]), m0625a);
        let m0625a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[13]), m0625a);
        let m0625a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[7]), m0625a);
        let m0625a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[1]), m0625a);
        let m0625a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[3]), m0625a);
        let m0625a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[9]), m0625a);
        let m0625a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[14]), m0625a);
        let m0625a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[8]), m0625a);
        let m0625a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[2]), m0625a);
        let m0625b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[5]));
        let m0625b = _mm_fmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[11]), m0625b);
        let m0625b = _mm_fnmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[12]), m0625b);
        let m0625b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[6]), m0625b);
        let m0625b = _mm_fnmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[0]), m0625b);
        let m0625b = _mm_fmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[4]), m0625b);
        let m0625b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[10]), m0625b);
        let m0625b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[13]), m0625b);
        let m0625b = _mm_fnmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[7]), m0625b);
        let m0625b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[1]), m0625b);
        let m0625b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[3]), m0625b);
        let m0625b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[9]), m0625b);
        let m0625b = _mm_fnmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[14]), m0625b);
        let m0625b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[8]), m0625b);
        let m0625b = _mm_fnmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[2]), m0625b);
        let (y06, y25) = AvxButterfly::butterfly2_f32_m128(m0625a, m0625b);

        let m0724a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[6]), v[0]);
        let m0724a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[13]), m0724a);
        let m0724a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[9]), m0724a);
        let m0724a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[2]), m0724a);
        let m0724a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[3]), m0724a);
        let m0724a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[10]), m0724a);
        let m0724a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[12]), m0724a);
        let m0724a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[5]), m0724a);
        let m0724a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[0]), m0724a);
        let m0724a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[7]), m0724a);
        let m0724a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[14]), m0724a);
        let m0724a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[8]), m0724a);
        let m0724a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[1]), m0724a);
        let m0724a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[4]), m0724a);
        let m0724a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[11]), m0724a);
        let m0724b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[6]));
        let m0724b = _mm_fmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[13]), m0724b);
        let m0724b = _mm_fnmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[9]), m0724b);
        let m0724b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[2]), m0724b);
        let m0724b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[3]), m0724b);
        let m0724b = _mm_fmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[10]), m0724b);
        let m0724b = _mm_fnmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[12]), m0724b);
        let m0724b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[5]), m0724b);
        let m0724b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[0]), m0724b);
        let m0724b = _mm_fmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[7]), m0724b);
        let m0724b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[14]), m0724b);
        let m0724b = _mm_fnmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[8]), m0724b);
        let m0724b = _mm_fnmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[1]), m0724b);
        let m0724b = _mm_fmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[4]), m0724b);
        let m0724b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[11]), m0724b);
        let (y07, y24) = AvxButterfly::butterfly2_f32_m128(m0724a, m0724b);

        let m0823a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[7]), v[0]);
        let m0823a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[14]), m0823a);
        let m0823a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[6]), m0823a);
        let m0823a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[0]), m0823a);
        let m0823a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[8]), m0823a);
        let m0823a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[13]), m0823a);
        let m0823a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[5]), m0823a);
        let m0823a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[1]), m0823a);
        let m0823a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[9]), m0823a);
        let m0823a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[12]), m0823a);
        let m0823a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[4]), m0823a);
        let m0823a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[2]), m0823a);
        let m0823a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[10]), m0823a);
        let m0823a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[11]), m0823a);
        let m0823a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[3]), m0823a);
        let m0823b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[7]));
        let m0823b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[14]), m0823b);
        let m0823b = _mm_fnmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[6]), m0823b);
        let m0823b = _mm_fmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[0]), m0823b);
        let m0823b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[8]), m0823b);
        let m0823b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[13]), m0823b);
        let m0823b = _mm_fnmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[5]), m0823b);
        let m0823b = _mm_fmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[1]), m0823b);
        let m0823b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[9]), m0823b);
        let m0823b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[12]), m0823b);
        let m0823b = _mm_fnmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[4]), m0823b);
        let m0823b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[2]), m0823b);
        let m0823b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[10]), m0823b);
        let m0823b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[11]), m0823b);
        let m0823b = _mm_fnmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[3]), m0823b);
        let (y08, y23) = AvxButterfly::butterfly2_f32_m128(m0823a, m0823b);

        let m0922a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[8]), v[0]);
        let m0922a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[12]), m0922a);
        let m0922a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[3]), m0922a);
        let m0922a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[4]), m0922a);
        let m0922a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[13]), m0922a);
        let m0922a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[7]), m0922a);
        let m0922a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[0]), m0922a);
        let m0922a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[9]), m0922a);
        let m0922a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[11]), m0922a);
        let m0922a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[2]), m0922a);
        let m0922a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[5]), m0922a);
        let m0922a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[14]), m0922a);
        let m0922a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[6]), m0922a);
        let m0922a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[1]), m0922a);
        let m0922a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[10]), m0922a);
        let m0922b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[8]));
        let m0922b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[12]), m0922b);
        let m0922b = _mm_fnmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[3]), m0922b);
        let m0922b = _mm_fmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[4]), m0922b);
        let m0922b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[13]), m0922b);
        let m0922b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[7]), m0922b);
        let m0922b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[0]), m0922b);
        let m0922b = _mm_fmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[9]), m0922b);
        let m0922b = _mm_fnmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[11]), m0922b);
        let m0922b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[2]), m0922b);
        let m0922b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[5]), m0922b);
        let m0922b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[14]), m0922b);
        let m0922b = _mm_fnmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[6]), m0922b);
        let m0922b = _mm_fmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[1]), m0922b);
        let m0922b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[10]), m0922b);
        let (y09, y22) = AvxButterfly::butterfly2_f32_m128(m0922a, m0922b);

        let m1021a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[9]), v[0]);
        let m1021a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[10]), m1021a);
        let m1021a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[0]), m1021a);
        let m1021a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[8]), m1021a);
        let m1021a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[11]), m1021a);
        let m1021a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[1]), m1021a);
        let m1021a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[7]), m1021a);
        let m1021a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[12]), m1021a);
        let m1021a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[2]), m1021a);
        let m1021a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[6]), m1021a);
        let m1021a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[13]), m1021a);
        let m1021a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[3]), m1021a);
        let m1021a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[5]), m1021a);
        let m1021a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[14]), m1021a);
        let m1021a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[4]), m1021a);
        let m1021b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[9]));
        let m1021b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[10]), m1021b);
        let m1021b = _mm_fnmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[0]), m1021b);
        let m1021b = _mm_fmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[8]), m1021b);
        let m1021b = _mm_fnmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[11]), m1021b);
        let m1021b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[1]), m1021b);
        let m1021b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[7]), m1021b);
        let m1021b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[12]), m1021b);
        let m1021b = _mm_fnmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[2]), m1021b);
        let m1021b = _mm_fmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[6]), m1021b);
        let m1021b = _mm_fnmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[13]), m1021b);
        let m1021b = _mm_fnmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[3]), m1021b);
        let m1021b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[5]), m1021b);
        let m1021b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[14]), m1021b);
        let m1021b = _mm_fnmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[4]), m1021b);
        let (y10, y21) = AvxButterfly::butterfly2_f32_m128(m1021a, m1021b);

        let m1120a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[10]), v[0]);
        let m1120a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[8]), m1120a);
        let m1120a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[1]), m1120a);
        let m1120a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[12]), m1120a);
        let m1120a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[6]), m1120a);
        let m1120a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[3]), m1120a);
        let m1120a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[14]), m1120a);
        let m1120a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[4]), m1120a);
        let m1120a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[5]), m1120a);
        let m1120a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[13]), m1120a);
        let m1120a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[2]), m1120a);
        let m1120a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[7]), m1120a);
        let m1120a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[11]), m1120a);
        let m1120a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[0]), m1120a);
        let m1120a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[9]), m1120a);
        let m1120b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[10]));
        let m1120b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[8]), m1120b);
        let m1120b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[1]), m1120b);
        let m1120b = _mm_fmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[12]), m1120b);
        let m1120b = _mm_fnmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[6]), m1120b);
        let m1120b = _mm_fmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[3]), m1120b);
        let m1120b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[14]), m1120b);
        let m1120b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[4]), m1120b);
        let m1120b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[5]), m1120b);
        let m1120b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[13]), m1120b);
        let m1120b = _mm_fnmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[2]), m1120b);
        let m1120b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[7]), m1120b);
        let m1120b = _mm_fnmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[11]), m1120b);
        let m1120b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[0]), m1120b);
        let m1120b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[9]), m1120b);
        let (y11, y20) = AvxButterfly::butterfly2_f32_m128(m1120a, m1120b);

        let m1219a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[11]), v[0]);
        let m1219a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[6]), m1219a);
        let m1219a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[4]), m1219a);
        let m1219a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[13]), m1219a);
        let m1219a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[1]), m1219a);
        let m1219a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[9]), m1219a);
        let m1219a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[8]), m1219a);
        let m1219a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[2]), m1219a);
        let m1219a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[14]), m1219a);
        let m1219a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[3]), m1219a);
        let m1219a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[7]), m1219a);
        let m1219a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[10]), m1219a);
        let m1219a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[0]), m1219a);
        let m1219a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[12]), m1219a);
        let m1219a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[5]), m1219a);
        let m1219b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[11]));
        let m1219b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[6]), m1219b);
        let m1219b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[4]), m1219b);
        let m1219b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[13]), m1219b);
        let m1219b = _mm_fnmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[1]), m1219b);
        let m1219b = _mm_fmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[9]), m1219b);
        let m1219b = _mm_fnmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[8]), m1219b);
        let m1219b = _mm_fmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[2]), m1219b);
        let m1219b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[14]), m1219b);
        let m1219b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[3]), m1219b);
        let m1219b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[7]), m1219b);
        let m1219b = _mm_fnmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[10]), m1219b);
        let m1219b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[0]), m1219b);
        let m1219b = _mm_fmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[12]), m1219b);
        let m1219b = _mm_fnmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[5]), m1219b);
        let (y12, y19) = AvxButterfly::butterfly2_f32_m128(m1219a, m1219b);

        let m1318a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[12]), v[0]);
        let m1318a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[4]), m1318a);
        let m1318a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[7]), m1318a);
        let m1318a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[9]), m1318a);
        let m1318a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[2]), m1318a);
        let m1318a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[14]), m1318a);
        let m1318a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[1]), m1318a);
        let m1318a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[10]), m1318a);
        let m1318a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[6]), m1318a);
        let m1318a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[5]), m1318a);
        let m1318a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[11]), m1318a);
        let m1318a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[0]), m1318a);
        let m1318a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[13]), m1318a);
        let m1318a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[3]), m1318a);
        let m1318a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[8]), m1318a);
        let m1318b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[12]));
        let m1318b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[4]), m1318b);
        let m1318b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[7]), m1318b);
        let m1318b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[9]), m1318b);
        let m1318b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[2]), m1318b);
        let m1318b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[14]), m1318b);
        let m1318b = _mm_fnmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[1]), m1318b);
        let m1318b = _mm_fmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[10]), m1318b);
        let m1318b = _mm_fnmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[6]), m1318b);
        let m1318b = _mm_fmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[5]), m1318b);
        let m1318b = _mm_fnmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[11]), m1318b);
        let m1318b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[0]), m1318b);
        let m1318b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[13]), m1318b);
        let m1318b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[3]), m1318b);
        let m1318b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[8]), m1318b);
        let (y13, y18) = AvxButterfly::butterfly2_f32_m128(m1318a, m1318b);

        let m1417a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[13]), v[0]);
        let m1417a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[2]), m1417a);
        let m1417a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[10]), m1417a);
        let m1417a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[5]), m1417a);
        let m1417a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[7]), m1417a);
        let m1417a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[8]), m1417a);
        let m1417a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[4]), m1417a);
        let m1417a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[11]), m1417a);
        let m1417a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[1]), m1417a);
        let m1417a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[14]), m1417a);
        let m1417a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[0]), m1417a);
        let m1417a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[12]), m1417a);
        let m1417a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[3]), m1417a);
        let m1417a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[9]), m1417a);
        let m1417a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[6]), m1417a);
        let m1417b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[13]));
        let m1417b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[2]), m1417b);
        let m1417b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[10]), m1417b);
        let m1417b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[5]), m1417b);
        let m1417b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[7]), m1417b);
        let m1417b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[8]), m1417b);
        let m1417b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[4]), m1417b);
        let m1417b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[11]), m1417b);
        let m1417b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[1]), m1417b);
        let m1417b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[14]), m1417b);
        let m1417b = _mm_fnmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[0]), m1417b);
        let m1417b = _mm_fmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[12]), m1417b);
        let m1417b = _mm_fnmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[3]), m1417b);
        let m1417b = _mm_fmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[9]), m1417b);
        let m1417b = _mm_fnmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[6]), m1417b);
        let (y14, y17) = AvxButterfly::butterfly2_f32_m128(m1417a, m1417b);

        let m1516a = _mm_fmadd_ps(x1p30, _mm_set1_ps(self.twiddles_re[14]), v[0]);
        let m1516a = _mm_fmadd_ps(x2p29, _mm_set1_ps(self.twiddles_re[0]), m1516a);
        let m1516a = _mm_fmadd_ps(x3p28, _mm_set1_ps(self.twiddles_re[13]), m1516a);
        let m1516a = _mm_fmadd_ps(x4p27, _mm_set1_ps(self.twiddles_re[1]), m1516a);
        let m1516a = _mm_fmadd_ps(x5p26, _mm_set1_ps(self.twiddles_re[12]), m1516a);
        let m1516a = _mm_fmadd_ps(x6p25, _mm_set1_ps(self.twiddles_re[2]), m1516a);
        let m1516a = _mm_fmadd_ps(x7p24, _mm_set1_ps(self.twiddles_re[11]), m1516a);
        let m1516a = _mm_fmadd_ps(x8p23, _mm_set1_ps(self.twiddles_re[3]), m1516a);
        let m1516a = _mm_fmadd_ps(x9p22, _mm_set1_ps(self.twiddles_re[10]), m1516a);
        let m1516a = _mm_fmadd_ps(x10p21, _mm_set1_ps(self.twiddles_re[4]), m1516a);
        let m1516a = _mm_fmadd_ps(x11p20, _mm_set1_ps(self.twiddles_re[9]), m1516a);
        let m1516a = _mm_fmadd_ps(x12p19, _mm_set1_ps(self.twiddles_re[5]), m1516a);
        let m1516a = _mm_fmadd_ps(x13p18, _mm_set1_ps(self.twiddles_re[8]), m1516a);
        let m1516a = _mm_fmadd_ps(x14p17, _mm_set1_ps(self.twiddles_re[6]), m1516a);
        let m1516a = _mm_fmadd_ps(x15p16, _mm_set1_ps(self.twiddles_re[7]), m1516a);
        let m1516b = _mm_mul_ps(x1m30, _mm_set1_ps(self.twiddles_im[14]));
        let m1516b = _mm_fnmadd_ps(x2m29, _mm_set1_ps(self.twiddles_im[0]), m1516b);
        let m1516b = _mm_fmadd_ps(x3m28, _mm_set1_ps(self.twiddles_im[13]), m1516b);
        let m1516b = _mm_fnmadd_ps(x4m27, _mm_set1_ps(self.twiddles_im[1]), m1516b);
        let m1516b = _mm_fmadd_ps(x5m26, _mm_set1_ps(self.twiddles_im[12]), m1516b);
        let m1516b = _mm_fnmadd_ps(x6m25, _mm_set1_ps(self.twiddles_im[2]), m1516b);
        let m1516b = _mm_fmadd_ps(x7m24, _mm_set1_ps(self.twiddles_im[11]), m1516b);
        let m1516b = _mm_fnmadd_ps(x8m23, _mm_set1_ps(self.twiddles_im[3]), m1516b);
        let m1516b = _mm_fmadd_ps(x9m22, _mm_set1_ps(self.twiddles_im[10]), m1516b);
        let m1516b = _mm_fnmadd_ps(x10m21, _mm_set1_ps(self.twiddles_im[4]), m1516b);
        let m1516b = _mm_fmadd_ps(x11m20, _mm_set1_ps(self.twiddles_im[9]), m1516b);
        let m1516b = _mm_fnmadd_ps(x12m19, _mm_set1_ps(self.twiddles_im[5]), m1516b);
        let m1516b = _mm_fmadd_ps(x13m18, _mm_set1_ps(self.twiddles_im[8]), m1516b);
        let m1516b = _mm_fnmadd_ps(x14m17, _mm_set1_ps(self.twiddles_im[6]), m1516b);
        let m1516b = _mm_fmadd_ps(x15m16, _mm_set1_ps(self.twiddles_im[7]), m1516b);
        let (y15, y16) = AvxButterfly::butterfly2_f32_m128(m1516a, m1516b);

        [
            y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15, y16,
            y17, y18, y19, y20, y21, y22, y23, y24, y25, y26, y27, y28, y29, y30,
        ]
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 31 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(62) {
                let (u0, u1, u2, u3) = shift_load4!(chunk, 31, 0);
                let (u4, u5, u6, u7) = shift_load4!(chunk, 31, 4);
                let (u8, u9, u10, u11) = shift_load4!(chunk, 31, 8);
                let (u12, u13, u14, u15) = shift_load4!(chunk, 31, 12);
                let (u16, u17, u18, u19) = shift_load4!(chunk, 31, 16);
                let (u20, u21, u22, u23) = shift_load4!(chunk, 31, 20);
                let (u24, u25, u26, u27) = shift_load4!(chunk, 31, 24);
                let (_, u28, u29, u30) = shift_load4!(chunk, 31, 27);

                let q = self.kernel_f32([
                    u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17,
                    u18, u19, u20, u21, u22, u23, u24, u25, u26, u27, u28, u29, u30,
                ]);

                shift_store4!(chunk, 31, 0, q[0], q[1], q[2], q[3]);
                shift_store4!(chunk, 31, 4, q[4], q[5], q[6], q[7]);
                shift_store4!(chunk, 31, 8, q[8], q[9], q[10], q[11]);
                shift_store4!(chunk, 31, 12, q[12], q[13], q[14], q[15]);
                shift_store4!(chunk, 31, 16, q[16], q[17], q[18], q[19]);
                shift_store4!(chunk, 31, 20, q[20], q[21], q[22], q[23]);
                shift_store4!(chunk, 31, 24, q[24], q[25], q[26], q[27]);
                shift_store4!(chunk, 31, 27, q[27], q[28], q[29], q[30]);
            }

            let rem = in_place.chunks_exact_mut(62).into_remainder();

            for chunk in rem.chunks_exact_mut(31) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16u17u18u19 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());
                let u20u21u22u23 = _mm256_loadu_ps(chunk.get_unchecked(20..).as_ptr().cast());
                let u24u25u26u27 = _mm256_loadu_ps(chunk.get_unchecked(24..).as_ptr().cast());
                let u27u28u29u30 = _mm256_loadu_ps(chunk.get_unchecked(27..).as_ptr().cast());

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
                let u27u28 = _mm256_castps256_ps128(u27u28u29u30);
                let u29u30 = _mm256_extractf128_ps::<1>(u27u28u29u30);

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
                    _mm_unpackhi_ps64(u27u28, u27u28),
                    u29u30,
                    _mm_unpackhi_ps64(u29u30, u29u30),
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

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(27..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[27], q[28]),
                        _mm_unpacklo_ps64(q[29], q[30]),
                    ),
                );
            }
        }

        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly31<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        31
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;
    use rand::Rng;

    test_avx_butterfly!(test_avx_butterfly31, f32, AvxButterfly31, 31, 1e-4);
    test_avx_butterfly!(test_avx_butterfly31_f64, f64, AvxButterfly31, 31, 1e-7);
}
