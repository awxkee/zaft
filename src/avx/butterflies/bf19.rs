// Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::avx::butterflies::AvxButterfly;
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd,
    _mm256_create_ps,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly19<T> {
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
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly19<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            rotate: unsafe { AvxRotate::new(FftDirection::Inverse) },
            twiddle1: compute_twiddle(1, 19, fft_direction),
            twiddle2: compute_twiddle(2, 19, fft_direction),
            twiddle3: compute_twiddle(3, 19, fft_direction),
            twiddle4: compute_twiddle(4, 19, fft_direction),
            twiddle5: compute_twiddle(5, 19, fft_direction),
            twiddle6: compute_twiddle(6, 19, fft_direction),
            twiddle7: compute_twiddle(7, 19, fft_direction),
            twiddle8: compute_twiddle(8, 19, fft_direction),
            twiddle9: compute_twiddle(9, 19, fft_direction),
        }
    }
}

impl AvxButterfly19<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 19 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(19) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = _mm256_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());
                let u18 = _mm_loadu_pd(chunk.get_unchecked(18..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let y00 = _mm256_castpd256_pd128(u0u1);
                let (x1p18, x1m18) =
                    AvxButterfly::butterfly2_f64_m128(_mm256_extractf128_pd::<1>(u0u1), u18);
                let x1m18 = self.rotate.rotate_m128d(x1m18);
                let y00 = _mm_add_pd(y00, x1p18);
                let (x2p17, x2m17) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u16u17),
                );
                let x2m17 = self.rotate.rotate_m128d(x2m17);
                let y00 = _mm_add_pd(y00, x2p17);
                let (x3p16, x3m16) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u16u17),
                );
                let x3m16 = self.rotate.rotate_m128d(x3m16);
                let y00 = _mm_add_pd(y00, x3p16);
                let (x4p15, x4m15) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u14u15),
                );
                let x4m15 = self.rotate.rotate_m128d(x4m15);
                let y00 = _mm_add_pd(y00, x4p15);
                let (x5p14, x5m14) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_castpd256_pd128(u14u15),
                );
                let x5m14 = self.rotate.rotate_m128d(x5m14);
                let y00 = _mm_add_pd(y00, x5p14);
                let (x6p13, x6m13) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u12u13),
                );
                let x6m13 = self.rotate.rotate_m128d(x6m13);
                let y00 = _mm_add_pd(y00, x6p13);
                let (x7p12, x7m12) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u6u7),
                    _mm256_castpd256_pd128(u12u13),
                );
                let x7m12 = self.rotate.rotate_m128d(x7m12);
                let y00 = _mm_add_pd(y00, x7p12);
                let (x8p11, x8m11) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u8u9),
                    _mm256_extractf128_pd::<1>(u10u11),
                );
                let x8m11 = self.rotate.rotate_m128d(x8m11);
                let y00 = _mm_add_pd(y00, x8p11);
                let (x9p10, x9m10) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u8u9),
                    _mm256_castpd256_pd128(u10u11),
                );
                let x9m10 = self.rotate.rotate_m128d(x9m10);
                let y00 = _mm_add_pd(y00, x9p10);

                let m0118a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle1.re), u0);
                let m0118a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle2.re), m0118a);
                let m0118a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle3.re), m0118a);
                let m0118a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle4.re), m0118a);
                let m0118a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle5.re), m0118a);
                let m0118a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle6.re), m0118a);
                let m0118a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle7.re), m0118a);
                let m0118a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle8.re), m0118a);
                let m0118a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle9.re), m0118a);
                let m0118b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle1.im));
                let m0118b = _mm_fmadd_pd(x2m17, _mm_set1_pd(self.twiddle2.im), m0118b);
                let m0118b = _mm_fmadd_pd(x3m16, _mm_set1_pd(self.twiddle3.im), m0118b);
                let m0118b = _mm_fmadd_pd(x4m15, _mm_set1_pd(self.twiddle4.im), m0118b);
                let m0118b = _mm_fmadd_pd(x5m14, _mm_set1_pd(self.twiddle5.im), m0118b);
                let m0118b = _mm_fmadd_pd(x6m13, _mm_set1_pd(self.twiddle6.im), m0118b);
                let m0118b = _mm_fmadd_pd(x7m12, _mm_set1_pd(self.twiddle7.im), m0118b);
                let m0118b = _mm_fmadd_pd(x8m11, _mm_set1_pd(self.twiddle8.im), m0118b);
                let m0118b = _mm_fmadd_pd(x9m10, _mm_set1_pd(self.twiddle9.im), m0118b);
                let (y01, y18) = AvxButterfly::butterfly2_f64_m128(m0118a, m0118b);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), _mm256_create_pd(y00, y01));
                _mm_storeu_pd(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y18);

                let m0217a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle2.re), x1p18, u0);
                let m0217a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle4.re), m0217a);
                let m0217a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle6.re), m0217a);
                let m0217a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle8.re), m0217a);
                let m0217a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle9.re), m0217a);
                let m0217a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle7.re), m0217a);
                let m0217a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle5.re), m0217a);
                let m0217a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle3.re), m0217a);
                let m0217a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle1.re), m0217a);
                let m0217b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle2.im));
                let m0217b = _mm_fmadd_pd(x2m17, _mm_set1_pd(self.twiddle4.im), m0217b);
                let m0217b = _mm_fmadd_pd(x3m16, _mm_set1_pd(self.twiddle6.im), m0217b);
                let m0217b = _mm_fmadd_pd(x4m15, _mm_set1_pd(self.twiddle8.im), m0217b);
                let m0217b = _mm_fnmadd_pd(x5m14, _mm_set1_pd(self.twiddle9.im), m0217b);
                let m0217b = _mm_fnmadd_pd(x6m13, _mm_set1_pd(self.twiddle7.im), m0217b);
                let m0217b = _mm_fnmadd_pd(x7m12, _mm_set1_pd(self.twiddle5.im), m0217b);
                let m0217b = _mm_fnmadd_pd(x8m11, _mm_set1_pd(self.twiddle3.im), m0217b);
                let m0217b = _mm_fnmadd_pd(x9m10, _mm_set1_pd(self.twiddle1.im), m0217b);
                let (y02, y17) = AvxButterfly::butterfly2_f64_m128(m0217a, m0217b);

                let m0316a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle3.re), u0);
                let m0316a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle6.re), m0316a);
                let m0316a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle9.re), m0316a);
                let m0316a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle7.re), m0316a);
                let m0316a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle4.re), m0316a);
                let m0316a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle1.re), m0316a);
                let m0316a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle2.re), m0316a);
                let m0316a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle5.re), m0316a);
                let m0316a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle8.re), m0316a);
                let m0316b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle3.im));
                let m0316b = _mm_fmadd_pd(x2m17, _mm_set1_pd(self.twiddle6.im), m0316b);
                let m0316b = _mm_fmadd_pd(x3m16, _mm_set1_pd(self.twiddle9.im), m0316b);
                let m0316b = _mm_fnmadd_pd(x4m15, _mm_set1_pd(self.twiddle7.im), m0316b);
                let m0316b = _mm_fnmadd_pd(x5m14, _mm_set1_pd(self.twiddle4.im), m0316b);
                let m0316b = _mm_fnmadd_pd(x6m13, _mm_set1_pd(self.twiddle1.im), m0316b);
                let m0316b = _mm_fmadd_pd(x7m12, _mm_set1_pd(self.twiddle2.im), m0316b);
                let m0316b = _mm_fmadd_pd(x8m11, _mm_set1_pd(self.twiddle5.im), m0316b);
                let m0316b = _mm_fmadd_pd(x9m10, _mm_set1_pd(self.twiddle8.im), m0316b);
                let (y03, y16) = AvxButterfly::butterfly2_f64_m128(m0316a, m0316b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y02, y03),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_create_pd(y16, y17),
                );

                let m0415a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle4.re), u0);
                let m0415a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle8.re), m0415a);
                let m0415a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle7.re), m0415a);
                let m0415a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle3.re), m0415a);
                let m0415a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle1.re), m0415a);
                let m0415a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle5.re), m0415a);
                let m0415a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle9.re), m0415a);
                let m0415a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle6.re), m0415a);
                let m0415a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle2.re), m0415a);
                let m0415b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle4.im));
                let m0415b = _mm_fmadd_pd(x2m17, _mm_set1_pd(self.twiddle8.im), m0415b);
                let m0415b = _mm_fnmadd_pd(x3m16, _mm_set1_pd(self.twiddle7.im), m0415b);
                let m0415b = _mm_fnmadd_pd(x4m15, _mm_set1_pd(self.twiddle3.im), m0415b);
                let m0415b = _mm_fmadd_pd(x5m14, _mm_set1_pd(self.twiddle1.im), m0415b);
                let m0415b = _mm_fmadd_pd(x6m13, _mm_set1_pd(self.twiddle5.im), m0415b);
                let m0415b = _mm_fmadd_pd(x7m12, _mm_set1_pd(self.twiddle9.im), m0415b);
                let m0415b = _mm_fnmadd_pd(x8m11, _mm_set1_pd(self.twiddle6.im), m0415b);
                let m0415b = _mm_fnmadd_pd(x9m10, _mm_set1_pd(self.twiddle2.im), m0415b);
                let (y04, y15) = AvxButterfly::butterfly2_f64_m128(m0415a, m0415b);

                let m0514a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle5.re), u0);
                let m0514a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle9.re), m0514a);
                let m0514a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle4.re), m0514a);
                let m0514a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle1.re), m0514a);
                let m0514a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle6.re), m0514a);
                let m0514a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle8.re), m0514a);
                let m0514a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle3.re), m0514a);
                let m0514a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle2.re), m0514a);
                let m0514a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle7.re), m0514a);
                let m0514b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle5.im));
                let m0514b = _mm_fnmadd_pd(x2m17, _mm_set1_pd(self.twiddle9.im), m0514b);
                let m0514b = _mm_fnmadd_pd(x3m16, _mm_set1_pd(self.twiddle4.im), m0514b);
                let m0514b = _mm_fmadd_pd(x4m15, _mm_set1_pd(self.twiddle1.im), m0514b);
                let m0514b = _mm_fmadd_pd(x5m14, _mm_set1_pd(self.twiddle6.im), m0514b);
                let m0514b = _mm_fnmadd_pd(x6m13, _mm_set1_pd(self.twiddle8.im), m0514b);
                let m0514b = _mm_fnmadd_pd(x7m12, _mm_set1_pd(self.twiddle3.im), m0514b);
                let m0514b = _mm_fmadd_pd(x8m11, _mm_set1_pd(self.twiddle2.im), m0514b);
                let m0514b = _mm_fmadd_pd(x9m10, _mm_set1_pd(self.twiddle7.im), m0514b);
                let (y05, y14) = AvxButterfly::butterfly2_f64_m128(m0514a, m0514b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y04, y05),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_create_pd(y14, y15),
                );

                let m0613a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle6.re), u0);
                let m0613a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle7.re), m0613a);
                let m0613a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle1.re), m0613a);
                let m0613a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle5.re), m0613a);
                let m0613a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle8.re), m0613a);
                let m0613a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle2.re), m0613a);
                let m0613a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle4.re), m0613a);
                let m0613a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle9.re), m0613a);
                let m0613a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle3.re), m0613a);
                let m0613b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle6.im));
                let m0613b = _mm_fnmadd_pd(x2m17, _mm_set1_pd(self.twiddle7.im), m0613b);
                let m0613b = _mm_fnmadd_pd(x3m16, _mm_set1_pd(self.twiddle1.im), m0613b);
                let m0613b = _mm_fmadd_pd(x4m15, _mm_set1_pd(self.twiddle5.im), m0613b);
                let m0613b = _mm_fnmadd_pd(x5m14, _mm_set1_pd(self.twiddle8.im), m0613b);
                let m0613b = _mm_fnmadd_pd(x6m13, _mm_set1_pd(self.twiddle2.im), m0613b);
                let m0613b = _mm_fmadd_pd(x7m12, _mm_set1_pd(self.twiddle4.im), m0613b);
                let m0613b = _mm_fnmadd_pd(x8m11, _mm_set1_pd(self.twiddle9.im), m0613b);
                let m0613b = _mm_fnmadd_pd(x9m10, _mm_set1_pd(self.twiddle3.im), m0613b);
                let (y06, y13) = AvxButterfly::butterfly2_f64_m128(m0613a, m0613b);

                let m0712a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle7.re), u0);
                let m0712a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle5.re), m0712a);
                let m0712a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle2.re), m0712a);
                let m0712a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle9.re), m0712a);
                let m0712a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle3.re), m0712a);
                let m0712a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle4.re), m0712a);
                let m0712a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle8.re), m0712a);
                let m0712a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle1.re), m0712a);
                let m0712a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle6.re), m0712a);
                let m0712b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle7.im));
                let m0712b = _mm_fnmadd_pd(x2m17, _mm_set1_pd(self.twiddle5.im), m0712b);
                let m0712b = _mm_fmadd_pd(x3m16, _mm_set1_pd(self.twiddle2.im), m0712b);
                let m0712b = _mm_fmadd_pd(x4m15, _mm_set1_pd(self.twiddle9.im), m0712b);
                let m0712b = _mm_fnmadd_pd(x5m14, _mm_set1_pd(self.twiddle3.im), m0712b);
                let m0712b = _mm_fmadd_pd(x6m13, _mm_set1_pd(self.twiddle4.im), m0712b);
                let m0712b = _mm_fnmadd_pd(x7m12, _mm_set1_pd(self.twiddle8.im), m0712b);
                let m0712b = _mm_fnmadd_pd(x8m11, _mm_set1_pd(self.twiddle1.im), m0712b);
                let m0712b = _mm_fmadd_pd(x9m10, _mm_set1_pd(self.twiddle6.im), m0712b);
                let (y07, y12) = AvxButterfly::butterfly2_f64_m128(m0712a, m0712b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y06, y07),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_pd(y12, y13),
                );

                let m0811a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle8.re), u0);
                let m0811a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle3.re), m0811a);
                let m0811a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle5.re), m0811a);
                let m0811a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle6.re), m0811a);
                let m0811a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle2.re), m0811a);
                let m0811a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle9.re), m0811a);
                let m0811a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle1.re), m0811a);
                let m0811a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle7.re), m0811a);
                let m0811a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle4.re), m0811a);
                let m0811b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle8.im));
                let m0811b = _mm_fnmadd_pd(x2m17, _mm_set1_pd(self.twiddle3.im), m0811b);
                let m0811b = _mm_fmadd_pd(x3m16, _mm_set1_pd(self.twiddle5.im), m0811b);
                let m0811b = _mm_fnmadd_pd(x4m15, _mm_set1_pd(self.twiddle6.im), m0811b);
                let m0811b = _mm_fmadd_pd(x5m14, _mm_set1_pd(self.twiddle2.im), m0811b);
                let m0811b = _mm_fnmadd_pd(x6m13, _mm_set1_pd(self.twiddle9.im), m0811b);
                let m0811b = _mm_fnmadd_pd(x7m12, _mm_set1_pd(self.twiddle1.im), m0811b);
                let m0811b = _mm_fmadd_pd(x8m11, _mm_set1_pd(self.twiddle7.im), m0811b);
                let m0811b = _mm_fnmadd_pd(x9m10, _mm_set1_pd(self.twiddle4.im), m0811b);
                let (y08, y11) = AvxButterfly::butterfly2_f64_m128(m0811a, m0811b);

                let m0910a = _mm_fmadd_pd(x1p18, _mm_set1_pd(self.twiddle9.re), u0);
                let m0910a = _mm_fmadd_pd(x2p17, _mm_set1_pd(self.twiddle1.re), m0910a);
                let m0910a = _mm_fmadd_pd(x3p16, _mm_set1_pd(self.twiddle8.re), m0910a);
                let m0910a = _mm_fmadd_pd(x4p15, _mm_set1_pd(self.twiddle2.re), m0910a);
                let m0910a = _mm_fmadd_pd(x5p14, _mm_set1_pd(self.twiddle7.re), m0910a);
                let m0910a = _mm_fmadd_pd(x6p13, _mm_set1_pd(self.twiddle3.re), m0910a);
                let m0910a = _mm_fmadd_pd(x7p12, _mm_set1_pd(self.twiddle6.re), m0910a);
                let m0910a = _mm_fmadd_pd(x8p11, _mm_set1_pd(self.twiddle4.re), m0910a);
                let m0910a = _mm_fmadd_pd(x9p10, _mm_set1_pd(self.twiddle5.re), m0910a);
                let m0910b = _mm_mul_pd(x1m18, _mm_set1_pd(self.twiddle9.im));
                let m0910b = _mm_fnmadd_pd(x2m17, _mm_set1_pd(self.twiddle1.im), m0910b);
                let m0910b = _mm_fmadd_pd(x3m16, _mm_set1_pd(self.twiddle8.im), m0910b);
                let m0910b = _mm_fnmadd_pd(x4m15, _mm_set1_pd(self.twiddle2.im), m0910b);
                let m0910b = _mm_fmadd_pd(x5m14, _mm_set1_pd(self.twiddle7.im), m0910b);
                let m0910b = _mm_fnmadd_pd(x6m13, _mm_set1_pd(self.twiddle3.im), m0910b);
                let m0910b = _mm_fmadd_pd(x7m12, _mm_set1_pd(self.twiddle6.im), m0910b);
                let m0910b = _mm_fnmadd_pd(x8m11, _mm_set1_pd(self.twiddle4.im), m0910b);
                let m0910b = _mm_fmadd_pd(x9m10, _mm_set1_pd(self.twiddle5.im), m0910b);
                let (y09, y10) = AvxButterfly::butterfly2_f64_m128(m0910a, m0910b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(y08, y09),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(y10, y11),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly19<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        19
    }
}

impl AvxButterfly19<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 19 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(19) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16u17 = _mm_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());
                let u18 = _m128s_load_f32x2(chunk.get_unchecked(18..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u14u15 = _mm256_extractf128_ps::<1>(u12u13u14u15);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u12u13 = _mm256_castps256_ps128(u12u13u14u15);
                let u8u9 = _mm256_castps256_ps128(u8u9u10u11);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);

                let u0 = u0u1;
                let y00 = u0;
                let (x1p18, x1m18) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u0u1, u0u1), u18);
                let x1m18 = self.rotate.rotate_m128(x1m18);
                let y00 = _mm_add_ps(y00, x1p18);
                let (x2p17, x2m17) =
                    AvxButterfly::butterfly2_f32_m128(u2u3, _mm_unpackhi_ps64(u16u17, u16u17)); // 2,17
                let x2m17 = self.rotate.rotate_m128(x2m17);
                let y00 = _mm_add_ps(y00, x2p17);
                let (x3p16, x3m16) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u2u3, u2u3), u16u17); // 3, 16
                let x3m16 = self.rotate.rotate_m128(x3m16);
                let y00 = _mm_add_ps(y00, x3p16);
                let (x4p15, x4m15) =
                    AvxButterfly::butterfly2_f32_m128(u4u5, _mm_unpackhi_ps64(u14u15, u14u15)); // 4,15
                let x4m15 = self.rotate.rotate_m128(x4m15);
                let y00 = _mm_add_ps(y00, x4p15);
                let (x5p14, x5m14) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u4u5, u4u5), u14u15); // 5,14
                let x5m14 = self.rotate.rotate_m128(x5m14);
                let y00 = _mm_add_ps(y00, x5p14);
                let (x6p13, x6m13) =
                    AvxButterfly::butterfly2_f32_m128(u6u7, _mm_unpackhi_ps64(u12u13, u12u13)); // 6,13
                let x6m13 = self.rotate.rotate_m128(x6m13);
                let y00 = _mm_add_ps(y00, x6p13);
                let (x7p12, x7m12) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u6u7, u6u7), u12u13); // 7,12
                let x7m12 = self.rotate.rotate_m128(x7m12);
                let y00 = _mm_add_ps(y00, x7p12);
                let (x8p11, x8m11) =
                    AvxButterfly::butterfly2_f32_m128(u8u9, _mm_unpackhi_ps64(u10u11, u10u11)); // 8,11
                let x8m11 = self.rotate.rotate_m128(x8m11);
                let y00 = _mm_add_ps(y00, x8p11);
                let (x9p10, x9m10) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u8u9, u8u9), u10u11); // 9,10
                let x9m10 = self.rotate.rotate_m128(x9m10);
                let y00 = _mm_add_ps(y00, x9p10);

                let m0118a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle1.re), u0);
                let m0118a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle2.re), m0118a);
                let m0118a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle3.re), m0118a);
                let m0118a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle4.re), m0118a);
                let m0118a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle5.re), m0118a);
                let m0118a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle6.re), m0118a);
                let m0118a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle7.re), m0118a);
                let m0118a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle8.re), m0118a);
                let m0118a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle9.re), m0118a);
                let m0118b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle1.im));
                let m0118b = _mm_fmadd_ps(x2m17, _mm_set1_ps(self.twiddle2.im), m0118b);
                let m0118b = _mm_fmadd_ps(x3m16, _mm_set1_ps(self.twiddle3.im), m0118b);
                let m0118b = _mm_fmadd_ps(x4m15, _mm_set1_ps(self.twiddle4.im), m0118b);
                let m0118b = _mm_fmadd_ps(x5m14, _mm_set1_ps(self.twiddle5.im), m0118b);
                let m0118b = _mm_fmadd_ps(x6m13, _mm_set1_ps(self.twiddle6.im), m0118b);
                let m0118b = _mm_fmadd_ps(x7m12, _mm_set1_ps(self.twiddle7.im), m0118b);
                let m0118b = _mm_fmadd_ps(x8m11, _mm_set1_ps(self.twiddle8.im), m0118b);
                let m0118b = _mm_fmadd_ps(x9m10, _mm_set1_ps(self.twiddle9.im), m0118b);
                let (y01, y18) = AvxButterfly::butterfly2_f32_m128(m0118a, m0118b);

                _m128s_store_f32x2(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y18);

                let m0217a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x1p18, u0);
                let m0217a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle4.re), m0217a);
                let m0217a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle6.re), m0217a);
                let m0217a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle8.re), m0217a);
                let m0217a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle9.re), m0217a);
                let m0217a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle7.re), m0217a);
                let m0217a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle5.re), m0217a);
                let m0217a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle3.re), m0217a);
                let m0217a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle1.re), m0217a);
                let m0217b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle2.im));
                let m0217b = _mm_fmadd_ps(x2m17, _mm_set1_ps(self.twiddle4.im), m0217b);
                let m0217b = _mm_fmadd_ps(x3m16, _mm_set1_ps(self.twiddle6.im), m0217b);
                let m0217b = _mm_fmadd_ps(x4m15, _mm_set1_ps(self.twiddle8.im), m0217b);
                let m0217b = _mm_fnmadd_ps(x5m14, _mm_set1_ps(self.twiddle9.im), m0217b);
                let m0217b = _mm_fnmadd_ps(x6m13, _mm_set1_ps(self.twiddle7.im), m0217b);
                let m0217b = _mm_fnmadd_ps(x7m12, _mm_set1_ps(self.twiddle5.im), m0217b);
                let m0217b = _mm_fnmadd_ps(x8m11, _mm_set1_ps(self.twiddle3.im), m0217b);
                let m0217b = _mm_fnmadd_ps(x9m10, _mm_set1_ps(self.twiddle1.im), m0217b);
                let (y02, y17) = AvxButterfly::butterfly2_f32_m128(m0217a, m0217b);

                let m0316a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle3.re), u0);
                let m0316a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle6.re), m0316a);
                let m0316a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle9.re), m0316a);
                let m0316a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle7.re), m0316a);
                let m0316a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle4.re), m0316a);
                let m0316a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle1.re), m0316a);
                let m0316a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle2.re), m0316a);
                let m0316a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle5.re), m0316a);
                let m0316a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle8.re), m0316a);
                let m0316b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle3.im));
                let m0316b = _mm_fmadd_ps(x2m17, _mm_set1_ps(self.twiddle6.im), m0316b);
                let m0316b = _mm_fmadd_ps(x3m16, _mm_set1_ps(self.twiddle9.im), m0316b);
                let m0316b = _mm_fnmadd_ps(x4m15, _mm_set1_ps(self.twiddle7.im), m0316b);
                let m0316b = _mm_fnmadd_ps(x5m14, _mm_set1_ps(self.twiddle4.im), m0316b);
                let m0316b = _mm_fnmadd_ps(x6m13, _mm_set1_ps(self.twiddle1.im), m0316b);
                let m0316b = _mm_fmadd_ps(x7m12, _mm_set1_ps(self.twiddle2.im), m0316b);
                let m0316b = _mm_fmadd_ps(x8m11, _mm_set1_ps(self.twiddle5.im), m0316b);
                let m0316b = _mm_fmadd_ps(x9m10, _mm_set1_ps(self.twiddle8.im), m0316b);
                let (y03, y16) = AvxButterfly::butterfly2_f32_m128(m0316a, m0316b);

                _mm256_storeu_ps(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03)),
                );
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm_unpacklo_ps64(y16, y17),
                );

                let m0415a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle4.re), u0);
                let m0415a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle8.re), m0415a);
                let m0415a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle7.re), m0415a);
                let m0415a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle3.re), m0415a);
                let m0415a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle1.re), m0415a);
                let m0415a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle5.re), m0415a);
                let m0415a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle9.re), m0415a);
                let m0415a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle6.re), m0415a);
                let m0415a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle2.re), m0415a);
                let m0415b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle4.im));
                let m0415b = _mm_fmadd_ps(x2m17, _mm_set1_ps(self.twiddle8.im), m0415b);
                let m0415b = _mm_fnmadd_ps(x3m16, _mm_set1_ps(self.twiddle7.im), m0415b);
                let m0415b = _mm_fnmadd_ps(x4m15, _mm_set1_ps(self.twiddle3.im), m0415b);
                let m0415b = _mm_fmadd_ps(x5m14, _mm_set1_ps(self.twiddle1.im), m0415b);
                let m0415b = _mm_fmadd_ps(x6m13, _mm_set1_ps(self.twiddle5.im), m0415b);
                let m0415b = _mm_fmadd_ps(x7m12, _mm_set1_ps(self.twiddle9.im), m0415b);
                let m0415b = _mm_fnmadd_ps(x8m11, _mm_set1_ps(self.twiddle6.im), m0415b);
                let m0415b = _mm_fnmadd_ps(x9m10, _mm_set1_ps(self.twiddle2.im), m0415b);
                let (y04, y15) = AvxButterfly::butterfly2_f32_m128(m0415a, m0415b);

                let m0514a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle5.re), u0);
                let m0514a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle9.re), m0514a);
                let m0514a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle4.re), m0514a);
                let m0514a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle1.re), m0514a);
                let m0514a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle6.re), m0514a);
                let m0514a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle8.re), m0514a);
                let m0514a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle3.re), m0514a);
                let m0514a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle2.re), m0514a);
                let m0514a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle7.re), m0514a);
                let m0514b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle5.im));
                let m0514b = _mm_fnmadd_ps(x2m17, _mm_set1_ps(self.twiddle9.im), m0514b);
                let m0514b = _mm_fnmadd_ps(x3m16, _mm_set1_ps(self.twiddle4.im), m0514b);
                let m0514b = _mm_fmadd_ps(x4m15, _mm_set1_ps(self.twiddle1.im), m0514b);
                let m0514b = _mm_fmadd_ps(x5m14, _mm_set1_ps(self.twiddle6.im), m0514b);
                let m0514b = _mm_fnmadd_ps(x6m13, _mm_set1_ps(self.twiddle8.im), m0514b);
                let m0514b = _mm_fnmadd_ps(x7m12, _mm_set1_ps(self.twiddle3.im), m0514b);
                let m0514b = _mm_fmadd_ps(x8m11, _mm_set1_ps(self.twiddle2.im), m0514b);
                let m0514b = _mm_fmadd_ps(x9m10, _mm_set1_ps(self.twiddle7.im), m0514b);
                let (y05, y14) = AvxButterfly::butterfly2_f32_m128(m0514a, m0514b);

                let m0613a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle6.re), u0);
                let m0613a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle7.re), m0613a);
                let m0613a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle1.re), m0613a);
                let m0613a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle5.re), m0613a);
                let m0613a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle8.re), m0613a);
                let m0613a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle2.re), m0613a);
                let m0613a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle4.re), m0613a);
                let m0613a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle9.re), m0613a);
                let m0613a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle3.re), m0613a);
                let m0613b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle6.im));
                let m0613b = _mm_fnmadd_ps(x2m17, _mm_set1_ps(self.twiddle7.im), m0613b);
                let m0613b = _mm_fnmadd_ps(x3m16, _mm_set1_ps(self.twiddle1.im), m0613b);
                let m0613b = _mm_fmadd_ps(x4m15, _mm_set1_ps(self.twiddle5.im), m0613b);
                let m0613b = _mm_fnmadd_ps(x5m14, _mm_set1_ps(self.twiddle8.im), m0613b);
                let m0613b = _mm_fnmadd_ps(x6m13, _mm_set1_ps(self.twiddle2.im), m0613b);
                let m0613b = _mm_fmadd_ps(x7m12, _mm_set1_ps(self.twiddle4.im), m0613b);
                let m0613b = _mm_fnmadd_ps(x8m11, _mm_set1_ps(self.twiddle9.im), m0613b);
                let m0613b = _mm_fnmadd_ps(x9m10, _mm_set1_ps(self.twiddle3.im), m0613b);
                let (y06, y13) = AvxButterfly::butterfly2_f32_m128(m0613a, m0613b);

                let m0712a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle7.re), u0);
                let m0712a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle5.re), m0712a);
                let m0712a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle2.re), m0712a);
                let m0712a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle9.re), m0712a);
                let m0712a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle3.re), m0712a);
                let m0712a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle4.re), m0712a);
                let m0712a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle8.re), m0712a);
                let m0712a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle1.re), m0712a);
                let m0712a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle6.re), m0712a);
                let m0712b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle7.im));
                let m0712b = _mm_fnmadd_ps(x2m17, _mm_set1_ps(self.twiddle5.im), m0712b);
                let m0712b = _mm_fmadd_ps(x3m16, _mm_set1_ps(self.twiddle2.im), m0712b);
                let m0712b = _mm_fmadd_ps(x4m15, _mm_set1_ps(self.twiddle9.im), m0712b);
                let m0712b = _mm_fnmadd_ps(x5m14, _mm_set1_ps(self.twiddle3.im), m0712b);
                let m0712b = _mm_fmadd_ps(x6m13, _mm_set1_ps(self.twiddle4.im), m0712b);
                let m0712b = _mm_fnmadd_ps(x7m12, _mm_set1_ps(self.twiddle8.im), m0712b);
                let m0712b = _mm_fnmadd_ps(x8m11, _mm_set1_ps(self.twiddle1.im), m0712b);
                let m0712b = _mm_fmadd_ps(x9m10, _mm_set1_ps(self.twiddle6.im), m0712b);
                let (y07, y12) = AvxButterfly::butterfly2_f32_m128(m0712a, m0712b);

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(y12, y13), _mm_unpacklo_ps64(y14, y15)),
                );

                let m0811a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle8.re), u0);
                let m0811a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle3.re), m0811a);
                let m0811a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle5.re), m0811a);
                let m0811a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle6.re), m0811a);
                let m0811a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle2.re), m0811a);
                let m0811a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle9.re), m0811a);
                let m0811a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle1.re), m0811a);
                let m0811a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle7.re), m0811a);
                let m0811a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle4.re), m0811a);
                let m0811b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle8.im));
                let m0811b = _mm_fnmadd_ps(x2m17, _mm_set1_ps(self.twiddle3.im), m0811b);
                let m0811b = _mm_fmadd_ps(x3m16, _mm_set1_ps(self.twiddle5.im), m0811b);
                let m0811b = _mm_fnmadd_ps(x4m15, _mm_set1_ps(self.twiddle6.im), m0811b);
                let m0811b = _mm_fmadd_ps(x5m14, _mm_set1_ps(self.twiddle2.im), m0811b);
                let m0811b = _mm_fnmadd_ps(x6m13, _mm_set1_ps(self.twiddle9.im), m0811b);
                let m0811b = _mm_fnmadd_ps(x7m12, _mm_set1_ps(self.twiddle1.im), m0811b);
                let m0811b = _mm_fmadd_ps(x8m11, _mm_set1_ps(self.twiddle7.im), m0811b);
                let m0811b = _mm_fnmadd_ps(x9m10, _mm_set1_ps(self.twiddle4.im), m0811b);
                let (y08, y11) = AvxButterfly::butterfly2_f32_m128(m0811a, m0811b);

                let m0910a = _mm_fmadd_ps(x1p18, _mm_set1_ps(self.twiddle9.re), u0);
                let m0910a = _mm_fmadd_ps(x2p17, _mm_set1_ps(self.twiddle1.re), m0910a);
                let m0910a = _mm_fmadd_ps(x3p16, _mm_set1_ps(self.twiddle8.re), m0910a);
                let m0910a = _mm_fmadd_ps(x4p15, _mm_set1_ps(self.twiddle2.re), m0910a);
                let m0910a = _mm_fmadd_ps(x5p14, _mm_set1_ps(self.twiddle7.re), m0910a);
                let m0910a = _mm_fmadd_ps(x6p13, _mm_set1_ps(self.twiddle3.re), m0910a);
                let m0910a = _mm_fmadd_ps(x7p12, _mm_set1_ps(self.twiddle6.re), m0910a);
                let m0910a = _mm_fmadd_ps(x8p11, _mm_set1_ps(self.twiddle4.re), m0910a);
                let m0910a = _mm_fmadd_ps(x9p10, _mm_set1_ps(self.twiddle5.re), m0910a);
                let m0910b = _mm_mul_ps(x1m18, _mm_set1_ps(self.twiddle9.im));
                let m0910b = _mm_fnmadd_ps(x2m17, _mm_set1_ps(self.twiddle1.im), m0910b);
                let m0910b = _mm_fmadd_ps(x3m16, _mm_set1_ps(self.twiddle8.im), m0910b);
                let m0910b = _mm_fnmadd_ps(x4m15, _mm_set1_ps(self.twiddle2.im), m0910b);
                let m0910b = _mm_fmadd_ps(x5m14, _mm_set1_ps(self.twiddle7.im), m0910b);
                let m0910b = _mm_fnmadd_ps(x6m13, _mm_set1_ps(self.twiddle3.im), m0910b);
                let m0910b = _mm_fmadd_ps(x7m12, _mm_set1_ps(self.twiddle6.im), m0910b);
                let m0910b = _mm_fnmadd_ps(x8m11, _mm_set1_ps(self.twiddle4.im), m0910b);
                let m0910b = _mm_fmadd_ps(x9m10, _mm_set1_ps(self.twiddle5.im), m0910b);
                let (y09, y10) = AvxButterfly::butterfly2_f32_m128(m0910a, m0910b);

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(y08, y09), _mm_unpacklo_ps64(y10, y11)),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly19<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        19
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly19;
    use rand::Rng;

    #[test]
    fn test_butterfly19_f32() {
        for i in 1..4 {
            let size = 19usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly19::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly19::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly19::new(FftDirection::Forward);

            radix_forward.execute(&mut input).unwrap();
            radix_forward_ref.execute(&mut ref0).unwrap();

            input
                .iter()
                .zip(ref0.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-4,
                        "forward at {idx} a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-4,
                        "forward at {idx} a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 19f32)).collect();

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-5,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-5,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }

    #[test]
    fn test_butterfly19_f64() {
        for i in 1..4 {
            let size = 19usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly19::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly19::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly19::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 19f64)).collect();

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
