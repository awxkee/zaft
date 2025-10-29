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
    _m128s_load_f32x2, _m128s_store_f32x2, _m128s_storeh_f32x2, _mm_unpackhi_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly17<T> {
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
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly17<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            rotate: unsafe { AvxRotate::new(FftDirection::Inverse) },
            twiddle1: compute_twiddle(1, 17, fft_direction),
            twiddle2: compute_twiddle(2, 17, fft_direction),
            twiddle3: compute_twiddle(3, 17, fft_direction),
            twiddle4: compute_twiddle(4, 17, fft_direction),
            twiddle5: compute_twiddle(5, 17, fft_direction),
            twiddle6: compute_twiddle(6, 17, fft_direction),
            twiddle7: compute_twiddle(7, 17, fft_direction),
            twiddle8: compute_twiddle(8, 17, fft_direction),
        }
    }
}

impl AvxButterfly17<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 17 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(17) {
                let u0u1 = _mm256_loadu_pd(chunk.as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());
                let u16 = _mm_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());

                let y00 = _mm256_castpd256_pd128(u0u1);
                let u0 = _mm256_castpd256_pd128(u0u1);
                let (x1p16, x1m16) =
                    AvxButterfly::butterfly2_f64_m128(_mm256_extractf128_pd::<1>(u0u1), u16);
                let x1m16 = self.rotate.rotate_m128d(x1m16);
                let y00 = _mm_add_pd(y00, x1p16);
                let (x2p15, x2m15) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_extractf128_pd::<1>(u14u15),
                );
                let x2m15 = self.rotate.rotate_m128d(x2m15);
                let y00 = _mm_add_pd(y00, x2p15);
                let (x3p14, x3m14) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u14u15),
                );
                let x3m14 = self.rotate.rotate_m128d(x3m14);
                let y00 = _mm_add_pd(y00, x3p14);
                let (x4p13, x4m13) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_extractf128_pd::<1>(u12u13),
                );
                let x4m13 = self.rotate.rotate_m128d(x4m13);
                let y00 = _mm_add_pd(y00, x4p13);
                let (x5p12, x5m12) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_castpd256_pd128(u12u13),
                );
                let x5m12 = self.rotate.rotate_m128d(x5m12);
                let y00 = _mm_add_pd(y00, x5p12);
                let (x6p11, x6m11) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u10u11),
                );
                let x6m11 = self.rotate.rotate_m128d(x6m11);
                let y00 = _mm_add_pd(y00, x6p11);
                let (x7p10, x7m10) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_extractf128_pd::<1>(u6u7),
                    _mm256_castpd256_pd128(u10u11),
                );
                let x7m10 = self.rotate.rotate_m128d(x7m10);
                let y00 = _mm_add_pd(y00, x7p10);
                let (x8p9, x8m9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u8u9),
                    _mm256_extractf128_pd::<1>(u8u9),
                );
                let x8m9 = self.rotate.rotate_m128d(x8m9);
                let y00 = _mm_add_pd(y00, x8p9);

                let m0116a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle1.re), u0);
                let m0116a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle2.re), m0116a);
                let m0116a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle3.re), m0116a);
                let m0116a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle4.re), m0116a);
                let m0116a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle5.re), m0116a);
                let m0116a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle6.re), m0116a);
                let m0116a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle7.re), m0116a);
                let m0116a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle8.re), m0116a);
                let m0116b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle1.im));
                let m0116b = _mm_fmadd_pd(x2m15, _mm_set1_pd(self.twiddle2.im), m0116b);
                let m0116b = _mm_fmadd_pd(x3m14, _mm_set1_pd(self.twiddle3.im), m0116b);
                let m0116b = _mm_fmadd_pd(x4m13, _mm_set1_pd(self.twiddle4.im), m0116b);
                let m0116b = _mm_fmadd_pd(x5m12, _mm_set1_pd(self.twiddle5.im), m0116b);
                let m0116b = _mm_fmadd_pd(x6m11, _mm_set1_pd(self.twiddle6.im), m0116b);
                let m0116b = _mm_fmadd_pd(x7m10, _mm_set1_pd(self.twiddle7.im), m0116b);
                let m0116b = _mm_fmadd_pd(x8m9, _mm_set1_pd(self.twiddle8.im), m0116b);
                let (y01, y16) = AvxButterfly::butterfly2_f64_m128(m0116a, m0116b);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), _mm256_create_pd(y00, y01));
                _mm_storeu_pd(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), y16);

                let m0215a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle2.re), u0);
                let m0215a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle4.re), m0215a);
                let m0215a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle6.re), m0215a);
                let m0215a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle8.re), m0215a);
                let m0215a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle7.re), m0215a);
                let m0215a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle5.re), m0215a);
                let m0215a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle3.re), m0215a);
                let m0215a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle1.re), m0215a);
                let m0215b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle2.im));
                let m0215b = _mm_fmadd_pd(x2m15, _mm_set1_pd(self.twiddle4.im), m0215b);
                let m0215b = _mm_fmadd_pd(x3m14, _mm_set1_pd(self.twiddle6.im), m0215b);
                let m0215b = _mm_fmadd_pd(x4m13, _mm_set1_pd(self.twiddle8.im), m0215b);
                let m0215b = _mm_fnmadd_pd(x5m12, _mm_set1_pd(self.twiddle7.im), m0215b);
                let m0215b = _mm_fnmadd_pd(x6m11, _mm_set1_pd(self.twiddle5.im), m0215b);
                let m0215b = _mm_fnmadd_pd(x7m10, _mm_set1_pd(self.twiddle3.im), m0215b);
                let m0215b = _mm_fnmadd_pd(x8m9, _mm_set1_pd(self.twiddle1.im), m0215b);
                let (y02, y15) = AvxButterfly::butterfly2_f64_m128(m0215a, m0215b);

                let m0314a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle3.re), u0);
                let m0314a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle6.re), m0314a);
                let m0314a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle8.re), m0314a);
                let m0314a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle5.re), m0314a);
                let m0314a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle2.re), m0314a);
                let m0314a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle1.re), m0314a);
                let m0314a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle4.re), m0314a);
                let m0314a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle7.re), m0314a);
                let m0314b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle3.im));
                let m0314b = _mm_fmadd_pd(x2m15, _mm_set1_pd(self.twiddle6.im), m0314b);
                let m0314b = _mm_fnmadd_pd(x3m14, _mm_set1_pd(self.twiddle8.im), m0314b);
                let m0314b = _mm_fnmadd_pd(x4m13, _mm_set1_pd(self.twiddle5.im), m0314b);
                let m0314b = _mm_fnmadd_pd(x5m12, _mm_set1_pd(self.twiddle2.im), m0314b);
                let m0314b = _mm_fmadd_pd(x6m11, _mm_set1_pd(self.twiddle1.im), m0314b);
                let m0314b = _mm_fmadd_pd(x7m10, _mm_set1_pd(self.twiddle4.im), m0314b);
                let m0314b = _mm_fmadd_pd(x8m9, _mm_set1_pd(self.twiddle7.im), m0314b);
                let (y03, y14) = AvxButterfly::butterfly2_f64_m128(m0314a, m0314b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y02, y03),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_create_pd(y14, y15),
                );

                let m0413a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle4.re), u0);
                let m0413a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle8.re), m0413a);
                let m0413a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle5.re), m0413a);
                let m0413a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle1.re), m0413a);
                let m0413a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle3.re), m0413a);
                let m0413a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle7.re), m0413a);
                let m0413a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle6.re), m0413a);
                let m0413a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle2.re), m0413a);
                let m0413b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle4.im));
                let m0413b = _mm_fmadd_pd(x2m15, _mm_set1_pd(self.twiddle8.im), m0413b);
                let m0413b = _mm_fnmadd_pd(x3m14, _mm_set1_pd(self.twiddle5.im), m0413b);
                let m0413b = _mm_fnmadd_pd(x4m13, _mm_set1_pd(self.twiddle1.im), m0413b);
                let m0413b = _mm_fmadd_pd(x5m12, _mm_set1_pd(self.twiddle3.im), m0413b);
                let m0413b = _mm_fmadd_pd(x6m11, _mm_set1_pd(self.twiddle7.im), m0413b);
                let m0413b = _mm_fnmadd_pd(x7m10, _mm_set1_pd(self.twiddle6.im), m0413b);
                let m0413b = _mm_fnmadd_pd(x8m9, _mm_set1_pd(self.twiddle2.im), m0413b);
                let (y04, y13) = AvxButterfly::butterfly2_f64_m128(m0413a, m0413b);

                let m0512a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle5.re), u0);
                let m0512a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle7.re), m0512a);
                let m0512a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle2.re), m0512a);
                let m0512a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle3.re), m0512a);
                let m0512a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle8.re), m0512a);
                let m0512a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle4.re), m0512a);
                let m0512a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle1.re), m0512a);
                let m0512a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle6.re), m0512a);
                let m0512b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle5.im));
                let m0512b = _mm_fnmadd_pd(x2m15, _mm_set1_pd(self.twiddle7.im), m0512b);
                let m0512b = _mm_fnmadd_pd(x3m14, _mm_set1_pd(self.twiddle2.im), m0512b);
                let m0512b = _mm_fmadd_pd(x4m13, _mm_set1_pd(self.twiddle3.im), m0512b);
                let m0512b = _mm_fmadd_pd(x5m12, _mm_set1_pd(self.twiddle8.im), m0512b);
                let m0512b = _mm_fnmadd_pd(x6m11, _mm_set1_pd(self.twiddle4.im), m0512b);
                let m0512b = _mm_fmadd_pd(x7m10, _mm_set1_pd(self.twiddle1.im), m0512b);
                let m0512b = _mm_fmadd_pd(x8m9, _mm_set1_pd(self.twiddle6.im), m0512b);
                let (y05, y12) = AvxButterfly::butterfly2_f64_m128(m0512a, m0512b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y04, y05),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_pd(y12, y13),
                );

                let m0611a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle6.re), u0);
                let m0611a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle5.re), m0611a);
                let m0611a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle1.re), m0611a);
                let m0611a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle7.re), m0611a);
                let m0611a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle4.re), m0611a);
                let m0611a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle2.re), m0611a);
                let m0611a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle8.re), m0611a);
                let m0611a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle3.re), m0611a);
                let m0611b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle6.im));
                let m0611b = _mm_fnmadd_pd(x2m15, _mm_set1_pd(self.twiddle5.im), m0611b);
                let m0611b = _mm_fmadd_pd(x3m14, _mm_set1_pd(self.twiddle1.im), m0611b);
                let m0611b = _mm_fmadd_pd(x4m13, _mm_set1_pd(self.twiddle7.im), m0611b);
                let m0611b = _mm_fnmadd_pd(x5m12, _mm_set1_pd(self.twiddle4.im), m0611b);
                let m0611b = _mm_fmadd_pd(x6m11, _mm_set1_pd(self.twiddle2.im), m0611b);
                let m0611b = _mm_fmadd_pd(x7m10, _mm_set1_pd(self.twiddle8.im), m0611b);
                let m0611b = _mm_fnmadd_pd(x8m9, _mm_set1_pd(self.twiddle3.im), m0611b);
                let (y06, y11) = AvxButterfly::butterfly2_f64_m128(m0611a, m0611b);

                let m0710a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle7.re), u0);
                let m0710a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle3.re), m0710a);
                let m0710a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle4.re), m0710a);
                let m0710a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle6.re), m0710a);
                let m0710a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle1.re), m0710a);
                let m0710a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle8.re), m0710a);
                let m0710a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle2.re), m0710a);
                let m0710a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle5.re), m0710a);
                let m0710b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle7.im));
                let m0710b = _mm_fnmadd_pd(x2m15, _mm_set1_pd(self.twiddle3.im), m0710b);
                let m0710b = _mm_fmadd_pd(x3m14, _mm_set1_pd(self.twiddle4.im), m0710b);
                let m0710b = _mm_fnmadd_pd(x4m13, _mm_set1_pd(self.twiddle6.im), m0710b);
                let m0710b = _mm_fmadd_pd(x5m12, _mm_set1_pd(self.twiddle1.im), m0710b);
                let m0710b = _mm_fmadd_pd(x6m11, _mm_set1_pd(self.twiddle8.im), m0710b);
                let m0710b = _mm_fnmadd_pd(x7m10, _mm_set1_pd(self.twiddle2.im), m0710b);
                let m0710b = _mm_fmadd_pd(x8m9, _mm_set1_pd(self.twiddle5.im), m0710b);
                let (y07, y10) = AvxButterfly::butterfly2_f64_m128(m0710a, m0710b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y06, y07),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(y10, y11),
                );

                let m0809a = _mm_fmadd_pd(x1p16, _mm_set1_pd(self.twiddle8.re), u0);
                let m0809a = _mm_fmadd_pd(x2p15, _mm_set1_pd(self.twiddle1.re), m0809a);
                let m0809a = _mm_fmadd_pd(x3p14, _mm_set1_pd(self.twiddle7.re), m0809a);
                let m0809a = _mm_fmadd_pd(x4p13, _mm_set1_pd(self.twiddle2.re), m0809a);
                let m0809a = _mm_fmadd_pd(x5p12, _mm_set1_pd(self.twiddle6.re), m0809a);
                let m0809a = _mm_fmadd_pd(x6p11, _mm_set1_pd(self.twiddle3.re), m0809a);
                let m0809a = _mm_fmadd_pd(x7p10, _mm_set1_pd(self.twiddle5.re), m0809a);
                let m0809a = _mm_fmadd_pd(x8p9, _mm_set1_pd(self.twiddle4.re), m0809a);
                let m0809b = _mm_mul_pd(x1m16, _mm_set1_pd(self.twiddle8.im));
                let m0809b = _mm_fnmadd_pd(x2m15, _mm_set1_pd(self.twiddle1.im), m0809b);
                let m0809b = _mm_fmadd_pd(x3m14, _mm_set1_pd(self.twiddle7.im), m0809b);
                let m0809b = _mm_fnmadd_pd(x4m13, _mm_set1_pd(self.twiddle2.im), m0809b);
                let m0809b = _mm_fmadd_pd(x5m12, _mm_set1_pd(self.twiddle6.im), m0809b);
                let m0809b = _mm_fnmadd_pd(x6m11, _mm_set1_pd(self.twiddle3.im), m0809b);
                let m0809b = _mm_fmadd_pd(x7m10, _mm_set1_pd(self.twiddle5.im), m0809b);
                let m0809b = _mm_fnmadd_pd(x8m9, _mm_set1_pd(self.twiddle4.im), m0809b);
                let (y08, y09) = AvxButterfly::butterfly2_f64_m128(m0809a, m0809b);

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(y08, y09),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly17<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        17
    }
}

impl AvxButterfly17<f32> {
    #[inline]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn kernel_f32(&self, v: [__m128; 17]) -> [__m128; 17] {
        unsafe {
            let y00 = v[0];
            let u0 = v[0];
            let (x1p16, x1m16) = AvxButterfly::butterfly2_f32_m128(v[1], v[16]); // 1, 16
            let x1m16 = self.rotate.rotate_m128(x1m16);
            let y00 = _mm_add_ps(y00, x1p16);
            let (x2p15, x2m15) = AvxButterfly::butterfly2_f32_m128(v[2], v[15]); // 2, 15
            let x2m15 = self.rotate.rotate_m128(x2m15);
            let y00 = _mm_add_ps(y00, x2p15);
            let (x3p14, x3m14) = AvxButterfly::butterfly2_f32_m128(v[3], v[14]); // 3, 14
            let x3m14 = self.rotate.rotate_m128(x3m14);
            let y00 = _mm_add_ps(y00, x3p14);
            let (x4p13, x4m13) = AvxButterfly::butterfly2_f32_m128(v[4], v[13]); // 4, 13
            let x4m13 = self.rotate.rotate_m128(x4m13);
            let y00 = _mm_add_ps(y00, x4p13);
            let (x5p12, x5m12) = AvxButterfly::butterfly2_f32_m128(v[5], v[12]); // 5, 12
            let x5m12 = self.rotate.rotate_m128(x5m12);
            let y00 = _mm_add_ps(y00, x5p12);
            let (x6p11, x6m11) = AvxButterfly::butterfly2_f32_m128(v[6], v[11]); // 6, 11
            let x6m11 = self.rotate.rotate_m128(x6m11);
            let y00 = _mm_add_ps(y00, x6p11);
            let (x7p10, x7m10) = AvxButterfly::butterfly2_f32_m128(v[7], v[10]); // 7, 10
            let x7m10 = self.rotate.rotate_m128(x7m10);
            let y00 = _mm_add_ps(y00, x7p10);
            let (x8p9, x8m9) = AvxButterfly::butterfly2_f32_m128(v[8], v[9]); // 8, 9
            let x8m9 = self.rotate.rotate_m128(x8m9);
            let y00 = _mm_add_ps(y00, x8p9);

            let m0116a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle1.re), u0);
            let m0116a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle2.re), m0116a);
            let m0116a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle3.re), m0116a);
            let m0116a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle4.re), m0116a);
            let m0116a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle5.re), m0116a);
            let m0116a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle6.re), m0116a);
            let m0116a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle7.re), m0116a);
            let m0116a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle8.re), m0116a);
            let m0116b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle1.im));
            let m0116b = _mm_fmadd_ps(x2m15, _mm_set1_ps(self.twiddle2.im), m0116b);
            let m0116b = _mm_fmadd_ps(x3m14, _mm_set1_ps(self.twiddle3.im), m0116b);
            let m0116b = _mm_fmadd_ps(x4m13, _mm_set1_ps(self.twiddle4.im), m0116b);
            let m0116b = _mm_fmadd_ps(x5m12, _mm_set1_ps(self.twiddle5.im), m0116b);
            let m0116b = _mm_fmadd_ps(x6m11, _mm_set1_ps(self.twiddle6.im), m0116b);
            let m0116b = _mm_fmadd_ps(x7m10, _mm_set1_ps(self.twiddle7.im), m0116b);
            let m0116b = _mm_fmadd_ps(x8m9, _mm_set1_ps(self.twiddle8.im), m0116b);
            let (y01, y16) = AvxButterfly::butterfly2_f32_m128(m0116a, m0116b);

            let m0215a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle2.re), u0);
            let m0215a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle4.re), m0215a);
            let m0215a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle6.re), m0215a);
            let m0215a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle8.re), m0215a);
            let m0215a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle7.re), m0215a);
            let m0215a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle5.re), m0215a);
            let m0215a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle3.re), m0215a);
            let m0215a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle1.re), m0215a);
            let m0215b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle2.im));
            let m0215b = _mm_fmadd_ps(x2m15, _mm_set1_ps(self.twiddle4.im), m0215b);
            let m0215b = _mm_fmadd_ps(x3m14, _mm_set1_ps(self.twiddle6.im), m0215b);
            let m0215b = _mm_fmadd_ps(x4m13, _mm_set1_ps(self.twiddle8.im), m0215b);
            let m0215b = _mm_fnmadd_ps(x5m12, _mm_set1_ps(self.twiddle7.im), m0215b);
            let m0215b = _mm_fnmadd_ps(x6m11, _mm_set1_ps(self.twiddle5.im), m0215b);
            let m0215b = _mm_fnmadd_ps(x7m10, _mm_set1_ps(self.twiddle3.im), m0215b);
            let m0215b = _mm_fnmadd_ps(x8m9, _mm_set1_ps(self.twiddle1.im), m0215b);
            let (y02, y15) = AvxButterfly::butterfly2_f32_m128(m0215a, m0215b);

            let m0314a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle3.re), u0);
            let m0314a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle6.re), m0314a);
            let m0314a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle8.re), m0314a);
            let m0314a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle5.re), m0314a);
            let m0314a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle2.re), m0314a);
            let m0314a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle1.re), m0314a);
            let m0314a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle4.re), m0314a);
            let m0314a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle7.re), m0314a);
            let m0314b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle3.im));
            let m0314b = _mm_fmadd_ps(x2m15, _mm_set1_ps(self.twiddle6.im), m0314b);
            let m0314b = _mm_fnmadd_ps(x3m14, _mm_set1_ps(self.twiddle8.im), m0314b);
            let m0314b = _mm_fnmadd_ps(x4m13, _mm_set1_ps(self.twiddle5.im), m0314b);
            let m0314b = _mm_fnmadd_ps(x5m12, _mm_set1_ps(self.twiddle2.im), m0314b);
            let m0314b = _mm_fmadd_ps(x6m11, _mm_set1_ps(self.twiddle1.im), m0314b);
            let m0314b = _mm_fmadd_ps(x7m10, _mm_set1_ps(self.twiddle4.im), m0314b);
            let m0314b = _mm_fmadd_ps(x8m9, _mm_set1_ps(self.twiddle7.im), m0314b);
            let (y03, y14) = AvxButterfly::butterfly2_f32_m128(m0314a, m0314b);

            let m0413a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle4.re), u0);
            let m0413a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle8.re), m0413a);
            let m0413a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle5.re), m0413a);
            let m0413a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle1.re), m0413a);
            let m0413a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle3.re), m0413a);
            let m0413a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle7.re), m0413a);
            let m0413a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle6.re), m0413a);
            let m0413a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle2.re), m0413a);
            let m0413b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle4.im));
            let m0413b = _mm_fmadd_ps(x2m15, _mm_set1_ps(self.twiddle8.im), m0413b);
            let m0413b = _mm_fnmadd_ps(x3m14, _mm_set1_ps(self.twiddle5.im), m0413b);
            let m0413b = _mm_fnmadd_ps(x4m13, _mm_set1_ps(self.twiddle1.im), m0413b);
            let m0413b = _mm_fmadd_ps(x5m12, _mm_set1_ps(self.twiddle3.im), m0413b);
            let m0413b = _mm_fmadd_ps(x6m11, _mm_set1_ps(self.twiddle7.im), m0413b);
            let m0413b = _mm_fnmadd_ps(x7m10, _mm_set1_ps(self.twiddle6.im), m0413b);
            let m0413b = _mm_fnmadd_ps(x8m9, _mm_set1_ps(self.twiddle2.im), m0413b);
            let (y04, y13) = AvxButterfly::butterfly2_f32_m128(m0413a, m0413b);

            let m0512a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle5.re), u0);
            let m0512a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle7.re), m0512a);
            let m0512a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle2.re), m0512a);
            let m0512a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle3.re), m0512a);
            let m0512a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle8.re), m0512a);
            let m0512a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle4.re), m0512a);
            let m0512a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle1.re), m0512a);
            let m0512a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle6.re), m0512a);
            let m0512b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle5.im));
            let m0512b = _mm_fnmadd_ps(x2m15, _mm_set1_ps(self.twiddle7.im), m0512b);
            let m0512b = _mm_fnmadd_ps(x3m14, _mm_set1_ps(self.twiddle2.im), m0512b);
            let m0512b = _mm_fmadd_ps(x4m13, _mm_set1_ps(self.twiddle3.im), m0512b);
            let m0512b = _mm_fmadd_ps(x5m12, _mm_set1_ps(self.twiddle8.im), m0512b);
            let m0512b = _mm_fnmadd_ps(x6m11, _mm_set1_ps(self.twiddle4.im), m0512b);
            let m0512b = _mm_fmadd_ps(x7m10, _mm_set1_ps(self.twiddle1.im), m0512b);
            let m0512b = _mm_fmadd_ps(x8m9, _mm_set1_ps(self.twiddle6.im), m0512b);
            let (y05, y12) = AvxButterfly::butterfly2_f32_m128(m0512a, m0512b);

            let m0611a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle6.re), u0);
            let m0611a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle5.re), m0611a);
            let m0611a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle1.re), m0611a);
            let m0611a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle7.re), m0611a);
            let m0611a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle4.re), m0611a);
            let m0611a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle2.re), m0611a);
            let m0611a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle8.re), m0611a);
            let m0611a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle3.re), m0611a);
            let m0611b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle6.im));
            let m0611b = _mm_fnmadd_ps(x2m15, _mm_set1_ps(self.twiddle5.im), m0611b);
            let m0611b = _mm_fmadd_ps(x3m14, _mm_set1_ps(self.twiddle1.im), m0611b);
            let m0611b = _mm_fmadd_ps(x4m13, _mm_set1_ps(self.twiddle7.im), m0611b);
            let m0611b = _mm_fnmadd_ps(x5m12, _mm_set1_ps(self.twiddle4.im), m0611b);
            let m0611b = _mm_fmadd_ps(x6m11, _mm_set1_ps(self.twiddle2.im), m0611b);
            let m0611b = _mm_fmadd_ps(x7m10, _mm_set1_ps(self.twiddle8.im), m0611b);
            let m0611b = _mm_fnmadd_ps(x8m9, _mm_set1_ps(self.twiddle3.im), m0611b);
            let (y06, y11) = AvxButterfly::butterfly2_f32_m128(m0611a, m0611b);

            let m0710a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle7.re), u0);
            let m0710a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle3.re), m0710a);
            let m0710a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle4.re), m0710a);
            let m0710a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle6.re), m0710a);
            let m0710a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle1.re), m0710a);
            let m0710a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle8.re), m0710a);
            let m0710a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle2.re), m0710a);
            let m0710a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle5.re), m0710a);
            let m0710b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle7.im));
            let m0710b = _mm_fnmadd_ps(x2m15, _mm_set1_ps(self.twiddle3.im), m0710b);
            let m0710b = _mm_fmadd_ps(x3m14, _mm_set1_ps(self.twiddle4.im), m0710b);
            let m0710b = _mm_fnmadd_ps(x4m13, _mm_set1_ps(self.twiddle6.im), m0710b);
            let m0710b = _mm_fmadd_ps(x5m12, _mm_set1_ps(self.twiddle1.im), m0710b);
            let m0710b = _mm_fmadd_ps(x6m11, _mm_set1_ps(self.twiddle8.im), m0710b);
            let m0710b = _mm_fnmadd_ps(x7m10, _mm_set1_ps(self.twiddle2.im), m0710b);
            let m0710b = _mm_fmadd_ps(x8m9, _mm_set1_ps(self.twiddle5.im), m0710b);
            let (y07, y10) = AvxButterfly::butterfly2_f32_m128(m0710a, m0710b);

            let m0809a = _mm_fmadd_ps(x1p16, _mm_set1_ps(self.twiddle8.re), u0);
            let m0809a = _mm_fmadd_ps(x2p15, _mm_set1_ps(self.twiddle1.re), m0809a);
            let m0809a = _mm_fmadd_ps(x3p14, _mm_set1_ps(self.twiddle7.re), m0809a);
            let m0809a = _mm_fmadd_ps(x4p13, _mm_set1_ps(self.twiddle2.re), m0809a);
            let m0809a = _mm_fmadd_ps(x5p12, _mm_set1_ps(self.twiddle6.re), m0809a);
            let m0809a = _mm_fmadd_ps(x6p11, _mm_set1_ps(self.twiddle3.re), m0809a);
            let m0809a = _mm_fmadd_ps(x7p10, _mm_set1_ps(self.twiddle5.re), m0809a);
            let m0809a = _mm_fmadd_ps(x8p9, _mm_set1_ps(self.twiddle4.re), m0809a);
            let m0809b = _mm_mul_ps(x1m16, _mm_set1_ps(self.twiddle8.im));
            let m0809b = _mm_fnmadd_ps(x2m15, _mm_set1_ps(self.twiddle1.im), m0809b);
            let m0809b = _mm_fmadd_ps(x3m14, _mm_set1_ps(self.twiddle7.im), m0809b);
            let m0809b = _mm_fnmadd_ps(x4m13, _mm_set1_ps(self.twiddle2.im), m0809b);
            let m0809b = _mm_fmadd_ps(x5m12, _mm_set1_ps(self.twiddle6.im), m0809b);
            let m0809b = _mm_fnmadd_ps(x6m11, _mm_set1_ps(self.twiddle3.im), m0809b);
            let m0809b = _mm_fmadd_ps(x7m10, _mm_set1_ps(self.twiddle5.im), m0809b);
            let m0809b = _mm_fnmadd_ps(x8m9, _mm_set1_ps(self.twiddle4.im), m0809b);
            let (y08, y09) = AvxButterfly::butterfly2_f32_m128(m0809a, m0809b);
            [
                y00, y01, y02, y03, y04, y05, y06, y07, y08, y09, y10, y11, y12, y13, y14, y15, y16,
            ]
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 17 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(34) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16 = _m128s_load_f32x2(chunk.get_unchecked(16..).as_ptr().cast());

                let chunk1 = chunk.get_unchecked_mut(17..);

                let u0u1u2u3_1 = _mm256_loadu_ps(chunk1.as_ptr().cast());
                let u4u5u6u7_1 = _mm256_loadu_ps(chunk1.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11_1 = _mm256_loadu_ps(chunk1.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15_1 = _mm256_loadu_ps(chunk1.get_unchecked(12..).as_ptr().cast());
                let u16_1 = _m128s_load_f32x2(chunk1.get_unchecked(16..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u14u15 = _mm256_extractf128_ps::<1>(u12u13u14u15);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u12u13 = _mm256_castps256_ps128(u12u13u14u15);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u8u9 = _mm256_castps256_ps128(u8u9u10u11);

                let u2u3_1 = _mm256_extractf128_ps::<1>(u0u1u2u3_1);
                let u14u15_1 = _mm256_extractf128_ps::<1>(u12u13u14u15_1);
                let u4u5_1 = _mm256_castps256_ps128(u4u5u6u7_1);
                let u12u13_1 = _mm256_castps256_ps128(u12u13u14u15_1);
                let u10u11_1 = _mm256_extractf128_ps::<1>(u8u9u10u11_1);
                let u6u7_1 = _mm256_extractf128_ps::<1>(u4u5u6u7_1);
                let u8u9_1 = _mm256_castps256_ps128(u8u9u10u11_1);

                let q = self.kernel_f32([
                    _mm_unpacklo_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3_1),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3_1),
                    ),
                    _mm_unpacklo_ps64(u2u3, u2u3_1),
                    _mm_unpackhi_ps64(u2u3, u2u3_1), // 3
                    _mm_unpacklo_ps64(u4u5, u4u5_1),
                    _mm_unpackhi_ps64(u4u5, u4u5_1),
                    _mm_unpacklo_ps64(u6u7, u6u7_1), // 6
                    _mm_unpackhi_ps64(u6u7, u6u7_1),
                    _mm_unpacklo_ps64(u8u9, u8u9_1),
                    _mm_unpackhi_ps64(u8u9, u8u9_1),
                    _mm_unpacklo_ps64(u10u11, u10u11_1),
                    _mm_unpackhi_ps64(u10u11, u10u11_1),
                    _mm_unpacklo_ps64(u12u13, u12u13_1),
                    _mm_unpackhi_ps64(u12u13, u12u13_1),
                    _mm_unpacklo_ps64(u14u15, u14u15_1),
                    _mm_unpackhi_ps64(u14u15, u14u15_1),
                    _mm_unpacklo_ps64(u16, u16_1),
                ]);

                _m128s_store_f32x2(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), q[16]);

                _mm256_storeu_ps(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(q[0], q[1]), _mm_unpacklo_ps64(q[2], q[3])),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[12], q[13]),
                        _mm_unpacklo_ps64(q[14], q[15]),
                    ),
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

                // -- hi

                let chunk1 = chunk.get_unchecked_mut(17..);
                _m128s_storeh_f32x2(chunk1.get_unchecked_mut(16..).as_mut_ptr().cast(), q[16]);

                _mm256_storeu_ps(
                    chunk1.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpackhi_ps64(q[0], q[1]), _mm_unpackhi_ps64(q[2], q[3])),
                );

                _mm256_storeu_ps(
                    chunk1.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpackhi_ps64(q[12], q[13]),
                        _mm_unpackhi_ps64(q[14], q[15]),
                    ),
                );

                _mm256_storeu_ps(
                    chunk1.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpackhi_ps64(q[4], q[5]), _mm_unpackhi_ps64(q[6], q[7])),
                );

                _mm256_storeu_ps(
                    chunk1.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpackhi_ps64(q[8], q[9]),
                        _mm_unpackhi_ps64(q[10], q[11]),
                    ),
                );
            }

            let rem = in_place.chunks_exact_mut(34).into_remainder();

            for chunk in rem.chunks_exact_mut(17) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16 = _m128s_load_f32x2(chunk.get_unchecked(16..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u14u15 = _mm256_extractf128_ps::<1>(u12u13u14u15);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u12u13 = _mm256_castps256_ps128(u12u13u14u15);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u8u9 = _mm256_castps256_ps128(u8u9u10u11);

                let q = self.kernel_f32([
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_castps256_ps128(u0u1u2u3),
                    ),
                    u2u3,
                    _mm_unpackhi_ps64(u2u3, u2u3), // 3
                    u4u5,
                    _mm_unpackhi_ps64(u4u5, u4u5),
                    u6u7, // 6
                    _mm_unpackhi_ps64(u6u7, u6u7),
                    u8u9,
                    _mm_unpackhi_ps64(u8u9, u8u9),
                    u10u11,
                    _mm_unpackhi_ps64(u10u11, u10u11),
                    u12u13,
                    _mm_unpackhi_ps64(u12u13, u12u13),
                    u14u15,
                    _mm_unpackhi_ps64(u14u15, u14u15),
                    u16,
                ]);

                _m128s_store_f32x2(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), q[16]);

                _mm256_storeu_ps(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(q[0], q[1]), _mm_unpacklo_ps64(q[2], q[3])),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(q[12], q[13]),
                        _mm_unpacklo_ps64(q[14], q[15]),
                    ),
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
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly17<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        17
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly17;
    use rand::Rng;

    #[test]
    fn test_butterfly17_f32() {
        for i in 1..4 {
            let size = 17usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly17::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly17::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly17::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 17f32)).collect();

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
    fn test_butterfly17_f64() {
        for i in 1..4 {
            let size = 17usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly17::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly17::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly17::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 17f64)).collect();

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
