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
use crate::avx::mixed::{AvxStoreD, ColumnButterfly11d};
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxButterfly11d {
    direction: FftDirection,
    bf11: ColumnButterfly11d,
}

impl AvxButterfly11d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe { Self::new_init(fft_direction) }
    }
    #[target_feature(enable = "avx2")]
    fn new_init(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf11: ColumnButterfly11d::new(fft_direction),
        }
    }
}

impl FftExecutor<f64> for AvxButterfly11d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        11
    }
}

impl AvxButterfly11d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(11) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let mut rows = [AvxStoreD::zero(); 11];
            for chunk in in_place.chunks_exact_mut(22) {
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;
                for i in 0..5 {
                    let q0 = _mm256_loadu_pd(chunk.get_unchecked(i * 2..).as_ptr().cast());
                    let q1 = _mm256_loadu_pd(chunk.get_unchecked(i * 2 + 11..).as_ptr().cast());
                    rows[i * 2] = AvxStoreD::raw(_mm256_permute2f128_pd::<LO_LO>(q0, q1));
                    rows[i * 2 + 1] = AvxStoreD::raw(_mm256_permute2f128_pd::<HI_HI>(q0, q1));
                }

                let q0 = _mm_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let q1 = _mm_loadu_pd(chunk.get_unchecked(10 + 11..).as_ptr().cast());
                rows[10] = AvxStoreD::raw(_mm256_create_pd(q0, q1));

                rows = self.bf11.exec(rows);
                for i in 0..5 {
                    let r0 = rows[i * 2];
                    let r1 = rows[i * 2 + 1];
                    let new_row0 = AvxStoreD::raw(_mm256_permute2f128_pd::<LO_LO>(r0.v, r1.v));
                    let new_row1 = AvxStoreD::raw(_mm256_permute2f128_pd::<HI_HI>(r0.v, r1.v));
                    new_row0.write(chunk.get_unchecked_mut(i * 2..));
                    new_row1.write(chunk.get_unchecked_mut(i * 2 + 11..));
                }

                let r0 = rows[10];
                r0.write_lo(chunk.get_unchecked_mut(10..));
                r0.write_hi(chunk.get_unchecked_mut(10 + 11..));
            }
            let rem = in_place.chunks_exact_mut(22).into_remainder();
            for chunk in rem.chunks_exact_mut(11) {
                for i in 0..11 {
                    rows[i] = AvxStoreD::from_complex(chunk.get_unchecked(i));
                }
                rows = self.bf11.exec(rows);
                for i in 0..11 {
                    rows[i].write_lo(chunk.get_unchecked_mut(i..));
                }
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly11d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_impl(src, dst) }
    }
}

impl AvxButterfly11d {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(11) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(11) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let mut rows = [AvxStoreD::zero(); 11];

            for (dst, src) in dst.chunks_exact_mut(22).zip(src.chunks_exact(22)) {
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;
                for i in 0..5 {
                    let q0 = _mm256_loadu_pd(src.get_unchecked(i * 2..).as_ptr().cast());
                    let q1 = _mm256_loadu_pd(src.get_unchecked(i * 2 + 11..).as_ptr().cast());
                    rows[i * 2] = AvxStoreD::raw(_mm256_permute2f128_pd::<LO_LO>(q0, q1));
                    rows[i * 2 + 1] = AvxStoreD::raw(_mm256_permute2f128_pd::<HI_HI>(q0, q1));
                }

                let q0 = _mm_loadu_pd(src.get_unchecked(10..).as_ptr().cast());
                let q1 = _mm_loadu_pd(src.get_unchecked(10 + 11..).as_ptr().cast());
                rows[10] = AvxStoreD::raw(_mm256_create_pd(q0, q1));

                rows = self.bf11.exec(rows);
                for i in 0..5 {
                    let r0 = rows[i * 2];
                    let r1 = rows[i * 2 + 1];
                    let new_row0 = AvxStoreD::raw(_mm256_permute2f128_pd::<LO_LO>(r0.v, r1.v));
                    let new_row1 = AvxStoreD::raw(_mm256_permute2f128_pd::<HI_HI>(r0.v, r1.v));
                    new_row0.write(dst.get_unchecked_mut(i * 2..));
                    new_row1.write(dst.get_unchecked_mut(i * 2 + 11..));
                }

                let r0 = rows[10];
                r0.write_lo(dst.get_unchecked_mut(10..));
                r0.write_hi(dst.get_unchecked_mut(10 + 11..));
            }

            let rem_dst = dst.chunks_exact_mut(22).into_remainder();
            let rem_src = src.chunks_exact(22).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(11).zip(rem_src.chunks_exact(11)) {
                for i in 0..11 {
                    rows[i] = AvxStoreD::from_complex(src.get_unchecked(i));
                }
                rows = self.bf11.exec(rows);
                for i in 0..11 {
                    rows[i].write_lo(dst.get_unchecked_mut(i..));
                }
            }
        }
        Ok(())
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly11d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

pub(crate) struct AvxButterfly11<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly11<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 11, fft_direction),
            twiddle2: compute_twiddle(2, 11, fft_direction),
            twiddle3: compute_twiddle(3, 11, fft_direction),
            twiddle4: compute_twiddle(4, 11, fft_direction),
            twiddle5: compute_twiddle(5, 11, fft_direction),
        }
    }
}

impl AvxButterfly11<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(11) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

        unsafe {
            for chunk in in_place.chunks_exact_mut(22) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(chunk.get_unchecked(7..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(chunk.get_unchecked(11..).as_ptr().cast());
                let u4u5u6u7_2 = _mm256_loadu_ps(chunk.get_unchecked(15..).as_ptr().cast());
                let u7u8u9u10_2 = _mm256_loadu_ps(chunk.get_unchecked(18..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let u0u1_2 = _mm256_castps256_ps128(u0u1u2u3_2);
                let u9u10_2 = _mm256_extractf128_ps::<1>(u7u8u9u10_2);
                let u2u3_2 = _mm256_extractf128_ps::<1>(u0u1u2u3_2);
                let u7u8_2 = _mm256_castps256_ps128(u7u8u9u10_2);
                let u4u5_2 = _mm256_castps256_ps128(u4u5u6u7_2);
                let u6u7_2 = _mm256_extractf128_ps::<1>(u4u5u6u7_2);

                let u0 = _mm_unpacklo_ps64(u0u1, u0u1_2);
                let y00 = u0;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u0u1, u0u1_2),
                    _mm_unpackhi_ps64(u9u10, u9u10_2),
                ); // u1, u10
                let x1m10 = rotate.rotate_m128(x1m10);
                let y00 = _mm_add_ps(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpacklo_ps64(u2u3, u2u3_2),
                    _mm_unpacklo_ps64(u9u10, u9u10_2),
                ); // u2, u9
                let x2m9 = rotate.rotate_m128(x2m9);
                let y00 = _mm_add_ps(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u2u3, u2u3_2),
                    _mm_unpackhi_ps64(u7u8, u7u8_2),
                ); // u3, u8
                let x3m8 = rotate.rotate_m128(x3m8);
                let y00 = _mm_add_ps(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpacklo_ps64(u4u5, u4u5_2),
                    _mm_unpacklo_ps64(u7u8, u7u8_2),
                ); // u4, u7
                let x4m7 = rotate.rotate_m128(x4m7);
                let y00 = _mm_add_ps(y00, x4p7);
                let (x5p6, x5m6) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u4u5, u4u5_2),
                    _mm_unpacklo_ps64(u6u7, u6u7_2),
                ); // u5, u6
                let x5m6 = rotate.rotate_m128(x5m6);
                let y00 = _mm_add_ps(y00, x5p6);

                let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                let y0000 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y0001 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y07, y08), _mm_unpacklo_ps64(y09, y10));

                let y0000_2 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y00, y01), _mm_unpackhi_ps64(y02, y03));
                let y0001_2 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y04, y05), _mm_unpackhi_ps64(y06, y07));
                let y0002_2 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y07, y08), _mm_unpackhi_ps64(y09, y10));

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);

                _mm256_storeu_ps(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), y0000_2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), y0001_2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y0002_2);
            }

            let rem = in_place.chunks_exact_mut(22).into_remainder();

            for chunk in rem.chunks_exact_mut(11) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(chunk.get_unchecked(7..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let u0 = u0u1;
                let y00 = u0u1;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    _mm_unpackhi_ps64(u9u10, u9u10),
                );
                let x1m10 = rotate.rotate_m128(x1m10);
                let y00 = _mm_add_ps(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(u2u3, u9u10);
                let x2m9 = rotate.rotate_m128(x2m9);
                let y00 = _mm_add_ps(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    _mm_unpackhi_ps64(u7u8, u7u8),
                );
                let x3m8 = rotate.rotate_m128(x3m8);
                let y00 = _mm_add_ps(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f32_m128(u4u5, u7u8);
                let x4m7 = rotate.rotate_m128(x4m7);
                let y00 = _mm_add_ps(y00, x4p7);
                let (x5p6, x5m6) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u4u5, u4u5), u6u7);
                let x5m6 = rotate.rotate_m128(x5m6);
                let y00 = _mm_add_ps(y00, x5p6);

                let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                let y0000 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y0001 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y07, y08), _mm_unpacklo_ps64(y09, y10));

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe {
            if !src.len().is_multiple_of(11) {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if !dst.len().is_multiple_of(11) {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            for (dst, src) in dst.chunks_exact_mut(22).zip(src.chunks_exact(22)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(src.get_unchecked(7..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(src.get_unchecked(11..).as_ptr().cast());
                let u4u5u6u7_2 = _mm256_loadu_ps(src.get_unchecked(15..).as_ptr().cast());
                let u7u8u9u10_2 = _mm256_loadu_ps(src.get_unchecked(18..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let u0u1_2 = _mm256_castps256_ps128(u0u1u2u3_2);
                let u9u10_2 = _mm256_extractf128_ps::<1>(u7u8u9u10_2);
                let u2u3_2 = _mm256_extractf128_ps::<1>(u0u1u2u3_2);
                let u7u8_2 = _mm256_castps256_ps128(u7u8u9u10_2);
                let u4u5_2 = _mm256_castps256_ps128(u4u5u6u7_2);
                let u6u7_2 = _mm256_extractf128_ps::<1>(u4u5u6u7_2);

                let u0 = _mm_unpacklo_ps64(u0u1, u0u1_2);
                let y00 = u0;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u0u1, u0u1_2),
                    _mm_unpackhi_ps64(u9u10, u9u10_2),
                ); // u1, u10
                let x1m10 = rotate.rotate_m128(x1m10);
                let y00 = _mm_add_ps(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpacklo_ps64(u2u3, u2u3_2),
                    _mm_unpacklo_ps64(u9u10, u9u10_2),
                ); // u2, u9
                let x2m9 = rotate.rotate_m128(x2m9);
                let y00 = _mm_add_ps(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u2u3, u2u3_2),
                    _mm_unpackhi_ps64(u7u8, u7u8_2),
                ); // u3, u8
                let x3m8 = rotate.rotate_m128(x3m8);
                let y00 = _mm_add_ps(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpacklo_ps64(u4u5, u4u5_2),
                    _mm_unpacklo_ps64(u7u8, u7u8_2),
                ); // u4, u7
                let x4m7 = rotate.rotate_m128(x4m7);
                let y00 = _mm_add_ps(y00, x4p7);
                let (x5p6, x5m6) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u4u5, u4u5_2),
                    _mm_unpacklo_ps64(u6u7, u6u7_2),
                ); // u5, u6
                let x5m6 = rotate.rotate_m128(x5m6);
                let y00 = _mm_add_ps(y00, x5p6);

                let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                let y0000 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y0001 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y07, y08), _mm_unpacklo_ps64(y09, y10));

                let y0000_2 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y00, y01), _mm_unpackhi_ps64(y02, y03));
                let y0001_2 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y04, y05), _mm_unpackhi_ps64(y06, y07));
                let y0002_2 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y07, y08), _mm_unpackhi_ps64(y09, y10));

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);

                _mm256_storeu_ps(dst.get_unchecked_mut(11..).as_mut_ptr().cast(), y0000_2);
                _mm256_storeu_ps(dst.get_unchecked_mut(15..).as_mut_ptr().cast(), y0001_2);
                _mm256_storeu_ps(dst.get_unchecked_mut(18..).as_mut_ptr().cast(), y0002_2);
            }

            let rem_dst = dst.chunks_exact_mut(22).into_remainder();
            let rem_src = src.chunks_exact(22).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(11).zip(rem_src.chunks_exact(11)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u7u8u9u10 = _mm256_loadu_ps(src.get_unchecked(7..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u9u10 = _mm256_extractf128_ps::<1>(u7u8u9u10);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u7u8 = _mm256_castps256_ps128(u7u8u9u10);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);

                let u0 = u0u1;
                let y00 = u0u1;
                let (x1p10, x1m10) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    _mm_unpackhi_ps64(u9u10, u9u10),
                );
                let x1m10 = rotate.rotate_m128(x1m10);
                let y00 = _mm_add_ps(y00, x1p10);
                let (x2p9, x2m9) = AvxButterfly::butterfly2_f32_m128(u2u3, u9u10);
                let x2m9 = rotate.rotate_m128(x2m9);
                let y00 = _mm_add_ps(y00, x2p9);
                let (x3p8, x3m8) = AvxButterfly::butterfly2_f32_m128(
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    _mm_unpackhi_ps64(u7u8, u7u8),
                );
                let x3m8 = rotate.rotate_m128(x3m8);
                let y00 = _mm_add_ps(y00, x3p8);
                let (x4p7, x4m7) = AvxButterfly::butterfly2_f32_m128(u4u5, u7u8);
                let x4m7 = rotate.rotate_m128(x4m7);
                let y00 = _mm_add_ps(y00, x4p7);
                let (x5p6, x5m6) =
                    AvxButterfly::butterfly2_f32_m128(_mm_unpackhi_ps64(u4u5, u4u5), u6u7);
                let x5m6 = rotate.rotate_m128(x5m6);
                let y00 = _mm_add_ps(y00, x5p6);

                let m0110a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle1.re), u0);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p9, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p8, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.re), x4p7, m0110a);
                let m0110a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle5.re), x5p6, m0110a);
                let m0110b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle1.im));
                let m0110b = _mm_fmadd_ps(x2m9, _mm_set1_ps(self.twiddle2.im), m0110b);
                let m0110b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle3.im), m0110b);
                let m0110b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle4.im), m0110b);
                let m0110b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle5.im), m0110b);
                let (y01, y10) = AvxButterfly::butterfly2_f32_m128(m0110a, m0110b);

                let m0209a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle2.re), u0);
                let m0209a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle4.re), m0209a);
                let m0209a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle5.re), m0209a);
                let m0209a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle3.re), m0209a);
                let m0209a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle1.re), m0209a);
                let m0209b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle2.im));
                let m0209b = _mm_fmadd_ps(_mm_set1_ps(self.twiddle4.im), x2m9, m0209b);
                let m0209b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle5.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle3.im), m0209b);
                let m0209b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle1.im), m0209b);
                let (y02, y09) = AvxButterfly::butterfly2_f32_m128(m0209a, m0209b);

                let m0308a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle3.re), u0);
                let m0308a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle5.re), m0308a);
                let m0308a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle2.re), m0308a);
                let m0308a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle1.re), m0308a);
                let m0308a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle4.re), m0308a);
                let m0308b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle3.im));
                let m0308b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle5.im), m0308b);
                let m0308b = _mm_fnmadd_ps(x3m8, _mm_set1_ps(self.twiddle2.im), m0308b);
                let m0308b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle1.im), m0308b);
                let m0308b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle4.im), m0308b);
                let (y03, y08) = AvxButterfly::butterfly2_f32_m128(m0308a, m0308b);

                let m0407a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle4.re), u0);
                let m0407a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle3.re), m0407a);
                let m0407a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle1.re), m0407a);
                let m0407a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle5.re), m0407a);
                let m0407a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle2.re), m0407a);
                let m0407b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle4.im));
                let m0407b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle3.im), m0407b);
                let m0407b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle1.im), m0407b);
                let m0407b = _mm_fmadd_ps(x4m7, _mm_set1_ps(self.twiddle5.im), m0407b);
                let m0407b = _mm_fnmadd_ps(x5m6, _mm_set1_ps(self.twiddle2.im), m0407b);
                let (y04, y07) = AvxButterfly::butterfly2_f32_m128(m0407a, m0407b);

                let m0506a = _mm_fmadd_ps(x1p10, _mm_set1_ps(self.twiddle5.re), u0);
                let m0506a = _mm_fmadd_ps(x2p9, _mm_set1_ps(self.twiddle1.re), m0506a);
                let m0506a = _mm_fmadd_ps(x3p8, _mm_set1_ps(self.twiddle4.re), m0506a);
                let m0506a = _mm_fmadd_ps(x4p7, _mm_set1_ps(self.twiddle2.re), m0506a);
                let m0506a = _mm_fmadd_ps(x5p6, _mm_set1_ps(self.twiddle3.re), m0506a);
                let m0506b = _mm_mul_ps(x1m10, _mm_set1_ps(self.twiddle5.im));
                let m0506b = _mm_fnmadd_ps(x2m9, _mm_set1_ps(self.twiddle1.im), m0506b);
                let m0506b = _mm_fmadd_ps(x3m8, _mm_set1_ps(self.twiddle4.im), m0506b);
                let m0506b = _mm_fnmadd_ps(x4m7, _mm_set1_ps(self.twiddle2.im), m0506b);
                let m0506b = _mm_fmadd_ps(x5m6, _mm_set1_ps(self.twiddle3.im), m0506b);
                let (y05, y06) = AvxButterfly::butterfly2_f32_m128(m0506a, m0506b);

                let y0000 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y0001 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y04, y05), _mm_unpacklo_ps64(y06, y07));
                let y0002 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y07, y08), _mm_unpacklo_ps64(y09, y10));

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0000);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y0001);
                _mm256_storeu_ps(dst.get_unchecked_mut(7..).as_mut_ptr().cast(), y0002);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly11<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly11<f32> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly11<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        11
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly11, f32, AvxButterfly11, 11, 1e-5);
    test_avx_butterfly!(test_avx_butterfly11_f64, f64, AvxButterfly11d, 11, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly11, f32, AvxButterfly11, 11, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly11_f64, f64, AvxButterfly11d, 11, 1e-7);
}
