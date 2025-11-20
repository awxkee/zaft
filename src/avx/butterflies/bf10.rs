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
use crate::avx::butterflies::fast_bf5::{AvxFastButterfly5d, AvxFastButterfly5f};
use crate::avx::util::{
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm_unpacklohi_ps64, _mm256_create_pd, _mm256_create_ps,
    _mm256_permute4x64_ps, shuffle,
};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxButterfly10d {
    direction: FftDirection,
    bf5: AvxFastButterfly5d,
}

impl AvxButterfly10d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5d::new(fft_direction) },
        }
    }
}

impl AvxButterfly10d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(20) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());

                let u0u1_2 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u2u3_2 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u4u5_2 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());
                let u6u7_2 = _mm256_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());
                let u8u9_2 = _mm256_loadu_pd(chunk.get_unchecked(18..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let mid0 = self.bf5.exec(
                    _mm256_permute2f128_pd::<LO_LO>(u0u1, u0u1_2),
                    _mm256_permute2f128_pd::<LO_LO>(u2u3, u2u3_2),
                    _mm256_permute2f128_pd::<LO_LO>(u4u5, u4u5_2),
                    _mm256_permute2f128_pd::<LO_LO>(u6u7, u6u7_2),
                    _mm256_permute2f128_pd::<LO_LO>(u8u9, u8u9_2),
                );
                let mid1 = self.bf5.exec(
                    _mm256_permute2f128_pd::<HI_HI>(u4u5, u4u5_2),
                    _mm256_permute2f128_pd::<HI_HI>(u6u7, u6u7_2),
                    _mm256_permute2f128_pd::<HI_HI>(u8u9, u8u9_2),
                    _mm256_permute2f128_pd::<HI_HI>(u0u1, u0u1_2),
                    _mm256_permute2f128_pd::<HI_HI>(u2u3, u2u3_2),
                );

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f64(mid0.0, mid1.0);
                let (y2, y3) = AvxButterfly::butterfly2_f64(mid0.1, mid1.1);
                let (y4, y5) = AvxButterfly::butterfly2_f64(mid0.2, mid1.2);
                let (y6, y7) = AvxButterfly::butterfly2_f64(mid0.3, mid1.3);
                let (y8, y9) = AvxButterfly::butterfly2_f64(mid0.4, mid1.4);

                _mm256_storeu_pd(
                    chunk.as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y0, y3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y4, y7),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y8, y1),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y2, y5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y6, y9),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y0, y3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y4, y7),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y8, y1),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y2, y5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y6, y9),
                );
            }

            let remainder = in_place.chunks_exact_mut(20).into_remainder();

            for chunk in remainder.chunks_exact_mut(10) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;

                let u0u5 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u4u5);
                let u2u7 = _mm256_permute2f128_pd::<LO_HI>(u2u3, u6u7);
                let u4u9 = _mm256_permute2f128_pd::<LO_HI>(u4u5, u8u9);
                let u6u1 = _mm256_permute2f128_pd::<LO_HI>(u6u7, u0u1);
                let u8u3 = _mm256_permute2f128_pd::<LO_HI>(u8u9, u2u3);

                let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.0),
                    _mm256_extractf128_pd::<1>(mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.1),
                    _mm256_extractf128_pd::<1>(mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.2),
                    _mm256_extractf128_pd::<1>(mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.3),
                    _mm256_extractf128_pd::<1>(mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.4),
                    _mm256_extractf128_pd::<1>(mid0mid1.4),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_create_pd(y0, y3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y4, y7),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y8, y1),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y2, y5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(y6, y9),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe {
            if src.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            for (dst, src) in dst.chunks_exact_mut(20).zip(src.chunks_exact(20)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());

                let u0u1_2 = _mm256_loadu_pd(src.get_unchecked(10..).as_ptr().cast());
                let u2u3_2 = _mm256_loadu_pd(src.get_unchecked(12..).as_ptr().cast());
                let u4u5_2 = _mm256_loadu_pd(src.get_unchecked(14..).as_ptr().cast());
                let u6u7_2 = _mm256_loadu_pd(src.get_unchecked(16..).as_ptr().cast());
                let u8u9_2 = _mm256_loadu_pd(src.get_unchecked(18..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let mid0 = self.bf5.exec(
                    _mm256_permute2f128_pd::<LO_LO>(u0u1, u0u1_2),
                    _mm256_permute2f128_pd::<LO_LO>(u2u3, u2u3_2),
                    _mm256_permute2f128_pd::<LO_LO>(u4u5, u4u5_2),
                    _mm256_permute2f128_pd::<LO_LO>(u6u7, u6u7_2),
                    _mm256_permute2f128_pd::<LO_LO>(u8u9, u8u9_2),
                );
                let mid1 = self.bf5.exec(
                    _mm256_permute2f128_pd::<HI_HI>(u4u5, u4u5_2),
                    _mm256_permute2f128_pd::<HI_HI>(u6u7, u6u7_2),
                    _mm256_permute2f128_pd::<HI_HI>(u8u9, u8u9_2),
                    _mm256_permute2f128_pd::<HI_HI>(u0u1, u0u1_2),
                    _mm256_permute2f128_pd::<HI_HI>(u2u3, u2u3_2),
                );

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f64(mid0.0, mid1.0);
                let (y2, y3) = AvxButterfly::butterfly2_f64(mid0.1, mid1.1);
                let (y4, y5) = AvxButterfly::butterfly2_f64(mid0.2, mid1.2);
                let (y6, y7) = AvxButterfly::butterfly2_f64(mid0.3, mid1.3);
                let (y8, y9) = AvxButterfly::butterfly2_f64(mid0.4, mid1.4);

                _mm256_storeu_pd(
                    dst.as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y0, y3),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y4, y7),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y8, y1),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y2, y5),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<LO_LO>(y6, y9),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y0, y3),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y4, y7),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y8, y1),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y2, y5),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_permute2f128_pd::<HI_HI>(y6, y9),
                );
            }

            let rem_src = src.chunks_exact(20).remainder();
            let rem_dst = dst.chunks_exact_mut(20).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(10).zip(rem_src.chunks_exact(10)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;

                let u0u5 = _mm256_permute2f128_pd::<LO_HI>(u0u1, u4u5);
                let u2u7 = _mm256_permute2f128_pd::<LO_HI>(u2u3, u6u7);
                let u4u9 = _mm256_permute2f128_pd::<LO_HI>(u4u5, u8u9);
                let u6u1 = _mm256_permute2f128_pd::<LO_HI>(u6u7, u0u1);
                let u8u3 = _mm256_permute2f128_pd::<LO_HI>(u8u9, u2u3);

                let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.0),
                    _mm256_extractf128_pd::<1>(mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.1),
                    _mm256_extractf128_pd::<1>(mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.2),
                    _mm256_extractf128_pd::<1>(mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.3),
                    _mm256_extractf128_pd::<1>(mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(mid0mid1.4),
                    _mm256_extractf128_pd::<1>(mid0mid1.4),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm256_create_pd(y0, y3),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(y4, y7),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(y8, y1),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(y2, y5),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(y6, y9),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly10d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly10d {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly10d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        10
    }
}

pub(crate) struct AvxButterfly10f {
    direction: FftDirection,
    bf5: AvxFastButterfly5f,
}

impl AvxButterfly10f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5f::new(fft_direction) },
        }
    }
}

impl AvxButterfly10f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(20) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7u8u9 = _mm256_loadu_ps(chunk.get_unchecked(6..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(chunk.get_unchecked(10..).as_ptr().cast());
                let u4u5u6u7_2 = _mm256_loadu_ps(chunk.get_unchecked(14..).as_ptr().cast());
                let u6u7u8u9_2 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u8u9 = _mm256_extractf128_ps::<1>(u6u7u8u9);

                let u2u3_2 = _mm256_extractf128_ps::<1>(u0u1u2u3_2);
                let u8u9_2 = _mm256_extractf128_ps::<1>(u6u7u8u9_2);

                let u0u5 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm256_castps256_ps128(u4u5u6u7),
                );

                let u0u5_2 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u0u1u2u3_2),
                    _mm256_castps256_ps128(u4u5u6u7_2),
                );

                let u2u7 = _mm_unpacklohi_ps64(u2u3, _mm256_castps256_ps128(u6u7u8u9));
                let u2u7_2 = _mm_unpacklohi_ps64(u2u3_2, _mm256_castps256_ps128(u6u7u8u9_2));

                let u4u9 = _mm_unpacklohi_ps64(_mm256_castps256_ps128(u4u5u6u7), u8u9);
                let u4u9_2 = _mm_unpacklohi_ps64(_mm256_castps256_ps128(u4u5u6u7_2), u8u9_2);

                let u6u1 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u6u7u8u9),
                    _mm256_castps256_ps128(u0u1u2u3),
                );

                let u6u1_2 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u6u7u8u9_2),
                    _mm256_castps256_ps128(u0u1u2u3_2),
                );

                let u8u3 = _mm_unpacklohi_ps64(u8u9, u2u3);
                let u8u3_2 = _mm_unpacklohi_ps64(u8u9_2, u2u3_2);

                const SH: i32 = shuffle(3, 1, 2, 0);

                let mid0mid1 = self.bf5._m256_exec(
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u0u5, u0u5_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u2u7, u2u7_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u4u9, u4u9_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u6u1, u6u1_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u8u3, u8u3_2)),
                );

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.0),
                    _mm256_extractf128_ps::<1>(mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.1),
                    _mm256_extractf128_ps::<1>(mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.2),
                    _mm256_extractf128_ps::<1>(mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.3),
                    _mm256_extractf128_ps::<1>(mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.4),
                    _mm256_extractf128_ps::<1>(mid0mid1.4),
                );

                let y0y3 = _mm_unpacklo_ps64(y0, y3);
                let y4y7 = _mm_unpacklo_ps64(y4, y7);
                let y8y1 = _mm_unpacklo_ps64(y8, y1);
                let y2y5 = _mm_unpacklo_ps64(y2, y5);
                let y6y9 = _mm_unpacklo_ps64(y6, y9);

                let y0y3_2 = _mm_unpackhi_ps64(y0, y3);
                let y4y7_2 = _mm_unpackhi_ps64(y4, y7);
                let y8y1_2 = _mm_unpackhi_ps64(y8, y1);
                let y2y5_2 = _mm_unpackhi_ps64(y2, y5);
                let y6y9_2 = _mm_unpackhi_ps64(y6, y9);

                let yyyy0 = _mm256_create_ps(y0y3, y4y7);
                let yyyy1 = _mm256_create_ps(y8y1, y2y5);

                let yyyy0_2 = _mm256_create_ps(y0y3_2, y4y7_2);
                let yyyy1_2 = _mm256_create_ps(y8y1_2, y2y5_2);

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), yyyy0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), yyyy1);
                _mm_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y6y9);

                _mm256_storeu_ps(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), yyyy0_2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), yyyy1_2);
                _mm_storeu_ps(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), y6y9_2);
            }

            let rem = in_place.chunks_exact_mut(20).into_remainder();

            for chunk in rem.chunks_exact_mut(10) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7u8u9 = _mm256_loadu_ps(chunk.get_unchecked(6..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u8u9 = _mm256_extractf128_ps::<1>(u6u7u8u9);

                let u0u5 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm256_castps256_ps128(u4u5u6u7),
                );
                let u2u7 = _mm_unpacklohi_ps64(u2u3, _mm256_castps256_ps128(u6u7u8u9));
                let u4u9 = _mm_unpacklohi_ps64(_mm256_castps256_ps128(u4u5u6u7), u8u9);
                let u6u1 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u6u7u8u9),
                    _mm256_castps256_ps128(u0u1u2u3),
                );
                let u8u3 = _mm_unpacklohi_ps64(u8u9, u2u3);

                let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.0,
                    _mm_unpackhi_ps64(mid0mid1.0, mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.1,
                    _mm_unpackhi_ps64(mid0mid1.1, mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.2,
                    _mm_unpackhi_ps64(mid0mid1.2, mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.3,
                    _mm_unpackhi_ps64(mid0mid1.3, mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.4,
                    _mm_unpackhi_ps64(mid0mid1.4, mid0mid1.4),
                );

                let y0y3 = _mm_unpacklo_ps64(y0, y3);
                let y4y7 = _mm_unpacklo_ps64(y4, y7);
                let y8y1 = _mm_unpacklo_ps64(y8, y1);
                let y2y5 = _mm_unpacklo_ps64(y2, y5);
                let y6y9 = _mm_unpacklo_ps64(y6, y9);

                let yyyy0 = _mm256_create_ps(y0y3, y4y7);
                let yyyy1 = _mm256_create_ps(y8y1, y2y5);

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), yyyy0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), yyyy1);
                _mm_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y6y9);
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
            if src.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 10 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            for (dst, src) in dst.chunks_exact_mut(20).zip(src.chunks_exact(20)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u6u7u8u9 = _mm256_loadu_ps(src.get_unchecked(6..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(src.get_unchecked(10..).as_ptr().cast());
                let u4u5u6u7_2 = _mm256_loadu_ps(src.get_unchecked(14..).as_ptr().cast());
                let u6u7u8u9_2 = _mm256_loadu_ps(src.get_unchecked(16..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u8u9 = _mm256_extractf128_ps::<1>(u6u7u8u9);

                let u2u3_2 = _mm256_extractf128_ps::<1>(u0u1u2u3_2);
                let u8u9_2 = _mm256_extractf128_ps::<1>(u6u7u8u9_2);

                let u0u5 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm256_castps256_ps128(u4u5u6u7),
                );

                let u0u5_2 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u0u1u2u3_2),
                    _mm256_castps256_ps128(u4u5u6u7_2),
                );

                let u2u7 = _mm_unpacklohi_ps64(u2u3, _mm256_castps256_ps128(u6u7u8u9));
                let u2u7_2 = _mm_unpacklohi_ps64(u2u3_2, _mm256_castps256_ps128(u6u7u8u9_2));

                let u4u9 = _mm_unpacklohi_ps64(_mm256_castps256_ps128(u4u5u6u7), u8u9);
                let u4u9_2 = _mm_unpacklohi_ps64(_mm256_castps256_ps128(u4u5u6u7_2), u8u9_2);

                let u6u1 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u6u7u8u9),
                    _mm256_castps256_ps128(u0u1u2u3),
                );

                let u6u1_2 = _mm_unpacklohi_ps64(
                    _mm256_castps256_ps128(u6u7u8u9_2),
                    _mm256_castps256_ps128(u0u1u2u3_2),
                );

                let u8u3 = _mm_unpacklohi_ps64(u8u9, u2u3);
                let u8u3_2 = _mm_unpacklohi_ps64(u8u9_2, u2u3_2);

                const SH: i32 = shuffle(3, 1, 2, 0);

                let mid0mid1 = self.bf5._m256_exec(
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u0u5, u0u5_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u2u7, u2u7_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u4u9, u4u9_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u6u1, u6u1_2)),
                    _mm256_permute4x64_ps::<SH>(_mm256_create_ps(u8u3, u8u3_2)),
                );

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.0),
                    _mm256_extractf128_ps::<1>(mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.1),
                    _mm256_extractf128_ps::<1>(mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.2),
                    _mm256_extractf128_ps::<1>(mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.3),
                    _mm256_extractf128_ps::<1>(mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(mid0mid1.4),
                    _mm256_extractf128_ps::<1>(mid0mid1.4),
                );

                let y0y3 = _mm_unpacklo_ps64(y0, y3);
                let y4y7 = _mm_unpacklo_ps64(y4, y7);
                let y8y1 = _mm_unpacklo_ps64(y8, y1);
                let y2y5 = _mm_unpacklo_ps64(y2, y5);
                let y6y9 = _mm_unpacklo_ps64(y6, y9);

                let y0y3_2 = _mm_unpackhi_ps64(y0, y3);
                let y4y7_2 = _mm_unpackhi_ps64(y4, y7);
                let y8y1_2 = _mm_unpackhi_ps64(y8, y1);
                let y2y5_2 = _mm_unpackhi_ps64(y2, y5);
                let y6y9_2 = _mm_unpackhi_ps64(y6, y9);

                let yyyy0 = _mm256_create_ps(y0y3, y4y7);
                let yyyy1 = _mm256_create_ps(y8y1, y2y5);

                let yyyy0_2 = _mm256_create_ps(y0y3_2, y4y7_2);
                let yyyy1_2 = _mm256_create_ps(y8y1_2, y2y5_2);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), yyyy0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), yyyy1);
                _mm_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y6y9);

                _mm256_storeu_ps(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), yyyy0_2);
                _mm256_storeu_ps(dst.get_unchecked_mut(14..).as_mut_ptr().cast(), yyyy1_2);
                _mm_storeu_ps(dst.get_unchecked_mut(18..).as_mut_ptr().cast(), y6y9_2);
            }

            let rem_dst = dst.chunks_exact_mut(20).into_remainder();
            let rem_src = src.chunks_exact(20).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(10).zip(rem_src.chunks_exact(10)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u6u7u8u9 = _mm256_loadu_ps(src.get_unchecked(6..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u8u9 = _mm256_extractf128_ps::<1>(u6u7u8u9);

                let u0u5 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm256_castps256_ps128(u4u5u6u7),
                );
                let u2u7 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    u2u3,
                    _mm256_castps256_ps128(u6u7u8u9),
                );
                let u4u9 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    _mm256_castps256_ps128(u4u5u6u7),
                    u8u9,
                );
                let u6u1 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(
                    _mm256_castps256_ps128(u6u7u8u9),
                    _mm256_castps256_ps128(u0u1u2u3),
                );
                let u8u3 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(u8u9, u2u3);

                let mid0mid1 = self.bf5.exec(u0u5, u2u7, u4u9, u6u1, u8u3);

                // Since this is good-thomas algorithm, we don't need twiddle factors
                let (y0, y1) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.0,
                    _mm_unpackhi_ps64(mid0mid1.0, mid0mid1.0),
                );
                let (y2, y3) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.1,
                    _mm_unpackhi_ps64(mid0mid1.1, mid0mid1.1),
                );
                let (y4, y5) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.2,
                    _mm_unpackhi_ps64(mid0mid1.2, mid0mid1.2),
                );
                let (y6, y7) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.3,
                    _mm_unpackhi_ps64(mid0mid1.3, mid0mid1.3),
                );
                let (y8, y9) = AvxButterfly::butterfly2_f32_m128(
                    mid0mid1.4,
                    _mm_unpackhi_ps64(mid0mid1.4, mid0mid1.4),
                );

                let y0y3 = _mm_unpacklo_ps64(y0, y3);
                let y4y7 = _mm_unpacklo_ps64(y4, y7);
                let y8y1 = _mm_unpacklo_ps64(y8, y1);
                let y2y5 = _mm_unpacklo_ps64(y2, y5);
                let y6y9 = _mm_unpacklo_ps64(y6, y9);

                let yyyy0 = _mm256_create_ps(y0y3, y4y7);
                let yyyy1 = _mm256_create_ps(y8y1, y2y5);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), yyyy0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), yyyy1);
                _mm_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y6y9);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly10f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly10f {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly10f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        10
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly10, f32, AvxButterfly10f, 10, 1e-5);
    test_avx_butterfly!(test_avx_butterfly10_f64, f64, AvxButterfly10d, 10, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly10, f32, AvxButterfly10f, 10, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly10_f64, f64, AvxButterfly10d, 10, 1e-7);
}
