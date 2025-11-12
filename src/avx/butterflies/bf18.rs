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
use crate::avx::butterflies::fast_bf9::AvxFastButterfly9f;
use crate::avx::butterflies::fast_bf9d::AvxFastButterfly9d;
use crate::avx::butterflies::{
    AvxButterfly, shift_load2, shift_load2dd, shift_load4, shift_store2, shift_store2dd,
    shift_store4,
};
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly18d {
    direction: FftDirection,
    bf9: AvxFastButterfly9d,
}

impl AvxButterfly18d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf9: unsafe { AvxFastButterfly9d::new(fft_direction) },
        }
    }
}

impl AvxButterfly18d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 18 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(36) {
                let (u0, u3) = shift_load2dd!(chunk, 18, 0);
                let (u4, u7) = shift_load2dd!(chunk, 18, 2);
                let (u8, u11) = shift_load2dd!(chunk, 18, 4);
                let (u12, u15) = shift_load2dd!(chunk, 18, 6);
                let (u16, u1) = shift_load2dd!(chunk, 18, 8);
                let (u2, u5) = shift_load2dd!(chunk, 18, 10);
                let (u6, u9) = shift_load2dd!(chunk, 18, 12);
                let (u10, u13) = shift_load2dd!(chunk, 18, 14);
                let (u14, u17) = shift_load2dd!(chunk, 18, 16);

                let (t0, t1) = AvxButterfly::butterfly2_f64(u0, u1);
                let (t2, t3) = AvxButterfly::butterfly2_f64(u2, u3);
                let (t4, t5) = AvxButterfly::butterfly2_f64(u4, u5);
                let (t6, t7) = AvxButterfly::butterfly2_f64(u6, u7);
                let (t8, t9) = AvxButterfly::butterfly2_f64(u8, u9);
                let (t10, t11) = AvxButterfly::butterfly2_f64(u10, u11);
                let (t12, t13) = AvxButterfly::butterfly2_f64(u12, u13);
                let (t14, t15) = AvxButterfly::butterfly2_f64(u14, u15);
                let (t16, t17) = AvxButterfly::butterfly2_f64(u16, u17);

                let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                    self.bf9.exec(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                    self.bf9.exec(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                shift_store2dd!(chunk, 18, 0, u0, u1);
                shift_store2dd!(chunk, 18, 2, u2, u3);
                shift_store2dd!(chunk, 18, 4, u4, u5);
                shift_store2dd!(chunk, 18, 6, u6, u7);
                shift_store2dd!(chunk, 18, 8, u8, u9);
                shift_store2dd!(chunk, 18, 10, u10, u11);
                shift_store2dd!(chunk, 18, 12, u12, u13);
                shift_store2dd!(chunk, 18, 14, u14, u15);
                shift_store2dd!(chunk, 18, 16, u16, u17);
            }

            let rem = in_place.chunks_exact_mut(36).into_remainder();

            for chunk in rem.chunks_exact_mut(18) {
                let u0u3 = _mm256_loadu_pd(chunk.as_ptr().cast());
                let u4u7 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u8u11 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u12u15 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u16u1 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u2u5 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u6u9 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u10u13 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());
                let u14u17 = _mm256_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());

                let (t0, t1) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u0u3),
                    _mm256_extractf128_pd::<1>(u16u1),
                );
                let (t2, t3) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u2u5),
                    _mm256_extractf128_pd::<1>(u0u3),
                );
                let (t4, t5) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u4u7),
                    _mm256_extractf128_pd::<1>(u2u5),
                );
                let (t6, t7) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u6u9),
                    _mm256_extractf128_pd::<1>(u4u7),
                );
                let (t8, t9) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u8u11),
                    _mm256_extractf128_pd::<1>(u6u9),
                );
                let (t10, t11) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u10u13),
                    _mm256_extractf128_pd::<1>(u8u11),
                );
                let (t12, t13) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u12u15),
                    _mm256_extractf128_pd::<1>(u10u13),
                );
                let (t14, t15) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u14u17),
                    _mm256_extractf128_pd::<1>(u12u15),
                );
                let (t16, t17) = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(u16u1),
                    _mm256_extractf128_pd::<1>(u14u17),
                );

                let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                    self.bf9.exec_m128d(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                    self.bf9.exec_m128d(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), _mm256_create_pd(u0, u1));
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(u2, u3),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(u4, u5),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(u6, u7),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(u8, u9),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(u10, u11),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_pd(u12, u13),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_create_pd(u14, u15),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_create_pd(u16, u17),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly18d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        18
    }
}

pub(crate) struct AvxButterfly18f {
    direction: FftDirection,
    bf9: AvxFastButterfly9f,
}

impl AvxButterfly18f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf9: unsafe { AvxFastButterfly9f::new(fft_direction) },
        }
    }
}

impl AvxButterfly18f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 18 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(36) {
                let (u0, u3, u4, u7) = shift_load4!(chunk, 18, 0);
                let (u8, u11, u12, u15) = shift_load4!(chunk, 18, 4);
                let (u16, u1, u2, u5) = shift_load4!(chunk, 18, 8);
                let (u6, u9, u10, u13) = shift_load4!(chunk, 18, 12);
                let (u14, u17) = shift_load2!(chunk, 18, 16);

                let (t0, t1) = AvxButterfly::butterfly2_f32_m128(u0, u1);
                let (t2, t3) = AvxButterfly::butterfly2_f32_m128(u2, u3);
                let (t4, t5) = AvxButterfly::butterfly2_f32_m128(u4, u5);
                let (t6, t7) = AvxButterfly::butterfly2_f32_m128(u6, u7);
                let (t8, t9) = AvxButterfly::butterfly2_f32_m128(u8, u9);
                let (t10, t11) = AvxButterfly::butterfly2_f32_m128(u10, u11);
                let (t12, t13) = AvxButterfly::butterfly2_f32_m128(u12, u13);
                let (t14, t15) = AvxButterfly::butterfly2_f32_m128(u14, u15);
                let (t16, t17) = AvxButterfly::butterfly2_f32_m128(u16, u17);

                let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                    self.bf9.exec(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                    self.bf9.exec(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                shift_store4!(chunk, 18, 0, u0, u1, u2, u3);
                shift_store4!(chunk, 18, 4, u4, u5, u6, u7);
                shift_store4!(chunk, 18, 8, u8, u9, u10, u11);
                shift_store4!(chunk, 18, 12, u12, u13, u14, u15);
                shift_store2!(chunk, 18, 16, u16, u17);
            }

            let rem = in_place.chunks_exact_mut(36).into_remainder();

            for chunk in rem.chunks_exact_mut(18) {
                let u0u3u4u7 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u8u11u12u15 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u16u1u2u5 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u6u9u10u13 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u17 = _mm_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());

                let u4u7 = _mm256_extractf128_ps::<1>(u0u3u4u7);
                let u16u1 = _mm256_castps256_ps128(u16u1u2u5);
                let u0u3 = _mm256_castps256_ps128(u0u3u4u7);
                let u6u9 = _mm256_castps256_ps128(u6u9u10u13);
                let u12u15 = _mm256_extractf128_ps::<1>(u8u11u12u15);
                let u10u13 = _mm256_extractf128_ps::<1>(u6u9u10u13);
                let u2u5 = _mm256_extractf128_ps::<1>(u16u1u2u5);

                let (t0, t1) =
                    AvxButterfly::butterfly2_f32_m128(u0u3, _mm_unpackhi_ps64(u16u1, u16u1));
                let (t2, t3) =
                    AvxButterfly::butterfly2_f32_m128(u2u5, _mm_unpackhi_ps64(u0u3, u0u3));
                let (t4, t5) =
                    AvxButterfly::butterfly2_f32_m128(u4u7, _mm_unpackhi_ps64(u2u5, u2u5));
                let (t6, t7) =
                    AvxButterfly::butterfly2_f32_m128(u6u9, _mm_unpackhi_ps64(u4u7, u4u7));
                let (t8, t9) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(u8u11u12u15),
                    _mm_unpackhi_ps64(u6u9, u6u9),
                );
                let (t10, t11) = AvxButterfly::butterfly2_f32_m128(
                    u10u13,
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u8u11u12u15),
                        _mm256_castps256_ps128(u8u11u12u15),
                    ),
                );
                let (t12, t13) =
                    AvxButterfly::butterfly2_f32_m128(u12u15, _mm_unpackhi_ps64(u10u13, u10u13));
                let (t14, t15) =
                    AvxButterfly::butterfly2_f32_m128(u14u17, _mm_unpackhi_ps64(u12u15, u12u15));
                let (t16, t17) =
                    AvxButterfly::butterfly2_f32_m128(u16u1, _mm_unpackhi_ps64(u14u17, u14u17));

                let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                    self.bf9.exec(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                    self.bf9.exec(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                _mm256_storeu_ps(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(u0, u1), _mm_unpacklo_ps64(u2, u3)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(u4, u5), _mm_unpacklo_ps64(u6, u7)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(u8, u9), _mm_unpacklo_ps64(u10, u11)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(u12, u13), _mm_unpacklo_ps64(u14, u15)),
                );
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm_unpacklo_ps64(u16, u17),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly18f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        18
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly18, f32, AvxButterfly18f, 18, 1e-5);
    test_avx_butterfly!(test_avx_butterfly18_f64, f64, AvxButterfly18d, 18, 1e-7);
}
