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
    AvxFastButterfly4, AvxFastButterfly5d, AvxFastButterfly5f, shift_load2dd, shift_load4,
    shift_store2dd, shift_store4,
};
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly20d {
    direction: FftDirection,
    bf5: AvxFastButterfly5d,
    bf4: AvxFastButterfly4<f64>,
}

impl AvxButterfly20d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5d::new(fft_direction) },
            bf4: unsafe { AvxFastButterfly4::new(fft_direction) },
        }
    }
}

impl AvxButterfly20d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 20 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(40) {
                let (u0, u5) = shift_load2dd!(chunk, 20, 0);
                let (u10, u15) = shift_load2dd!(chunk, 20, 2);
                let (u16, u1) = shift_load2dd!(chunk, 20, 4);
                let (u6, u11) = shift_load2dd!(chunk, 20, 6);
                let (u12, u17) = shift_load2dd!(chunk, 20, 8);
                let (u2, u7) = shift_load2dd!(chunk, 20, 10);
                let (u8, u13) = shift_load2dd!(chunk, 20, 12);
                let (u18, u3) = shift_load2dd!(chunk, 20, 14);
                let (u4, u9) = shift_load2dd!(chunk, 20, 16);
                let (u14, u19) = shift_load2dd!(chunk, 20, 18);

                let (t0, t1, t2, t3) = self.bf4.exec(u0, u1, u2, u3);
                let (t4, t5, t6, t7) = self.bf4.exec(u4, u5, u6, u7);
                let (t8, t9, t10, t11) = self.bf4.exec(u8, u9, u10, u11);
                let (t12, t13, t14, t15) = self.bf4.exec(u12, u13, u14, u15);
                let (t16, t17, t18, t19) = self.bf4.exec(u16, u17, u18, u19);

                let (u0, u4, u8, u12, u16) = self.bf5.exec(t0, t4, t8, t12, t16);
                let (u5, u9, u13, u17, u1) = self.bf5.exec(t1, t5, t9, t13, t17);
                let (u10, u14, u18, u2, u6) = self.bf5.exec(t2, t6, t10, t14, t18);
                let (u15, u19, u3, u7, u11) = self.bf5.exec(t3, t7, t11, t15, t19);

                shift_store2dd!(chunk, 20, 0, u0, u1);
                shift_store2dd!(chunk, 20, 2, u2, u3);
                shift_store2dd!(chunk, 20, 4, u4, u5);
                shift_store2dd!(chunk, 20, 6, u6, u7);
                shift_store2dd!(chunk, 20, 8, u8, u9);
                shift_store2dd!(chunk, 20, 10, u10, u11);
                shift_store2dd!(chunk, 20, 12, u12, u13);
                shift_store2dd!(chunk, 20, 14, u14, u15);
                shift_store2dd!(chunk, 20, 16, u16, u17);
                shift_store2dd!(chunk, 20, 18, u18, u19);
            }

            let rem = in_place.chunks_exact_mut(36).into_remainder();

            for chunk in rem.chunks_exact_mut(20) {
                let u0u5 = _mm256_loadu_pd(chunk.as_ptr().cast());
                let u10u15 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());

                let u16u1 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u11 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());

                let u12u17 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u2u7 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());

                let u8u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u18u3 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());

                let u4u9 = _mm256_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());
                let u14u19 = _mm256_loadu_pd(chunk.get_unchecked(18..).as_ptr().cast());

                let (t0, t1, t2, t3) = self.bf4.exec_short(
                    _mm256_castpd256_pd128(u0u5),
                    _mm256_extractf128_pd::<1>(u16u1),
                    _mm256_castpd256_pd128(u2u7),
                    _mm256_extractf128_pd::<1>(u18u3),
                );
                let (t4, t5, t6, t7) = self.bf4.exec_short(
                    _mm256_castpd256_pd128(u4u9),
                    _mm256_extractf128_pd::<1>(u0u5),
                    _mm256_castpd256_pd128(u6u11),
                    _mm256_extractf128_pd::<1>(u2u7),
                );
                let (t8, t9, t10, t11) = self.bf4.exec_short(
                    _mm256_castpd256_pd128(u8u13),
                    _mm256_extractf128_pd::<1>(u4u9),
                    _mm256_castpd256_pd128(u10u15),
                    _mm256_extractf128_pd::<1>(u6u11),
                );
                let (t12, t13, t14, t15) = self.bf4.exec_short(
                    _mm256_castpd256_pd128(u12u17),
                    _mm256_extractf128_pd::<1>(u8u13),
                    _mm256_castpd256_pd128(u14u19),
                    _mm256_extractf128_pd::<1>(u10u15),
                );
                let (t16, t17, t18, t19) = self.bf4.exec_short(
                    _mm256_castpd256_pd128(u16u1),
                    _mm256_extractf128_pd::<1>(u12u17),
                    _mm256_castpd256_pd128(u18u3),
                    _mm256_extractf128_pd::<1>(u14u19),
                );

                let (u0, u4, u8, u12, u16) = self.bf5.exec_short(t0, t4, t8, t12, t16);
                let (u5, u9, u13, u17, u1) = self.bf5.exec_short(t1, t5, t9, t13, t17);
                let (u10, u14, u18, u2, u6) = self.bf5.exec_short(t2, t6, t10, t14, t18);
                let (u15, u19, u3, u7, u11) = self.bf5.exec_short(t3, t7, t11, t15, t19);

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
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_create_pd(u18, u19),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly20d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        20
    }
}

pub(crate) struct AvxButterfly20f {
    direction: FftDirection,
    bf5: AvxFastButterfly5f,
    bf4: AvxFastButterfly4<f32>,
}

impl AvxButterfly20f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            bf5: unsafe { AvxFastButterfly5f::new(fft_direction) },
            bf4: unsafe { AvxFastButterfly4::new(fft_direction) },
        }
    }
}

impl AvxButterfly20f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 20 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            for chunk in in_place.chunks_exact_mut(40) {
                let (u0, u5, u10, u15) = shift_load4!(chunk, 20, 0);
                let (u16, u1, u6, u11) = shift_load4!(chunk, 20, 4);
                let (u12, u17, u2, u7) = shift_load4!(chunk, 20, 8);
                let (u8, u13, u18, u3) = shift_load4!(chunk, 20, 12);
                let (u4, u9, u14, u19) = shift_load4!(chunk, 20, 16);

                let (t0, t1, t2, t3) = self.bf4.exec_short(u0, u1, u2, u3);
                let (t4, t5, t6, t7) = self.bf4.exec_short(u4, u5, u6, u7);
                let (t8, t9, t10, t11) = self.bf4.exec_short(u8, u9, u10, u11);
                let (t12, t13, t14, t15) = self.bf4.exec_short(u12, u13, u14, u15);
                let (t16, t17, t18, t19) = self.bf4.exec_short(u16, u17, u18, u19);

                let (u0, u4, u8, u12, u16) = self.bf5.exec(t0, t4, t8, t12, t16);
                let (u5, u9, u13, u17, u1) = self.bf5.exec(t1, t5, t9, t13, t17);
                let (u10, u14, u18, u2, u6) = self.bf5.exec(t2, t6, t10, t14, t18);
                let (u15, u19, u3, u7, u11) = self.bf5.exec(t3, t7, t11, t15, t19);

                shift_store4!(chunk, 20, 0, u0, u1, u2, u3);
                shift_store4!(chunk, 20, 4, u4, u5, u6, u7);
                shift_store4!(chunk, 20, 8, u8, u9, u10, u11);
                shift_store4!(chunk, 20, 12, u12, u13, u14, u15);
                shift_store4!(chunk, 20, 16, u16, u17, u18, u19);
            }

            let rem = in_place.chunks_exact_mut(40).into_remainder();

            for chunk in rem.chunks_exact_mut(20) {
                let u0u5u10u15 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u16u1u6u11 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u12u17u2u7 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u8u13u18u3 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u4u9u14u19 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());

                let u0u5 = _mm256_castps256_ps128(u0u5u10u15);
                let u10u15 = _mm256_extractf128_ps::<1>(u0u5u10u15);
                let u16u1 = _mm256_castps256_ps128(u16u1u6u11);
                let u6u11 = _mm256_extractf128_ps::<1>(u16u1u6u11);
                let u12u17 = _mm256_castps256_ps128(u12u17u2u7);
                let u2u7 = _mm256_extractf128_ps::<1>(u12u17u2u7);
                let u8u13 = _mm256_castps256_ps128(u8u13u18u3);
                let u18u3 = _mm256_extractf128_ps::<1>(u8u13u18u3);
                let u4u9 = _mm256_castps256_ps128(u4u9u14u19);
                let u14u19 = _mm256_extractf128_ps::<1>(u4u9u14u19);

                let (t0, t1, t2, t3) = self.bf4.exec_short(
                    u0u5,
                    _mm_unpackhi_ps64(u16u1, u16u1),
                    u2u7,
                    _mm_unpackhi_ps64(u18u3, u18u3),
                );
                let (t4, t5, t6, t7) = self.bf4.exec_short(
                    u4u9,
                    _mm_unpackhi_ps64(u0u5, u0u5),
                    u6u11,
                    _mm_unpackhi_ps64(u2u7, u2u7),
                );
                let (t8, t9, t10, t11) = self.bf4.exec_short(
                    u8u13,
                    _mm_unpackhi_ps64(u4u9, u4u9),
                    u10u15,
                    _mm_unpackhi_ps64(u6u11, u6u11),
                );
                let (t12, t13, t14, t15) = self.bf4.exec_short(
                    u12u17,
                    _mm_unpackhi_ps64(u8u13, u8u13),
                    u14u19,
                    _mm_unpackhi_ps64(u10u15, u10u15),
                );
                let (t16, t17, t18, t19) = self.bf4.exec_short(
                    u16u1,
                    _mm_unpackhi_ps64(u12u17, u12u17),
                    u18u3,
                    _mm_unpackhi_ps64(u14u19, u14u19),
                );

                let (u0, u4, u8, u12, u16) = self.bf5.exec(t0, t4, t8, t12, t16);
                let (u5, u9, u13, u17, u1) = self.bf5.exec(t1, t5, t9, t13, t17);
                let (u10, u14, u18, u2, u6) = self.bf5.exec(t2, t6, t10, t14, t18);
                let (u15, u19, u3, u7, u11) = self.bf5.exec(t3, t7, t11, t15, t19);

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
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_unpacklo_ps64(u16, u17), _mm_unpacklo_ps64(u18, u19)),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly20f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        20
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::test_avx_butterfly;

    test_avx_butterfly!(test_avx_butterfly20, f32, AvxButterfly20f, 20, 1e-5);
    test_avx_butterfly!(test_avx_butterfly20_f64, f64, AvxButterfly20d, 20, 1e-7);
}
