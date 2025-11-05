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
use crate::avx::butterflies::fast_bf9::AvxFastButterfly9f;
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm_unpackhilo_ps64,
    _mm_unpacklo_ps64, _mm_unpacklohi_ps64, _mm256_create_ps, _mm256_fcmul_ps, _mm256_set4_complex,
};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly27f {
    direction: FftDirection,
    bf9: AvxFastButterfly9f,
    tw1tw2tw2tw4: __m256,
    tw3tw6tw4tw8: __m256,
    tw5tw9tw6tw10: __m256,
    tw7tw11tw8tw12: __m256,
}

impl AvxButterfly27f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let t1 = compute_twiddle(1, 27, fft_direction);
            let t2 = compute_twiddle(2, 27, fft_direction);
            let t3 = compute_twiddle(3, 27, fft_direction);
            let t4 = compute_twiddle(4, 27, fft_direction);
            let t5 = compute_twiddle(5, 27, fft_direction);
            let t6 = compute_twiddle(6, 27, fft_direction);
            let t7 = compute_twiddle(7, 27, fft_direction);
            let t8 = compute_twiddle(8, 27, fft_direction);
            let t9 = compute_twiddle(10, 27, fft_direction);
            let t10 = compute_twiddle(12, 27, fft_direction);
            let t11 = compute_twiddle(14, 27, fft_direction);
            let t12 = compute_twiddle(16, 27, fft_direction);
            Self {
                direction: fft_direction,
                tw1tw2tw2tw4: _mm256_set4_complex(t1, t2, t2, t4),
                tw3tw6tw4tw8: _mm256_set4_complex(t3, t6, t4, t8),
                tw5tw9tw6tw10: _mm256_set4_complex(t5, t9, t6, t10),
                tw7tw11tw8tw12: _mm256_set4_complex(t7, t11, t8, t12),
                bf9: AvxFastButterfly9f::new(fft_direction),
            }
        }
    }
}

impl AvxButterfly27f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(27) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16u17u18u19 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());
                let u20u21u22u23 = _mm256_loadu_ps(chunk.get_unchecked(20..).as_ptr().cast());
                let u23u24u25u26 = _mm256_loadu_ps(chunk.get_unchecked(23..).as_ptr().cast());

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

                //      let s0 = self.bf9.exec(u0, u3, u6, u9, u12, u15, u18, u21, u24);
                let s0 = self.bf9.exec(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    u6u7,
                    _mm_unpackhi_ps64(u8u9, u8u9),
                    u12u13,
                    _mm_unpackhi_ps64(u14u15, u14u15),
                    u18u19,
                    _mm_unpackhi_ps64(u20u21, u20u21),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u23u24u25u26),
                        _mm256_castps256_ps128(u23u24u25u26),
                    ),
                );

                // let mut s1 = self.bf9.exec(u1, u4, u7, u10, u13, u16, u19, u22, u25);
                // let mut s2 = self.bf9.exec(u2, u5, u8, u11, u14, u17, u20, u23, u26);
                let u25u26 = _mm256_extractf128_ps::<1>(u23u24u25u26);
                let s1s2 = self.bf9.exec(
                    _mm_unpackhilo_ps64(u0u1, u2u3),
                    u4u5,
                    _mm_unpackhilo_ps64(u6u7, u8u9),
                    u10u11,
                    _mm_unpackhilo_ps64(u12u13, u14u15),
                    u16u17,
                    _mm_unpackhilo_ps64(u18u19, u20u21),
                    u22u23,
                    _mm_unpacklohi_ps64(u25u26, u25u26),
                );

                let s1s2_1_2 = _mm256_fcmul_ps(_mm256_create_ps(s1s2.1, s1s2.2), self.tw1tw2tw2tw4); // s1.1 s2.1 s1.2 s2.2
                let s1s2_3_4 = _mm256_fcmul_ps(_mm256_create_ps(s1s2.3, s1s2.4), self.tw3tw6tw4tw8); // s1.3 s2.3 s1.4 s2.4
                let s1s2_5_6 =
                    _mm256_fcmul_ps(_mm256_create_ps(s1s2.5, s1s2.6), self.tw5tw9tw6tw10); // s1.5 s2.5 s1.6 s2.6
                let s1s2_7_8 =
                    _mm256_fcmul_ps(_mm256_create_ps(s1s2.7, s1s2.8), self.tw7tw11tw8tw12); // s1.7 s2.7 s1.8 s2.8

                let s1s2_1_2_hi = _mm256_extractf128_ps::<1>(s1s2_1_2);

                let z0z1z2z3 = self.bf9.bf3.exec(
                    _mm256_create_ps(_mm_unpacklo_ps64(s0.0, s0.1), _mm_unpacklo_ps64(s0.2, s0.3)),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(s1s2.0, _mm256_castps256_ps128(s1s2_1_2)),
                        _mm_unpacklo_ps64(s1s2_1_2_hi, _mm256_castps256_ps128(s1s2_3_4)),
                    ),
                    _mm256_create_ps(
                        _mm_unpackhi_ps64(s1s2.0, _mm256_castps256_ps128(s1s2_1_2)),
                        _mm_unpackhi_ps64(s1s2_1_2_hi, _mm256_castps256_ps128(s1s2_3_4)),
                    ),
                );

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), z0z1z2z3.0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), z0z1z2z3.1);
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    z0z1z2z3.2,
                );

                let s1s2_3_4_hi = _mm256_extractf128_ps::<1>(s1s2_3_4);
                let s1s2_5_6_hi = _mm256_extractf128_ps::<1>(s1s2_5_6);

                let z4z5z6z7 = self.bf9.bf3.exec(
                    _mm256_create_ps(_mm_unpacklo_ps64(s0.4, s0.5), _mm_unpacklo_ps64(s0.6, s0.7)),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(s1s2_3_4_hi, _mm256_castps256_ps128(s1s2_5_6)),
                        _mm_unpacklo_ps64(s1s2_5_6_hi, _mm256_castps256_ps128(s1s2_7_8)),
                    ),
                    _mm256_create_ps(
                        _mm_unpackhi_ps64(s1s2_3_4_hi, _mm256_castps256_ps128(s1s2_5_6)),
                        _mm_unpackhi_ps64(s1s2_5_6_hi, _mm256_castps256_ps128(s1s2_7_8)),
                    ),
                );

                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), z4z5z6z7.0);
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(13..).as_mut_ptr().cast(),
                    z4z5z6z7.1,
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    z4z5z6z7.2,
                );

                let s1s2_7_8_hi = _mm256_extractf128_ps::<1>(s1s2_7_8);

                let z8 = self.bf9.bf3.exec_m128(
                    s0.8,
                    _mm_unpacklo_ps64(s1s2_7_8_hi, s1s2_7_8_hi),
                    _mm_unpackhi_ps64(s1s2_7_8_hi, s1s2_7_8_hi),
                );
                _m128s_store_f32x2(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), z8.0);
                _m128s_store_f32x2(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), z8.1);
                _m128s_store_f32x2(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), z8.2);
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
        if src.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(27).zip(src.chunks_exact(27)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(src.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(src.get_unchecked(12..).as_ptr().cast());
                let u16u17u18u19 = _mm256_loadu_ps(src.get_unchecked(16..).as_ptr().cast());
                let u20u21u22u23 = _mm256_loadu_ps(src.get_unchecked(20..).as_ptr().cast());
                let u24u25 = _mm_loadu_ps(src.get_unchecked(24..).as_ptr().cast());
                let u26 = _m128s_load_f32x2(src.get_unchecked(26..).as_ptr().cast());

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

                //      let s0 = self.bf9.exec(u0, u3, u6, u9, u12, u15, u18, u21, u24);
                let s0 = self.bf9.exec(
                    _mm256_castps256_ps128(u0u1u2u3),
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    u6u7,
                    _mm_unpackhi_ps64(u8u9, u8u9),
                    u12u13,
                    _mm_unpackhi_ps64(u14u15, u14u15),
                    u18u19,
                    _mm_unpackhi_ps64(u20u21, u20u21),
                    u24u25,
                );

                // let mut s1 = self.bf9.exec(u1, u4, u7, u10, u13, u16, u19, u22, u25);
                // let mut s2 = self.bf9.exec(u2, u5, u8, u11, u14, u17, u20, u23, u26);
                let s1s2 = self.bf9.exec(
                    _mm_unpackhilo_ps64(u0u1, u2u3),
                    u4u5,
                    _mm_unpackhilo_ps64(u6u7, u8u9),
                    u10u11,
                    _mm_unpackhilo_ps64(u12u13, u14u15),
                    u16u17,
                    _mm_unpackhilo_ps64(u18u19, u20u21),
                    u22u23,
                    _mm_unpackhilo_ps64(u24u25, u26),
                );

                let s1s2_1_2 = _mm256_fcmul_ps(_mm256_create_ps(s1s2.1, s1s2.2), self.tw1tw2tw2tw4); // s1.1 s2.1 s1.2 s2.2
                let s1s2_3_4 = _mm256_fcmul_ps(_mm256_create_ps(s1s2.3, s1s2.4), self.tw3tw6tw4tw8); // s1.3 s2.3 s1.4 s2.4
                let s1s2_5_6 =
                    _mm256_fcmul_ps(_mm256_create_ps(s1s2.5, s1s2.6), self.tw5tw9tw6tw10); // s1.5 s2.5 s1.6 s2.6
                let s1s2_7_8 =
                    _mm256_fcmul_ps(_mm256_create_ps(s1s2.7, s1s2.8), self.tw7tw11tw8tw12); // s1.7 s2.7 s1.8 s2.8

                let s1s2_1_2_hi = _mm256_extractf128_ps::<1>(s1s2_1_2);

                let z0z1z2z3 = self.bf9.bf3.exec(
                    _mm256_create_ps(_mm_unpacklo_ps64(s0.0, s0.1), _mm_unpacklo_ps64(s0.2, s0.3)),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(s1s2.0, _mm256_castps256_ps128(s1s2_1_2)),
                        _mm_unpacklo_ps64(s1s2_1_2_hi, _mm256_castps256_ps128(s1s2_3_4)),
                    ),
                    _mm256_create_ps(
                        _mm_unpackhi_ps64(s1s2.0, _mm256_castps256_ps128(s1s2_1_2)),
                        _mm_unpackhi_ps64(s1s2_1_2_hi, _mm256_castps256_ps128(s1s2_3_4)),
                    ),
                );

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), z0z1z2z3.0);
                _mm256_storeu_ps(dst.get_unchecked_mut(9..).as_mut_ptr().cast(), z0z1z2z3.1);
                _mm256_storeu_ps(dst.get_unchecked_mut(18..).as_mut_ptr().cast(), z0z1z2z3.2);

                let s1s2_3_4_hi = _mm256_extractf128_ps::<1>(s1s2_3_4);
                let s1s2_5_6_hi = _mm256_extractf128_ps::<1>(s1s2_5_6);

                let z4z5z6z7 = self.bf9.bf3.exec(
                    _mm256_create_ps(_mm_unpacklo_ps64(s0.4, s0.5), _mm_unpacklo_ps64(s0.6, s0.7)),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(s1s2_3_4_hi, _mm256_castps256_ps128(s1s2_5_6)),
                        _mm_unpacklo_ps64(s1s2_5_6_hi, _mm256_castps256_ps128(s1s2_7_8)),
                    ),
                    _mm256_create_ps(
                        _mm_unpackhi_ps64(s1s2_3_4_hi, _mm256_castps256_ps128(s1s2_5_6)),
                        _mm_unpackhi_ps64(s1s2_5_6_hi, _mm256_castps256_ps128(s1s2_7_8)),
                    ),
                );

                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), z4z5z6z7.0);
                _mm256_storeu_ps(dst.get_unchecked_mut(13..).as_mut_ptr().cast(), z4z5z6z7.1);
                _mm256_storeu_ps(dst.get_unchecked_mut(22..).as_mut_ptr().cast(), z4z5z6z7.2);

                let s1s2_7_8_hi = _mm256_extractf128_ps::<1>(s1s2_7_8);

                let z8 = self.bf9.bf3.exec_m128(
                    s0.8,
                    _mm_unpacklo_ps64(s1s2_7_8_hi, s1s2_7_8_hi),
                    _mm_unpackhi_ps64(s1s2_7_8_hi, s1s2_7_8_hi),
                );
                _m128s_store_f32x2(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), z8.0);
                _m128s_store_f32x2(dst.get_unchecked_mut(17..).as_mut_ptr().cast(), z8.1);
                _m128s_store_f32x2(dst.get_unchecked_mut(26..).as_mut_ptr().cast(), z8.2);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly27f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly27f {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly27f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        27
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};
    use rand::Rng;

    test_avx_butterfly!(test_avx_butterfly27, f32, AvxButterfly27f, 27, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly27, f32, AvxButterfly27f, 27, 1e-5);
}
