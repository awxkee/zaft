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
use crate::avx::butterflies::fast_bf8::AvxFastButterfly8;
use crate::avx::util::{
    _mm_fcmul_pd, _mm_fcmul_pd_conj_b, _mm_fcmul_ps, _mm_fcmul_ps_conj_b, _mm_unpackhi_ps64,
    _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly16<T> {
    direction: FftDirection,
    bf8: AvxFastButterfly8<T>,
    twiddle1: [T; 8],
    twiddle2: [T; 8],
    twiddle3: [T; 8],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly16<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 16, fft_direction);
        let tw2 = compute_twiddle(2, 16, fft_direction);
        let tw3 = compute_twiddle(3, 16, fft_direction);
        Self {
            direction: fft_direction,
            bf8: unsafe { AvxFastButterfly8::new(fft_direction) },
            twiddle1: [
                tw1.re, tw1.im, tw1.re, tw1.im, tw1.re, tw1.im, tw1.re, tw1.im,
            ],
            twiddle2: [
                tw2.re, tw2.im, tw2.re, tw2.im, tw2.re, tw2.im, tw2.re, tw2.im,
            ],
            twiddle3: [
                tw3.re, tw3.im, tw3.re, tw3.im, tw3.re, tw3.im, tw3.re, tw3.im,
            ],
        }
    }
}

impl AvxButterfly16<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 16 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let tw1 = _mm_loadu_pd(self.twiddle1.as_ptr());
            let tw2 = _mm_loadu_pd(self.twiddle2.as_ptr());
            let tw3 = _mm_loadu_pd(self.twiddle3.as_ptr());

            for chunk in in_place.chunks_exact_mut(16) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());

                let evens = self.bf8.exec_short(
                    _mm256_castpd256_pd128(u0u1),
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_castpd256_pd128(u8u9),
                    _mm256_castpd256_pd128(u10u11),
                    _mm256_castpd256_pd128(u12u13),
                    _mm256_castpd256_pd128(u14u15),
                );

                let mut odds_1 = AvxButterfly::butterfly4h_f64(
                    _mm256_extractf128_pd::<1>(u0u1),
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_extractf128_pd::<1>(u8u9),
                    _mm256_extractf128_pd::<1>(u12u13),
                    _mm256_castpd256_pd128(self.bf8.rotate.rot_flag),
                );
                let mut odds_2 = AvxButterfly::butterfly4h_f64(
                    _mm256_extractf128_pd::<1>(u14u15),
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_extractf128_pd::<1>(u6u7),
                    _mm256_extractf128_pd::<1>(u10u11),
                    _mm256_castpd256_pd128(self.bf8.rotate.rot_flag),
                );

                odds_1.1 = _mm_fcmul_pd(odds_1.1, tw1);
                odds_2.1 = _mm_fcmul_pd_conj_b(odds_2.1, tw1);

                odds_1.2 = _mm_fcmul_pd(odds_1.2, tw2);
                odds_2.2 = _mm_fcmul_pd_conj_b(odds_2.2, tw2);

                odds_1.3 = _mm_fcmul_pd(odds_1.3, tw3);
                odds_2.3 = _mm_fcmul_pd_conj_b(odds_2.3, tw3);

                // step 4: cross FFTs
                let (o01, o02) = AvxButterfly::butterfly2_f64_m128(odds_1.0, odds_2.0);
                odds_1.0 = o01;
                odds_2.0 = o02;

                let (o03, o04) = AvxButterfly::butterfly2_f64_m128(odds_1.1, odds_2.1);
                odds_1.1 = o03;
                odds_2.1 = o04;
                let (o05, o06) = AvxButterfly::butterfly2_f64_m128(odds_1.2, odds_2.2);
                odds_1.2 = o05;
                odds_2.2 = o06;
                let (o07, o08) = AvxButterfly::butterfly2_f64_m128(odds_1.3, odds_2.3);
                odds_1.3 = o07;
                odds_2.3 = o08;

                // apply the butterfly 4 twiddle factor, which is just a rotation
                odds_2.0 = self.bf8.rotate.rotate_m128d(odds_2.0);
                odds_2.1 = self.bf8.rotate.rotate_m128d(odds_2.1);
                odds_2.2 = self.bf8.rotate.rotate_m128d(odds_2.2);
                odds_2.3 = self.bf8.rotate.rotate_m128d(odds_2.3);

                _mm256_storeu_pd(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.0, odds_1.0), _mm_add_pd(evens.1, odds_1.1)),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.2, odds_1.2), _mm_add_pd(evens.3, odds_1.3)),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.4, odds_2.0), _mm_add_pd(evens.5, odds_2.1)),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.6, odds_2.2), _mm_add_pd(evens.7, odds_2.3)),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.0, odds_1.0), _mm_sub_pd(evens.1, odds_1.1)),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.2, odds_1.2), _mm_sub_pd(evens.3, odds_1.3)),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.4, odds_2.0), _mm_sub_pd(evens.5, odds_2.1)),
                );

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.6, odds_2.2), _mm_sub_pd(evens.7, odds_2.3)),
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
            if src.len() % 16 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 16 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let tw1 = _mm_loadu_pd(self.twiddle1.as_ptr());
            let tw2 = _mm_loadu_pd(self.twiddle2.as_ptr());
            let tw3 = _mm_loadu_pd(self.twiddle3.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(src.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(src.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(src.get_unchecked(14..).as_ptr().cast());

                let evens = self.bf8.exec_short(
                    _mm256_castpd256_pd128(u0u1),
                    _mm256_castpd256_pd128(u2u3),
                    _mm256_castpd256_pd128(u4u5),
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_castpd256_pd128(u8u9),
                    _mm256_castpd256_pd128(u10u11),
                    _mm256_castpd256_pd128(u12u13),
                    _mm256_castpd256_pd128(u14u15),
                );

                let mut odds_1 = AvxButterfly::butterfly4h_f64(
                    _mm256_extractf128_pd::<1>(u0u1),
                    _mm256_extractf128_pd::<1>(u4u5),
                    _mm256_extractf128_pd::<1>(u8u9),
                    _mm256_extractf128_pd::<1>(u12u13),
                    _mm256_castpd256_pd128(self.bf8.rotate.rot_flag),
                );
                let mut odds_2 = AvxButterfly::butterfly4h_f64(
                    _mm256_extractf128_pd::<1>(u14u15),
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_extractf128_pd::<1>(u6u7),
                    _mm256_extractf128_pd::<1>(u10u11),
                    _mm256_castpd256_pd128(self.bf8.rotate.rot_flag),
                );

                odds_1.1 = _mm_fcmul_pd(odds_1.1, tw1);
                odds_2.1 = _mm_fcmul_pd_conj_b(odds_2.1, tw1);

                odds_1.2 = _mm_fcmul_pd(odds_1.2, tw2);
                odds_2.2 = _mm_fcmul_pd_conj_b(odds_2.2, tw2);

                odds_1.3 = _mm_fcmul_pd(odds_1.3, tw3);
                odds_2.3 = _mm_fcmul_pd_conj_b(odds_2.3, tw3);

                // step 4: cross FFTs
                let (o01, o02) = AvxButterfly::butterfly2_f64_m128(odds_1.0, odds_2.0);
                odds_1.0 = o01;
                odds_2.0 = o02;

                let (o03, o04) = AvxButterfly::butterfly2_f64_m128(odds_1.1, odds_2.1);
                odds_1.1 = o03;
                odds_2.1 = o04;
                let (o05, o06) = AvxButterfly::butterfly2_f64_m128(odds_1.2, odds_2.2);
                odds_1.2 = o05;
                odds_2.2 = o06;
                let (o07, o08) = AvxButterfly::butterfly2_f64_m128(odds_1.3, odds_2.3);
                odds_1.3 = o07;
                odds_2.3 = o08;

                // apply the butterfly 4 twiddle factor, which is just a rotation
                odds_2.0 = self.bf8.rotate.rotate_m128d(odds_2.0);
                odds_2.1 = self.bf8.rotate.rotate_m128d(odds_2.1);
                odds_2.2 = self.bf8.rotate.rotate_m128d(odds_2.2);
                odds_2.3 = self.bf8.rotate.rotate_m128d(odds_2.3);

                _mm256_storeu_pd(
                    dst.as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.0, odds_1.0), _mm_add_pd(evens.1, odds_1.1)),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.2, odds_1.2), _mm_add_pd(evens.3, odds_1.3)),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.4, odds_2.0), _mm_add_pd(evens.5, odds_2.1)),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_add_pd(evens.6, odds_2.2), _mm_add_pd(evens.7, odds_2.3)),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.0, odds_1.0), _mm_sub_pd(evens.1, odds_1.1)),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.2, odds_1.2), _mm_sub_pd(evens.3, odds_1.3)),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.4, odds_2.0), _mm_sub_pd(evens.5, odds_2.1)),
                );

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_create_pd(_mm_sub_pd(evens.6, odds_2.2), _mm_sub_pd(evens.7, odds_2.3)),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly16<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly16<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly16<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

impl AvxButterfly16<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 16 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let tw1 = _mm_loadu_ps(self.twiddle1.as_ptr());
            let tw2 = _mm_loadu_ps(self.twiddle2.as_ptr());
            let tw3 = _mm_loadu_ps(self.twiddle3.as_ptr());

            for chunk in in_place.chunks_exact_mut(16) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u8u9 = _mm256_castps256_ps128(u8u9u10u11);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);
                let u12u13 = _mm256_castps256_ps128(u12u13u14u15);
                let u14u15 = _mm256_extractf128_ps::<1>(u12u13u14u15);

                let evens = self.bf8.exec_short(
                    _mm256_castps256_ps128(u0u1u2u3),
                    u2u3,
                    u4u5,
                    u6u7,
                    u8u9,
                    u10u11,
                    u12u13,
                    u14u15,
                );

                let mut odds_1 = AvxButterfly::butterfly4h_f32(
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    _mm_unpackhi_ps64(u4u5, u4u5),
                    _mm_unpackhi_ps64(u8u9, u8u9),
                    _mm_unpackhi_ps64(u12u13, u12u13),
                    _mm_castpd_ps(_mm256_castpd256_pd128(self.bf8.rotate.rot_flag)),
                );
                let mut odds_2 = AvxButterfly::butterfly4h_f32(
                    _mm_unpackhi_ps64(u14u15, u14u15),
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    _mm_unpackhi_ps64(u6u7, u6u7),
                    _mm_unpackhi_ps64(u10u11, u10u11),
                    _mm_castpd_ps(_mm256_castpd256_pd128(self.bf8.rotate.rot_flag)),
                );

                odds_1.1 = _mm_fcmul_ps(odds_1.1, tw1);
                odds_2.1 = _mm_fcmul_ps_conj_b(odds_2.1, tw1);

                odds_1.2 = _mm_fcmul_ps(odds_1.2, tw2);
                odds_2.2 = _mm_fcmul_ps_conj_b(odds_2.2, tw2);

                odds_1.3 = _mm_fcmul_ps(odds_1.3, tw3);
                odds_2.3 = _mm_fcmul_ps_conj_b(odds_2.3, tw3);

                // step 4: cross FFTs
                let (o01, o02) = AvxButterfly::butterfly2_f32_m128(odds_1.0, odds_2.0);
                odds_1.0 = o01;
                odds_2.0 = o02;

                let (o03, o04) = AvxButterfly::butterfly2_f32_m128(odds_1.1, odds_2.1);
                odds_1.1 = o03;
                odds_2.1 = o04;
                let (o05, o06) = AvxButterfly::butterfly2_f32_m128(odds_1.2, odds_2.2);
                odds_1.2 = o05;
                odds_2.2 = o06;
                let (o07, o08) = AvxButterfly::butterfly2_f32_m128(odds_1.3, odds_2.3);
                odds_1.3 = o07;
                odds_2.3 = o08;

                // apply the butterfly 4 twiddle factor, which is just a rotation
                odds_2.0 = self.bf8.rotate.rotate_m128(odds_2.0);
                odds_2.1 = self.bf8.rotate.rotate_m128(odds_2.1);
                odds_2.2 = self.bf8.rotate.rotate_m128(odds_2.2);
                odds_2.3 = self.bf8.rotate.rotate_m128(odds_2.3);

                _mm256_storeu_ps(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.0, odds_1.0),
                            _mm_add_ps(evens.1, odds_1.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.2, odds_1.2),
                            _mm_add_ps(evens.3, odds_1.3),
                        ),
                    ),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.4, odds_2.0),
                            _mm_add_ps(evens.5, odds_2.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.6, odds_2.2),
                            _mm_add_ps(evens.7, odds_2.3),
                        ),
                    ),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.0, odds_1.0),
                            _mm_sub_ps(evens.1, odds_1.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.2, odds_1.2),
                            _mm_sub_ps(evens.3, odds_1.3),
                        ),
                    ),
                );

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.4, odds_2.0),
                            _mm_sub_ps(evens.5, odds_2.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.6, odds_2.2),
                            _mm_sub_ps(evens.7, odds_2.3),
                        ),
                    ),
                );
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
            if src.len() % 16 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 16 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let tw1 = _mm_loadu_ps(self.twiddle1.as_ptr());
            let tw2 = _mm_loadu_ps(self.twiddle2.as_ptr());
            let tw3 = _mm_loadu_ps(self.twiddle3.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(src.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(src.get_unchecked(12..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u4u5 = _mm256_castps256_ps128(u4u5u6u7);
                let u6u7 = _mm256_extractf128_ps::<1>(u4u5u6u7);
                let u8u9 = _mm256_castps256_ps128(u8u9u10u11);
                let u10u11 = _mm256_extractf128_ps::<1>(u8u9u10u11);
                let u12u13 = _mm256_castps256_ps128(u12u13u14u15);
                let u14u15 = _mm256_extractf128_ps::<1>(u12u13u14u15);

                let evens = self.bf8.exec_short(
                    _mm256_castps256_ps128(u0u1u2u3),
                    u2u3,
                    u4u5,
                    u6u7,
                    u8u9,
                    u10u11,
                    u12u13,
                    u14u15,
                );

                let mut odds_1 = AvxButterfly::butterfly4h_f32(
                    _mm_unpackhi_ps64(u0u1, u0u1),
                    _mm_unpackhi_ps64(u4u5, u4u5),
                    _mm_unpackhi_ps64(u8u9, u8u9),
                    _mm_unpackhi_ps64(u12u13, u12u13),
                    _mm_castpd_ps(_mm256_castpd256_pd128(self.bf8.rotate.rot_flag)),
                );
                let mut odds_2 = AvxButterfly::butterfly4h_f32(
                    _mm_unpackhi_ps64(u14u15, u14u15),
                    _mm_unpackhi_ps64(u2u3, u2u3),
                    _mm_unpackhi_ps64(u6u7, u6u7),
                    _mm_unpackhi_ps64(u10u11, u10u11),
                    _mm_castpd_ps(_mm256_castpd256_pd128(self.bf8.rotate.rot_flag)),
                );

                odds_1.1 = _mm_fcmul_ps(odds_1.1, tw1);
                odds_2.1 = _mm_fcmul_ps_conj_b(odds_2.1, tw1);

                odds_1.2 = _mm_fcmul_ps(odds_1.2, tw2);
                odds_2.2 = _mm_fcmul_ps_conj_b(odds_2.2, tw2);

                odds_1.3 = _mm_fcmul_ps(odds_1.3, tw3);
                odds_2.3 = _mm_fcmul_ps_conj_b(odds_2.3, tw3);

                // step 4: cross FFTs
                let (o01, o02) = AvxButterfly::butterfly2_f32_m128(odds_1.0, odds_2.0);
                odds_1.0 = o01;
                odds_2.0 = o02;

                let (o03, o04) = AvxButterfly::butterfly2_f32_m128(odds_1.1, odds_2.1);
                odds_1.1 = o03;
                odds_2.1 = o04;
                let (o05, o06) = AvxButterfly::butterfly2_f32_m128(odds_1.2, odds_2.2);
                odds_1.2 = o05;
                odds_2.2 = o06;
                let (o07, o08) = AvxButterfly::butterfly2_f32_m128(odds_1.3, odds_2.3);
                odds_1.3 = o07;
                odds_2.3 = o08;

                // apply the butterfly 4 twiddle factor, which is just a rotation
                odds_2.0 = self.bf8.rotate.rotate_m128(odds_2.0);
                odds_2.1 = self.bf8.rotate.rotate_m128(odds_2.1);
                odds_2.2 = self.bf8.rotate.rotate_m128(odds_2.2);
                odds_2.3 = self.bf8.rotate.rotate_m128(odds_2.3);

                _mm256_storeu_ps(
                    dst.as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.0, odds_1.0),
                            _mm_add_ps(evens.1, odds_1.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.2, odds_1.2),
                            _mm_add_ps(evens.3, odds_1.3),
                        ),
                    ),
                );

                _mm256_storeu_ps(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.4, odds_2.0),
                            _mm_add_ps(evens.5, odds_2.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_add_ps(evens.6, odds_2.2),
                            _mm_add_ps(evens.7, odds_2.3),
                        ),
                    ),
                );

                _mm256_storeu_ps(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.0, odds_1.0),
                            _mm_sub_ps(evens.1, odds_1.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.2, odds_1.2),
                            _mm_sub_ps(evens.3, odds_1.3),
                        ),
                    ),
                );

                _mm256_storeu_ps(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.4, odds_2.0),
                            _mm_sub_ps(evens.5, odds_2.1),
                        ),
                        _mm_unpacklo_ps64(
                            _mm_sub_ps(evens.6, odds_2.2),
                            _mm_sub_ps(evens.7, odds_2.3),
                        ),
                    ),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly16<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly16<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly16<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        16
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::butterflies::Butterfly16;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly16_f32() {
        for i in 1..4 {
            let size = 16usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly16::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly16::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly16::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 16f32)).collect();

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
    fn test_butterfly16_f64() {
        for i in 1..4 {
            let size = 16usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly16::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly16::new(FftDirection::Inverse);

            let radix_forward_ref = Butterfly16::new(FftDirection::Forward);

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

            input = input.iter().map(|&x| x * (1.0 / 16f64)).collect();

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

    #[test]
    fn test_butterfly16_out_of_place_f64() {
        for i in 1..4 {
            let size = 16usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f64>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = AvxButterfly16::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly16::new(FftDirection::Inverse);

            let reference_dft = Dft::new(16, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
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

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 16f64)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly16_out_of_place_f32() {
        for i in 1..4 {
            let size = 16usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f32>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = AvxButterfly16::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly16::new(FftDirection::Inverse);

            let reference_dft = Dft::new(16, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
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

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 16f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
