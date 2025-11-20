// Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use crate::avx::util::{
    _m128s_store_f32x2, _mm256_create_pd, _mm256_fcmul_pd, _mm256_fcmul_ps, _mm256_permute4x64_ps,
    _mm256_unpackhi_ps64, _mm256_unpacklo_ps64, shuffle,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::any::TypeId;
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxButterfly9<T> {
    direction: FftDirection,
    tw1tw2: [T; 8],
    tw2tw4: [T; 8],
    tw3_re: T,
    tw3_im: [T; 8],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly9<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 9, fft_direction);
        let tw2 = compute_twiddle(2, 9, fft_direction);
        let tw4 = compute_twiddle(4, 9, fft_direction);
        let tw3_bf3 = compute_twiddle::<T>(1, 3, fft_direction);
        Self {
            direction: fft_direction,
            tw1tw2: if TypeId::of::<T>() == TypeId::of::<f64>() {
                [
                    tw1.re, tw1.im, tw2.re, tw2.im, tw1.re, tw1.im, tw2.re, tw2.im,
                ]
            } else {
                [
                    1f64.as_(),
                    0f64.as_(),
                    tw1.re,
                    tw1.im,
                    tw2.re,
                    tw2.im,
                    tw2.re,
                    tw2.im,
                ]
            },
            tw2tw4: if TypeId::of::<T>() == TypeId::of::<f64>() {
                [
                    tw2.re, tw2.im, tw4.re, tw4.im, tw2.re, tw2.im, tw4.re, tw4.im,
                ]
            } else {
                [
                    1f64.as_(),
                    0f64.as_(),
                    tw2.re,
                    tw2.im,
                    tw4.re,
                    tw4.im,
                    tw4.re,
                    tw4.im,
                ]
            },
            tw3_re: tw3_bf3.re,
            tw3_im: [
                -tw3_bf3.im,
                tw3_bf3.im,
                -tw3_bf3.im,
                tw3_bf3.im,
                -tw3_bf3.im,
                tw3_bf3.im,
                -tw3_bf3.im,
                tw3_bf3.im,
            ],
        }
    }
}

impl AvxButterfly9<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 9 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let tw3_re = _mm256_set1_pd(self.tw3_re);
            let tw3_im = _mm256_loadu_pd(self.tw3_im.as_ptr().cast());
            let tw1tw2 = _mm256_loadu_pd(self.tw1tw2.as_ptr().cast());
            let tw2tw4 = _mm256_loadu_pd(self.tw2tw4.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(9) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8 = _mm_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());

                const HI_LO: i32 = 0b0010_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u2 = _mm256_castpd256_pd128(u2u3);
                let u5 = _mm256_extractf128_pd::<1>(u4u5);

                // Radix-9 butterfly

                let (u0u1, u3u4, u6u7) = AvxButterfly::butterfly3_f64(
                    u0u1,
                    _mm256_permute2f128_pd::<HI_LO>(u2u3, u4u5),
                    u6u7,
                    tw3_re,
                    tw3_im,
                );
                let mut u4 = _mm256_extractf128_pd::<1>(u3u4);
                let mut u7 = _mm256_extractf128_pd::<1>(u6u7);
                let (u2, mut u5, mut u8) = AvxButterfly::butterfly3_f64_m128(
                    u2,
                    u5,
                    u8,
                    _mm256_castpd256_pd128(tw3_re),
                    _mm256_castpd256_pd128(tw3_im),
                );

                let u4u7 = _mm256_fcmul_pd(_mm256_create_pd(u4, u7), tw1tw2);
                u4 = _mm256_castpd256_pd128(u4u7);
                u7 = _mm256_extractf128_pd::<1>(u4u7);
                let u5u8 = _mm256_fcmul_pd(_mm256_create_pd(u5, u8), tw2tw4);
                u5 = _mm256_castpd256_pd128(u5u8);
                u8 = _mm256_extractf128_pd::<1>(u5u8);

                let (zu0zu1, zu3zu4, zu6zu7) = AvxButterfly::butterfly3_f64(
                    _mm256_permute2f128_pd::<LO_LO>(u0u1, u3u4),
                    _mm256_permute2f128_pd::<HI_LO>(u0u1, _mm256_castpd128_pd256(u4)),
                    _mm256_create_pd(u2, u5),
                    tw3_re,
                    tw3_im,
                );
                let (zu2, zu5, zu8) = AvxButterfly::butterfly3_f64_m128(
                    _mm256_castpd256_pd128(u6u7),
                    u7,
                    u8,
                    _mm256_castpd256_pd128(tw3_re),
                    _mm256_castpd256_pd128(tw3_im),
                );

                let y0 = zu0zu1;
                let y1 = _mm256_permute2f128_pd::<LO_LO>(_mm256_castpd128_pd256(zu2), zu3zu4);
                let y2 = _mm256_permute2f128_pd::<HI_LO>(zu3zu4, _mm256_castpd128_pd256(zu5));
                let y3 = zu6zu7;

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
                _mm256_storeu_pd(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);
                _mm_storeu_pd(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), zu8);
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
            if src.len() % 9 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let tw3_re = _mm256_set1_pd(self.tw3_re);
            let tw3_im = _mm256_loadu_pd(self.tw3_im.as_ptr().cast());
            let tw1tw2 = _mm256_loadu_pd(self.tw1tw2.as_ptr().cast());
            let tw2tw4 = _mm256_loadu_pd(self.tw2tw4.as_ptr().cast());

            for (dst, src) in dst.chunks_exact_mut(9).zip(src.chunks_exact(9)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8 = _mm_loadu_pd(src.get_unchecked(8..).as_ptr().cast());

                const HI_LO: i32 = 0b0010_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u2 = _mm256_castpd256_pd128(u2u3);
                let u5 = _mm256_extractf128_pd::<1>(u4u5);

                // Radix-9 butterfly

                let (u0u1, u3u4, u6u7) = AvxButterfly::butterfly3_f64(
                    u0u1,
                    _mm256_permute2f128_pd::<HI_LO>(u2u3, u4u5),
                    u6u7,
                    tw3_re,
                    tw3_im,
                );
                let mut u4 = _mm256_extractf128_pd::<1>(u3u4);
                let mut u7 = _mm256_extractf128_pd::<1>(u6u7);
                let (u2, mut u5, mut u8) = AvxButterfly::butterfly3_f64_m128(
                    u2,
                    u5,
                    u8,
                    _mm256_castpd256_pd128(tw3_re),
                    _mm256_castpd256_pd128(tw3_im),
                );

                let u4u7 = _mm256_fcmul_pd(_mm256_create_pd(u4, u7), tw1tw2);
                u4 = _mm256_castpd256_pd128(u4u7);
                u7 = _mm256_extractf128_pd::<1>(u4u7);
                let u5u8 = _mm256_fcmul_pd(_mm256_create_pd(u5, u8), tw2tw4);
                u5 = _mm256_castpd256_pd128(u5u8);
                u8 = _mm256_extractf128_pd::<1>(u5u8);

                let (zu0zu1, zu3zu4, zu6zu7) = AvxButterfly::butterfly3_f64(
                    _mm256_permute2f128_pd::<LO_LO>(u0u1, u3u4),
                    _mm256_permute2f128_pd::<HI_LO>(u0u1, _mm256_castpd128_pd256(u4)),
                    _mm256_create_pd(u2, u5),
                    tw3_re,
                    tw3_im,
                );
                let (zu2, zu5, zu8) = AvxButterfly::butterfly3_f64_m128(
                    _mm256_castpd256_pd128(u6u7),
                    u7,
                    u8,
                    _mm256_castpd256_pd128(tw3_re),
                    _mm256_castpd256_pd128(tw3_im),
                );

                let y0 = zu0zu1;
                let y1 = _mm256_permute2f128_pd::<LO_LO>(_mm256_castpd128_pd256(zu2), zu3zu4);
                let y2 = _mm256_permute2f128_pd::<HI_LO>(zu3zu4, _mm256_castpd128_pd256(zu5));
                let y3 = zu6zu7;

                _mm256_storeu_pd(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y1);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y3);
                _mm_storeu_pd(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), zu8);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly9<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly9<f64> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly9<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
    }
}

impl AvxButterfly9<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 9 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let tw3_re = _mm256_set1_ps(self.tw3_re);
            let tw3_im = _mm256_loadu_ps(self.tw3_im.as_ptr().cast());
            let tw1tw2 = _mm256_loadu_ps(self.tw1tw2.as_ptr().cast());
            let tw2tw4 = _mm256_loadu_ps(self.tw2tw4.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(18) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(chunk.get_unchecked(3..).as_ptr().cast());
                let u5u6u7u8 = _mm256_loadu_ps(chunk.get_unchecked(5..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(chunk.get_unchecked(9..).as_ptr().cast());
                let u3u4u5u6_2 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u5u6u7u8_2 = _mm256_loadu_ps(chunk.get_unchecked(14..).as_ptr().cast());

                let u6u7u8 = _mm256_permute4x64_ps::<{ shuffle(0, 3, 2, 1) }>(u5u6u7u8);
                let u6u7u8_2 = _mm256_permute4x64_ps::<{ shuffle(0, 3, 2, 1) }>(u5u6u7u8_2);

                let (u0u1u2u3, zu3u4u5, zu6u7u8) =
                    AvxButterfly::butterfly3_f32(u0u1u2u3, u3u4u5u6, u6u7u8, tw3_re, tw3_im);
                let (u0u1u2u3_2, zu3u4u5_2, zu6u7u8_2) =
                    AvxButterfly::butterfly3_f32(u0u1u2u3_2, u3u4u5u6_2, u6u7u8_2, tw3_re, tw3_im);

                // Radix-9 butterfly

                let g3u4u5 = _mm256_fcmul_ps(zu3u4u5, tw1tw2);
                let g6u7u8 = _mm256_fcmul_ps(zu6u7u8, tw2tw4);

                let g3u4u5_2 = _mm256_fcmul_ps(zu3u4u5_2, tw1tw2);
                let g6u7u8_2 = _mm256_fcmul_ps(zu6u7u8_2, tw2tw4);

                const HI_HI: i32 = 0b0011_0001;
                const LO_HI: i32 = 0b0011_0000;

                let u3u6u5u8 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u4u7 = _mm256_unpackhi_ps64(g3u4u5, g6u7u8);
                let r0 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u5u8 = _mm256_permute2f128_ps::<HI_HI>(r0, r0);

                let u3u6u5u8_2 = _mm256_unpacklo_ps64(g3u4u5_2, g6u7u8_2);
                let u4u7_2 = _mm256_unpackhi_ps64(g3u4u5_2, g6u7u8_2);
                let r0_2 = _mm256_unpacklo_ps64(g3u4u5_2, g6u7u8_2);
                let u5u8_2 = _mm256_permute2f128_ps::<HI_HI>(r0_2, r0_2);

                let u0u3u6 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 0) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u3u6u5u8)),
                );
                let u1u4u7 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 1) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u4u7)),
                );
                let u2u5u8 = _mm256_permute4x64_ps::<{ shuffle(3, 1, 0, 2) }>(
                    _mm256_permute2f128_ps::<LO_HI>(u5u8, u0u1u2u3),
                );

                let u0u3u6_2 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 0) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3_2, _mm256_castps256_ps128(u3u6u5u8_2)),
                );
                let u1u4u7_2 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 1) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3_2, _mm256_castps256_ps128(u4u7_2)),
                );
                let u2u5u8_2 = _mm256_permute4x64_ps::<{ shuffle(3, 1, 0, 2) }>(
                    _mm256_permute2f128_ps::<LO_HI>(u5u8_2, u0u1u2u3_2),
                );

                let (y0y1y2, y3y4y5, y6y7y8) =
                    AvxButterfly::butterfly3_f32(u0u3u6, u1u4u7, u2u5u8, tw3_re, tw3_im);
                let (y0y1y2_2, y3y4y5_2, y6y7y8_2) =
                    AvxButterfly::butterfly3_f32(u0u3u6_2, u1u4u7_2, u2u5u8_2, tw3_re, tw3_im);

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), y0y1y2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5);
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y6y7y8),
                );
                _m128s_store_f32x2(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y6y7y8),
                );

                _mm256_storeu_ps(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), y0y1y2_2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y3y4y5_2);
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y6y7y8_2),
                );
                _m128s_store_f32x2(
                    chunk.get_unchecked_mut(17..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y6y7y8_2),
                );
            }

            let rem = in_place.chunks_exact_mut(18).into_remainder();

            for chunk in rem.chunks_exact_mut(9) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(chunk.get_unchecked(3..).as_ptr().cast());
                let u5u6u7u8 = _mm256_loadu_ps(chunk.get_unchecked(5..).as_ptr().cast());

                let u6u7u8 = _mm256_permute4x64_ps::<{ shuffle(0, 3, 2, 1) }>(u5u6u7u8);

                let (u0u1u2u3, zu3u4u5, zu6u7u8) =
                    AvxButterfly::butterfly3_f32(u0u1u2u3, u3u4u5u6, u6u7u8, tw3_re, tw3_im);

                // Radix-9 butterfly

                let g3u4u5 = _mm256_fcmul_ps(zu3u4u5, tw1tw2);
                let g6u7u8 = _mm256_fcmul_ps(zu6u7u8, tw2tw4);

                const HI_HI: i32 = 0b0011_0001;
                const LO_HI: i32 = 0b0011_0000;

                let u3u6u5u8 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u4u7 = _mm256_unpackhi_ps64(g3u4u5, g6u7u8);
                let r0 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u5u8 = _mm256_permute2f128_ps::<HI_HI>(r0, r0);

                let u0u3u6 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 0) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u3u6u5u8)),
                );
                let u1u4u7 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 1) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u4u7)),
                );
                let u2u5u8 = _mm256_permute4x64_ps::<{ shuffle(3, 1, 0, 2) }>(
                    _mm256_permute2f128_ps::<LO_HI>(u5u8, u0u1u2u3),
                );

                let (y0y1y2, y3y4y5, y6y7y8) =
                    AvxButterfly::butterfly3_f32(u0u3u6, u1u4u7, u2u5u8, tw3_re, tw3_im);

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), y0y1y2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5);
                _mm_storeu_ps(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y6y7y8),
                );
                _m128s_store_f32x2(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y6y7y8),
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
            if src.len() % 9 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 9 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let tw3_re = _mm256_set1_ps(self.tw3_re);
            let tw3_im = _mm256_loadu_ps(self.tw3_im.as_ptr().cast());
            let tw1tw2 = _mm256_loadu_ps(self.tw1tw2.as_ptr().cast());
            let tw2tw4 = _mm256_loadu_ps(self.tw2tw4.as_ptr().cast());

            for (dst, src) in dst.chunks_exact_mut(18).zip(src.chunks_exact(18)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(src.get_unchecked(3..).as_ptr().cast());
                let u5u6u7u8 = _mm256_loadu_ps(src.get_unchecked(5..).as_ptr().cast());

                let u0u1u2u3_2 = _mm256_loadu_ps(src.get_unchecked(9..).as_ptr().cast());
                let u3u4u5u6_2 = _mm256_loadu_ps(src.get_unchecked(12..).as_ptr().cast());
                let u5u6u7u8_2 = _mm256_loadu_ps(src.get_unchecked(14..).as_ptr().cast());

                let u6u7u8 = _mm256_permute4x64_ps::<{ shuffle(0, 3, 2, 1) }>(u5u6u7u8);
                let u6u7u8_2 = _mm256_permute4x64_ps::<{ shuffle(0, 3, 2, 1) }>(u5u6u7u8_2);

                let (u0u1u2u3, zu3u4u5, zu6u7u8) =
                    AvxButterfly::butterfly3_f32(u0u1u2u3, u3u4u5u6, u6u7u8, tw3_re, tw3_im);
                let (u0u1u2u3_2, zu3u4u5_2, zu6u7u8_2) =
                    AvxButterfly::butterfly3_f32(u0u1u2u3_2, u3u4u5u6_2, u6u7u8_2, tw3_re, tw3_im);

                // Radix-9 butterfly

                let g3u4u5 = _mm256_fcmul_ps(zu3u4u5, tw1tw2);
                let g6u7u8 = _mm256_fcmul_ps(zu6u7u8, tw2tw4);

                let g3u4u5_2 = _mm256_fcmul_ps(zu3u4u5_2, tw1tw2);
                let g6u7u8_2 = _mm256_fcmul_ps(zu6u7u8_2, tw2tw4);

                const HI_HI: i32 = 0b0011_0001;
                const LO_HI: i32 = 0b0011_0000;

                let u3u6u5u8 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u4u7 = _mm256_unpackhi_ps64(g3u4u5, g6u7u8);
                let r0 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u5u8 = _mm256_permute2f128_ps::<HI_HI>(r0, r0);

                let u3u6u5u8_2 = _mm256_unpacklo_ps64(g3u4u5_2, g6u7u8_2);
                let u4u7_2 = _mm256_unpackhi_ps64(g3u4u5_2, g6u7u8_2);
                let r0_2 = _mm256_unpacklo_ps64(g3u4u5_2, g6u7u8_2);
                let u5u8_2 = _mm256_permute2f128_ps::<HI_HI>(r0_2, r0_2);

                let u0u3u6 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 0) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u3u6u5u8)),
                );
                let u1u4u7 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 1) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u4u7)),
                );
                let u2u5u8 = _mm256_permute4x64_ps::<{ shuffle(3, 1, 0, 2) }>(
                    _mm256_permute2f128_ps::<LO_HI>(u5u8, u0u1u2u3),
                );

                let u0u3u6_2 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 0) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3_2, _mm256_castps256_ps128(u3u6u5u8_2)),
                );
                let u1u4u7_2 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 1) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3_2, _mm256_castps256_ps128(u4u7_2)),
                );
                let u2u5u8_2 = _mm256_permute4x64_ps::<{ shuffle(3, 1, 0, 2) }>(
                    _mm256_permute2f128_ps::<LO_HI>(u5u8_2, u0u1u2u3_2),
                );

                let (y0y1y2, y3y4y5, y6y7y8) =
                    AvxButterfly::butterfly3_f32(u0u3u6, u1u4u7, u2u5u8, tw3_re, tw3_im);
                let (y0y1y2_2, y3y4y5_2, y6y7y8_2) =
                    AvxButterfly::butterfly3_f32(u0u3u6_2, u1u4u7_2, u2u5u8_2, tw3_re, tw3_im);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), y0y1y2);
                _mm256_storeu_ps(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5);
                _mm_storeu_ps(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y6y7y8),
                );
                _m128s_store_f32x2(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y6y7y8),
                );

                _mm256_storeu_ps(dst.get_unchecked_mut(9..).as_mut_ptr().cast(), y0y1y2_2);
                _mm256_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), y3y4y5_2);
                _mm_storeu_ps(
                    dst.get_unchecked_mut(15..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y6y7y8_2),
                );
                _m128s_store_f32x2(
                    dst.get_unchecked_mut(17..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y6y7y8_2),
                );
            }

            let rem_dst = dst.chunks_exact_mut(18).into_remainder();
            let rem_src = src.chunks_exact(18).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(9).zip(rem_src.chunks_exact(9)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(src.get_unchecked(3..).as_ptr().cast());
                let u5u6u7u8 = _mm256_loadu_ps(src.get_unchecked(5..).as_ptr().cast());

                let u6u7u8 = _mm256_permute4x64_ps::<{ shuffle(0, 3, 2, 1) }>(u5u6u7u8);

                let (u0u1u2u3, zu3u4u5, zu6u7u8) =
                    AvxButterfly::butterfly3_f32(u0u1u2u3, u3u4u5u6, u6u7u8, tw3_re, tw3_im);

                // Radix-9 butterfly

                let g3u4u5 = _mm256_fcmul_ps(zu3u4u5, tw1tw2);
                let g6u7u8 = _mm256_fcmul_ps(zu6u7u8, tw2tw4);

                const HI_HI: i32 = 0b0011_0001;
                const LO_HI: i32 = 0b0011_0000;

                let u3u6u5u8 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u4u7 = _mm256_unpackhi_ps64(g3u4u5, g6u7u8);
                let r0 = _mm256_unpacklo_ps64(g3u4u5, g6u7u8);
                let u5u8 = _mm256_permute2f128_ps::<HI_HI>(r0, r0);

                let u0u3u6 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 0) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u3u6u5u8)),
                );
                let u1u4u7 = _mm256_permute4x64_ps::<{ shuffle(3, 3, 2, 1) }>(
                    _mm256_insertf128_ps::<1>(u0u1u2u3, _mm256_castps256_ps128(u4u7)),
                );
                let u2u5u8 = _mm256_permute4x64_ps::<{ shuffle(3, 1, 0, 2) }>(
                    _mm256_permute2f128_ps::<LO_HI>(u5u8, u0u1u2u3),
                );

                let (y0y1y2, y3y4y5, y6y7y8) =
                    AvxButterfly::butterfly3_f32(u0u3u6, u1u4u7, u2u5u8, tw3_re, tw3_im);

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), y0y1y2);
                _mm256_storeu_ps(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5);
                _mm_storeu_ps(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_castps256_ps128(y6y7y8),
                );
                _m128s_store_f32x2(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_extractf128_ps::<1>(y6y7y8),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly9<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly9<f32> {
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly9<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        9
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly9, f32, AvxButterfly9, 9, 1e-5);
    test_avx_butterfly!(test_avx_butterfly9_f64, f64, AvxButterfly9, 9, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly9, f32, AvxButterfly9, 9, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly9_f64, f64, AvxButterfly9, 9, 1e-7);
}
