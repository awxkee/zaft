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
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, shuffle,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly6<T> {
    direction: FftDirection,
    twiddle_re: T,
    twiddle_im: [T; 8],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly6<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle::<T>(1, 3, fft_direction);
        Self {
            direction: fft_direction,
            twiddle_re: twiddle.re,
            twiddle_im: [
                -twiddle.im,
                twiddle.im,
                -twiddle.im,
                twiddle.im,
                -twiddle.im,
                twiddle.im,
                -twiddle.im,
                twiddle.im,
            ],
        }
    }
}

impl AvxButterfly6<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 6 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let twiddle_re = _mm256_set1_pd(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_pd(self.twiddle_im.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(12) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_LO>(u0u1, u6u7);
                let u1 = _mm256_permute2f128_pd::<HI_HI>(u0u1, u6u7);
                let u2 = _mm256_permute2f128_pd::<LO_LO>(u2u3, u8u9);
                let u3 = _mm256_permute2f128_pd::<HI_HI>(u2u3, u8u9);
                let u4 = _mm256_permute2f128_pd::<LO_LO>(u4u5, u10u11);
                let u5 = _mm256_permute2f128_pd::<HI_HI>(u4u5, u10u11);

                let (t0, t2, t4) =
                    AvxButterfly::butterfly3_f64(u0, u2, u4, twiddle_re, twiddle_w_2);
                let (t1, t3, t5) =
                    AvxButterfly::butterfly3_f64(u3, u5, u1, twiddle_re, twiddle_w_2);
                let (y0, y3) = AvxButterfly::butterfly2_f64(t0, t1);
                let (y4, y1) = AvxButterfly::butterfly2_f64(t2, t3);
                let (y2, y5) = AvxButterfly::butterfly2_f64(t4, t5);

                let y0y1 = _mm256_permute2f128_pd::<LO_LO>(y0, y1);
                let y2y3 = _mm256_permute2f128_pd::<LO_LO>(y2, y3);
                let y4y5 = _mm256_permute2f128_pd::<LO_LO>(y4, y5);
                let y6y7 = _mm256_permute2f128_pd::<HI_HI>(y0, y1);
                let y8y9 = _mm256_permute2f128_pd::<HI_HI>(y2, y3);
                let y10y11 = _mm256_permute2f128_pd::<HI_HI>(y4, y5);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
                _mm256_storeu_pd(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), y6y7);
                _mm256_storeu_pd(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8y9);
                _mm256_storeu_pd(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), y10y11);
            }

            let rem = in_place.chunks_exact_mut(12).into_remainder();

            for chunk in rem.chunks_exact_mut(6) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let u1 = _mm256_extractf128_pd::<1>(u0u1);
                let u2 = _mm256_castpd256_pd128(u2u3);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm256_castpd256_pd128(u4u5);
                let u5 = _mm256_extractf128_pd::<1>(u4u5);

                let (t0, t2, t4) = AvxButterfly::butterfly3_f64_m128(
                    u0,
                    u2,
                    u4,
                    _mm256_castpd256_pd128(twiddle_re),
                    _mm256_castpd256_pd128(twiddle_w_2),
                );
                let (t1, t3, t5) = AvxButterfly::butterfly3_f64_m128(
                    u3,
                    u5,
                    u1,
                    _mm256_castpd256_pd128(twiddle_re),
                    _mm256_castpd256_pd128(twiddle_w_2),
                );
                let (y0, y3) = AvxButterfly::butterfly2_f64_m128(t0, t1);
                let (y4, y1) = AvxButterfly::butterfly2_f64_m128(t2, t3);
                let (y2, y5) = AvxButterfly::butterfly2_f64_m128(t4, t5);

                let y0y1 = _mm256_create_pd(y0, y1);
                let y2y3 = _mm256_create_pd(y2, y3);
                let y4y5 = _mm256_create_pd(y4, y5);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
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
            if src.len() % 6 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 6 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let twiddle_re = _mm256_set1_pd(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_pd(self.twiddle_im.as_ptr().cast());

            for (dst, src) in dst.chunks_exact_mut(12).zip(src.chunks_exact(12)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(src.get_unchecked(10..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_LO>(u0u1, u6u7);
                let u1 = _mm256_permute2f128_pd::<HI_HI>(u0u1, u6u7);
                let u2 = _mm256_permute2f128_pd::<LO_LO>(u2u3, u8u9);
                let u3 = _mm256_permute2f128_pd::<HI_HI>(u2u3, u8u9);
                let u4 = _mm256_permute2f128_pd::<LO_LO>(u4u5, u10u11);
                let u5 = _mm256_permute2f128_pd::<HI_HI>(u4u5, u10u11);

                let (t0, t2, t4) =
                    AvxButterfly::butterfly3_f64(u0, u2, u4, twiddle_re, twiddle_w_2);
                let (t1, t3, t5) =
                    AvxButterfly::butterfly3_f64(u3, u5, u1, twiddle_re, twiddle_w_2);
                let (y0, y3) = AvxButterfly::butterfly2_f64(t0, t1);
                let (y4, y1) = AvxButterfly::butterfly2_f64(t2, t3);
                let (y2, y5) = AvxButterfly::butterfly2_f64(t4, t5);

                let y0y1 = _mm256_permute2f128_pd::<LO_LO>(y0, y1);
                let y2y3 = _mm256_permute2f128_pd::<LO_LO>(y2, y3);
                let y4y5 = _mm256_permute2f128_pd::<LO_LO>(y4, y5);
                let y6y7 = _mm256_permute2f128_pd::<HI_HI>(y0, y1);
                let y8y9 = _mm256_permute2f128_pd::<HI_HI>(y2, y3);
                let y10y11 = _mm256_permute2f128_pd::<HI_HI>(y4, y5);

                _mm256_storeu_pd(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), y6y7);
                _mm256_storeu_pd(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y8y9);
                _mm256_storeu_pd(dst.get_unchecked_mut(10..).as_mut_ptr().cast(), y10y11);
            }

            let rem_src = src.chunks_exact(12).remainder();
            let rem_dst = dst.chunks_exact_mut(12).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(6).zip(rem_src.chunks_exact(6)) {
                let u0u1 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(u0u1);
                let u1 = _mm256_extractf128_pd::<1>(u0u1);
                let u2 = _mm256_castpd256_pd128(u2u3);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm256_castpd256_pd128(u4u5);
                let u5 = _mm256_extractf128_pd::<1>(u4u5);

                let (t0, t2, t4) = AvxButterfly::butterfly3_f64_m128(
                    u0,
                    u2,
                    u4,
                    _mm256_castpd256_pd128(twiddle_re),
                    _mm256_castpd256_pd128(twiddle_w_2),
                );
                let (t1, t3, t5) = AvxButterfly::butterfly3_f64_m128(
                    u3,
                    u5,
                    u1,
                    _mm256_castpd256_pd128(twiddle_re),
                    _mm256_castpd256_pd128(twiddle_w_2),
                );
                let (y0, y3) = AvxButterfly::butterfly2_f64_m128(t0, t1);
                let (y4, y1) = AvxButterfly::butterfly2_f64_m128(t2, t3);
                let (y2, y5) = AvxButterfly::butterfly2_f64_m128(t4, t5);

                let y0y1 = _mm256_create_pd(y0, y1);
                let y2y3 = _mm256_create_pd(y2, y3);
                let y4y5 = _mm256_create_pd(y4, y5);

                _mm256_storeu_pd(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly6<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly6<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly6<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        6
    }
}

impl AvxButterfly6<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 6 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let twiddle_re = _mm256_set1_ps(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_ps(self.twiddle_im.as_ptr().cast());

            for chunk in in_place.chunks_exact_mut(12) {
                let u0u1u2u3 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());

                let u0u1 = _mm256_castpd256_pd128(u0u1u2u3); // u0 u1
                let u2u3 = _mm256_extractf128_pd::<1>(u0u1u2u3); // u2 u3

                let u4u5 = _mm256_castpd256_pd128(u4u5u6u7); // u4 u5
                let u6u7 = _mm256_extractf128_pd::<1>(u4u5u6u7); // u6 u7

                let u8u9 = _mm256_castpd256_pd128(u8u9u10u11); // u8 u9
                let u10u11 = _mm256_extractf128_pd::<1>(u8u9u10u11); // u10 u11

                let u0 = _mm_castpd_ps(_mm_shuffle_pd::<0x0>(u0u1, u6u7));
                let u1 = _mm_castpd_ps(_mm_shuffle_pd::<0b11>(u0u1, u6u7));
                let u2 = _mm_castpd_ps(_mm_shuffle_pd::<0x0>(u2u3, u8u9));
                let u3 = _mm_castpd_ps(_mm_shuffle_pd::<0b11>(u2u3, u8u9));
                let u4 = _mm_castpd_ps(_mm_shuffle_pd::<0x0>(u4u5, u10u11));
                let u5 = _mm_castpd_ps(_mm_shuffle_pd::<0b11>(u4u5, u10u11));

                let (t0t1, t2t3, t4t5) = AvxButterfly::butterfly3_f32(
                    _mm256_create_ps(u0, u3),
                    _mm256_create_ps(u2, u5),
                    _mm256_create_ps(u4, u1),
                    twiddle_re,
                    twiddle_w_2,
                );

                let (y0, y3) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(t0t1),
                    _mm256_extractf128_ps::<1>(t0t1),
                );
                let (y4, y1) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(t2t3),
                    _mm256_extractf128_ps::<1>(t2t3),
                );
                let (y2, y5) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(t4t5),
                    _mm256_extractf128_ps::<1>(t4t5),
                );

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y0, y1), _mm_unpacklo_ps64(y2, y3));
                let y4y5y6y7 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y4, y5), _mm_unpackhi_ps64(y0, y1));
                let y8y9y10y11 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y2, y3), _mm_unpackhi_ps64(y4, y5));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5y6y7);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y8y9y10y11);
            }

            let rem = in_place.chunks_exact_mut(12).into_remainder();

            for chunk in rem.chunks_exact_mut(6) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5 = _mm_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);

                let u0 = _mm256_castps256_ps128(u0u1u2u3);
                let u1 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u0, u0);
                let u2 = u2u3;
                let u3 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u2u3, u2u3);
                let u4 = u4u5;
                let u5 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u4u5, u4u5);

                let (t0, t2, t4) = AvxButterfly::butterfly3_f32_m128(
                    u0,
                    u2,
                    u4,
                    _mm256_castps256_ps128(twiddle_re),
                    _mm256_castps256_ps128(twiddle_w_2),
                );
                let (t1, t3, t5) = AvxButterfly::butterfly3_f32_m128(
                    u3,
                    u5,
                    u1,
                    _mm256_castps256_ps128(twiddle_re),
                    _mm256_castps256_ps128(twiddle_w_2),
                );
                let (y0, y3) = AvxButterfly::butterfly2_f32_m128(t0, t1);
                let (y4, y1) = AvxButterfly::butterfly2_f32_m128(t2, t3);
                let (y2, y5) = AvxButterfly::butterfly2_f32_m128(t4, t5);

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y0, y1), _mm_unpacklo_ps64(y2, y3));
                let y2y3 = _mm_unpacklo_ps64(y4, y5);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y2y3);
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
            if src.len() % 6 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
            }
            if dst.len() % 6 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
            }

            let twiddle_re = _mm256_set1_ps(self.twiddle_re);
            let twiddle_w_2 = _mm256_loadu_ps(self.twiddle_im.as_ptr().cast());

            for (dst, src) in dst.chunks_exact_mut(12).zip(src.chunks_exact(12)) {
                let u0u1u2u3 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());

                let u0u1 = _mm256_castpd256_pd128(u0u1u2u3); // u0 u1
                let u2u3 = _mm256_extractf128_pd::<1>(u0u1u2u3); // u2 u3

                let u4u5 = _mm256_castpd256_pd128(u4u5u6u7); // u4 u5
                let u6u7 = _mm256_extractf128_pd::<1>(u4u5u6u7); // u6 u7

                let u8u9 = _mm256_castpd256_pd128(u8u9u10u11); // u8 u9
                let u10u11 = _mm256_extractf128_pd::<1>(u8u9u10u11); // u10 u11

                let u0 = _mm_castpd_ps(_mm_shuffle_pd::<0x0>(u0u1, u6u7));
                let u1 = _mm_castpd_ps(_mm_shuffle_pd::<0b11>(u0u1, u6u7));
                let u2 = _mm_castpd_ps(_mm_shuffle_pd::<0x0>(u2u3, u8u9));
                let u3 = _mm_castpd_ps(_mm_shuffle_pd::<0b11>(u2u3, u8u9));
                let u4 = _mm_castpd_ps(_mm_shuffle_pd::<0x0>(u4u5, u10u11));
                let u5 = _mm_castpd_ps(_mm_shuffle_pd::<0b11>(u4u5, u10u11));

                let (t0t1, t2t3, t4t5) = AvxButterfly::butterfly3_f32(
                    _mm256_create_ps(u0, u3),
                    _mm256_create_ps(u2, u5),
                    _mm256_create_ps(u4, u1),
                    twiddle_re,
                    twiddle_w_2,
                );

                let (y0, y3) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(t0t1),
                    _mm256_extractf128_ps::<1>(t0t1),
                );
                let (y4, y1) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(t2t3),
                    _mm256_extractf128_ps::<1>(t2t3),
                );
                let (y2, y5) = AvxButterfly::butterfly2_f32_m128(
                    _mm256_castps256_ps128(t4t5),
                    _mm256_extractf128_ps::<1>(t4t5),
                );

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y0, y1), _mm_unpacklo_ps64(y2, y3));
                let y4y5y6y7 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y4, y5), _mm_unpackhi_ps64(y0, y1));
                let y8y9y10y11 =
                    _mm256_create_ps(_mm_unpackhi_ps64(y2, y3), _mm_unpackhi_ps64(y4, y5));

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5y6y7);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y8y9y10y11);
            }

            let rem_src = src.chunks_exact(12).remainder();
            let rem_dst = dst.chunks_exact_mut(12).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(6).zip(rem_src.chunks_exact(6)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());
                let u4u5 = _mm_loadu_ps(src.get_unchecked(4..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);

                let u0 = _mm256_castps256_ps128(u0u1u2u3);
                let u1 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u0, u0);
                let u2 = u2u3;
                let u3 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u2u3, u2u3);
                let u4 = u4u5;
                let u5 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u4u5, u4u5);

                let (t0, t2, t4) = AvxButterfly::butterfly3_f32_m128(
                    u0,
                    u2,
                    u4,
                    _mm256_castps256_ps128(twiddle_re),
                    _mm256_castps256_ps128(twiddle_w_2),
                );
                let (t1, t3, t5) = AvxButterfly::butterfly3_f32_m128(
                    u3,
                    u5,
                    u1,
                    _mm256_castps256_ps128(twiddle_re),
                    _mm256_castps256_ps128(twiddle_w_2),
                );
                let (y0, y3) = AvxButterfly::butterfly2_f32_m128(t0, t1);
                let (y4, y1) = AvxButterfly::butterfly2_f32_m128(t2, t3);
                let (y2, y5) = AvxButterfly::butterfly2_f32_m128(t4, t5);

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y0, y1), _mm_unpacklo_ps64(y2, y3));
                let y2y3 = _mm_unpacklo_ps64(y4, y5);

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y2y3);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly6<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly6<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly6<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        6
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly6_f32() {
        for i in 1..5 {
            let size = 6usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly6::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly6::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 6f32)).collect();

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

    #[test]
    fn test_butterfly6_f64() {
        for i in 1..5 {
            let size = 6usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly6::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly6::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 6f64)).collect();

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
    fn test_butterfly6_out_of_place_f64() {
        for i in 1..5 {
            let size = 6usize.pow(i);
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
            let radix_forward = AvxButterfly6::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly6::new(FftDirection::Inverse);

            let reference_dft = Dft::new(6, FftDirection::Forward).unwrap();
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

            input = input.iter().map(|&x| x * (1.0 / 6f64)).collect();

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
    fn test_butterfly6_out_of_place_f32() {
        for i in 1..5 {
            let size = 6usize.pow(i);
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
            let radix_forward = AvxButterfly6::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly6::new(FftDirection::Inverse);

            let reference_dft = Dft::new(6, FftDirection::Forward).unwrap();
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

            input = input.iter().map(|&x| x * (1.0 / 6f32)).collect();

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
