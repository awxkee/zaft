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

use crate::avx::util::{
    _mm_unpackhi_ps64, _mm256_create_ps, _mm256_permute4x64_ps, _mm256s_deinterleave3_epi64,
    _mm256s_interleave3_epi64, shuffle,
};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxButterfly3<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
    tw1: [T; 8],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly3<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let twiddle = compute_twiddle(1, 3, fft_direction);
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
            twiddle,
            tw1: [
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

impl AvxButterfly3<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let twiddle_re = _mm256_set1_ps(self.twiddle.re);
            let tw1 = _mm256_loadu_ps(self.tw1.as_ptr());

            for chunk in in_place.chunks_exact_mut(12) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u6u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());

                let (u0, u1, u2) = _mm256s_deinterleave3_epi64(u0u1u2u3, u4u6u6u7, u8u9u10u11);

                let xp = _mm256_add_ps(u1, u2);
                let xn = _mm256_sub_ps(u1, u2);
                let sum = _mm256_add_ps(u0, xp);

                let w_1 = _mm256_fmadd_ps(twiddle_re, xp, u0);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let xn_rot = _mm256_shuffle_ps::<SH>(xn, xn);

                let zy0 = sum;
                let zy1 = _mm256_fmadd_ps(tw1, xn_rot, w_1);
                let zy2 = _mm256_fnmadd_ps(tw1, xn_rot, w_1);

                let (y0, y1, y2) = _mm256s_interleave3_epi64(zy0, zy1, zy2);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y1);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y2);
            }

            let rem = in_place.chunks_exact_mut(12).into_remainder();

            for chunk in rem.chunks_exact_mut(6) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast()); // u0, u1,
                let u4u6 = _mm_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());

                const SH0312: i32 = shuffle(1, 2, 3, 0);
                let u0u3u2u1 = _mm256_permute4x64_ps::<SH0312>(u0u1u2u3);
                let u2u1 = _mm256_extractf128_ps::<1>(u0u3u2u1);

                let u0 = _mm256_castps256_ps128(u0u3u2u1);
                let u1 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(u2u1, u4u6);
                let u2 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(u2u1, u4u6);

                let xp = _mm_add_ps(u1, u2);
                let xn = _mm_sub_ps(u1, u2);
                let sum = _mm_add_ps(u0, xp);

                let w_1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_re), xp, u0);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

                let zy0 = sum;
                let zy1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);
                let zy2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);

                let y0 = _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(zy0, zy1);
                let y1 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(zy2, zy0);
                let y2 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(zy1, zy2);

                let y0y1 = _mm256_create_ps(y0, y1);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
            }

            let rem = rem.chunks_exact_mut(6).into_remainder();

            for chunk in rem.chunks_exact_mut(3) {
                let uz = _mm_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());

                let u0 = uz;
                let u1 = _mm_unpackhi_ps64(uz, uz);
                let u2 = _mm_castsi128_ps(_mm_loadu_si64(chunk.get_unchecked(2..).as_ptr().cast()));

                let xp = _mm_add_ps(u1, u2);
                let xn = _mm_sub_ps(u1, u2);
                let sum = _mm_add_ps(u0, xp);

                let w_1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_re), xp, u0);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

                let y0 = sum;
                let y1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);
                let y2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);

                _mm_storeu_pd(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm_unpacklo_pd(_mm_castps_pd(y0), _mm_castps_pd(y1)),
                );
                _mm_storeu_si64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm_castps_si128(y2),
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
        if src.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let twiddle_re = _mm256_set1_ps(self.twiddle.re);
            let tw1 = _mm256_loadu_ps(self.tw1.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(12).zip(src.chunks_exact(12)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());
                let u4u6u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(src.get_unchecked(8..).as_ptr().cast());

                let (u0, u1, u2) = _mm256s_deinterleave3_epi64(u0u1u2u3, u4u6u6u7, u8u9u10u11);

                let xp = _mm256_add_ps(u1, u2);
                let xn = _mm256_sub_ps(u1, u2);
                let sum = _mm256_add_ps(u0, xp);

                let w_1 = _mm256_fmadd_ps(twiddle_re, xp, u0);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let xn_rot = _mm256_shuffle_ps::<SH>(xn, xn);

                let zy0 = sum;
                let zy1 = _mm256_fmadd_ps(tw1, xn_rot, w_1);
                let zy2 = _mm256_fnmadd_ps(tw1, xn_rot, w_1);

                let (y0, y1, y2) = _mm256s_interleave3_epi64(zy0, zy1, zy2);

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y1);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y2);
            }

            let rem_src = src.chunks_exact(12).remainder();
            let rem_dst = dst.chunks_exact_mut(12).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(6).zip(rem_src.chunks_exact(6)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast()); // u0, u1,
                let u4u6 = _mm_loadu_ps(src.get_unchecked(4..).as_ptr().cast());

                const SH0312: i32 = shuffle(1, 2, 3, 0);
                let u0u3u2u1 = _mm256_permute4x64_ps::<SH0312>(u0u1u2u3);
                let u2u1 = _mm256_extractf128_ps::<1>(u0u3u2u1);

                let u0 = _mm256_castps256_ps128(u0u3u2u1);
                let u1 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(u2u1, u4u6);
                let u2 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(u2u1, u4u6);

                let xp = _mm_add_ps(u1, u2);
                let xn = _mm_sub_ps(u1, u2);
                let sum = _mm_add_ps(u0, xp);

                let w_1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_re), xp, u0);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

                let zy0 = sum;
                let zy1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);
                let zy2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);

                let y0 = _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(zy0, zy1);
                let y1 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(zy2, zy0);
                let y2 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(zy1, zy2);

                let y0y1 = _mm256_create_ps(y0, y1);

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y2);
            }

            let rem_src = rem_src.chunks_exact(6).remainder();
            let rem_dst = rem_dst.chunks_exact_mut(6).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(3).zip(rem_src.chunks_exact(3)) {
                let uz = _mm_loadu_ps(src.get_unchecked(0..).as_ptr().cast());

                let u0 = uz;
                let u1 = _mm_unpackhi_ps64(uz, uz);
                let u2 = _mm_castsi128_ps(_mm_loadu_si64(src.get_unchecked(2..).as_ptr().cast()));

                let xp = _mm_add_ps(u1, u2);
                let xn = _mm_sub_ps(u1, u2);
                let sum = _mm_add_ps(u0, xp);

                let w_1 = _mm_fmadd_ps(_mm256_castps256_ps128(twiddle_re), xp, u0);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

                let y0 = sum;
                let y1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);
                let y2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1), xn_rot, w_1);

                _mm_storeu_pd(
                    dst.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    _mm_unpacklo_pd(_mm_castps_pd(y0), _mm_castps_pd(y1)),
                );
                _mm_storeu_si64(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm_castps_si128(y2),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly3<f32> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly3<f32> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

impl AvxButterfly3<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let twiddle_re = _mm256_set1_pd(self.twiddle.re);
            let tw1 = _mm256_loadu_pd(self.tw1.as_ptr());

            for chunk in in_place.chunks_exact_mut(6) {
                let uz01 = _mm256_loadu_pd(chunk.as_ptr().cast());
                let uz23 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let uz46 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_HI>(uz01, uz23);
                let u1 = _mm256_permute2f128_pd::<HI_LO>(uz01, uz46);
                let u2 = _mm256_permute2f128_pd::<LO_HI>(uz23, uz46);

                let xp = _mm256_add_pd(u1, u2);
                let xn = _mm256_sub_pd(u1, u2);
                let sum = _mm256_add_pd(u0, xp);

                let w_1 = _mm256_fmadd_pd(twiddle_re, xp, u0);

                let xn_rot = _mm256_permute_pd::<0b0101>(xn);

                let zy0 = sum;
                let zy1 = _mm256_fmadd_pd(tw1, xn_rot, w_1);
                let zy2 = _mm256_fnmadd_pd(tw1, xn_rot, w_1);

                let u0 = _mm256_permute2f128_pd::<LO_LO>(zy0, zy1);
                let u1 = _mm256_permute2f128_pd::<LO_HI>(zy2, zy0);
                let u2 = _mm256_permute2f128_pd::<HI_HI>(zy1, zy2);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), u0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), u1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), u2);
            }

            let rem = in_place.chunks_exact_mut(6).into_remainder();

            for chunk in rem.chunks_exact_mut(3) {
                let uz0 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(uz0);
                let u1 = _mm256_extractf128_pd::<1>(uz0);
                let u2 = _mm_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());

                let xp = _mm_add_pd(u1, u2);
                let xn = _mm_sub_pd(u1, u2);
                let sum = _mm_add_pd(u0, xp);

                let w_1 = _mm_fmadd_pd(_mm256_castpd256_pd128(twiddle_re), xp, u0);

                let xn_rot = _mm_shuffle_pd::<0b01>(xn, xn);

                let y0 = sum;
                let y1 = _mm_fmadd_pd(_mm256_castpd256_pd128(tw1), xn_rot, w_1);
                let y2 = _mm_fnmadd_pd(_mm256_castpd256_pd128(tw1), xn_rot, w_1);

                let y01 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(y0), y1);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y01);
                _mm_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f64_impl(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let twiddle_re = _mm256_set1_pd(self.twiddle.re);
            let tw1 = _mm256_loadu_pd(self.tw1.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(6).zip(src.chunks_exact(6)) {
                let uz01 = _mm256_loadu_pd(src.as_ptr().cast());
                let uz23 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let uz46 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());

                const LO_HI: i32 = 0b0011_0000;
                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let u0 = _mm256_permute2f128_pd::<LO_HI>(uz01, uz23);
                let u1 = _mm256_permute2f128_pd::<HI_LO>(uz01, uz46);
                let u2 = _mm256_permute2f128_pd::<LO_HI>(uz23, uz46);

                let xp = _mm256_add_pd(u1, u2);
                let xn = _mm256_sub_pd(u1, u2);
                let sum = _mm256_add_pd(u0, xp);

                let w_1 = _mm256_fmadd_pd(twiddle_re, xp, u0);

                let xn_rot = _mm256_permute_pd::<0b0101>(xn);

                let zy0 = sum;
                let zy1 = _mm256_fmadd_pd(tw1, xn_rot, w_1);
                let zy2 = _mm256_fnmadd_pd(tw1, xn_rot, w_1);

                let u0 = _mm256_permute2f128_pd::<LO_LO>(zy0, zy1);
                let u1 = _mm256_permute2f128_pd::<LO_HI>(zy2, zy0);
                let u2 = _mm256_permute2f128_pd::<HI_HI>(zy1, zy2);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), u0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), u1);
                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), u2);
            }

            let rem_src = src.chunks_exact(6).remainder();
            let rem_dst = dst.chunks_exact_mut(6).into_remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(3).zip(rem_src.chunks_exact(3)) {
                let uz0 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(uz0);
                let u1 = _mm256_extractf128_pd::<1>(uz0);
                let u2 = _mm_loadu_pd(src.get_unchecked(2..).as_ptr().cast());

                let xp = _mm_add_pd(u1, u2);
                let xn = _mm_sub_pd(u1, u2);
                let sum = _mm_add_pd(u0, xp);

                let w_1 = _mm_fmadd_pd(_mm256_castpd256_pd128(twiddle_re), xp, u0);

                let xn_rot = _mm_shuffle_pd::<0b01>(xn, xn);

                let y0 = sum;
                let y1 = _mm_fmadd_pd(_mm256_castpd256_pd128(tw1), xn_rot, w_1);
                let y2 = _mm_fnmadd_pd(_mm256_castpd256_pd128(tw1), xn_rot, w_1);

                let y01 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(y0), y1);

                _mm256_storeu_pd(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y01);
                _mm_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly3<f64> {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64_impl(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly3<f64> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f32> for AvxButterfly3<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        3
    }
}

impl FftExecutor<f64> for AvxButterfly3<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};
    use rand::Rng;

    test_avx_butterfly!(test_avx_butterfly3, f32, AvxButterfly3, 3, 1e-5);
    test_avx_butterfly!(test_avx_butterfly3_f64, f64, AvxButterfly3, 3, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly3, f32, AvxButterfly3, 3, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly3_f64, f64, AvxButterfly3, 3, 1e-7);
}
