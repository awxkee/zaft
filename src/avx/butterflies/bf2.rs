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

use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, shuffle};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxButterfly2<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly2<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            phantom_data: PhantomData,
        }
    }
}

impl AvxButterfly2<f64> {
    #[target_feature(enable = "avx2")]
    fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let a = _mm256_loadu_pd(chunk.as_ptr().cast());
                let b = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());

                let u0 = _mm256_permute2f128_pd::<0x20>(a, b);
                let u1 = _mm256_permute2f128_pd::<0x31>(a, b);

                let y0 = _mm256_add_pd(u0, u1);
                let y1 = _mm256_sub_pd(u0, u1);

                let zy0 = _mm256_permute2f128_pd::<0x20>(y0, y1);
                let zy1 = _mm256_permute2f128_pd::<0x31>(y0, y1);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), zy0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), zy1);
            }
        }

        let rem = in_place.chunks_exact_mut(4).into_remainder();

        for chunk in rem.chunks_exact_mut(2) {
            unsafe {
                let uz0 = _mm256_loadu_pd(chunk.as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(uz0);
                let u1 = _mm256_extractf128_pd::<1>(uz0);

                let y0 = _mm_add_pd(u0, u1);
                let y1 = _mm_sub_pd(u0, u1);

                let y0y1 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(y0), y1);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), y0y1);
            }
        }
        Ok(())
    }

    #[allow(unused)]
    #[target_feature(enable = "avx2")]
    fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            unsafe {
                let a = _mm256_loadu_pd(src.as_ptr().cast());
                let b = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());

                let u0 = _mm256_permute2f128_pd::<0x20>(a, b);
                let u1 = _mm256_permute2f128_pd::<0x31>(a, b);

                let y0 = _mm256_add_pd(u0, u1);
                let y1 = _mm256_sub_pd(u0, u1);

                let zy0 = _mm256_permute2f128_pd::<0x20>(y0, y1);
                let zy1 = _mm256_permute2f128_pd::<0x31>(y0, y1);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), zy0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), zy1);
            }
        }

        let rem_src = src.chunks_exact(4).remainder();
        let rem_dst = dst.chunks_exact_mut(4).into_remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(2).zip(rem_src.chunks_exact(2)) {
            unsafe {
                let uz0 = _mm256_loadu_pd(src.as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(uz0);
                let u1 = _mm256_extractf128_pd::<1>(uz0);

                let y0 = _mm_add_pd(u0, u1);
                let y1 = _mm_sub_pd(u0, u1);

                let y0y1 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(y0), y1);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), y0y1);
            }
        }
        Ok(())
    }
}

impl AvxButterfly2<f32> {
    #[target_feature(enable = "avx2")]
    fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let a = _mm256_loadu_ps(chunk.as_ptr().cast());

                const SH: i32 = shuffle(3, 1, 2, 0);
                let ab =
                    _mm256_castsi256_ps(_mm256_permute4x64_epi64::<SH>(_mm256_castps_si256(a)));
                let u0 = _mm256_castps256_ps128(ab);
                let u1 = _mm256_extractf128_ps::<1>(ab);

                let y0 = _mm_add_ps(u0, u1);
                let y1 = _mm_sub_ps(u0, u1);

                let y0y1 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(y0), y1);

                let vy0y1 =
                    _mm256_castsi256_ps(_mm256_permute4x64_epi64::<SH>(_mm256_castps_si256(y0y1)));

                _mm256_storeu_ps(chunk.as_mut_ptr().cast(), vy0y1);
            }
        }

        let rem = in_place.chunks_exact_mut(4).into_remainder();

        for chunk in rem.chunks_exact_mut(2) {
            unsafe {
                let uz0 = _mm_loadu_ps(chunk.as_ptr().cast());

                let u0 = _mm_unpacklo_ps64(uz0, uz0);
                let u1 = _mm_unpackhi_ps64(uz0, uz0);

                let y0 = _mm_add_ps(u0, u1);
                let y1 = _mm_sub_ps(u0, u1);

                let y0y1 = _mm_unpacklo_ps64(y0, y1);

                _mm_storeu_ps(chunk.as_mut_ptr().cast(), y0y1);
            }
        }

        Ok(())
    }

    #[allow(unused)]
    #[target_feature(enable = "avx2")]
    fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(2) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            unsafe {
                let a = _mm256_loadu_ps(src.as_ptr().cast());

                const SH: i32 = shuffle(3, 1, 2, 0);
                let ab =
                    _mm256_castsi256_ps(_mm256_permute4x64_epi64::<SH>(_mm256_castps_si256(a)));
                let u0 = _mm256_castps256_ps128(ab);
                let u1 = _mm256_extractf128_ps::<1>(ab);

                let y0 = _mm_add_ps(u0, u1);
                let y1 = _mm_sub_ps(u0, u1);

                let y0y1 = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(y0), y1);

                let vy0y1 =
                    _mm256_castsi256_ps(_mm256_permute4x64_epi64::<SH>(_mm256_castps_si256(y0y1)));

                _mm256_storeu_ps(dst.as_mut_ptr().cast(), vy0y1);
            }
        }

        let rem_src = src.chunks_exact(4).remainder();
        let rem_dst = dst.chunks_exact_mut(4).into_remainder();

        for (dst, src) in rem_dst.chunks_exact_mut(2).zip(rem_src.chunks_exact(2)) {
            unsafe {
                let uz0 = _mm_loadu_ps(src.as_ptr().cast());

                let u0 = _mm_unpacklo_ps64(uz0, uz0);
                let u1 = _mm_unpackhi_ps64(uz0, uz0);

                let y0 = _mm_add_ps(u0, u1);
                let y1 = _mm_sub_ps(u0, u1);

                let y0y1 = _mm_unpacklo_ps64(y0, y1);

                _mm_storeu_ps(dst.as_mut_ptr().cast(), y0y1);
            }
        }

        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly2<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f64>],
        dst: &mut [Complex<f64>],
        scratch: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        2
    }

    fn scratch_length(&self) -> usize {
        0
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

impl FftExecutor<f32> for AvxButterfly2<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f32>],
        dst: &mut [Complex<f32>],
        scratch: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        2
    }

    fn scratch_length(&self) -> usize {
        0
    }

    fn out_of_place_scratch_length(&self) -> usize {
        0
    }

    fn destructive_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};

    test_avx_butterfly!(test_avx_butterfly2, f32, AvxButterfly2, 2, 1e-5);
    test_avx_butterfly!(test_avx_butterfly2_f64, f64, AvxButterfly2, 2, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly2, f32, AvxButterfly2, 2, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly2_f64, f64, AvxButterfly2, 2, 1e-7);
}
