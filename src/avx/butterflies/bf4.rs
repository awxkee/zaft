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
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256s_deinterleave4_epi64, _mm256s_interleave4_epi64,
    shuffle,
};
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly4<T> {
    direction: FftDirection,
    multiplier: [T; 8],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            multiplier: match fft_direction {
                FftDirection::Inverse => [
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                ],
                FftDirection::Forward => [
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                    0.0.as_(),
                    -0.0.as_(),
                ],
            },
        }
    }
}

impl AvxButterfly4<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            let v_i_multiplier = _mm256_loadu_ps(self.multiplier.as_ptr());

            for chunk in in_place.chunks_exact_mut(16) {
                let aaaa0 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let bbbb0 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let cccc0 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let dddd0 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());

                let (a, b, c, d) = _mm256s_deinterleave4_epi64(aaaa0, bbbb0, cccc0, dddd0);

                let t0 = _mm256_add_ps(a, c);
                let t1 = _mm256_sub_ps(a, c);
                let t2 = _mm256_add_ps(b, d);
                let mut t3 = _mm256_sub_ps(b, d);
                const SH: i32 = shuffle(2, 3, 0, 1);
                t3 = _mm256_xor_ps(_mm256_shuffle_ps::<SH>(t3, t3), v_i_multiplier);

                let xy0 = _mm256_add_ps(t0, t2);
                let xy1 = _mm256_add_ps(t1, t3);
                let xy2 = _mm256_sub_ps(t0, t2);
                let xy3 = _mm256_sub_ps(t1, t3);

                let (y0, y1, y2, y3) = _mm256s_interleave4_epi64(xy0, xy1, xy2, xy3);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y1);
                _mm256_storeu_ps(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), y2);
                _mm256_storeu_ps(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), y3);
            }

            let rem = in_place.chunks_exact_mut(16).into_remainder();

            for chunk in rem.chunks_exact_mut(4) {
                let aaaa0 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());

                let a = _mm256_castps256_ps128(aaaa0);
                let b =
                    _mm_unpackhi_ps64(_mm256_castps256_ps128(aaaa0), _mm256_castps256_ps128(aaaa0));

                let aa0 = _mm256_extractf128_ps::<1>(aaaa0);

                let c = aa0;
                let d = _mm_unpackhi_ps64(aa0, aa0);

                let t0 = _mm_add_ps(a, c);
                let t1 = _mm_sub_ps(a, c);
                let t2 = _mm_add_ps(b, d);
                let mut t3 = _mm_sub_ps(b, d);
                const SH: i32 = shuffle(2, 3, 0, 1);
                t3 = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(t3, t3),
                    _mm256_castps256_ps128(v_i_multiplier),
                );

                let yy0 = _mm_unpacklo_ps64(_mm_add_ps(t0, t2), _mm_add_ps(t1, t3));
                let yy1 = _mm_unpacklo_ps64(_mm_sub_ps(t0, t2), _mm_sub_ps(t1, t3));
                let yyyy = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(yy0), yy1);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), yyyy);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            let v_i_multiplier = _mm256_loadu_ps(self.multiplier.as_ptr());

            for (dst, src) in dst.chunks_exact_mut(16).zip(src.chunks_exact(16)) {
                let aaaa0 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());
                let bbbb0 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let cccc0 = _mm256_loadu_ps(src.get_unchecked(8..).as_ptr().cast());
                let dddd0 = _mm256_loadu_ps(src.get_unchecked(12..).as_ptr().cast());

                let (a, b, c, d) = _mm256s_deinterleave4_epi64(aaaa0, bbbb0, cccc0, dddd0);

                let t0 = _mm256_add_ps(a, c);
                let t1 = _mm256_sub_ps(a, c);
                let t2 = _mm256_add_ps(b, d);
                let mut t3 = _mm256_sub_ps(b, d);
                const SH: i32 = shuffle(2, 3, 0, 1);
                t3 = _mm256_xor_ps(_mm256_shuffle_ps::<SH>(t3, t3), v_i_multiplier);

                let xy0 = _mm256_add_ps(t0, t2);
                let xy1 = _mm256_add_ps(t1, t3);
                let xy2 = _mm256_sub_ps(t0, t2);
                let xy3 = _mm256_sub_ps(t1, t3);

                let (y0, y1, y2, y3) = _mm256s_interleave4_epi64(xy0, xy1, xy2, xy3);

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), y0);
                _mm256_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), y1);
                _mm256_storeu_ps(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), y2);
                _mm256_storeu_ps(dst.get_unchecked_mut(12..).as_mut_ptr().cast(), y3);
            }

            let rem_dst = dst.chunks_exact_mut(16).into_remainder();
            let rem_src = src.chunks_exact(16).remainder();

            for (dst, src) in rem_dst.chunks_exact_mut(4).zip(rem_src.chunks_exact(4)) {
                let aaaa0 = _mm256_loadu_ps(src.get_unchecked(0..).as_ptr().cast());

                let a = _mm256_castps256_ps128(aaaa0);
                let b =
                    _mm_unpackhi_ps64(_mm256_castps256_ps128(aaaa0), _mm256_castps256_ps128(aaaa0));

                let aa0 = _mm256_extractf128_ps::<1>(aaaa0);

                let c = aa0;
                let d = _mm_unpackhi_ps64(aa0, aa0);

                let t0 = _mm_add_ps(a, c);
                let t1 = _mm_sub_ps(a, c);
                let t2 = _mm_add_ps(b, d);
                let mut t3 = _mm_sub_ps(b, d);
                const SH: i32 = shuffle(2, 3, 0, 1);
                t3 = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(t3, t3),
                    _mm256_castps256_ps128(v_i_multiplier),
                );

                let yy0 = _mm_unpacklo_ps64(_mm_add_ps(t0, t2), _mm_add_ps(t1, t3));
                let yy1 = _mm_unpacklo_ps64(_mm_sub_ps(t0, t2), _mm_sub_ps(t1, t3));
                let yyyy = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(yy0), yy1);

                _mm256_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), yyyy);
            }
        }
        Ok(())
    }
}

impl AvxButterfly4<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let v_i_multiplier = unsafe { _mm_loadu_pd(self.multiplier.as_ptr()) };

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let aa0 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let bb0 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());

                let a = _mm256_castpd256_pd128(aa0);
                let b = _mm256_extractf128_pd::<1>(aa0);
                let c = _mm256_castpd256_pd128(bb0);
                let d = _mm256_extractf128_pd::<1>(bb0);

                let t0 = _mm_add_pd(a, c);
                let t1 = _mm_sub_pd(a, c);
                let t2 = _mm_add_pd(b, d);
                let mut t3 = _mm_sub_pd(b, d);
                t3 = _mm_xor_pd(_mm_shuffle_pd::<0b01>(t3, t3), v_i_multiplier);

                let yy0 = _mm256_insertf128_pd::<1>(
                    _mm256_castpd128_pd256(_mm_add_pd(t0, t2)),
                    _mm_add_pd(t1, t3),
                );
                let yy1 = _mm256_insertf128_pd::<1>(
                    _mm256_castpd128_pd256(_mm_sub_pd(t0, t2)),
                    _mm_sub_pd(t1, t3),
                );

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), yy0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), yy1);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let v_i_multiplier = unsafe { _mm_loadu_pd(self.multiplier.as_ptr()) };

        for (dst, src) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
            unsafe {
                let aa0 = _mm256_loadu_pd(src.get_unchecked(0..).as_ptr().cast());
                let bb0 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());

                let a = _mm256_castpd256_pd128(aa0);
                let b = _mm256_extractf128_pd::<1>(aa0);
                let c = _mm256_castpd256_pd128(bb0);
                let d = _mm256_extractf128_pd::<1>(bb0);

                let t0 = _mm_add_pd(a, c);
                let t1 = _mm_sub_pd(a, c);
                let t2 = _mm_add_pd(b, d);
                let mut t3 = _mm_sub_pd(b, d);
                t3 = _mm_xor_pd(_mm_shuffle_pd::<0b01>(t3, t3), v_i_multiplier);

                let yy0 = _mm256_insertf128_pd::<1>(
                    _mm256_castpd128_pd256(_mm_add_pd(t0, t2)),
                    _mm_add_pd(t1, t3),
                );
                let yy1 = _mm256_insertf128_pd::<1>(
                    _mm256_castpd128_pd256(_mm_sub_pd(t0, t2)),
                    _mm_sub_pd(t1, t3),
                );

                _mm256_storeu_pd(dst.get_unchecked_mut(0..).as_mut_ptr().cast(), yy0);
                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), yy1);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly4<f32> {
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
        4
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

impl FftExecutor<f64> for AvxButterfly4<f64> {
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
        4
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

    test_avx_butterfly!(test_avx_butterfly4, f32, AvxButterfly4, 4, 1e-5);
    test_avx_butterfly!(test_avx_butterfly4_f64, f64, AvxButterfly4, 4, 1e-7);

    test_oof_avx_butterfly!(test_oof_avx_butterfly4, f32, AvxButterfly4, 4, 1e-5);
    test_oof_avx_butterfly!(test_oof_avx_butterfly4_f64, f64, AvxButterfly4, 4, 1e-7);
}
