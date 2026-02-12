/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
use crate::FftDirection;
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
use crate::avx::util::shuffle;
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub(crate) fn gen_butterfly_twiddles_f32<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [AvxStoreF; N] {
    let mut twiddles = [AvxStoreF::zero(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 4;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = AvxStoreF::set_complex4(
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR), size, direction),
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), size, direction),
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 2), size, direction),
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 3), size, direction),
            );
            q += 1;
        }
    }
    twiddles
}

#[target_feature(enable = "avx2")]
pub(crate) fn gen_butterfly_twiddles_f64<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [AvxStoreD; N] {
    let mut twiddles = [AvxStoreD::zero(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 2;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = AvxStoreD::set_complex2(
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR), size, direction),
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), size, direction),
            );
            q += 1;
        }
    }
    twiddles
}

#[target_feature(enable = "avx2")]
pub(crate) fn gen_butterfly_separate_cols_twiddles_f64<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [AvxStoreD; N] {
    let mut twiddles = [AvxStoreD::zero(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 1;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = AvxStoreD::set_complex(&compute_twiddle(
                y * (x * COMPLEX_PER_VECTOR),
                size,
                direction,
            ));
            q += 1;
        }
    }
    twiddles
}

#[target_feature(enable = "avx2")]
pub(crate) fn gen_butterfly_separate_cols_twiddles_f32<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [AvxStoreF; N] {
    let mut twiddles = [AvxStoreF::zero(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 1;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = AvxStoreF::set_complex(compute_twiddle(
                y * (x * COMPLEX_PER_VECTOR),
                size,
                direction,
            ));
            q += 1;
        }
    }
    twiddles
}

pub(crate) struct AvxButterfly {}

impl AvxButterfly {
    #[inline(always)]
    pub(crate) fn butterfly3_f32(
        u0: __m256,
        u1: __m256,
        u2: __m256,
        tw_re: __m256,
        tw_w_2: __m256,
    ) -> (__m256, __m256, __m256) {
        unsafe {
            let xp = _mm256_add_ps(u1, u2);
            let xn = _mm256_sub_ps(u1, u2);
            let sum = _mm256_add_ps(u0, xp);

            const SH: i32 = shuffle(2, 3, 0, 1);
            let w_1 = _mm256_fmadd_ps(tw_re, xp, u0);
            let xn_rot = _mm256_shuffle_ps::<SH>(xn, xn);

            let y0 = sum;
            let y1 = _mm256_fmadd_ps(xn_rot, tw_w_2, w_1);
            let y2 = _mm256_fnmadd_ps(xn_rot, tw_w_2, w_1);
            (y0, y1, y2)
        }
    }

    #[inline(always)]
    pub(crate) fn butterfly2_f32(u0: __m256, u1: __m256) -> (__m256, __m256) {
        unsafe {
            let t = _mm256_add_ps(u0, u1);
            let y1 = _mm256_sub_ps(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline(always)]
    pub(crate) fn butterfly3_f32_m128(
        u0: __m128,
        u1: __m128,
        u2: __m128,
        tw_re: __m128,
        tw_w_2: __m128,
    ) -> (__m128, __m128, __m128) {
        unsafe {
            let xp = _mm_add_ps(u1, u2);
            let xn = _mm_sub_ps(u1, u2);
            let sum = _mm_add_ps(u0, xp);

            const SH: i32 = shuffle(2, 3, 0, 1);
            let w_1 = _mm_fmadd_ps(tw_re, xp, u0);
            let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

            let y0 = sum;
            let y1 = _mm_fmadd_ps(tw_w_2, xn_rot, w_1);
            let y2 = _mm_fnmadd_ps(tw_w_2, xn_rot, w_1);
            (y0, y1, y2)
        }
    }

    #[inline(always)]
    pub(crate) fn butterfly2_f32_m128(u0: __m128, u1: __m128) -> (__m128, __m128) {
        unsafe {
            let t = _mm_add_ps(u0, u1);
            let y1 = _mm_sub_ps(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline(always)]
    pub(crate) fn butterfly3_f64(
        u0: __m256d,
        u1: __m256d,
        u2: __m256d,
        tw_re: __m256d,
        tw_w_2: __m256d,
    ) -> (__m256d, __m256d, __m256d) {
        unsafe {
            let xp = _mm256_add_pd(u1, u2);
            let xn = _mm256_sub_pd(u1, u2);
            let sum = _mm256_add_pd(u0, xp);

            let w_1 = _mm256_fmadd_pd(tw_re, xp, u0);
            let xn_rot = _mm256_shuffle_pd::<0b0101>(xn, xn);

            let y0 = sum;
            let y1 = _mm256_fmadd_pd(tw_w_2, xn_rot, w_1);
            let y2 = _mm256_fnmadd_pd(tw_w_2, xn_rot, w_1);
            (y0, y1, y2)
        }
    }

    #[inline(always)]
    pub(crate) fn butterfly3_f64_m128(
        u0: __m128d,
        u1: __m128d,
        u2: __m128d,
        tw_re: __m128d,
        tw_w_2: __m128d,
    ) -> (__m128d, __m128d, __m128d) {
        unsafe {
            let xp = _mm_add_pd(u1, u2);
            let xn = _mm_sub_pd(u1, u2);
            let sum = _mm_add_pd(u0, xp);

            let w_1 = _mm_fmadd_pd(tw_re, xp, u0);
            let xn_rot = _mm_shuffle_pd::<0b01>(xn, xn);

            let y0 = sum;
            let y1 = _mm_fmadd_pd(tw_w_2, xn_rot, w_1);
            let y2 = _mm_fnmadd_pd(tw_w_2, xn_rot, w_1);
            (y0, y1, y2)
        }
    }

    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn butterfly2_f64_m128(u0: __m128d, u1: __m128d) -> (__m128d, __m128d) {
        let t = _mm_add_pd(u0, u1);
        let y1 = _mm_sub_pd(u0, u1);
        let y0 = t;
        (y0, y1)
    }

    #[inline(always)]
    pub(crate) fn butterfly2_f64(u0: __m256d, u1: __m256d) -> (__m256d, __m256d) {
        unsafe {
            let t = _mm256_add_pd(u0, u1);
            let y1 = _mm256_sub_pd(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }
}

macro_rules! boring_avx_butterfly {
    ($bf_name: ident, $f_type: ident, $size: expr) => {
        impl $bf_name {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_impl(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.chunks_exact_mut($size) {
                    use crate::store::InPlaceStore;
                    self.run(&mut InPlaceStore::new(chunk));
                }

                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_oof_impl(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, $size);

                for (dst, src) in dst.chunks_exact_mut($size).zip(src.chunks_exact($size)) {
                    use crate::store::BiStore;
                    self.run(&mut BiStore::new(src, dst));
                }
                Ok(())
            }
        }

        impl FftExecutor<$f_type> for $bf_name {
            fn execute(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                FftExecutor::execute_with_scratch(self, in_place, &mut [])
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                $size
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
    };
}

pub(crate) use boring_avx_butterfly;

macro_rules! boring_avx_butterfly2 {
    ($bf_name: ident, $f_type: ident, $size: expr) => {
        impl $bf_name {
            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_impl(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.chunks_exact_mut($size * 2) {
                    use crate::store::InPlaceStore;
                    self.run2(&mut InPlaceStore::new(chunk));
                }

                let rem = in_place.chunks_exact_mut($size * 2).into_remainder();

                for chunk in rem.chunks_exact_mut($size) {
                    use crate::store::InPlaceStore;
                    self.run(&mut InPlaceStore::new(chunk));
                }

                Ok(())
            }

            #[target_feature(enable = "avx2", enable = "fma")]
            fn execute_oof_impl(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, $size);

                for (dst, src) in dst
                    .chunks_exact_mut($size * 2)
                    .zip(src.chunks_exact($size * 2))
                {
                    use crate::store::BiStore;
                    self.run2(&mut BiStore::new(src, dst));
                }

                let rem_dst = dst.chunks_exact_mut($size * 2).into_remainder();
                let rem_src = src.chunks_exact($size * 2).remainder();

                for (dst, src) in rem_dst
                    .chunks_exact_mut($size)
                    .zip(rem_src.chunks_exact($size))
                {
                    use crate::store::BiStore;
                    self.run(&mut BiStore::new(src, dst));
                }
                Ok(())
            }
        }

        impl FftExecutor<$f_type> for $bf_name {
            fn execute(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                FftExecutor::execute_with_scratch(self, in_place, &mut [])
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            fn length(&self) -> usize {
                $size
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
    };
}

pub(crate) use boring_avx_butterfly2;
