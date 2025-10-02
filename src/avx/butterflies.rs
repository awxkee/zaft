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
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd,
    _mm256_create_ps, shuffle,
};
use crate::radix6::Radix6Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;
use std::marker::PhantomData;

pub(crate) struct AvxButterfly {}

impl AvxButterfly {
    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn butterfly3_f32(
        u0: __m256,
        u1: __m256,
        u2: __m256,
        tw_re: __m256,
        tw_w_2: __m256,
    ) -> (__m256, __m256, __m256) {
        let xp = _mm256_add_ps(u1, u2);
        let xn = _mm256_sub_ps(u1, u2);
        let sum = _mm256_add_ps(u0, xp);

        const SH: i32 = shuffle(2, 3, 0, 1);
        let w_1 = _mm256_fmadd_ps(tw_re, xp, u0);
        let vw_2 = _mm256_mul_ps(tw_w_2, _mm256_shuffle_ps::<SH>(xn, xn));

        let y0 = sum;
        let y1 = _mm256_add_ps(w_1, vw_2);
        let y2 = _mm256_sub_ps(w_1, vw_2);
        (y0, y1, y2)
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn butterfly2_f32(u0: __m256, u1: __m256) -> (__m256, __m256) {
        let t = _mm256_add_ps(u0, u1);
        let y1 = _mm256_sub_ps(u0, u1);
        let y0 = t;
        (y0, y1)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn butterfly3_f32_m128(
        u0: __m128,
        u1: __m128,
        u2: __m128,
        tw_re: __m128,
        tw_w_2: __m128,
    ) -> (__m128, __m128, __m128) {
        let xp = _mm_add_ps(u1, u2);
        let xn = _mm_sub_ps(u1, u2);
        let sum = _mm_add_ps(u0, xp);

        const SH: i32 = shuffle(2, 3, 0, 1);
        let w_1 = _mm_fmadd_ps(tw_re, xp, u0);
        let vw_2 = _mm_mul_ps(tw_w_2, _mm_shuffle_ps::<SH>(xn, xn));

        let y0 = sum;
        let y1 = _mm_add_ps(w_1, vw_2);
        let y2 = _mm_sub_ps(w_1, vw_2);
        (y0, y1, y2)
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn butterfly2_f32_m128(u0: __m128, u1: __m128) -> (__m128, __m128) {
        let t = _mm_add_ps(u0, u1);
        let y1 = _mm_sub_ps(u0, u1);
        let y0 = t;
        (y0, y1)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn butterfly3_f64(
        u0: __m256d,
        u1: __m256d,
        u2: __m256d,
        tw_re: __m256d,
        tw_w_2: __m256d,
    ) -> (__m256d, __m256d, __m256d) {
        let xp = _mm256_add_pd(u1, u2);
        let xn = _mm256_sub_pd(u1, u2);
        let sum = _mm256_add_pd(u0, xp);

        let w_1 = _mm256_fmadd_pd(tw_re, xp, u0);
        let vw_2 = _mm256_mul_pd(tw_w_2, _mm256_permute_pd::<0b0101>(xn));

        let y0 = sum;
        let y1 = _mm256_add_pd(w_1, vw_2);
        let y2 = _mm256_sub_pd(w_1, vw_2);
        (y0, y1, y2)
    }

    #[target_feature(enable = "avx", enable = "fma")]
    #[inline]
    pub(crate) unsafe fn butterfly3_f64_m128(
        u0: __m128d,
        u1: __m128d,
        u2: __m128d,
        tw_re: __m128d,
        tw_w_2: __m128d,
    ) -> (__m128d, __m128d, __m128d) {
        let xp = _mm_add_pd(u1, u2);
        let xn = _mm_sub_pd(u1, u2);
        let sum = _mm_add_pd(u0, xp);

        let w_1 = _mm_fmadd_pd(tw_re, xp, u0);
        let vw_2 = _mm_mul_pd(tw_w_2, _mm_shuffle_pd::<0b01>(xn, xn));

        let y0 = sum;
        let y1 = _mm_add_pd(w_1, vw_2);
        let y2 = _mm_sub_pd(w_1, vw_2);
        (y0, y1, y2)
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn butterfly2_f64_m128(u0: __m128d, u1: __m128d) -> (__m128d, __m128d) {
        let t = _mm_add_pd(u0, u1);
        let y1 = _mm_sub_pd(u0, u1);
        let y0 = t;
        (y0, y1)
    }

    #[target_feature(enable = "avx")]
    #[inline]
    pub(crate) unsafe fn butterfly2_f64(u0: __m256d, u1: __m256d) -> (__m256d, __m256d) {
        let t = _mm256_add_pd(u0, u1);
        let y1 = _mm256_sub_pd(u0, u1);
        let y0 = t;
        (y0, y1)
    }
}

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
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 2 != 0 {
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
}

impl AvxButterfly2<f32> {
    #[target_feature(enable = "avx2")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 2 != 0 {
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
}

impl FftExecutor<f64> for AvxButterfly2<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        2
    }
}

impl FftExecutor<f32> for AvxButterfly2<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        2
    }
}

pub(crate) struct AvxButterfly3<T> {
    phantom_data: PhantomData<T>,
    direction: FftDirection,
    twiddle: Complex<T>,
    tw1: [T; 4],
    tw2: [T; 4],
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
            tw1: [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im],
            tw2: [twiddle.im, -twiddle.im, twiddle.im, -twiddle.im],
        }
    }
}

impl AvxButterfly3<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 3));
        }

        let twiddle_re = _mm_set1_ps(self.twiddle.re);
        let tw1 = unsafe { _mm_loadu_ps(self.tw1.as_ptr()) };
        let tw2 = unsafe { _mm_loadu_ps(self.tw2.as_ptr()) };

        for chunk in in_place.chunks_exact_mut(3) {
            unsafe {
                let uz = _mm_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());

                let u0 = uz;
                let u1 = _mm_unpackhi_ps64(uz, uz);
                let u2 =
                    _mm_castsi128_ps(_mm_loadu_epi64(chunk.get_unchecked(2..).as_ptr().cast()));

                let xp = _mm_add_ps(u1, u2);
                let xn = _mm_sub_ps(u1, u2);
                let sum = _mm_add_ps(u0, xp);

                let w_1 = _mm_fmadd_ps(twiddle_re, xp, u0);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let xn_rot = _mm_shuffle_ps::<SH>(xn, xn);

                let y0 = sum;
                let y1 = _mm_fmadd_ps(tw1, xn_rot, w_1);
                let y2 = _mm_fmadd_ps(tw2, xn_rot, w_1);

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
}

impl AvxButterfly3<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 3 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 3));
        }

        let twiddle_re = _mm_set1_pd(self.twiddle.re);
        let tw1 = unsafe { _mm_loadu_pd(self.tw1.as_ptr()) };
        let tw2 = unsafe { _mm_loadu_pd(self.tw2.as_ptr()) };

        for chunk in in_place.chunks_exact_mut(3) {
            unsafe {
                let uz0 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());

                let u0 = _mm256_castpd256_pd128(uz0);
                let u1 = _mm256_extractf128_pd::<1>(uz0);
                let u2 = _mm_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());

                let xp = _mm_add_pd(u1, u2);
                let xn = _mm_sub_pd(u1, u2);
                let sum = _mm_add_pd(u0, xp);

                let w_1 = _mm_fmadd_pd(twiddle_re, xp, u0);

                let xn_rot = _mm_shuffle_pd::<0b01>(xn, xn);

                let y0 = sum;
                let y1 = _mm_fmadd_pd(tw1, xn_rot, w_1);
                let y2 = _mm_fmadd_pd(tw2, xn_rot, w_1);

                let y01 = _mm256_insertf128_pd::<1>(_mm256_castpd128_pd256(y0), y1);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y01);
                _mm_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2);
            }
        }
        Ok(())
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

pub(crate) struct AvxButterfly4<T> {
    direction: FftDirection,
    multiplier: [T; 4],
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float>
    AvxButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            multiplier: match fft_direction {
                FftDirection::Inverse => [-0.0.as_(), 0.0.as_(), -0.0.as_(), 0.0.as_()],
                FftDirection::Forward => [0.0.as_(), -0.0.as_(), 0.0.as_(), -0.0.as_()],
            },
        }
    }
}

impl AvxButterfly4<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let v_i_multiplier = unsafe { _mm_loadu_ps(self.multiplier.as_ptr()) };

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
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
                t3 = _mm_xor_ps(_mm_shuffle_ps::<SH>(t3, t3), v_i_multiplier);

                let yy0 = _mm_unpacklo_ps64(_mm_add_ps(t0, t2), _mm_add_ps(t1, t3));
                let yy1 = _mm_unpacklo_ps64(_mm_sub_ps(t0, t2), _mm_sub_ps(t1, t3));
                let yyyy = _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(yy0), yy1);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), yyyy);
            }
        }
        Ok(())
    }
}

impl AvxButterfly4<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
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
}

impl FftExecutor<f32> for AvxButterfly4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        4
    }
}

impl FftExecutor<f64> for AvxButterfly4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        4
    }
}

pub(crate) struct AvxButterfly5<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
}

impl<T: Default + Clone + Radix6Twiddles + 'static + Copy + FftTrigonometry + Float>
    AvxButterfly5<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
        }
    }
}

impl AvxButterfly5<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 5));
        }

        let tw1_re = _mm256_set1_ps(self.twiddle1.re);
        let tw1_im = _mm256_set1_ps(self.twiddle1.im);
        let tw2_re = _mm256_set1_ps(self.twiddle2.re);
        let tw2_im = _mm256_set1_ps(self.twiddle2.im);
        let rot_sign =
            unsafe { _mm256_loadu_ps([-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0].as_ptr()) };

        unsafe {
            for chunk in in_place.chunks_exact_mut(10) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9 = _mm_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());

                let lo = _mm256_castps256_ps128(u0u1u2u3); // (u0,u1)
                let hi = _mm256_extractf128_ps::<1>(u0u1u2u3); // (u2, u3)

                // u8..u11
                let hi2 = _mm256_castps256_ps128(u4u5u6u7); // (u4,u5)
                let hi3 = _mm256_extractf128_ps::<1>(u4u5u6u7); // (u6, u7)

                let u0 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(lo, hi2); // (u0,u5)
                let u1 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(lo, hi3); // (u1,u6)
                let u2 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(hi, hi3); // (u2,u7)
                let u3 = _mm_shuffle_ps::<{ shuffle(1, 0, 3, 2) }>(hi, u8u9); // (u3,u8)
                let u4 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(hi2, u8u9); // (u4,u9)

                // Radix-5 butterfly

                let x14p = _mm_add_ps(u1, u4);
                let x14n = _mm_sub_ps(u1, u4);
                let x23p = _mm_add_ps(u2, u3);
                let x23n = _mm_sub_ps(u2, u3);
                let y0 = _mm_add_ps(_mm_add_ps(u0, x14p), x23p);

                let temp_b1_1 = _mm_mul_ps(_mm256_castps256_ps128(tw1_im), x14n);
                let temp_b2_1 = _mm_mul_ps(_mm256_castps256_ps128(tw2_im), x14n);

                let temp_a1 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw2_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw1_re), x14p, u0),
                );
                let temp_a2 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw1_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw2_re), x14p, u0),
                );

                let temp_b1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
                let temp_b2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let temp_b1_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b1, temp_b1),
                    _mm256_castps256_ps128(rot_sign),
                );
                let temp_b2_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b2, temp_b2),
                    _mm256_castps256_ps128(rot_sign),
                );

                let y1 = _mm_add_ps(temp_a1, temp_b1_rot);
                let y2 = _mm_add_ps(temp_a2, temp_b2_rot);
                let y3 = _mm_sub_ps(temp_a2, temp_b2_rot);
                let y4 = _mm_sub_ps(temp_a1, temp_b1_rot);

                let zu0 = _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(y0, y1); // (u0,u5)
                let zu1 = _mm_shuffle_ps::<{ shuffle(1, 0, 1, 0) }>(y2, y3); // (u1,u6)
                let zu2 = _mm_shuffle_ps::<{ shuffle(3, 2, 1, 0) }>(y4, y0); // (u2,u7)
                let zu3 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(y1, y2); // (u3,u8)
                let zu4 = _mm_unpackhi_ps64(y3, y4); // (u4,u9)

                let zu0u1 = _mm256_create_ps(zu0, zu1);
                let zu2u3 = _mm256_create_ps(zu2, zu3);

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), zu0u1);
                _mm256_storeu_ps(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), zu2u3);
                _mm_storeu_ps(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), zu4);
            }
        }

        let remainder = in_place.chunks_exact_mut(10).into_remainder();

        for chunk in remainder.chunks_exact_mut(5) {
            unsafe {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());

                let u0u1 = _mm256_castps256_ps128(u0u1u2u3);
                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);
                let u0 = u0u1;
                let u1 = _mm_unpackhi_ps64(u0u1, u0u1);
                let u2 = u2u3;
                let u3 = _mm_unpackhi_ps64(u2u3, u2u3);
                let u4 = _m128s_load_f32x2(chunk.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14p = _mm_add_ps(u1, u4);
                let x14n = _mm_sub_ps(u1, u4);
                let x23p = _mm_add_ps(u2, u3);
                let x23n = _mm_sub_ps(u2, u3);
                let y0 = _mm_add_ps(_mm_add_ps(u0, x14p), x23p);

                let temp_b1_1 = _mm_mul_ps(_mm256_castps256_ps128(tw1_im), x14n);
                let temp_b2_1 = _mm_mul_ps(_mm256_castps256_ps128(tw2_im), x14n);

                let temp_a1 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw2_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw1_re), x14p, u0),
                );
                let temp_a2 = _mm_fmadd_ps(
                    _mm256_castps256_ps128(tw1_re),
                    x23p,
                    _mm_fmadd_ps(_mm256_castps256_ps128(tw2_re), x14p, u0),
                );

                let temp_b1 = _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
                let temp_b2 = _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

                const SH: i32 = shuffle(2, 3, 0, 1);
                let temp_b1_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b1, temp_b1),
                    _mm256_castps256_ps128(rot_sign),
                );
                let temp_b2_rot = _mm_xor_ps(
                    _mm_shuffle_ps::<SH>(temp_b2, temp_b2),
                    _mm256_castps256_ps128(rot_sign),
                );

                let y1 = _mm_add_ps(temp_a1, temp_b1_rot);
                let y2 = _mm_add_ps(temp_a2, temp_b2_rot);
                let y3 = _mm_sub_ps(temp_a2, temp_b2_rot);
                let y4 = _mm_sub_ps(temp_a1, temp_b1_rot);

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y0, y1), _mm_unpacklo_ps64(y2, y3));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _m128s_store_f32x2(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly5<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

impl AvxButterfly5<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 5 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), 5));
        }

        let tw1_re = _mm256_set1_pd(self.twiddle1.re);
        let tw1_im = _mm256_set1_pd(self.twiddle1.im);
        let tw2_re = _mm256_set1_pd(self.twiddle2.re);
        let tw2_im = _mm256_set1_pd(self.twiddle2.im);
        let rot_sign = unsafe { _mm256_loadu_pd([-0.0f64, 0.0, -0.0, 0.0].as_ptr()) };

        for chunk in in_place.chunks_exact_mut(5) {
            unsafe {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u0 = _mm256_castpd256_pd128(u0u1);
                let u1 = _mm256_extractf128_pd::<1>(u0u1);
                let u2 = _mm256_castpd256_pd128(u2u3);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());

                // Radix-5 butterfly

                let x14p = _mm_add_pd(u1, u4);
                let x14n = _mm_sub_pd(u1, u4);
                let x23p = _mm_add_pd(u2, u3);
                let x23n = _mm_sub_pd(u2, u3);
                let y0 = _mm_add_pd(_mm_add_pd(u0, x14p), x23p);

                let temp_b1_1 = _mm_mul_pd(_mm256_castpd256_pd128(tw1_im), x14n);
                let temp_b2_1 = _mm_mul_pd(_mm256_castpd256_pd128(tw2_im), x14n);

                let temp_a1 = _mm_fmadd_pd(
                    _mm256_castpd256_pd128(tw2_re),
                    x23p,
                    _mm_fmadd_pd(_mm256_castpd256_pd128(tw1_re), x14p, u0),
                );
                let temp_a2 = _mm_fmadd_pd(
                    _mm256_castpd256_pd128(tw1_re),
                    x23p,
                    _mm_fmadd_pd(_mm256_castpd256_pd128(tw2_re), x14p, u0),
                );

                let temp_b1 = _mm_fmadd_pd(_mm256_castpd256_pd128(tw2_im), x23n, temp_b1_1);
                let temp_b2 = _mm_fnmadd_pd(_mm256_castpd256_pd128(tw1_im), x23n, temp_b2_1);

                let temp_b1_rot = _mm_xor_pd(
                    _mm_shuffle_pd::<0b01>(temp_b1, temp_b1),
                    _mm256_castpd256_pd128(rot_sign),
                );
                let temp_b2_rot = _mm_xor_pd(
                    _mm_shuffle_pd::<0b01>(temp_b2, temp_b2),
                    _mm256_castpd256_pd128(rot_sign),
                );

                let y1 = _mm_add_pd(temp_a1, temp_b1_rot);
                let y2 = _mm_add_pd(temp_a2, temp_b2_rot);
                let y3 = _mm_sub_pd(temp_a2, temp_b2_rot);
                let y4 = _mm_sub_pd(temp_a1, temp_b1_rot);

                let y0y1 = _mm256_create_pd(y0, y1);
                let y2y3 = _mm256_create_pd(y2, y3);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly5<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        5
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly2_f32() {
        for i in 1..6 {
            let size = 2usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly2::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly2::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 2f32)).collect();

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
    fn test_butterfly2_f64() {
        for i in 1..7 {
            let size = 2usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly2::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly2::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 2f64)).collect();

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
    fn test_butterfly3_f32() {
        for i in 1..6 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly3::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly3::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 3f32)).collect();

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
    fn test_butterfly3_f64() {
        for i in 1..6 {
            let size = 3usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly3::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly3::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 3f64)).collect();

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
    fn test_butterfly4_f32() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly4::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly4::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 4f32)).collect();

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
    fn test_butterfly4_f64() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly4::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly4::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 4f64)).collect();

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
    fn test_butterfly5_f32() {
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly5::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f32)).collect();

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
    fn test_butterfly5_f64() {
        for i in 1..6 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly5::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly5::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 5f64)).collect();

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
}
