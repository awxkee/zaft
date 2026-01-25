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
use crate::FftDirection;
use crate::avx::util::shuffle;
use num_traits::AsPrimitive;
use std::arch::x86_64::*;
use std::marker::PhantomData;
use std::ops::Neg;

pub(crate) struct AvxRotate<T> {
    pub(crate) rot_flag: __m256d,
    phantom_data: PhantomData<T>,
}

impl<T: Copy + 'static + Neg<Output = T>> AvxRotate<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2")]
    #[inline]
    pub(crate) fn new(fft_direction: FftDirection) -> AvxRotate<T> {
        let rot90: [T; 8] = [
            -0.0f64.as_(),
            0.0.as_(),
            -0.0.as_(),
            0.0.as_(),
            -0.0f64.as_(),
            0.0.as_(),
            -0.0.as_(),
            0.0.as_(),
        ];
        let rot270: [T; 8] = [
            0.0f64.as_(),
            -0.0.as_(),
            0.0f64.as_(),
            -0.0.as_(),
            0.0f64.as_(),
            -0.0.as_(),
            0.0f64.as_(),
            -0.0.as_(),
        ];
        unsafe {
            match fft_direction {
                FftDirection::Inverse => Self {
                    rot_flag: _mm256_loadu_pd(rot90.as_ptr().cast()),
                    phantom_data: PhantomData,
                },
                FftDirection::Forward => Self {
                    rot_flag: _mm256_loadu_pd(rot270.as_ptr().cast()),
                    phantom_data: PhantomData,
                },
            }
        }
    }
}

impl AvxRotate<f64> {
    #[target_feature(enable = "avx2")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn rotate_m128d(&self, v: __m128d) -> __m128d {
        _mm_xor_pd(
            _mm_shuffle_pd::<0b01>(v, v),
            _mm256_castpd256_pd128(self.rot_flag),
        )
    }

    #[target_feature(enable = "avx2")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn rotate_m256d(&self, v: __m256d) -> __m256d {
        _mm256_xor_pd(_mm256_shuffle_pd::<0b0101>(v, v), self.rot_flag)
    }
}

impl AvxRotate<f32> {
    #[target_feature(enable = "avx2")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn rotate_m128(&self, v: __m128) -> __m128 {
        const SH: i32 = shuffle(2, 3, 0, 1);
        _mm_xor_ps(
            _mm_shuffle_ps::<SH>(v, v),
            _mm_castpd_ps(_mm256_castpd256_pd128(self.rot_flag)),
        )
    }

    #[target_feature(enable = "avx2")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn rotate_m256(&self, v: __m256) -> __m256 {
        const SH: i32 = shuffle(2, 3, 0, 1);
        _mm256_xor_ps(
            _mm256_shuffle_ps::<SH>(v, v),
            _mm256_castpd_ps(self.rot_flag),
        )
    }
}
