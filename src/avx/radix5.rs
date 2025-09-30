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
    _m128d_fma_mul_complex, _m128s_fma_mul_complex, _m128s_load_f32x2, _m128s_store_f32x2,
    _m256d_mul_complex, _m256s_mul_complex, _mm_unpackhi_ps64, _mm_unpacklo_ps64,
    _mm256_unpackhi_pd2, _mm256_unpacklo_pd2, _mm256s_interleave_epi64, shuffle,
};
use crate::radix5::Radix5Twiddles;
use crate::traits::FftTrigonometry;
use crate::util::{compute_twiddle, digit_reverse_indices, is_power_of_five, permute_inplace};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxFmaRadix5<T> {
    twiddles: Vec<Complex<T>>,
    permutations: Vec<usize>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
}

impl<T: Default + Clone + Radix5Twiddles + 'static + Copy + FftTrigonometry + Float> AvxFmaRadix5<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix5<T>, ZaftError> {
        assert!(is_power_of_five(size), "Input length must be a power of 5");

        let twiddles = T::make_twiddles(size, fft_direction)?;
        let rev = digit_reverse_indices(size, 5)?;

        Ok(AvxFmaRadix5 {
            permutations: rev,
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 5, fft_direction),
            twiddle2: compute_twiddle(2, 5, fft_direction),
            direction: fft_direction,
        })
    }
}

impl AvxFmaRadix5<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Digit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 5;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let tw1_re = _mm256_set1_pd(self.twiddle1.re);
            let tw1_im = _mm256_set1_pd(self.twiddle1.im);
            let tw2_re = _mm256_set1_pd(self.twiddle2.re);
            let tw2_im = _mm256_set1_pd(self.twiddle2.im);
            let rot_sign =
                _mm256_loadu_pd([-0.0f64, 0.0, -0.0f64, 0.0, -0.0f64, 0.0, -0.0f64, 0.0].as_ptr());

            while len <= self.execution_length {
                let fifth = len / 5;

                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;
                    while j + 2 < fifth {
                        let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                        let tw1 = _mm256_loadu_pd(
                            m_twiddles.get_unchecked(4 * (j + 1)..).as_ptr().cast(),
                        );
                        let tw2 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());
                        let tw3 = _mm256_loadu_pd(
                            m_twiddles.get_unchecked(4 * (j + 1) + 2..).as_ptr().cast(),
                        );

                        let u1 = _m256d_mul_complex(
                            _mm256_loadu_pd(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            _mm256_unpacklo_pd2(tw0, tw1),
                        );
                        let u2 = _m256d_mul_complex(
                            _mm256_loadu_pd(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            _mm256_unpackhi_pd2(tw0, tw1),
                        );
                        let u3 = _m256d_mul_complex(
                            _mm256_loadu_pd(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            _mm256_unpacklo_pd2(tw2, tw3),
                        );
                        let u4 = _m256d_mul_complex(
                            _mm256_loadu_pd(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            _mm256_unpackhi_pd2(tw2, tw3),
                        );

                        // Radix-5 butterfly

                        let x14p = _mm256_add_pd(u1, u4);
                        let x14n = _mm256_sub_pd(u1, u4);
                        let x23p = _mm256_add_pd(u2, u3);
                        let x23n = _mm256_sub_pd(u2, u3);
                        let y0 = _mm256_add_pd(_mm256_add_pd(u0, x14p), x23p);

                        let temp_b1_1 = _mm256_mul_pd(tw1_im, x14n);
                        let temp_b2_1 = _mm256_mul_pd(tw2_im, x14n);

                        let temp_a1 =
                            _mm256_fmadd_pd(tw2_re, x23p, _mm256_fmadd_pd(tw1_re, x14p, u0));
                        let temp_a2 =
                            _mm256_fmadd_pd(tw1_re, x23p, _mm256_fmadd_pd(tw2_re, x14p, u0));

                        let temp_b1 = _mm256_fmadd_pd(tw2_im, x23n, temp_b1_1);
                        let temp_b2 = _mm256_fnmadd_pd(tw1_im, x23n, temp_b2_1);

                        let temp_b1_rot =
                            _mm256_xor_pd(_mm256_permute_pd::<0b0101>(temp_b1), rot_sign);
                        let temp_b2_rot =
                            _mm256_xor_pd(_mm256_permute_pd::<0b0101>(temp_b2), rot_sign);

                        let y1 = _mm256_add_pd(temp_a1, temp_b1_rot);
                        let y2 = _mm256_add_pd(temp_a2, temp_b2_rot);
                        let y3 = _mm256_sub_pd(temp_a2, temp_b2_rot);
                        let y4 = _mm256_sub_pd(temp_a1, temp_b1_rot);

                        _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                            y1,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        _mm256_storeu_pd(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );

                        j += 2;
                    }

                    for j in j..fifth {
                        let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                        let tw1 =
                            _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());

                        let u1 = _m128d_fma_mul_complex(
                            _mm_loadu_pd(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            _mm256_castpd256_pd128(tw0),
                        );
                        let u2 = _m128d_fma_mul_complex(
                            _mm_loadu_pd(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            _mm256_extractf128_pd::<1>(tw0),
                        );
                        let u3 = _m128d_fma_mul_complex(
                            _mm_loadu_pd(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            _mm256_castpd256_pd128(tw1),
                        );
                        let u4 = _m128d_fma_mul_complex(
                            _mm_loadu_pd(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            _mm256_extractf128_pd::<1>(tw1),
                        );

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
                        let temp_b2 =
                            _mm_fnmadd_pd(_mm256_castpd256_pd128(tw1_im), x23n, temp_b2_1);

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

                        _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm_storeu_pd(data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(), y1);
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        _mm_storeu_pd(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );
                    }
                }

                m_twiddles = &m_twiddles[fifth * 4..];
                len *= 5;
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix5<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}

impl AvxFmaRadix5<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if self.execution_length != in_place.len() {
            return Err(ZaftError::InvalidInPlaceLength(
                self.execution_length,
                in_place.len(),
            ));
        }

        // Digit-reversal permutation
        permute_inplace(in_place, &self.permutations);

        let mut len = 5;

        unsafe {
            let mut m_twiddles = self.twiddles.as_slice();

            let tw1_re = _mm256_set1_ps(self.twiddle1.re);
            let tw1_im = _mm256_set1_ps(self.twiddle1.im);
            let tw2_re = _mm256_set1_ps(self.twiddle2.re);
            let tw2_im = _mm256_set1_ps(self.twiddle2.im);
            let rot_sign =
                _mm256_loadu_ps([-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0].as_ptr());

            while len <= self.execution_length {
                let fifth = len / 5;

                for data in in_place.chunks_exact_mut(len) {
                    let mut j = 0usize;

                    while j + 4 < fifth {
                        let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                        let xw0 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                        let xw1 = _mm256_loadu_ps(
                            m_twiddles.get_unchecked(4 * (j + 1)..).as_ptr().cast(),
                        );
                        let xw2 =
                            _mm256_loadu_ps(m_twiddles.get_unchecked(4 * j + 4..).as_ptr().cast());
                        let xw3 = _mm256_loadu_ps(
                            m_twiddles.get_unchecked(4 * (j + 1) + 4..).as_ptr().cast(),
                        );

                        let (tw0, tw1, tw2, tw3) = _mm256s_interleave_epi64(xw0, xw1, xw2, xw3);

                        let u1 = _m256s_mul_complex(
                            _mm256_loadu_ps(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            tw0,
                        );
                        let u2 = _m256s_mul_complex(
                            _mm256_loadu_ps(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            tw1,
                        );
                        let u3 = _m256s_mul_complex(
                            _mm256_loadu_ps(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            tw2,
                        );
                        let u4 = _m256s_mul_complex(
                            _mm256_loadu_ps(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            tw3,
                        );

                        // Radix-5 butterfly

                        let x14p = _mm256_add_ps(u1, u4);
                        let x14n = _mm256_sub_ps(u1, u4);
                        let x23p = _mm256_add_ps(u2, u3);
                        let x23n = _mm256_sub_ps(u2, u3);
                        let y0 = _mm256_add_ps(_mm256_add_ps(u0, x14p), x23p);

                        let temp_b1_1 = _mm256_mul_ps(tw1_im, x14n);
                        let temp_b2_1 = _mm256_mul_ps(tw2_im, x14n);

                        let temp_a1 =
                            _mm256_fmadd_ps(tw2_re, x23p, _mm256_fmadd_ps(tw1_re, x14p, u0));
                        let temp_a2 =
                            _mm256_fmadd_ps(tw1_re, x23p, _mm256_fmadd_ps(tw2_re, x14p, u0));

                        let temp_b1 = _mm256_fmadd_ps(tw2_im, x23n, temp_b1_1);
                        let temp_b2 = _mm256_fnmadd_ps(tw1_im, x23n, temp_b2_1);

                        const SH: i32 = shuffle(2, 3, 0, 1);
                        let temp_b1_rot =
                            _mm256_xor_ps(_mm256_shuffle_ps::<SH>(temp_b1, temp_b1), rot_sign);
                        let temp_b2_rot =
                            _mm256_xor_ps(_mm256_shuffle_ps::<SH>(temp_b2, temp_b2), rot_sign);

                        let y1 = _mm256_add_ps(temp_a1, temp_b1_rot);
                        let y2 = _mm256_add_ps(temp_a2, temp_b2_rot);
                        let y3 = _mm256_sub_ps(temp_a2, temp_b2_rot);
                        let y4 = _mm256_sub_ps(temp_a1, temp_b1_rot);

                        _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                            y1,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        _mm256_storeu_ps(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );
                        j += 4;
                    }

                    while j + 2 < fifth {
                        let u0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = _mm_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                        let tw1 =
                            _mm_loadu_ps(m_twiddles.get_unchecked(4 * (j + 1)..).as_ptr().cast());
                        let tw2 =
                            _mm_loadu_ps(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());
                        let tw3 = _mm_loadu_ps(
                            m_twiddles.get_unchecked(4 * (j + 1) + 2..).as_ptr().cast(),
                        );

                        let u1 = _m128s_fma_mul_complex(
                            _mm_loadu_ps(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            _mm_unpacklo_ps64(tw0, tw1),
                        );
                        let u2 = _m128s_fma_mul_complex(
                            _mm_loadu_ps(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            _mm_unpackhi_ps64(tw0, tw1),
                        );
                        let u3 = _m128s_fma_mul_complex(
                            _mm_loadu_ps(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            _mm_unpacklo_ps64(tw2, tw3),
                        );
                        let u4 = _m128s_fma_mul_complex(
                            _mm_loadu_ps(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            _mm_unpackhi_ps64(tw2, tw3),
                        );

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
                        let temp_b2 =
                            _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

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

                        _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _mm_storeu_ps(data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(), y1);
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        _mm_storeu_ps(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );
                        j += 2;
                    }

                    for j in j..fifth {
                        let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());

                        let tw0 = _mm_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                        let tw2 =
                            _mm_loadu_ps(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());

                        let u1 = _m128s_fma_mul_complex(
                            _m128s_load_f32x2(data.get_unchecked(j + fifth..).as_ptr().cast()),
                            tw0,
                        );
                        let u2 = _m128s_fma_mul_complex(
                            _m128s_load_f32x2(data.get_unchecked(j + 2 * fifth..).as_ptr().cast()),
                            _mm_unpackhi_ps64(tw0, tw0),
                        );
                        let u3 = _m128s_fma_mul_complex(
                            _m128s_load_f32x2(data.get_unchecked(j + 3 * fifth..).as_ptr().cast()),
                            tw2,
                        );
                        let u4 = _m128s_fma_mul_complex(
                            _m128s_load_f32x2(data.get_unchecked(j + 4 * fifth..).as_ptr().cast()),
                            _mm_unpackhi_ps64(tw2, tw2),
                        );

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
                        let temp_b2 =
                            _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

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

                        _m128s_store_f32x2(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                            y1,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                            y2,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                            y3,
                        );
                        _m128s_store_f32x2(
                            data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                            y4,
                        );
                    }
                }

                m_twiddles = &m_twiddles[fifth * 4..];
                len *= 5;
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix5<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }
}
