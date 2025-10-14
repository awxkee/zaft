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
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{
    _m128d_fma_mul_complex, _m128s_fma_mul_complex, _m128s_load_f32x2, _m128s_store_f32x2,
    _m256_fcmul_ps, _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps,
    _mm256_fcmul_pd, _mm256_unpackhi_pd2, _mm256_unpacklo_pd2, _mm256s_deinterleave4_epi64,
    shuffle,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix5::Radix5Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_twiddle, is_power_of_five};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::x86_64::*;
use std::fmt::Display;

pub(crate) struct AvxFmaRadix5<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    direction: FftDirection,
    tw1tw2_im: [T; 8],
    tw2ntw1_im: [T; 8],
    tw1tw2_re: [T; 8],
    tw2tw1_re: [T; 8],
    butterfly: Box<dyn FftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix5Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + Send
        + Sync
        + AlgorithmFactory<T>
        + MulAdd<T, Output = T>
        + SpectrumOpsFactory<T>
        + Display
        + TransposeFactory<T>,
> AvxFmaRadix5<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix5<T>, ZaftError> {
        assert!(
            is_power_of_five(size as u64),
            "Input length must be a power of 5"
        );

        let twiddles = T::make_twiddles_with_base(5, size, fft_direction)?;

        let tw1 = compute_twiddle(1, 5, fft_direction);
        let tw2 = compute_twiddle(2, 5, fft_direction);

        Ok(AvxFmaRadix5 {
            execution_length: size,
            twiddles,
            twiddle1: tw1,
            twiddle2: tw2,
            tw1tw2_im: [
                tw1.im, tw1.im, tw2.im, tw2.im, tw1.im, tw1.im, tw2.im, tw2.im,
            ],
            tw2ntw1_im: [
                tw2.im, tw2.im, -tw1.im, -tw1.im, tw2.im, tw2.im, -tw1.im, -tw1.im,
            ],
            tw1tw2_re: [
                tw1.re, tw1.re, tw2.re, tw2.re, tw1.re, tw1.re, tw2.re, tw2.re,
            ],
            tw2tw1_re: [
                tw2.re, tw2.re, tw1.re, tw1.re, tw2.re, tw2.re, tw1.re, tw1.re,
            ],
            direction: fft_direction,
            butterfly: Zaft::strategy(5, fft_direction)?,
        })
    }
}

impl AvxFmaRadix5<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let tw1_re = _mm256_set1_pd(self.twiddle1.re);
            let tw1_im = _mm256_set1_pd(self.twiddle1.im);
            let tw2_re = _mm256_set1_pd(self.twiddle2.re);
            let tw2_im = _mm256_set1_pd(self.twiddle2.im);
            let rot_sign =
                _mm256_loadu_pd([-0.0f64, 0.0, -0.0f64, 0.0, -0.0f64, 0.0, -0.0f64, 0.0].as_ptr());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 5>(5, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 5;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 5;
                    let fifth = len / 5;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;
                        while j + 2 < fifth {
                            let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * (j + 1)..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * (j + 1) + 2..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                _mm256_unpacklo_pd2(tw0, tw1),
                            );
                            let u2 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                ),
                                _mm256_unpackhi_pd2(tw0, tw1),
                            );
                            let u3 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                ),
                                _mm256_unpacklo_pd2(tw2, tw3),
                            );
                            let u4 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                ),
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

                        let tw1tw2_im = _mm256_loadu_pd(self.tw1tw2_im.as_ptr().cast());
                        let tw2ntw1_im = _mm256_loadu_pd(self.tw2ntw1_im.as_ptr().cast());
                        let tw1tw2_re = _mm256_loadu_pd(self.tw1tw2_re.as_ptr().cast());
                        let tw2tw1_re = _mm256_loadu_pd(self.tw2tw1_re.as_ptr().cast());
                        let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

                        for j in j..fifth {
                            let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast(),
                            );

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

                            const HI_HI: i32 = 0b0011_0001;
                            const LO_LO: i32 = 0b0010_0000;

                            let u1u2 = _mm256_create_pd(u1, u2);
                            let u4u3 = _mm256_create_pd(u4, u3);

                            let u0u0 = _mm256_create_pd(u0, u0);

                            let x14px23p = _mm256_add_pd(u1u2, u4u3);
                            let x14nx23n = _mm256_sub_pd(u1u2, u4u3);
                            let y0 = _mm_add_pd(
                                _mm_add_pd(u0, _mm256_castpd256_pd128(x14px23p)),
                                _mm256_extractf128_pd::<1>(x14px23p),
                            );

                            let temp_b1_1_b2_1 = _mm256_mul_pd(
                                tw1tw2_im,
                                _mm256_permute2f128_pd::<LO_LO>(x14nx23n, x14nx23n),
                            );

                            let wx23p = _mm256_permute2f128_pd::<HI_HI>(x14px23p, x14px23p);
                            let wx23n = _mm256_permute2f128_pd::<HI_HI>(x14nx23n, x14nx23n);

                            let temp_a1_a2 = _mm256_fmadd_pd(
                                tw2tw1_re,
                                wx23p,
                                _mm256_fmadd_pd(
                                    _mm256_permute2f128_pd::<LO_LO>(x14px23p, x14px23p),
                                    tw1tw2_re,
                                    u0u0,
                                ),
                            );

                            let temp_b1_b2 = _mm256_fmadd_pd(tw2ntw1_im, wx23n, temp_b1_1_b2_1);

                            let temp_b1_b2_rot = rotate.rotate_m256d(temp_b1_b2);

                            let y1y2 = _mm256_add_pd(temp_a1_a2, temp_b1_b2_rot);
                            let y4y3 = _mm256_sub_pd(temp_a1_a2, temp_b1_b2_rot);

                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y0);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                _mm256_castpd256_pd128(y1y2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * fifth..).as_mut_ptr().cast(),
                                _mm256_extractf128_pd::<1>(y1y2),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * fifth..).as_mut_ptr().cast(),
                                _mm256_extractf128_pd::<1>(y4y3),
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 4 * fifth..).as_mut_ptr().cast(),
                                _mm256_castpd256_pd128(y4y3),
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 4..];
                }
                chunk.copy_from_slice(&scratch);
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
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let tw1_re = _mm256_set1_ps(self.twiddle1.re);
            let tw1_im = _mm256_set1_ps(self.twiddle1.im);
            let tw2_re = _mm256_set1_ps(self.twiddle2.re);
            let tw2_im = _mm256_set1_ps(self.twiddle2.im);
            static ROT_90: [f32; 8] = [-0.0f32, 0.0, -0.0, 0.0, -0.0f32, 0.0, -0.0, 0.0];
            let rot_sign = _mm256_loadu_ps(ROT_90.as_ptr());

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f32>, 5>(5, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 5;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 5;
                    let fifth = len / 5;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < fifth {
                            let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let xw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let xw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(4 * j + 4..).as_ptr().cast(),
                            );
                            let xw2 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(4 * j + 8..).as_ptr().cast(),
                            );
                            let xw3 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(4 * j + 12..).as_ptr().cast(),
                            );

                            let (tw0, tw1, tw2, tw3) =
                                _mm256s_deinterleave4_epi64(xw0, xw1, xw2, xw3);

                            let u1 = _m256_fcmul_ps(
                                _mm256_loadu_ps(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                tw0,
                            );
                            let u2 = _m256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let u3 = _m256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                ),
                                tw2,
                            );
                            let u4 = _m256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                ),
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

                            let tw0 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw1 = _mm_loadu_ps(
                                m_twiddles.get_unchecked(4 * (j + 1)..).as_ptr().cast(),
                            );
                            let tw2 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());
                            let tw3 = _mm_loadu_ps(
                                m_twiddles.get_unchecked(4 * (j + 1) + 2..).as_ptr().cast(),
                            );

                            const SH: i32 = shuffle(3, 1, 2, 0);

                            let tw01 = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<SH>(
                                _mm256_castps_si256(_mm256_create_ps(tw0, tw1)),
                            ));
                            let tw23 = _mm256_castsi256_ps(_mm256_permute4x64_epi64::<SH>(
                                _mm256_castps_si256(_mm256_create_ps(tw2, tw3)),
                            ));

                            let u1u2 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(data.get_unchecked(j + fifth..).as_ptr().cast()),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw01,
                            );
                            let u3u4 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw23,
                            );

                            let u1 = _mm256_castps256_ps128(u1u2);
                            let u2 = _mm256_extractf128_ps::<1>(u1u2);
                            let u3 = _mm256_castps256_ps128(u3u4);
                            let u4 = _mm256_extractf128_ps::<1>(u3u4);

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

                            let temp_b1 =
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
                            let temp_b2 =
                                _mm_fnmadd_ps(_mm256_castps256_ps128(tw1_im), x23n, temp_b2_1);

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
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + fifth..).as_mut_ptr().cast(),
                                y1,
                            );
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

                            let tw0 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(4 * j..).as_ptr().cast());
                            let tw2 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(4 * j + 2..).as_ptr().cast());

                            let u1u2 = _m128s_fma_mul_complex(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + fifth..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 2 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _m128s_fma_mul_complex(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 3 * fifth..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 4 * fifth..).as_ptr().cast(),
                                    ),
                                ),
                                tw2,
                            );

                            let u1 = u1u2;
                            let u2 = _mm_unpackhi_ps64(u1u2, u1u2);
                            let u3 = u3u4;
                            let u4 = _mm_unpackhi_ps64(u3u4, u3u4);

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

                            let temp_b1 =
                                _mm_fmadd_ps(_mm256_castps256_ps128(tw2_im), x23n, temp_b1_1);
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

                    m_twiddles = &m_twiddles[columns * 4..];
                }
                chunk.copy_from_slice(&scratch);
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_neon_radix5() {
        for i in 1..4 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxFmaRadix5::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix5::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| Complex::new(x.re as f64, x.im as f64) * (1.0f64 / input.len() as f64))
                .map(|x| Complex::new(x.re as f32, x.im as f32))
                .collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-4,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-4,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_neon_radix5_f64() {
        for i in 1..4 {
            let size = 5usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxFmaRadix5::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix5::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| x * (1.0f64 / input.len() as f64))
                .collect();

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
