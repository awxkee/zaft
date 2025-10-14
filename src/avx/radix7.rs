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
use crate::avx::butterflies::AvxButterfly;
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{
    _m128s_fma_mul_complex, _m128s_load_f32x2, _m128s_store_f32x2, _m256_fcmul_ps,
    _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, _mm256_fcmul_pd,
    _mm256_load4_f32x2, _mm256_permute4x64_ps, shuffle,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix7::Radix7Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{bitreversed_transpose, compute_twiddle, is_power_of_seven};
use crate::{FftDirection, FftExecutor, Zaft, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::x86_64::*;
use std::fmt::Display;

pub(crate) struct AvxFmaRadix7<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    direction: FftDirection,
    butterfly: Box<dyn FftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix7Twiddles
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
> AvxFmaRadix7<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix7<T>, ZaftError> {
        assert!(
            is_power_of_seven(size as u64),
            "Input length must be a power of 7"
        );

        let twiddles = T::make_twiddles_with_base(7, size, fft_direction)?;

        Ok(AvxFmaRadix7 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 7, fft_direction),
            twiddle2: compute_twiddle(2, 7, fft_direction),
            twiddle3: compute_twiddle(3, 7, fft_direction),
            direction: fft_direction,
            butterfly: Zaft::strategy(7, fft_direction)?,
        })
    }
}

impl AvxFmaRadix7<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f64>, 7>(7, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 7;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 7;
                    let seventh = len / 7;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < seventh {
                            let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            const HI_HI: i32 = 0b0011_0001;
                            const LO_LO: i32 = 0b0010_0000;

                            let twi = 6 * j;
                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );

                            let tw0_2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 6..).as_ptr().cast(),
                            );
                            let tw1_2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );
                            let tw2_2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 10..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                _mm256_permute2f128_pd::<LO_LO>(tw0, tw0_2),
                            );
                            let u2 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * seventh..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<HI_HI>(tw0, tw0_2),
                            );
                            let u3 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * seventh..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<LO_LO>(tw1, tw1_2),
                            );
                            let u4 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 4 * seventh..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<HI_HI>(tw1, tw1_2),
                            );
                            let u5 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 5 * seventh..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<LO_LO>(tw2, tw2_2),
                            );
                            let u6 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 6 * seventh..).as_ptr().cast(),
                                ),
                                _mm256_permute2f128_pd::<HI_HI>(tw2, tw2_2),
                            );

                            let (x1p6, x1m6) = AvxButterfly::butterfly2_f64(u1, u6);
                            let x1m6 = rotate.rotate_m256d(x1m6);
                            let y00 = _mm256_add_pd(u0, x1p6);
                            let (x2p5, x2m5) = AvxButterfly::butterfly2_f64(u2, u5);
                            let x2m5 = rotate.rotate_m256d(x2m5);
                            let y00 = _mm256_add_pd(y00, x2p5);
                            let (x3p4, x3m4) = AvxButterfly::butterfly2_f64(u3, u4);
                            let x3m4 = rotate.rotate_m256d(x3m4);
                            let y00 = _mm256_add_pd(y00, x3p4);

                            let m0106a =
                                _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle1.re), u0);
                            let m0106a =
                                _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle2.re), m0106a);
                            let m0106a =
                                _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle3.re), m0106a);
                            let m0106b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle1.im));
                            let m0106b =
                                _mm256_fmadd_pd(x2m5, _mm256_set1_pd(self.twiddle2.im), m0106b);
                            let m0106b =
                                _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle3.im), m0106b);
                            let (y01, y06) = AvxButterfly::butterfly2_f64(m0106a, m0106b);

                            let m0205a =
                                _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle2.re), u0);
                            let m0205a =
                                _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle3.re), m0205a);
                            let m0205a =
                                _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle1.re), m0205a);
                            let m0205b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle2.im));
                            let m0205b =
                                _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle3.im), m0205b);
                            let m0205b =
                                _mm256_fnmadd_pd(x3m4, _mm256_set1_pd(self.twiddle1.im), m0205b);
                            let (y02, y05) = AvxButterfly::butterfly2_f64(m0205a, m0205b);

                            let m0304a =
                                _mm256_fmadd_pd(x1p6, _mm256_set1_pd(self.twiddle3.re), u0);
                            let m0304a =
                                _mm256_fmadd_pd(x2p5, _mm256_set1_pd(self.twiddle1.re), m0304a);
                            let m0304a =
                                _mm256_fmadd_pd(x3p4, _mm256_set1_pd(self.twiddle2.re), m0304a);
                            let m0304b = _mm256_mul_pd(x1m6, _mm256_set1_pd(self.twiddle3.im));
                            let m0304b =
                                _mm256_fnmadd_pd(x2m5, _mm256_set1_pd(self.twiddle1.im), m0304b);
                            let m0304b =
                                _mm256_fmadd_pd(x3m4, _mm256_set1_pd(self.twiddle2.im), m0304b);
                            let (y03, y04) = AvxButterfly::butterfly2_f64(m0304a, m0304b);

                            // // Store results
                            _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );

                            j += 2;
                        }

                        for j in j..seventh {
                            let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let twi = 6 * j;
                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );

                            let u1u2 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 2 * seventh..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 3 * seventh..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 4 * seventh..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );
                            let u5u6 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 5 * seventh..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 6 * seventh..).as_ptr().cast(),
                                    ),
                                ),
                                tw2,
                            );

                            let (x1p6, x1m6) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u1u2),
                                _mm256_extractf128_pd::<1>(u5u6),
                            );
                            let x1m6 = rotate.rotate_m128d(x1m6);
                            let y00 = _mm_add_pd(u0, x1p6);
                            let (x2p5, x2m5) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_extractf128_pd::<1>(u1u2),
                                _mm256_castpd256_pd128(u5u6),
                            );
                            let x2m5 = rotate.rotate_m128d(x2m5);
                            let y00 = _mm_add_pd(y00, x2p5);
                            let (x3p4, x3m4) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u3u4),
                                _mm256_extractf128_pd::<1>(u3u4),
                            );
                            let x3m4 = rotate.rotate_m128d(x3m4);
                            let y00 = _mm_add_pd(y00, x3p4);

                            let m0106a = _mm_fmadd_pd(x1p6, _mm_set1_pd(self.twiddle1.re), u0);
                            let m0106a = _mm_fmadd_pd(x2p5, _mm_set1_pd(self.twiddle2.re), m0106a);
                            let m0106a = _mm_fmadd_pd(x3p4, _mm_set1_pd(self.twiddle3.re), m0106a);
                            let m0106b = _mm_mul_pd(x1m6, _mm_set1_pd(self.twiddle1.im));
                            let m0106b = _mm_fmadd_pd(x2m5, _mm_set1_pd(self.twiddle2.im), m0106b);
                            let m0106b = _mm_fmadd_pd(x3m4, _mm_set1_pd(self.twiddle3.im), m0106b);
                            let (y01, y06) = AvxButterfly::butterfly2_f64_m128(m0106a, m0106b);

                            let m0205a = _mm_fmadd_pd(x1p6, _mm_set1_pd(self.twiddle2.re), u0);
                            let m0205a = _mm_fmadd_pd(x2p5, _mm_set1_pd(self.twiddle3.re), m0205a);
                            let m0205a = _mm_fmadd_pd(x3p4, _mm_set1_pd(self.twiddle1.re), m0205a);
                            let m0205b = _mm_mul_pd(x1m6, _mm_set1_pd(self.twiddle2.im));
                            let m0205b = _mm_fnmadd_pd(x2m5, _mm_set1_pd(self.twiddle3.im), m0205b);
                            let m0205b = _mm_fnmadd_pd(x3m4, _mm_set1_pd(self.twiddle1.im), m0205b);
                            let (y02, y05) = AvxButterfly::butterfly2_f64_m128(m0205a, m0205b);

                            let m0304a = _mm_fmadd_pd(x1p6, _mm_set1_pd(self.twiddle3.re), u0);
                            let m0304a = _mm_fmadd_pd(x2p5, _mm_set1_pd(self.twiddle1.re), m0304a);
                            let m0304a = _mm_fmadd_pd(x3p4, _mm_set1_pd(self.twiddle2.re), m0304a);
                            let m0304b = _mm_mul_pd(x1m6, _mm_set1_pd(self.twiddle3.im));
                            let m0304b = _mm_fnmadd_pd(x2m5, _mm_set1_pd(self.twiddle1.im), m0304b);
                            let m0304b = _mm_fmadd_pd(x3m4, _mm_set1_pd(self.twiddle2.im), m0304b);
                            let (y03, y04) = AvxButterfly::butterfly2_f64_m128(m0304a, m0304b);

                            // // Store results
                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 6..];
                }
                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix7<f64> {
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

impl AvxFmaRadix7<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        unsafe {
            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            let mut scratch = try_vec![Complex::new(0., 0.); self.execution_length];
            for chunk in in_place.chunks_exact_mut(self.execution_length) {
                // Digit-reversal permutation
                bitreversed_transpose::<Complex<f32>, 7>(7, chunk, &mut scratch);

                self.butterfly.execute(&mut scratch)?;

                let mut len = 7;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 7;
                    let seventh = len / 7;

                    for data in scratch.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < seventh {
                            let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                            let twi = 6 * j;
                            let tw0tw1tw2tw3 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw4tw5tw6tw7 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw8tw9tw10tw11 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );

                            let tw3 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 12..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 16..).as_ptr().cast(),
                            );
                            let tw5 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 20..).as_ptr().cast(),
                            );

                            const LO_HI: i32 = 0b0011_0000;
                            const HI_LO: i32 = 0b0010_0001;

                            let a1 =
                                _mm256_loadu_ps(data.get_unchecked(j + seventh..).as_ptr().cast());
                            let a2 = _mm256_loadu_ps(
                                data.get_unchecked(j + 2 * seventh..).as_ptr().cast(),
                            );
                            let a3 = _mm256_loadu_ps(
                                data.get_unchecked(j + 3 * seventh..).as_ptr().cast(),
                            );
                            let a4 = _mm256_loadu_ps(
                                data.get_unchecked(j + 4 * seventh..).as_ptr().cast(),
                            );
                            let a5 = _mm256_loadu_ps(
                                data.get_unchecked(j + 5 * seventh..).as_ptr().cast(),
                            );
                            let a6 = _mm256_loadu_ps(
                                data.get_unchecked(j + 6 * seventh..).as_ptr().cast(),
                            );

                            let u1u2 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm256_castps256_ps128(a1),
                                    _mm256_castps256_ps128(a2),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<LO_HI>(tw0tw1tw2tw3, tw4tw5tw6tw7),
                                ),
                            );

                            let u3u4 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm256_castps256_ps128(a3),
                                    _mm256_castps256_ps128(a4),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<HI_LO>(tw0tw1tw2tw3, tw8tw9tw10tw11),
                                ),
                            );

                            let u5u6 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm256_castps256_ps128(a5),
                                    _mm256_castps256_ps128(a6),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<LO_HI>(tw4tw5tw6tw7, tw8tw9tw10tw11),
                                ),
                            );

                            let u1u2_2 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm256_extractf128_ps::<1>(a1),
                                    _mm256_extractf128_ps::<1>(a2),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<LO_HI>(tw3, tw4),
                                ),
                            );

                            let u3u4_2 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm256_extractf128_ps::<1>(a3),
                                    _mm256_extractf128_ps::<1>(a4),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<HI_LO>(tw3, tw5),
                                ),
                            );

                            let u5u6_2 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm256_extractf128_ps::<1>(a5),
                                    _mm256_extractf128_ps::<1>(a6),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<LO_HI>(tw4, tw5),
                                ),
                            );

                            const HI_HI: i32 = 0b0011_0001;
                            const LO_LO: i32 = 0b0010_0000;

                            let (x1p6, x1m6) = AvxButterfly::butterfly2_f32(
                                _mm256_permute2f128_ps::<LO_LO>(u1u2, u1u2_2),
                                _mm256_permute2f128_ps::<HI_HI>(u5u6, u5u6_2),
                            );
                            let x1m6 = rotate.rotate_m256(x1m6);
                            let y00 = _mm256_add_ps(u0, x1p6);
                            let (x2p5, x2m5) = AvxButterfly::butterfly2_f32(
                                _mm256_permute2f128_ps::<HI_HI>(u1u2, u1u2_2),
                                _mm256_permute2f128_ps::<LO_LO>(u5u6, u5u6_2),
                            );
                            let x2m5 = rotate.rotate_m256(x2m5);
                            let y00 = _mm256_add_ps(y00, x2p5);
                            let (x3p4, x3m4) = AvxButterfly::butterfly2_f32(
                                _mm256_permute2f128_ps::<LO_LO>(u3u4, u3u4_2),
                                _mm256_permute2f128_ps::<HI_HI>(u3u4, u3u4_2),
                            );
                            let x3m4 = rotate.rotate_m256(x3m4);
                            let y00 = _mm256_add_ps(y00, x3p4);

                            let m0106a =
                                _mm256_fmadd_ps(x1p6, _mm256_set1_ps(self.twiddle1.re), u0);
                            let m0106a =
                                _mm256_fmadd_ps(x2p5, _mm256_set1_ps(self.twiddle2.re), m0106a);
                            let m0106a =
                                _mm256_fmadd_ps(x3p4, _mm256_set1_ps(self.twiddle3.re), m0106a);
                            let m0106b = _mm256_mul_ps(x1m6, _mm256_set1_ps(self.twiddle1.im));
                            let m0106b =
                                _mm256_fmadd_ps(x2m5, _mm256_set1_ps(self.twiddle2.im), m0106b);
                            let m0106b =
                                _mm256_fmadd_ps(x3m4, _mm256_set1_ps(self.twiddle3.im), m0106b);
                            let (y01, y06) = AvxButterfly::butterfly2_f32(m0106a, m0106b);

                            let m0205a =
                                _mm256_fmadd_ps(x1p6, _mm256_set1_ps(self.twiddle2.re), u0);
                            let m0205a =
                                _mm256_fmadd_ps(x2p5, _mm256_set1_ps(self.twiddle3.re), m0205a);
                            let m0205a =
                                _mm256_fmadd_ps(x3p4, _mm256_set1_ps(self.twiddle1.re), m0205a);
                            let m0205b = _mm256_mul_ps(x1m6, _mm256_set1_ps(self.twiddle2.im));
                            let m0205b =
                                _mm256_fnmadd_ps(x2m5, _mm256_set1_ps(self.twiddle3.im), m0205b);
                            let m0205b =
                                _mm256_fnmadd_ps(x3m4, _mm256_set1_ps(self.twiddle1.im), m0205b);
                            let (y02, y05) = AvxButterfly::butterfly2_f32(m0205a, m0205b);

                            let m0304a =
                                _mm256_fmadd_ps(x1p6, _mm256_set1_ps(self.twiddle3.re), u0);
                            let m0304a =
                                _mm256_fmadd_ps(x2p5, _mm256_set1_ps(self.twiddle1.re), m0304a);
                            let m0304a =
                                _mm256_fmadd_ps(x3p4, _mm256_set1_ps(self.twiddle2.re), m0304a);
                            let m0304b = _mm256_mul_ps(x1m6, _mm256_set1_ps(self.twiddle3.im));
                            let m0304b =
                                _mm256_fnmadd_ps(x2m5, _mm256_set1_ps(self.twiddle1.im), m0304b);
                            let m0304b =
                                _mm256_fmadd_ps(x3m4, _mm256_set1_ps(self.twiddle2.im), m0304b);
                            let (y03, y04) = AvxButterfly::butterfly2_f32(m0304a, m0304b);

                            // Store results
                            _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );

                            j += 4;
                        }

                        while j + 2 < seventh {
                            let u0 = _mm_loadu_ps(data.get_unchecked(j..).as_ptr().cast());
                            let twi = 6 * j;
                            let tw0tw1tw2tw3 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw4tw5tw6tw7 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw8tw9tw10tw11 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );

                            const LO_HI: i32 = 0b0011_0000;
                            const HI_LO: i32 = 0b0010_0001;

                            let u1u2 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(data.get_unchecked(j + seventh..).as_ptr().cast()),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 2 * seventh..).as_ptr().cast(),
                                    ),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<LO_HI>(tw0tw1tw2tw3, tw4tw5tw6tw7),
                                ),
                            );

                            let u3u4 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 3 * seventh..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 4 * seventh..).as_ptr().cast(),
                                    ),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<HI_LO>(tw0tw1tw2tw3, tw8tw9tw10tw11),
                                ),
                            );

                            let u5u6 = _m256_fcmul_ps(
                                _mm256_create_ps(
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 5 * seventh..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_ps(
                                        data.get_unchecked(j + 6 * seventh..).as_ptr().cast(),
                                    ),
                                ),
                                _mm256_permute4x64_ps::<{ shuffle(3, 1, 2, 0) }>(
                                    _mm256_permute2f128_ps::<LO_HI>(tw4tw5tw6tw7, tw8tw9tw10tw11),
                                ),
                            );

                            let (x1p6, x1m6) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_castps256_ps128(u1u2),
                                _mm256_extractf128_ps::<1>(u5u6),
                            );
                            let x1m6 = rotate.rotate_m128(x1m6);
                            let y00 = _mm_add_ps(u0, x1p6);
                            let (x2p5, x2m5) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_extractf128_ps::<1>(u1u2),
                                _mm256_castps256_ps128(u5u6),
                            );
                            let x2m5 = rotate.rotate_m128(x2m5);
                            let y00 = _mm_add_ps(y00, x2p5);
                            let (x3p4, x3m4) = AvxButterfly::butterfly2_f32_m128(
                                _mm256_castps256_ps128(u3u4),
                                _mm256_extractf128_ps::<1>(u3u4),
                            );
                            let x3m4 = rotate.rotate_m128(x3m4);
                            let y00 = _mm_add_ps(y00, x3p4);

                            let m0106a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle1.re), u0);
                            let m0106a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle2.re), m0106a);
                            let m0106a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle3.re), m0106a);
                            let m0106b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle1.im));
                            let m0106b = _mm_fmadd_ps(x2m5, _mm_set1_ps(self.twiddle2.im), m0106b);
                            let m0106b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle3.im), m0106b);
                            let (y01, y06) = AvxButterfly::butterfly2_f32_m128(m0106a, m0106b);

                            let m0205a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle2.re), u0);
                            let m0205a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle3.re), m0205a);
                            let m0205a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle1.re), m0205a);
                            let m0205b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle2.im));
                            let m0205b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle3.im), m0205b);
                            let m0205b = _mm_fnmadd_ps(x3m4, _mm_set1_ps(self.twiddle1.im), m0205b);
                            let (y02, y05) = AvxButterfly::butterfly2_f32_m128(m0205a, m0205b);

                            let m0304a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle3.re), u0);
                            let m0304a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle1.re), m0304a);
                            let m0304a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle2.re), m0304a);
                            let m0304b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle3.im));
                            let m0304b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle1.im), m0304b);
                            let m0304b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle2.im), m0304b);
                            let (y03, y04) = AvxButterfly::butterfly2_f32_m128(m0304a, m0304b);

                            // Store results
                            _mm_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm_storeu_ps(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );

                            j += 2;
                        }

                        for j in j..seventh {
                            let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                            let twi = 6 * j;
                            let tw0tw1tw2tw3 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw4tw5 =
                                _mm_loadu_ps(m_twiddles.get_unchecked(twi + 4..).as_ptr().cast());

                            let u1u2u3u4 = _m256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + seventh..),
                                    data.get_unchecked(j + 2 * seventh..),
                                    data.get_unchecked(j + 3 * seventh..),
                                    data.get_unchecked(j + 4 * seventh..),
                                ),
                                tw0tw1tw2tw3,
                            );
                            let u5u6 = _m128s_fma_mul_complex(
                                _mm_unpacklo_ps64(
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 5 * seventh..).as_ptr().cast(),
                                    ),
                                    _m128s_load_f32x2(
                                        data.get_unchecked(j + 6 * seventh..).as_ptr().cast(),
                                    ),
                                ),
                                tw4tw5,
                            );

                            let u1u2 = _mm256_castps256_ps128(u1u2u3u4);
                            let u3u4 = _mm256_extractf128_ps::<1>(u1u2u3u4);
                            let u6 = _mm_unpackhi_ps64(u5u6, u5u6);
                            let u2 = _mm_unpackhi_ps64(u1u2, u1u2);
                            let u4 = _mm_unpackhi_ps64(u3u4, u3u4);

                            let (x1p6, x1m6) = AvxButterfly::butterfly2_f32_m128(u1u2, u6);
                            let x1m6 = rotate.rotate_m128(x1m6);
                            let y00 = _mm_add_ps(u0, x1p6);
                            let (x2p5, x2m5) = AvxButterfly::butterfly2_f32_m128(u2, u5u6);
                            let x2m5 = rotate.rotate_m128(x2m5);
                            let y00 = _mm_add_ps(y00, x2p5);
                            let (x3p4, x3m4) = AvxButterfly::butterfly2_f32_m128(u3u4, u4);
                            let x3m4 = rotate.rotate_m128(x3m4);
                            let y00 = _mm_add_ps(y00, x3p4);

                            let m0106a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle1.re), u0);
                            let m0106a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle2.re), m0106a);
                            let m0106a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle3.re), m0106a);
                            let m0106b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle1.im));
                            let m0106b = _mm_fmadd_ps(x2m5, _mm_set1_ps(self.twiddle2.im), m0106b);
                            let m0106b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle3.im), m0106b);
                            let (y01, y06) = AvxButterfly::butterfly2_f32_m128(m0106a, m0106b);

                            let m0205a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle2.re), u0);
                            let m0205a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle3.re), m0205a);
                            let m0205a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle1.re), m0205a);
                            let m0205b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle2.im));
                            let m0205b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle3.im), m0205b);
                            let m0205b = _mm_fnmadd_ps(x3m4, _mm_set1_ps(self.twiddle1.im), m0205b);
                            let (y02, y05) = AvxButterfly::butterfly2_f32_m128(m0205a, m0205b);

                            let m0304a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle3.re), u0);
                            let m0304a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle1.re), m0304a);
                            let m0304a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle2.re), m0304a);
                            let m0304b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle3.im));
                            let m0304b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle1.im), m0304b);
                            let m0304b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle2.im), m0304b);
                            let (y03, y04) = AvxButterfly::butterfly2_f32_m128(m0304a, m0304b);

                            // Store results
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                y00,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + seventh..).as_mut_ptr().cast(),
                                y01,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 4 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 5 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 6 * seventh..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 6..];
                }
                chunk.copy_from_slice(&scratch);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix7<f32> {
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
    use crate::radix7::Radix7;
    use rand::Rng;
    #[test]
    fn test_avx_radix7() {
        for i in 1..5 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let radix7_reference = Radix7::new(size, FftDirection::Forward).unwrap();

            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = AvxFmaRadix7::new(size, FftDirection::Forward).unwrap();
            radix7_reference.execute(&mut z_ref).unwrap();
            radix_forward.execute(&mut input).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 0.01,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 0.01,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            let radix_inverse = AvxFmaRadix7::new(size, FftDirection::Inverse).unwrap();
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
    fn test_avx_radix7_f64() {
        for i in 1..5 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }

            let radix7_reference = Radix7::new(size, FftDirection::Forward).unwrap();

            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = AvxFmaRadix7::new(size, FftDirection::Forward).unwrap();
            let radix_inverse = AvxFmaRadix7::new(size, FftDirection::Inverse).unwrap();
            radix_forward.execute(&mut input).unwrap();
            radix7_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input
                .iter()
                .map(|&x| x * (1.0f64 / input.len() as f64))
                .collect();

            input
                .iter()
                .zip(src.iter())
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
                        "a_im {} != b_im {} for size {}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }
}
