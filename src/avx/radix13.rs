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
use crate::avx::mixed::AvxStoreF;
use crate::avx::rotate::AvxRotate;
use crate::avx::transpose::transpose_4x13;
use crate::avx::util::{
    _m128s_load_f32x2, _m128s_store_f32x2, _mm_unpackhi_ps64, _mm256_create_pd, _mm256_fcmul_pd,
    _mm256_fcmul_ps, _mm256_load4_f32x2, avx_bitreversed_transpose, create_avx4_1_twiddles,
};
use crate::err::try_vec;
use crate::factory::AlgorithmFactory;
use crate::radix13::Radix13Twiddles;
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{compute_logarithm, compute_twiddle, is_power_of_thirteen, reverse_bits};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::arch::x86_64::*;
use std::fmt::Display;
use std::sync::Arc;

pub(crate) struct AvxFmaRadix13<T> {
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
    direction: FftDirection,
    butterfly: Arc<dyn CompositeFftExecutor<T> + Send + Sync>,
}

impl<
    T: Default
        + Clone
        + Radix13Twiddles
        + 'static
        + Copy
        + FftTrigonometry
        + Float
        + Send
        + Sync
        + AlgorithmFactory<T>
        + SpectrumOpsFactory<T>
        + TransposeFactory<T>
        + MulAdd<T, Output = T>
        + Display,
> AvxFmaRadix13<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(size: usize, fft_direction: FftDirection) -> Result<AvxFmaRadix13<T>, ZaftError> {
        assert!(
            is_power_of_thirteen(size as u64),
            "Input length must be a power of 13"
        );

        let twiddles = create_avx4_1_twiddles::<T, 13>(13, size, fft_direction)?;

        Ok(AvxFmaRadix13 {
            execution_length: size,
            twiddles,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
            direction: fft_direction,
            butterfly: T::butterfly13(fft_direction)?,
        })
    }
}

impl AvxFmaRadix13<f64> {
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
                avx_bitreversed_transpose::<Complex<f64>, 13>(13, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 13;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 13;
                    let thirteenth = len / 13;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 2 < thirteenth {
                            let u0 = _mm256_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let twi = 12 * j;
                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 6..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );
                            let tw5 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 10..).as_ptr().cast(),
                            );

                            let u1 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + thirteenth..).as_ptr().cast(),
                                ),
                                tw0,
                            );
                            let u2 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 2 * thirteenth..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let u3 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 3 * thirteenth..).as_ptr().cast(),
                                ),
                                tw2,
                            );
                            let u4 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 4 * thirteenth..).as_ptr().cast(),
                                ),
                                tw3,
                            );
                            let u5 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 5 * thirteenth..).as_ptr().cast(),
                                ),
                                tw4,
                            );
                            let u6 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 6 * thirteenth..).as_ptr().cast(),
                                ),
                                tw5,
                            );

                            let tw7 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 12..).as_ptr().cast(),
                            );
                            let tw8 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 14..).as_ptr().cast(),
                            );
                            let tw9 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 16..).as_ptr().cast(),
                            );
                            let tw10 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 18..).as_ptr().cast(),
                            );
                            let tw11 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 20..).as_ptr().cast(),
                            );
                            let tw12 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 22..).as_ptr().cast(),
                            );

                            let u7 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 7 * thirteenth..).as_ptr().cast(),
                                ),
                                tw7,
                            );

                            let u8 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 8 * thirteenth..).as_ptr().cast(),
                                ),
                                tw8,
                            );

                            let u9 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 9 * thirteenth..).as_ptr().cast(),
                                ),
                                tw9,
                            );

                            let u10 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 10 * thirteenth..).as_ptr().cast(),
                                ),
                                tw10,
                            );

                            let u11 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 11 * thirteenth..).as_ptr().cast(),
                                ),
                                tw11,
                            );

                            let u12 = _mm256_fcmul_pd(
                                _mm256_loadu_pd(
                                    data.get_unchecked(j + 12 * thirteenth..).as_ptr().cast(),
                                ),
                                tw12,
                            );

                            let y00 = u0;
                            let (x1p12, x1m12) = AvxButterfly::butterfly2_f64(u1, u12); // u1, u12
                            let x1m12 = rotate.rotate_m256d(x1m12);
                            let y00 = _mm256_add_pd(y00, x1p12);
                            let (x2p11, x2m11) = AvxButterfly::butterfly2_f64(u2, u11); // u2, u11
                            let x2m11 = rotate.rotate_m256d(x2m11);
                            let y00 = _mm256_add_pd(y00, x2p11);
                            let (x3p10, x3m10) = AvxButterfly::butterfly2_f64(u3, u10); // u3, u10
                            let x3m10 = rotate.rotate_m256d(x3m10);
                            let y00 = _mm256_add_pd(y00, x3p10);
                            let (x4p9, x4m9) = AvxButterfly::butterfly2_f64(u4, u9); // u4, u9
                            let x4m9 = rotate.rotate_m256d(x4m9);
                            let y00 = _mm256_add_pd(y00, x4p9);
                            let (x5p8, x5m8) = AvxButterfly::butterfly2_f64(u5, u8); // u5, u8
                            let x5m8 = rotate.rotate_m256d(x5m8);
                            let y00 = _mm256_add_pd(y00, x5p8);
                            let (x6p7, x6m7) = AvxButterfly::butterfly2_f64(u6, u7); // u6, u7
                            let x6m7 = rotate.rotate_m256d(x6m7);
                            let y00 = _mm256_add_pd(y00, x6p7);

                            let m0112a =
                                _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle1.re), u0);
                            let m0112a =
                                _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle2.re), x2p11, m0112a);
                            let m0112a =
                                _mm256_fmadd_pd(_mm256_set1_pd(self.twiddle3.re), x3p10, m0112a);
                            let m0112a =
                                _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle4.re), m0112a);
                            let m0112a =
                                _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle5.re), m0112a);
                            let m0112a =
                                _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle6.re), m0112a);
                            let m0112b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle1.im));
                            let m0112b =
                                _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle2.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle3.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_pd(x4m9, _mm256_set1_pd(self.twiddle4.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle5.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle6.im), m0112b);
                            let (y01, y12) = AvxButterfly::butterfly2_f64(m0112a, m0112b);

                            let m0211a =
                                _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle2.re), u0);
                            let m0211a =
                                _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle4.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle6.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle5.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle3.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle1.re), m0211a);
                            let m0211b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle2.im));
                            let m0211b =
                                _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle4.im), m0211b);
                            let m0211b =
                                _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle6.im), m0211b);
                            let m0211b =
                                _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle5.im), m0211b);
                            let m0211b =
                                _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle3.im), m0211b);
                            let m0211b =
                                _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle1.im), m0211b);
                            let (y02, y11) = AvxButterfly::butterfly2_f64(m0211a, m0211b);

                            let m0310a =
                                _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle3.re), u0);
                            let m0310a =
                                _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle6.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle4.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle1.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle2.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle5.re), m0310a);
                            let m0310b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle3.im));
                            let m0310b =
                                _mm256_fmadd_pd(x2m11, _mm256_set1_pd(self.twiddle6.im), m0310b);
                            let m0310b =
                                _mm256_fnmadd_pd(x3m10, _mm256_set1_pd(self.twiddle4.im), m0310b);
                            let m0310b =
                                _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle1.im), m0310b);
                            let m0310b =
                                _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle2.im), m0310b);
                            let m0310b =
                                _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle5.im), m0310b);
                            let (y03, y10) = AvxButterfly::butterfly2_f64(m0310a, m0310b);

                            let m0409a =
                                _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle4.re), u0);
                            let m0409a =
                                _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle5.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle1.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle3.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle6.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle2.re), m0409a);
                            let m0409b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle4.im));
                            let m0409b =
                                _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle5.im), m0409b);
                            let m0409b =
                                _mm256_fnmadd_pd(x3m10, _mm256_set1_pd(self.twiddle1.im), m0409b);
                            let m0409b =
                                _mm256_fmadd_pd(x4m9, _mm256_set1_pd(self.twiddle3.im), m0409b);
                            let m0409b =
                                _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle6.im), m0409b);
                            let m0409b =
                                _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle2.im), m0409b);
                            let (y04, y09) = AvxButterfly::butterfly2_f64(m0409a, m0409b);

                            let m0508a =
                                _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle5.re), u0);
                            let m0508a =
                                _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle3.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle2.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle6.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle1.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle4.re), m0508a);
                            let m0508b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle5.im));
                            let m0508b =
                                _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle3.im), m0508b);
                            let m0508b =
                                _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle2.im), m0508b);
                            let m0508b =
                                _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle6.im), m0508b);
                            let m0508b =
                                _mm256_fnmadd_pd(x5m8, _mm256_set1_pd(self.twiddle1.im), m0508b);
                            let m0508b =
                                _mm256_fmadd_pd(x6m7, _mm256_set1_pd(self.twiddle4.im), m0508b);
                            let (y05, y08) = AvxButterfly::butterfly2_f64(m0508a, m0508b);

                            let m0607a =
                                _mm256_fmadd_pd(x1p12, _mm256_set1_pd(self.twiddle6.re), u0);
                            let m0607a =
                                _mm256_fmadd_pd(x2p11, _mm256_set1_pd(self.twiddle1.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_pd(x3p10, _mm256_set1_pd(self.twiddle5.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_pd(x4p9, _mm256_set1_pd(self.twiddle2.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_pd(x5p8, _mm256_set1_pd(self.twiddle4.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_pd(x6p7, _mm256_set1_pd(self.twiddle3.re), m0607a);
                            let m0607b = _mm256_mul_pd(x1m12, _mm256_set1_pd(self.twiddle6.im));
                            let m0607b =
                                _mm256_fnmadd_pd(x2m11, _mm256_set1_pd(self.twiddle1.im), m0607b);
                            let m0607b =
                                _mm256_fmadd_pd(x3m10, _mm256_set1_pd(self.twiddle5.im), m0607b);
                            let m0607b =
                                _mm256_fnmadd_pd(x4m9, _mm256_set1_pd(self.twiddle2.im), m0607b);
                            let m0607b =
                                _mm256_fmadd_pd(x5m8, _mm256_set1_pd(self.twiddle4.im), m0607b);
                            let m0607b =
                                _mm256_fnmadd_pd(x6m7, _mm256_set1_pd(self.twiddle3.im), m0607b);
                            let (y06, y07) = AvxButterfly::butterfly2_f64(m0607a, m0607b);

                            // // Store results
                            _mm256_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            _mm256_storeu_pd(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );

                            j += 2;
                        }

                        for j in j..thirteenth {
                            let u0 = _mm_loadu_pd(data.get_unchecked(j..).as_ptr().cast());

                            let twi = 12 * j;
                            let tw0 =
                                _mm256_loadu_pd(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 2..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw3 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 6..).as_ptr().cast(),
                            );
                            let tw4 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );
                            let tw5 = _mm256_loadu_pd(
                                m_twiddles.get_unchecked(twi + 10..).as_ptr().cast(),
                            );

                            let u1u2 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 2 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw0,
                            );
                            let u3u4 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 3 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 4 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw1,
                            );
                            let u5u6 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 5 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 6 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw2,
                            );
                            let u7u8 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 7 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 8 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw3,
                            );
                            let u9u10 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 9 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 10 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw4,
                            );
                            let u11u12 = _mm256_fcmul_pd(
                                _mm256_create_pd(
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 11 * thirteenth..).as_ptr().cast(),
                                    ),
                                    _mm_loadu_pd(
                                        data.get_unchecked(j + 12 * thirteenth..).as_ptr().cast(),
                                    ),
                                ),
                                tw5,
                            );

                            let y00 = u0;
                            let (x1p12, x1m12) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u1u2),
                                _mm256_extractf128_pd::<1>(u11u12),
                            ); // u1, u12
                            let x1m12 = rotate.rotate_m128d(x1m12);
                            let y00 = _mm_add_pd(y00, x1p12);
                            let (x2p11, x2m11) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_extractf128_pd::<1>(u1u2),
                                _mm256_castpd256_pd128(u11u12),
                            ); // u2, u11
                            let x2m11 = rotate.rotate_m128d(x2m11);
                            let y00 = _mm_add_pd(y00, x2p11);
                            let (x3p10, x3m10) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u3u4),
                                _mm256_extractf128_pd::<1>(u9u10),
                            ); // u3, u10
                            let x3m10 = rotate.rotate_m128d(x3m10);
                            let y00 = _mm_add_pd(y00, x3p10);
                            let (x4p9, x4m9) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_extractf128_pd::<1>(u3u4),
                                _mm256_castpd256_pd128(u9u10),
                            ); // u4, u9
                            let x4m9 = rotate.rotate_m128d(x4m9);
                            let y00 = _mm_add_pd(y00, x4p9);
                            let (x5p8, x5m8) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_castpd256_pd128(u5u6),
                                _mm256_extractf128_pd::<1>(u7u8),
                            ); // u5, u8
                            let x5m8 = rotate.rotate_m128d(x5m8);
                            let y00 = _mm_add_pd(y00, x5p8);
                            let (x6p7, x6m7) = AvxButterfly::butterfly2_f64_m128(
                                _mm256_extractf128_pd::<1>(u5u6),
                                _mm256_castpd256_pd128(u7u8),
                            ); // u6, u7
                            let x6m7 = rotate.rotate_m128d(x6m7);
                            let y00 = _mm_add_pd(y00, x6p7);

                            let m0112a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle1.re), u0);
                            let m0112a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle2.re), x2p11, m0112a);
                            let m0112a = _mm_fmadd_pd(_mm_set1_pd(self.twiddle3.re), x3p10, m0112a);
                            let m0112a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle4.re), m0112a);
                            let m0112a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle5.re), m0112a);
                            let m0112a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle6.re), m0112a);
                            let m0112b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle1.im));
                            let m0112b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle2.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle3.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x4m9, _mm_set1_pd(self.twiddle4.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle5.im), m0112b);
                            let m0112b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle6.im), m0112b);
                            let (y01, y12) = AvxButterfly::butterfly2_f64_m128(m0112a, m0112b);

                            let m0211a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle2.re), u0);
                            let m0211a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle4.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle6.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle5.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle3.re), m0211a);
                            let m0211a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle1.re), m0211a);
                            let m0211b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle2.im));
                            let m0211b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle4.im), m0211b);
                            let m0211b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle6.im), m0211b);
                            let m0211b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle5.im), m0211b);
                            let m0211b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle3.im), m0211b);
                            let m0211b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle1.im), m0211b);
                            let (y02, y11) = AvxButterfly::butterfly2_f64_m128(m0211a, m0211b);

                            let m0310a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle3.re), u0);
                            let m0310a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle6.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle4.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle1.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle2.re), m0310a);
                            let m0310a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle5.re), m0310a);
                            let m0310b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle3.im));
                            let m0310b = _mm_fmadd_pd(x2m11, _mm_set1_pd(self.twiddle6.im), m0310b);
                            let m0310b =
                                _mm_fnmadd_pd(x3m10, _mm_set1_pd(self.twiddle4.im), m0310b);
                            let m0310b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle1.im), m0310b);
                            let m0310b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle2.im), m0310b);
                            let m0310b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle5.im), m0310b);
                            let (y03, y10) = AvxButterfly::butterfly2_f64_m128(m0310a, m0310b);

                            let m0409a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle4.re), u0);
                            let m0409a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle5.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle1.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle3.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle6.re), m0409a);
                            let m0409a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle2.re), m0409a);
                            let m0409b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle4.im));
                            let m0409b =
                                _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle5.im), m0409b);
                            let m0409b =
                                _mm_fnmadd_pd(x3m10, _mm_set1_pd(self.twiddle1.im), m0409b);
                            let m0409b = _mm_fmadd_pd(x4m9, _mm_set1_pd(self.twiddle3.im), m0409b);
                            let m0409b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle6.im), m0409b);
                            let m0409b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle2.im), m0409b);
                            let (y04, y09) = AvxButterfly::butterfly2_f64_m128(m0409a, m0409b);

                            let m0508a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle5.re), u0);
                            let m0508a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle3.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle2.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle6.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle1.re), m0508a);
                            let m0508a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle4.re), m0508a);
                            let m0508b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle5.im));
                            let m0508b =
                                _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle3.im), m0508b);
                            let m0508b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle2.im), m0508b);
                            let m0508b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle6.im), m0508b);
                            let m0508b = _mm_fnmadd_pd(x5m8, _mm_set1_pd(self.twiddle1.im), m0508b);
                            let m0508b = _mm_fmadd_pd(x6m7, _mm_set1_pd(self.twiddle4.im), m0508b);
                            let (y05, y08) = AvxButterfly::butterfly2_f64_m128(m0508a, m0508b);

                            let m0607a = _mm_fmadd_pd(x1p12, _mm_set1_pd(self.twiddle6.re), u0);
                            let m0607a = _mm_fmadd_pd(x2p11, _mm_set1_pd(self.twiddle1.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x3p10, _mm_set1_pd(self.twiddle5.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x4p9, _mm_set1_pd(self.twiddle2.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x5p8, _mm_set1_pd(self.twiddle4.re), m0607a);
                            let m0607a = _mm_fmadd_pd(x6p7, _mm_set1_pd(self.twiddle3.re), m0607a);
                            let m0607b = _mm_mul_pd(x1m12, _mm_set1_pd(self.twiddle6.im));
                            let m0607b =
                                _mm_fnmadd_pd(x2m11, _mm_set1_pd(self.twiddle1.im), m0607b);
                            let m0607b = _mm_fmadd_pd(x3m10, _mm_set1_pd(self.twiddle5.im), m0607b);
                            let m0607b = _mm_fnmadd_pd(x4m9, _mm_set1_pd(self.twiddle2.im), m0607b);
                            let m0607b = _mm_fmadd_pd(x5m8, _mm_set1_pd(self.twiddle4.im), m0607b);
                            let m0607b = _mm_fnmadd_pd(x6m7, _mm_set1_pd(self.twiddle3.im), m0607b);
                            let (y06, y07) = AvxButterfly::butterfly2_f64_m128(m0607a, m0607b);

                            // // Store results
                            _mm_storeu_pd(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            _mm_storeu_pd(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 12..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxFmaRadix13<f64> {
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

#[target_feature(enable = "avx2")]
fn avx_bitreversed_transpose_f32_radix13(
    height: usize,
    input: &[Complex<f32>],
    output: &mut [Complex<f32>],
) {
    let width = input.len() / height;

    if width <= 1 {
        output.copy_from_slice(input);
        return;
    }

    const WIDTH: usize = 13;
    const HEIGHT: usize = 13;

    let rev_digits = compute_logarithm::<13>(width).unwrap();
    let strided_width = width / WIDTH;
    let strided_height = height / HEIGHT;

    for x in 0..strided_width {
        let x_rev = [
            reverse_bits::<WIDTH>(WIDTH * x, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 1, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 2, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 3, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 4, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 5, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 6, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 7, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 8, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 9, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 10, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 11, rev_digits) * height,
            reverse_bits::<WIDTH>(WIDTH * x + 12, rev_digits) * height,
        ];

        let mut cols = [AvxStoreF::zero(); 13];

        // Transposing 13×13 using 4×13 blocks
        //
        // The 13×13 matrix is partitioned into 4-row groups:
        //
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 A0      | 4×4 A1      | 4×4 A2      | 4×4 A3      | 4×4 A4      | 4×4 A5      | 4×1 A6      |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 B0      | 4×4 B1      | 4×4 B2      | 4×4 B3      | 4×4 B4      | 4×4 B5      | 4×1 B6      |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 C0      | 4×4 C1      | 4×4 C2      | 4×4 C3      | 4×4 C4      | 4×4 C5      | 4×1 C6      |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 1×4 D0      | 1×4 D1      | 1×4 D2      | 1×4 D3      | 1×4 D4      | 1×4 D5      | 1×1 D6      |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        //
        // After transposition:
        //
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 A0ᵀ     | 4×4 B0ᵀ     | 4×4 C0ᵀ     | 1×4 D0ᵀ     |             |             |             |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 A1ᵀ     | 4×4 B1ᵀ     | 4×4 C1ᵀ     | 1×4 D1ᵀ     |             |             |             |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 A2ᵀ     | 4×4 B2ᵀ     | 4×4 C2ᵀ     | 1×4 D2ᵀ     |             |             |             |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 A3ᵀ     | 4×4 B3ᵀ     | 4×4 C3ᵀ     | 1×4 D3ᵀ     |             |             |             |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 A4ᵀ     | 4×4 B4ᵀ     | 4×4 C4ᵀ     | 1×4 D4ᵀ     |             |             |             |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 4×4 A5ᵀ     | 4×4 B5ᵀ     | 4×4 C5ᵀ     | 1×4 D5ᵀ     |             |             |             |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+
        // | 1×4 A6ᵀ     | 1×4 B6ᵀ     | 1×4 C6ᵀ     | 1×1 D6ᵀ     |             |             |             |
        // +-------------+-------------+-------------+-------------+-------------+-------------+-------------+

        for y in 0..strided_height {
            let base_input_idx = (WIDTH * x) + y * HEIGHT * width;
            unsafe {
                for k in 0..3 {
                    for i in 0..13 {
                        cols[i] = AvxStoreF::from_complex_ref(
                            input.get_unchecked(base_input_idx + k * 4 + width * i..),
                        );
                    }
                    let x0 = x_rev[k * 4];
                    let x1 = x_rev[k * 4 + 1];
                    let x2 = x_rev[k * 4 + 2];
                    let x3 = x_rev[k * 4 + 3];
                    let transposed = transpose_4x13(cols);
                    for i in 0..3 {
                        transposed[i * 4]
                            .write(output.get_unchecked_mut(HEIGHT * y + x0 + i * 4..));
                        transposed[i * 4 + 1]
                            .write(output.get_unchecked_mut(HEIGHT * y + x1 + i * 4..));
                        transposed[i * 4 + 2]
                            .write(output.get_unchecked_mut(HEIGHT * y + x2 + i * 4..));
                        transposed[i * 4 + 3]
                            .write(output.get_unchecked_mut(HEIGHT * y + x3 + i * 4..));
                    }

                    transposed[12].write_lo1(output.get_unchecked_mut(HEIGHT * y + x0 + 12..));
                    transposed[13].write_lo1(output.get_unchecked_mut(HEIGHT * y + x1 + 12..));
                    transposed[14].write_lo1(output.get_unchecked_mut(HEIGHT * y + x2 + 12..));
                    transposed[15].write_lo1(output.get_unchecked_mut(HEIGHT * y + x3 + 12..));
                }

                {
                    let k = 3;
                    for i in 0..13 {
                        cols[i] = AvxStoreF::from_complex(
                            input.get_unchecked(base_input_idx + k * 4 + width * i),
                        );
                    }
                    let x0 = x_rev[k * 4];
                    let transposed = transpose_4x13(cols);
                    for i in 0..3 {
                        transposed[i * 4]
                            .write(output.get_unchecked_mut(HEIGHT * y + x0 + i * 4..));
                    }

                    transposed[12].write_lo1(output.get_unchecked_mut(HEIGHT * y + x0 + 12..));
                }
            }
        }
    }
}

impl AvxFmaRadix13<f32> {
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
                avx_bitreversed_transpose_f32_radix13(13, chunk, &mut scratch);

                self.butterfly.execute_out_of_place(&scratch, chunk)?;

                let mut len = 13;

                let mut m_twiddles = self.twiddles.as_slice();

                while len < self.execution_length {
                    let columns = len;
                    len *= 13;
                    let thirteenth = len / 13;

                    for data in chunk.chunks_exact_mut(len) {
                        let mut j = 0usize;

                        while j + 4 < thirteenth {
                            let u0 = _mm256_loadu_ps(data.get_unchecked(j..).as_ptr().cast());

                            let twi = 12 * j;
                            let tw0 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw1 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw2 = _mm256_loadu_ps(
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

                            let u1 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + thirteenth..).as_ptr().cast(),
                                ),
                                tw0,
                            );
                            let u2 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 2 * thirteenth..).as_ptr().cast(),
                                ),
                                tw1,
                            );
                            let u3 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 3 * thirteenth..).as_ptr().cast(),
                                ),
                                tw2,
                            );
                            let u4 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 4 * thirteenth..).as_ptr().cast(),
                                ),
                                tw3,
                            );
                            let u5 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 5 * thirteenth..).as_ptr().cast(),
                                ),
                                tw4,
                            );
                            let u6 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 6 * thirteenth..).as_ptr().cast(),
                                ),
                                tw5,
                            );

                            let tw7 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 24..).as_ptr().cast(),
                            );
                            let tw8 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 28..).as_ptr().cast(),
                            );
                            let tw9 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 32..).as_ptr().cast(),
                            );
                            let tw10 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 36..).as_ptr().cast(),
                            );
                            let tw11 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 40..).as_ptr().cast(),
                            );
                            let tw12 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 44..).as_ptr().cast(),
                            );

                            let u7 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 7 * thirteenth..).as_ptr().cast(),
                                ),
                                tw7,
                            );

                            let u8 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 8 * thirteenth..).as_ptr().cast(),
                                ),
                                tw8,
                            );

                            let u9 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 9 * thirteenth..).as_ptr().cast(),
                                ),
                                tw9,
                            );

                            let u10 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 10 * thirteenth..).as_ptr().cast(),
                                ),
                                tw10,
                            );

                            let u11 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 11 * thirteenth..).as_ptr().cast(),
                                ),
                                tw11,
                            );

                            let u12 = _mm256_fcmul_ps(
                                _mm256_loadu_ps(
                                    data.get_unchecked(j + 12 * thirteenth..).as_ptr().cast(),
                                ),
                                tw12,
                            );

                            let y00 = u0;
                            let (x1p12, x1m12) = AvxButterfly::butterfly2_f32(u1, u12); // u1, u12
                            let x1m12 = rotate.rotate_m256(x1m12);
                            let y00 = _mm256_add_ps(y00, x1p12);
                            let (x2p11, x2m11) = AvxButterfly::butterfly2_f32(u2, u11); // u2, u11
                            let x2m11 = rotate.rotate_m256(x2m11);
                            let y00 = _mm256_add_ps(y00, x2p11);
                            let (x3p10, x3m10) = AvxButterfly::butterfly2_f32(u3, u10); // u3, u10
                            let x3m10 = rotate.rotate_m256(x3m10);
                            let y00 = _mm256_add_ps(y00, x3p10);
                            let (x4p9, x4m9) = AvxButterfly::butterfly2_f32(u4, u9); // u4, u9
                            let x4m9 = rotate.rotate_m256(x4m9);
                            let y00 = _mm256_add_ps(y00, x4p9);
                            let (x5p8, x5m8) = AvxButterfly::butterfly2_f32(u5, u8); // u5, u8
                            let x5m8 = rotate.rotate_m256(x5m8);
                            let y00 = _mm256_add_ps(y00, x5p8);
                            let (x6p7, x6m7) = AvxButterfly::butterfly2_f32(u6, u7); // u6, u7
                            let x6m7 = rotate.rotate_m256(x6m7);
                            let y00 = _mm256_add_ps(y00, x6p7);

                            let m0112a =
                                _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle1.re), u0);
                            let m0112a =
                                _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle2.re), x2p11, m0112a);
                            let m0112a =
                                _mm256_fmadd_ps(_mm256_set1_ps(self.twiddle3.re), x3p10, m0112a);
                            let m0112a =
                                _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle4.re), m0112a);
                            let m0112a =
                                _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle5.re), m0112a);
                            let m0112a =
                                _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle6.re), m0112a);
                            let m0112b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle1.im));
                            let m0112b =
                                _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle2.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle3.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_ps(x4m9, _mm256_set1_ps(self.twiddle4.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle5.im), m0112b);
                            let m0112b =
                                _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle6.im), m0112b);
                            let (y01, y12) = AvxButterfly::butterfly2_f32(m0112a, m0112b);

                            let m0211a =
                                _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle2.re), u0);
                            let m0211a =
                                _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle4.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle6.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle5.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle3.re), m0211a);
                            let m0211a =
                                _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle1.re), m0211a);
                            let m0211b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle2.im));
                            let m0211b =
                                _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle4.im), m0211b);
                            let m0211b =
                                _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle6.im), m0211b);
                            let m0211b =
                                _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle5.im), m0211b);
                            let m0211b =
                                _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle3.im), m0211b);
                            let m0211b =
                                _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle1.im), m0211b);
                            let (y02, y11) = AvxButterfly::butterfly2_f32(m0211a, m0211b);

                            let m0310a =
                                _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle3.re), u0);
                            let m0310a =
                                _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle6.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle4.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle1.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle2.re), m0310a);
                            let m0310a =
                                _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle5.re), m0310a);
                            let m0310b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle3.im));
                            let m0310b =
                                _mm256_fmadd_ps(x2m11, _mm256_set1_ps(self.twiddle6.im), m0310b);
                            let m0310b =
                                _mm256_fnmadd_ps(x3m10, _mm256_set1_ps(self.twiddle4.im), m0310b);
                            let m0310b =
                                _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle1.im), m0310b);
                            let m0310b =
                                _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle2.im), m0310b);
                            let m0310b =
                                _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle5.im), m0310b);
                            let (y03, y10) = AvxButterfly::butterfly2_f32(m0310a, m0310b);

                            let m0409a =
                                _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle4.re), u0);
                            let m0409a =
                                _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle5.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle1.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle3.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle6.re), m0409a);
                            let m0409a =
                                _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle2.re), m0409a);
                            let m0409b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle4.im));
                            let m0409b =
                                _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle5.im), m0409b);
                            let m0409b =
                                _mm256_fnmadd_ps(x3m10, _mm256_set1_ps(self.twiddle1.im), m0409b);
                            let m0409b =
                                _mm256_fmadd_ps(x4m9, _mm256_set1_ps(self.twiddle3.im), m0409b);
                            let m0409b =
                                _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle6.im), m0409b);
                            let m0409b =
                                _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle2.im), m0409b);
                            let (y04, y09) = AvxButterfly::butterfly2_f32(m0409a, m0409b);

                            let m0508a =
                                _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle5.re), u0);
                            let m0508a =
                                _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle3.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle2.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle6.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle1.re), m0508a);
                            let m0508a =
                                _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle4.re), m0508a);
                            let m0508b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle5.im));
                            let m0508b =
                                _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle3.im), m0508b);
                            let m0508b =
                                _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle2.im), m0508b);
                            let m0508b =
                                _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle6.im), m0508b);
                            let m0508b =
                                _mm256_fnmadd_ps(x5m8, _mm256_set1_ps(self.twiddle1.im), m0508b);
                            let m0508b =
                                _mm256_fmadd_ps(x6m7, _mm256_set1_ps(self.twiddle4.im), m0508b);
                            let (y05, y08) = AvxButterfly::butterfly2_f32(m0508a, m0508b);

                            let m0607a =
                                _mm256_fmadd_ps(x1p12, _mm256_set1_ps(self.twiddle6.re), u0);
                            let m0607a =
                                _mm256_fmadd_ps(x2p11, _mm256_set1_ps(self.twiddle1.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_ps(x3p10, _mm256_set1_ps(self.twiddle5.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_ps(x4p9, _mm256_set1_ps(self.twiddle2.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_ps(x5p8, _mm256_set1_ps(self.twiddle4.re), m0607a);
                            let m0607a =
                                _mm256_fmadd_ps(x6p7, _mm256_set1_ps(self.twiddle3.re), m0607a);
                            let m0607b = _mm256_mul_ps(x1m12, _mm256_set1_ps(self.twiddle6.im));
                            let m0607b =
                                _mm256_fnmadd_ps(x2m11, _mm256_set1_ps(self.twiddle1.im), m0607b);
                            let m0607b =
                                _mm256_fmadd_ps(x3m10, _mm256_set1_ps(self.twiddle5.im), m0607b);
                            let m0607b =
                                _mm256_fnmadd_ps(x4m9, _mm256_set1_ps(self.twiddle2.im), m0607b);
                            let m0607b =
                                _mm256_fmadd_ps(x5m8, _mm256_set1_ps(self.twiddle4.im), m0607b);
                            let m0607b =
                                _mm256_fnmadd_ps(x6m7, _mm256_set1_ps(self.twiddle3.im), m0607b);
                            let (y06, y07) = AvxButterfly::butterfly2_f32(m0607a, m0607b);

                            // // Store results
                            _mm256_storeu_ps(data.get_unchecked_mut(j..).as_mut_ptr().cast(), y00);
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            _mm256_storeu_ps(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );

                            j += 4;
                        }

                        for j in j..thirteenth {
                            let u0 = _m128s_load_f32x2(data.get_unchecked(j..).as_ptr().cast());
                            let twi = 12 * j;
                            let tw0tw1tw2tw3 =
                                _mm256_loadu_ps(m_twiddles.get_unchecked(twi..).as_ptr().cast());
                            let tw4tw5tw6tw7 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 4..).as_ptr().cast(),
                            );
                            let tw8tw9tw10tw11 = _mm256_loadu_ps(
                                m_twiddles.get_unchecked(twi + 8..).as_ptr().cast(),
                            );

                            let u1u2u3u4 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + thirteenth..),
                                    data.get_unchecked(j + 2 * thirteenth..),
                                    data.get_unchecked(j + 3 * thirteenth..),
                                    data.get_unchecked(j + 4 * thirteenth..),
                                ),
                                tw0tw1tw2tw3,
                            );
                            let u5u6u7u8 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + 5 * thirteenth..),
                                    data.get_unchecked(j + 6 * thirteenth..),
                                    data.get_unchecked(j + 7 * thirteenth..),
                                    data.get_unchecked(j + 8 * thirteenth..),
                                ),
                                tw4tw5tw6tw7,
                            );

                            let u9u10u11u12 = _mm256_fcmul_ps(
                                _mm256_load4_f32x2(
                                    data.get_unchecked(j + 9 * thirteenth..),
                                    data.get_unchecked(j + 10 * thirteenth..),
                                    data.get_unchecked(j + 11 * thirteenth..),
                                    data.get_unchecked(j + 12 * thirteenth..),
                                ),
                                tw8tw9tw10tw11,
                            );

                            let u1u2 = _mm256_castps256_ps128(u1u2u3u4);
                            let u3u4 = _mm256_extractf128_ps::<1>(u1u2u3u4);
                            let u9u10 = _mm256_castps256_ps128(u9u10u11u12);
                            let u5u6 = _mm256_castps256_ps128(u5u6u7u8);
                            let u7u8 = _mm256_extractf128_ps::<1>(u5u6u7u8);
                            let u11u12 = _mm256_extractf128_ps::<1>(u9u10u11u12);

                            let y00 = u0;
                            let (x1p12, x1m12) = AvxButterfly::butterfly2_f32_m128(
                                u1u2,
                                _mm_unpackhi_ps64(u11u12, u11u12),
                            ); // u1, u12
                            let x1m12 = rotate.rotate_m128(x1m12);
                            let y00 = _mm_add_ps(y00, x1p12);
                            let (x2p11, x2m11) = AvxButterfly::butterfly2_f32_m128(
                                _mm_unpackhi_ps64(u1u2, u1u2),
                                u11u12,
                            ); // u2, u11
                            let x2m11 = rotate.rotate_m128(x2m11);
                            let y00 = _mm_add_ps(y00, x2p11);
                            let (x3p10, x3m10) = AvxButterfly::butterfly2_f32_m128(
                                u3u4,
                                _mm_unpackhi_ps64(u9u10, u9u10),
                            ); // u3, u10
                            let x3m10 = rotate.rotate_m128(x3m10);
                            let y00 = _mm_add_ps(y00, x3p10);
                            let (x4p9, x4m9) = AvxButterfly::butterfly2_f32_m128(
                                _mm_unpackhi_ps64(u3u4, u3u4),
                                u9u10,
                            ); // u4, u9
                            let x4m9 = rotate.rotate_m128(x4m9);
                            let y00 = _mm_add_ps(y00, x4p9);
                            let (x5p8, x5m8) = AvxButterfly::butterfly2_f32_m128(
                                u5u6,
                                _mm_unpackhi_ps64(u7u8, u7u8),
                            ); // u5, u8
                            let x5m8 = rotate.rotate_m128(x5m8);
                            let y00 = _mm_add_ps(y00, x5p8);
                            let (x6p7, x6m7) = AvxButterfly::butterfly2_f32_m128(
                                _mm_unpackhi_ps64(u5u6, u5u6),
                                u7u8,
                            ); // u6, u7
                            let x6m7 = rotate.rotate_m128(x6m7);
                            let y00 = _mm_add_ps(y00, x6p7);

                            let m0112a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle1.re), u0);
                            let m0112a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle2.re), x2p11, m0112a);
                            let m0112a = _mm_fmadd_ps(_mm_set1_ps(self.twiddle3.re), x3p10, m0112a);
                            let m0112a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle4.re), m0112a);
                            let m0112a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle5.re), m0112a);
                            let m0112a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle6.re), m0112a);
                            let m0112b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle1.im));
                            let m0112b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle2.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle3.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x4m9, _mm_set1_ps(self.twiddle4.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle5.im), m0112b);
                            let m0112b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle6.im), m0112b);
                            let (y01, y12) = AvxButterfly::butterfly2_f32_m128(m0112a, m0112b);

                            let m0211a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle2.re), u0);
                            let m0211a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle4.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle6.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle5.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle3.re), m0211a);
                            let m0211a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle1.re), m0211a);
                            let m0211b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle2.im));
                            let m0211b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle4.im), m0211b);
                            let m0211b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle6.im), m0211b);
                            let m0211b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle5.im), m0211b);
                            let m0211b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle3.im), m0211b);
                            let m0211b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle1.im), m0211b);
                            let (y02, y11) = AvxButterfly::butterfly2_f32_m128(m0211a, m0211b);

                            let m0310a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle3.re), u0);
                            let m0310a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle6.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle4.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle1.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle2.re), m0310a);
                            let m0310a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle5.re), m0310a);
                            let m0310b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle3.im));
                            let m0310b = _mm_fmadd_ps(x2m11, _mm_set1_ps(self.twiddle6.im), m0310b);
                            let m0310b =
                                _mm_fnmadd_ps(x3m10, _mm_set1_ps(self.twiddle4.im), m0310b);
                            let m0310b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle1.im), m0310b);
                            let m0310b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle2.im), m0310b);
                            let m0310b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle5.im), m0310b);
                            let (y03, y10) = AvxButterfly::butterfly2_f32_m128(m0310a, m0310b);

                            let m0409a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle4.re), u0);
                            let m0409a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle5.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle1.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle3.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle6.re), m0409a);
                            let m0409a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle2.re), m0409a);
                            let m0409b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle4.im));
                            let m0409b =
                                _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle5.im), m0409b);
                            let m0409b =
                                _mm_fnmadd_ps(x3m10, _mm_set1_ps(self.twiddle1.im), m0409b);
                            let m0409b = _mm_fmadd_ps(x4m9, _mm_set1_ps(self.twiddle3.im), m0409b);
                            let m0409b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle6.im), m0409b);
                            let m0409b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle2.im), m0409b);
                            let (y04, y09) = AvxButterfly::butterfly2_f32_m128(m0409a, m0409b);

                            let m0508a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle5.re), u0);
                            let m0508a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle3.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle2.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle6.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle1.re), m0508a);
                            let m0508a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle4.re), m0508a);
                            let m0508b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle5.im));
                            let m0508b =
                                _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle3.im), m0508b);
                            let m0508b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle2.im), m0508b);
                            let m0508b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle6.im), m0508b);
                            let m0508b = _mm_fnmadd_ps(x5m8, _mm_set1_ps(self.twiddle1.im), m0508b);
                            let m0508b = _mm_fmadd_ps(x6m7, _mm_set1_ps(self.twiddle4.im), m0508b);
                            let (y05, y08) = AvxButterfly::butterfly2_f32_m128(m0508a, m0508b);

                            let m0607a = _mm_fmadd_ps(x1p12, _mm_set1_ps(self.twiddle6.re), u0);
                            let m0607a = _mm_fmadd_ps(x2p11, _mm_set1_ps(self.twiddle1.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x3p10, _mm_set1_ps(self.twiddle5.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x4p9, _mm_set1_ps(self.twiddle2.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x5p8, _mm_set1_ps(self.twiddle4.re), m0607a);
                            let m0607a = _mm_fmadd_ps(x6p7, _mm_set1_ps(self.twiddle3.re), m0607a);
                            let m0607b = _mm_mul_ps(x1m12, _mm_set1_ps(self.twiddle6.im));
                            let m0607b =
                                _mm_fnmadd_ps(x2m11, _mm_set1_ps(self.twiddle1.im), m0607b);
                            let m0607b = _mm_fmadd_ps(x3m10, _mm_set1_ps(self.twiddle5.im), m0607b);
                            let m0607b = _mm_fnmadd_ps(x4m9, _mm_set1_ps(self.twiddle2.im), m0607b);
                            let m0607b = _mm_fmadd_ps(x5m8, _mm_set1_ps(self.twiddle4.im), m0607b);
                            let m0607b = _mm_fnmadd_ps(x6m7, _mm_set1_ps(self.twiddle3.im), m0607b);
                            let (y06, y07) = AvxButterfly::butterfly2_f32_m128(m0607a, m0607b);

                            // Store results
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j..).as_mut_ptr().cast(),
                                y00,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + thirteenth..).as_mut_ptr().cast(),
                                y01,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 2 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y02,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 3 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y03,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 4 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y04,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 5 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y05,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 6 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y06,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 7 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y07,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 8 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y08,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 9 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y09,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 10 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y10,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 11 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y11,
                            );
                            _m128s_store_f32x2(
                                data.get_unchecked_mut(j + 12 * thirteenth..)
                                    .as_mut_ptr()
                                    .cast(),
                                y12,
                            );
                        }
                    }

                    m_twiddles = &m_twiddles[columns * 12..];
                }
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxFmaRadix13<f32> {
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
    use crate::avx::test_avx_radix;

    test_avx_radix!(test_avx_radix13, f32, AvxFmaRadix13, 3, 13, 1e-3);
    test_avx_radix!(test_avx_radix13_f64, f64, AvxFmaRadix13, 3, 13, 1e-8);
}
