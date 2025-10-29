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

mod c2r;
mod c2r_twiddles;
mod r2c_twiddles;
mod real_to_complex;

use crate::ZaftError;
pub use c2r::C2RFftExecutor;
pub(crate) use c2r::{C2RFftEvenInterceptor, C2RFftOddInterceptor};
use num_complex::Complex;
use num_traits::AsPrimitive;
pub(crate) use r2c_twiddles::R2CTwiddlesHandler;
pub use real_to_complex::R2CFftExecutor;
pub(crate) use real_to_complex::{R2CFftEvenInterceptor, R2CFftOddInterceptor};
use std::marker::PhantomData;

pub(crate) struct OneSizedRealFft<T> {
    pub(crate) phantom_data: PhantomData<T>,
}

impl<T: Copy + 'static> R2CFftExecutor<T> for OneSizedRealFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if input.is_empty() {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 1));
        }
        if output.is_empty() {
            return Err(ZaftError::InvalidSizeMultiplier(output.len(), 1));
        }
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = Complex::new(*src, 0.0f64.as_())
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        1
    }

    fn complex_length(&self) -> usize {
        1
    }
}

impl<T: Copy + 'static> C2RFftExecutor<T> for OneSizedRealFft<T> {
    fn execute(&self, input: &[Complex<T>], output: &mut [T]) -> Result<(), ZaftError> {
        if input.is_empty() {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 1));
        }
        if output.is_empty() {
            return Err(ZaftError::InvalidSizeMultiplier(output.len(), 1));
        }
        for (dst, src) in output.iter_mut().zip(input.iter()) {
            *dst = src.re
        }
        Ok(())
    }

    fn complex_length(&self) -> usize {
        1
    }

    fn real_length(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use rand::Rng;

    #[test]
    fn test_r2c_and_c2r() {
        for i in 1..60 {
            let data = (0..i)
                .map(|_| {
                    Complex::<f32>::new(
                        rand::rng().random_range(-1.0..1.0),
                        rand::rng().random_range(0.0..1.0),
                    )
                })
                .collect::<Vec<_>>();

            let mut real_data = data.iter().map(|x| x.re).collect::<Vec<_>>();
            let real_data_ref = real_data.clone();

            let forward_r2c = Zaft::make_r2c_fft_f32(data.len()).unwrap();
            let inverse_r2c = Zaft::make_c2r_fft_f32(data.len()).unwrap();

            let mut complex_data = vec![Complex::<f32>::default(); data.len() / 2 + 1];
            forward_r2c.execute(&real_data, &mut complex_data).unwrap();
            inverse_r2c.execute(&complex_data, &mut real_data).unwrap();

            real_data = real_data
                .iter()
                .map(|&x| x * (1.0 / real_data.len() as f32))
                .collect();

            real_data
                .iter()
                .zip(real_data_ref)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a - b).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx} at size {}",
                        a,
                        b,
                        data.len()
                    );
                });
        }
    }

    #[test]
    fn test_r2c_and_c2r_f64() {
        for i in 1..60 {
            let data = (0..i)
                .map(|_| {
                    Complex::<f64>::new(
                        rand::rng().random_range(-1.0..1.0),
                        rand::rng().random_range(0.0..1.0),
                    )
                })
                .collect::<Vec<_>>();

            let mut real_data = data.iter().map(|x| x.re).collect::<Vec<_>>();
            let real_data_ref = real_data.clone();

            let forward_r2c = Zaft::make_r2c_fft_f64(data.len()).unwrap();
            let inverse_r2c = Zaft::make_c2r_fft_f64(data.len()).unwrap();

            let mut complex_data = vec![Complex::<f64>::default(); data.len() / 2 + 1];
            forward_r2c.execute(&real_data, &mut complex_data).unwrap();
            inverse_r2c.execute(&complex_data, &mut real_data).unwrap();

            real_data = real_data
                .iter()
                .map(|&x| x * (1.0 / real_data.len() as f64))
                .collect();

            real_data
                .iter()
                .zip(real_data_ref)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a - b).abs() < 1e-7,
                        "a_re {}, b_re {} at {idx} at size {}",
                        a,
                        b,
                        data.len()
                    );
                });
        }
    }
}
