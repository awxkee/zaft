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
mod factory_d;
mod factory_f;
mod mixed_radix_r2c_odd;
mod r2c_twiddles;
mod real_factory;
mod real_to_complex;
mod rfft_bluestein;
mod rfft_raders;
mod strategy;

use crate::ZaftError;
pub use c2r::C2RFftExecutor;
pub(crate) use c2r::{C2RFftEvenInterceptor, C2RFftOddInterceptor};
use num_complex::Complex;
use num_traits::AsPrimitive;
pub(crate) use r2c_twiddles::{R2CTwiddlesFactory, R2CTwiddlesHandler};
pub(crate) use real_factory::R2cAlgorithmFactory;
pub(crate) use real_to_complex::R2CFftEvenInterceptor;
pub use real_to_complex::R2CFftExecutor;
use std::marker::PhantomData;
pub(crate) use strategy::strategy_r2c;

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

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        R2CFftExecutor::execute(self, input, output)
    }

    fn real_length(&self) -> usize {
        1
    }

    fn complex_length(&self) -> usize {
        1
    }

    fn complex_scratch_length(&self) -> usize {
        0
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

    fn execute_with_scratch(
        &self,
        input: &[Complex<T>],
        output: &mut [T],
        _: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute(input, output)
    }

    fn complex_length(&self) -> usize {
        1
    }

    fn real_length(&self) -> usize {
        1
    }

    fn complex_scratch_length(&self) -> usize {
        0
    }
}

#[cfg(test)]
macro_rules! test_r2c_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use rand::Rng;
            let radix_forward = $butterfly::new(FftDirection::Forward);
            assert_eq!(radix_forward.real_length(), $scale);
            assert_eq!(radix_forward.complex_length(), $scale / 2 + 1);
            for i in 1..20 {
                let val = $scale as usize;
                let size = val * i;
                let mut input = vec![$data_type::default(); size];
                for z in input.iter_mut() {
                    *z = rand::rng().random();
                }
                let src = input.to_vec();
                use crate::dft::Dft;
                use crate::{FftDirection, FftExecutor};
                let reference_forward = Dft::new($scale, FftDirection::Forward).unwrap();

                let mut ref_src = src.iter().map(|x| Complex::new(*x, 0.)).collect::<Vec<_>>();
                reference_forward.execute(&mut ref_src).unwrap();

                let mut output = vec![Complex::<$data_type>::default(); ($scale / 2 + 1) * i];

                R2CFftExecutor::execute(&radix_forward, &input, &mut output).unwrap();

                let ref_src = ref_src
                    .chunks_exact($scale)
                    .flat_map(|x| (&x[..$scale / 2 + 1]).to_vec())
                    .collect::<Vec<_>>();

                output
                    .iter()
                    .zip(ref_src.iter())
                    .enumerate()
                    .for_each(|(idx, (a, b))| {
                        assert!(
                            (a.re - b.re).abs() < $tol,
                            "a_re {} != b_re {} for size {} at {idx}",
                            a.re,
                            b.re,
                            size
                        );
                        assert!(
                            (a.im - b.im).abs() < $tol,
                            "a_im {} != b_im {} for size {} at {idx}",
                            a.im,
                            b.im,
                            size
                        );
                    });
            }
        }
    };
}

#[cfg(test)]
pub(crate) use test_r2c_butterfly;

#[cfg(test)]
mod tests {
    use crate::*;
    use rand::Rng;

    #[test]
    fn test_r2c_and_c2r() {
        for i in 1..180 {
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
            forward_r2c
                .execute(&real_data, &mut complex_data)
                .expect(&format!("R2C Failed for size {i}"));
            inverse_r2c
                .execute(&complex_data, &mut real_data)
                .expect(&format!("R2C Failed for size {i}"));

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
        for i in 1..128 {
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
            forward_r2c
                .execute(&real_data, &mut complex_data)
                .expect(&format!("R2C Failed for size {i}"));
            inverse_r2c
                .execute(&complex_data, &mut real_data)
                .expect(&format!("R2C Failed for size {i}"));

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
