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
use crate::err::try_vec;
use crate::spectrum_arithmetic::{SpectrumOps, SpectrumOpsFactory};
use crate::traits::FftTrigonometry;
use crate::transpose::{TransposeExecutor, TransposeFactory};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num, Zero};
use std::fmt::Display;
use std::ops::{Add, Mul, Neg, Sub};

pub(crate) struct MixedRadix<T> {
    execution_length: usize,
    direction: FftDirection,
    twiddles: Vec<Complex<T>>,
    width_executor: Box<dyn FftExecutor<T> + Send + Sync>,
    width: usize,
    height_executor: Box<dyn FftExecutor<T> + Send + Sync>,
    height: usize,
    spectrum_ops: Box<dyn SpectrumOps<T> + Send + Sync>,
    transpose_executor: Box<dyn TransposeExecutor<T> + Send + Sync>,
}

impl<T: Copy + 'static + FftTrigonometry + Float + SpectrumOpsFactory<T> + TransposeFactory<T>>
    MixedRadix<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        width_executor: Box<dyn FftExecutor<T> + Send + Sync>,
        height_executor: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Self, ZaftError> {
        assert_eq!(
            width_executor.direction(),
            height_executor.direction(),
            "width_fft and height_fft must have the same direction. got width direction={}, height direction={}",
            width_executor.direction(),
            height_executor.direction()
        );

        let direction = width_executor.direction();

        let width = width_executor.length();
        let height = height_executor.length();

        let len = width * height;

        let mut twiddles = try_vec![Complex::zero(); len];
        for (x, row) in twiddles.chunks_exact_mut(height).enumerate() {
            for (y, dst) in row.iter_mut().enumerate() {
                *dst = compute_twiddle(x * y, len, direction);
            }
        }

        Ok(MixedRadix {
            execution_length: width * height,
            width_executor,
            width,
            height_executor,
            height,
            direction,
            twiddles,
            spectrum_ops: T::make_spectrum_arithmetic(),
            transpose_executor: T::transpose_strategy(width, height),
        })
    }
}

impl<
    T: Copy
        + Mul<T, Output = T>
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Num
        + 'static
        + Neg<Output = T>
        + MulAdd<T, Output = T>
        + Display,
> FftExecutor<T> for MixedRadix<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if in_place.len() % self.execution_length != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let mut scratch = try_vec![Complex::zero(); self.execution_length];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            // STEP 1: transpose
            self.transpose_executor
                .transpose(chunk, &mut scratch, self.width, self.height);

            // STEP 2: perform FFTs of size `height`
            self.height_executor.execute(&mut scratch)?;

            // STEP 3: Apply twiddle factors
            self.spectrum_ops.mul(&scratch, &self.twiddles, chunk);

            // STEP 4: transpose again
            self.transpose_executor
                .transpose(chunk, &mut scratch, self.height, self.width);

            // STEP 5: perform FFTs of size `width`
            self.width_executor.execute(&mut scratch)?;

            // STEP 6: transpose again
            self.transpose_executor
                .transpose(&scratch, chunk, self.width, self.height);
        }
        Ok(())
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
    use crate::good_thomas_small::GoodThomasSmallFft;
    use crate::mixed_radix::MixedRadix;
    use crate::{FftDirection, FftExecutor, Zaft};
    use num_complex::Complex;

    #[test]
    fn test_mixed_radixd() {
        let src: [Complex<f64>; 22] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
        ];
        let good_thomas20 = GoodThomasSmallFft::new(
            Zaft::strategy(11, FftDirection::Forward).unwrap(),
            Zaft::strategy(2, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mx = MixedRadix::new(
            Zaft::strategy(11, FftDirection::Forward).unwrap(),
            Zaft::strategy(2, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        let mut reference_value = src.to_vec();
        good_thomas20.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        mx.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }
}
