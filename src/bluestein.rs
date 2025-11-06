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
use crate::err::try_vec;
use crate::fast_divider::DividerU64;
use crate::spectrum_arithmetic::{SpectrumOps, SpectrumOpsFactory};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd, Num, Zero};
use std::ops::{Add, Mul, Neg, Rem, Sub};

pub(crate) struct BluesteinFft<T> {
    convolve_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    spectrum_ops: Box<dyn SpectrumOps<T> + Send + Sync>,
}

fn make_bluesteins_twiddles<T: Float + FftTrigonometry + 'static>(
    destination: &mut [Complex<T>],
    direction: FftDirection,
) where
    f64: AsPrimitive<T>,
{
    let twice_len = destination.len() * 2;

    // Standard bluestein's twiddle computation requires us to square the index before usingit to compute a twiddle factor
    // And since twiddle factors are cyclic, we can improve precision once the squared index gets converted to floating point by taking a modulo
    // Modulo is expensive, so we're going to use strength-reduction to keep it manageable

    // Strength-reduced u128s are very heavy, so we only want to use them if we need them - and we only need them if
    // len * len doesn't fit in a u64, AKA if len doesn't fit in a u32
    if destination.len() < u32::MAX as usize {
        let twice_len_reduced = DividerU64::new(twice_len as u64);

        for (i, e) in destination.iter_mut().enumerate() {
            let i_squared = i as u64 * i as u64;
            let i_mod = i_squared % twice_len_reduced;
            *e = compute_twiddle(i_mod as usize, twice_len, direction);
        }
    } else {
        // Sadly, the len doesn't fit in a u64, so we have to crank it up to u128 arithmetic
        let twice_len_reduced = twice_len as u128;

        for (i, e) in destination.iter_mut().enumerate() {
            // Standard bluestein's twiddle computation requires us to square the index before usingit to compute a twiddle factor
            // And since twiddle factors are cyclic, we can improve precision once the squared index gets converted to floating point by taking a modulo
            let i_squared = i as u128 * i as u128;
            let i_mod = i_squared.rem(twice_len_reduced);
            *e = compute_twiddle(i_mod as usize, twice_len, direction);
        }
    }
}

impl<
    T: Copy
        + Default
        + Clone
        + FftTrigonometry
        + Float
        + Zero
        + Default
        + SpectrumOpsFactory<T>
        + 'static,
> BluesteinFft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        size: usize,
        convolve_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<BluesteinFft<T>, ZaftError> {
        let convolve_fft_len = convolve_fft.length();
        assert!(
            size * 2 - 1 <= convolve_fft_len,
            "Bluestein requires convolve_fft.length() >= self.length() * 2 - 1. Expected >= {}, got {}",
            size * 2 - 1,
            convolve_fft_len
        );

        // when computing FFTs, we're going to run our inner multiply pairwise by some precomputed data, then run an inverse inner FFT. We need to precompute that inner data here
        let inner_fft_scale = (1f64 / convolve_fft_len as f64).as_();
        let direction = convolve_fft.direction();
        assert_eq!(
            direction, fft_direction,
            "Convolve FFT may not go with other direction"
        );

        // Compute twiddle factors that we'll run our inner FFT on
        let mut convolve_fft_twiddles = try_vec![Complex::zero(); convolve_fft_len];
        make_bluesteins_twiddles(&mut convolve_fft_twiddles[..size], direction.inverse());

        // Scale the computed twiddles and copy them to the end of the array
        convolve_fft_twiddles[0] = convolve_fft_twiddles[0] * inner_fft_scale;
        for i in 1..size {
            let twiddle = convolve_fft_twiddles[i] * inner_fft_scale;
            convolve_fft_twiddles[i] = twiddle;
            convolve_fft_twiddles[convolve_fft_len - i] = twiddle;
        }

        //Compute the inner fft
        convolve_fft.execute(&mut convolve_fft_twiddles)?;

        // also compute some more mundane twiddle factors to start and end with
        let mut twiddles = try_vec![Complex::zero(); size];
        make_bluesteins_twiddles(&mut twiddles, direction);

        Ok(BluesteinFft {
            convolve_fft,
            convolve_fft_twiddles,
            twiddles,
            execution_length: size,
            direction,
            spectrum_ops: T::make_spectrum_arithmetic(),
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
        + MulAdd<T, Output = T>,
> FftExecutor<T> for BluesteinFft<T>
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

        let mut scratch = try_vec![Complex::zero(); self.convolve_fft_twiddles.len()];

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            let (inner_input, _) = scratch.split_at_mut(self.convolve_fft_twiddles.len());

            // Copy the buffer into our inner FFT input. the buffer will only fill part of the FFT input, so zero fill the rest
            self.spectrum_ops
                .mul(chunk, &self.twiddles, &mut inner_input[..chunk.len()]);

            for inner in inner_input[chunk.len()..].iter_mut() {
                *inner = Complex::zero();
            }

            // run our inner forward FFT
            self.convolve_fft.execute(inner_input)?;

            // Multiply our inner FFT output by our precomputed data. Then, conjugate the result to set up for an inverse FFT
            self.spectrum_ops
                .mul_conjugate_in_place(inner_input, &self.convolve_fft_twiddles);

            // inverse FFT. we're computing a forward but we're massaging it into an inverse by conjugating the inputs and outputs
            self.convolve_fft.execute(inner_input)?;

            // copy our data back to the buffer, applying twiddle factors again as we go. Also conjugate inner_input to complete the inverse FFT
            self.spectrum_ops.conjugate_mul_by_b(
                &inner_input[..chunk.len()],
                &self.twiddles,
                chunk,
            );
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
