/*
 * // Copyright (c) Radzivon Bartoshyk 6/2025. All rights reserved.
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
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    all(feature = "fcma", target_arch = "aarch64"),
    feature(stdarch_neon_fcma)
)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod butterflies;
mod complex_fma;
mod dft;
mod err;
mod factory;
mod factory64;
mod mixed_radix;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod prime_factors;
mod radix2;
mod radix3;
mod radix4;
mod radix5;
mod radix6;
mod spectrum_arithmetic;
mod traits;
mod util;

pub use err::ZaftError;
use std::fmt::{Display, Formatter};

use crate::factory::AlgorithmFactory;
use crate::mixed_radix::MixedRadix;
use crate::prime_factors::PrimeFactors;
use crate::spectrum_arithmetic::SpectrumArithmeticFactory;
use crate::traits::FftTrigonometry;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};

pub trait FftExecutor<T> {
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError>;
    fn direction(&self) -> FftDirection;
    fn length(&self) -> usize;
}

pub struct Zaft {}

impl Zaft {
    fn make_mixed_radix<
        T: AlgorithmFactory<T>
            + FftTrigonometry
            + Float
            + 'static
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + SpectrumArithmeticFactory<T>
            + Copy
            + Display,
    >(
        direction: FftDirection,
        prime_factors: PrimeFactors,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        let factorization = prime_factors.factorization;

        let mut stages: Vec<Box<dyn FftExecutor<T> + Send + Sync>> = Vec::new();

        for (prime, exp) in factorization {
            let len = (prime as usize)
                .checked_pow(exp)
                .ok_or(ZaftError::InvalidPointerSize(
                    (prime as u128).saturating_pow(exp),
                ))?;
            let stage = Zaft::strategy(len, direction)?;
            stages.push(stage);
        }

        if stages.len() < 2 {
            unreachable!("This is an internal error, this should never happen");
        }

        // Take ownership via into_iter()
        let mut iter = stages.into_iter();
        let first = iter.next().unwrap();
        let second = iter.next().unwrap();
        let mut main_radix =
            Box::new(MixedRadix::new(first, second)?) as Box<dyn FftExecutor<T> + Send + Sync>;

        // Chain the rest
        for stage in iter {
            main_radix = Box::new(MixedRadix::new(main_radix, stage)?);
        }

        Ok(main_radix)
    }

    fn strategy<
        T: AlgorithmFactory<T>
            + FftTrigonometry
            + Float
            + 'static
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + SpectrumArithmeticFactory<T>
            + Copy
            + Display,
    >(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        if n == 1 {
            return T::dft(n, fft_direction);
        } else if n == 2 {
            return T::butterfly2(fft_direction);
        } else if n == 3 {
            return T::butterfly3(fft_direction);
        } else if n == 4 {
            return T::butterfly4(fft_direction);
        } else if n == 5 {
            return T::butterfly5(fft_direction);
        }
        let prime_factors = PrimeFactors::from_number(n as u64);
        if prime_factors.is_power_of_three {
            // Use Radix-3 if divisible by 3
            T::radix3(n, fft_direction)
        } else if prime_factors.is_power_of_five {
            // Use Radix-5 if power of 5
            T::radix5(n, fft_direction)
        } else if prime_factors.is_power_of_four {
            // Use Radix-4 if a power of 4
            T::radix4(n, fft_direction)
        } else if prime_factors.is_power_of_two {
            // Otherwise, fallback to Radix-2
            T::radix2(n, fft_direction)
        } else if prime_factors.is_power_of_six {
            T::radix6(n, fft_direction)
        } else if prime_factors.may_be_represented_in_mixed_radix() {
            Zaft::make_mixed_radix(fft_direction, prime_factors)
        } else {
            T::dft(n, fft_direction)
        }
    }

    pub fn make_forward_fft_f32(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Forward)
    }

    pub fn make_forward_fft_f64(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Forward)
    }

    pub fn make_inverse_fft_f32(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Inverse)
    }

    pub fn make_inverse_fft_f64(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Inverse)
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum FftDirection {
    Forward,
    Inverse,
}

impl Display for FftDirection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            FftDirection::Forward => f.write_str("FftDirection::Forward"),
            FftDirection::Inverse => f.write_str("FftDirection::Inverse"),
        }
    }
}

#[cfg(test)]
mod tests {}
