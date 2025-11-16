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
#![allow(
    clippy::manual_is_multiple_of,
    clippy::assign_op_pattern,
    clippy::only_used_in_recursion,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    all(feature = "fcma", target_arch = "aarch64"),
    feature(stdarch_neon_fcma)
)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
mod avx;
mod bluestein;
mod butterflies;
mod complex_fma;
mod dft;
mod err;
mod factory;
mod factory64;
mod fast_divider;
mod good_thomas;
mod good_thomas_small;
mod mixed_radix;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod prime_factors;
mod r2c;
mod raders;
mod radix10;
mod radix11;
mod radix13;
mod radix3;
mod radix4;
mod radix5;
mod radix6;
mod radix7;
mod spectrum_arithmetic;
mod traits;
mod transpose;
mod transpose_arbitrary;
mod util;

#[allow(unused_imports)]
use radix3::Radix3;
#[allow(unused_imports)]
use radix4::Radix4;
#[allow(unused_imports)]
use radix5::Radix5;
#[allow(unused_imports)]
use radix6::Radix6;
#[allow(unused_imports)]
use radix7::Radix7;
#[allow(unused_imports)]
use radix10::Radix10;
#[allow(unused_imports)]
use radix11::Radix11;
#[allow(unused_imports)]
use radix13::Radix13;

pub use err::ZaftError;
use std::fmt::{Display, Formatter};

use crate::factory::AlgorithmFactory;
use crate::prime_factors::{
    PrimeFactors, can_be_two_factors, split_factors_closest, try_greedy_pure_power_split,
};
use crate::r2c::{
    C2RFftEvenInterceptor, C2RFftOddInterceptor, OneSizedRealFft, R2CFftEvenInterceptor,
    R2CFftOddInterceptor,
};
use crate::spectrum_arithmetic::SpectrumOpsFactory;
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
pub use r2c::{C2RFftExecutor, R2CFftExecutor};

pub trait FftExecutor<T> {
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError>;
    fn direction(&self) -> FftDirection;
    fn length(&self) -> usize;
}

pub(crate) trait FftExecutorOutOfPlace<T> {
    #[allow(unused)]
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
}

pub(crate) trait CompositeFftExecutor<T>: FftExecutor<T> + FftExecutorOutOfPlace<T> {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<T> + Send + Sync>;
}

pub struct Zaft {}

impl Zaft {
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    fn could_do_split_mixed_radix() -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return false;
        }
        true
    }

    #[cfg(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    ))]
    fn try_split_mixed_radix_butterflies<
        T: AlgorithmFactory<T>
            + FftTrigonometry
            + Float
            + 'static
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + SpectrumOpsFactory<T>
            + TransposeFactory<T>
            + Copy
            + Display,
    >(
        n_length: u64,
        q_length: u64,
        direction: FftDirection,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return Ok(None);
        }
        let min_length = n_length.min(q_length);
        let max_length = n_length.max(q_length);
        if min_length == 2 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly2(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 3 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly3(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 4 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly4(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 5 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly5(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 6 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly6(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 7 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly7(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 8 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly8(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 9 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly9(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 10 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly10(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 11 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly11(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 12 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly12(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 13 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly13(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        } else if min_length == 16 {
            let q_fft = Zaft::strategy(max_length as usize, direction)?;
            let q_fft_opt = T::mixed_radix_butterfly16(q_fft)?;
            if let Some(q_fft_opt) = q_fft_opt {
                return Ok(Some(q_fft_opt));
            }
        }
        Ok(None)
    }

    fn make_mixed_radix<
        T: AlgorithmFactory<T>
            + FftTrigonometry
            + Float
            + 'static
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + SpectrumOpsFactory<T>
            + TransposeFactory<T>
            + Copy
            + Display,
    >(
        direction: FftDirection,
        prime_factors: PrimeFactors,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        let factorization = &prime_factors.factorization;
        let product = factorization.iter().map(|&x| x.0.pow(x.1)).product::<u64>();

        let (n_length, q_length) = if product <= 529 {
            match can_be_two_factors(&factorization) {
                None => match try_greedy_pure_power_split(&factorization) {
                    None => split_factors_closest(&factorization),
                    Some(values) => values,
                },
                Some(factors) => factors,
            }
        } else {
            match try_greedy_pure_power_split(&factorization) {
                None => split_factors_closest(&factorization),
                Some(values) => values,
            }
        };

        if prime_factors.is_power_of_two_and_three() {
            if product % 36 == 0 && product / 36 > 1 && product / 36 <= 16 {
                return if let Some(executor) =
                    Zaft::try_split_mixed_radix_butterflies(product / 36, 36, direction)?
                {
                    Ok(executor)
                } else {
                    let p_fft = Zaft::strategy((product / 36) as usize, direction)?;
                    let q_fft = Zaft::strategy(36, direction)?;
                    T::mixed_radix(p_fft, q_fft)
                };
            }
            let product2 = prime_factors
                .factorization
                .iter()
                .filter(|x| x.0 == 2)
                .map(|x| x.0.pow(x.1))
                .product::<u64>();
            let product3 = prime_factors
                .factorization
                .iter()
                .filter(|x| x.0 == 3)
                .map(|x| x.0.pow(x.1))
                .product::<u64>();
            return if let Some(executor) =
                Zaft::try_split_mixed_radix_butterflies(product2, product3, direction)?
            {
                Ok(executor)
            } else {
                let p_fft = Zaft::strategy(product2 as usize, direction)?;
                let q_fft = Zaft::strategy(product3 as usize, direction)?;
                T::mixed_radix(p_fft, q_fft)
            };
        }

        #[cfg(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        ))]
        {
            match Zaft::try_split_mixed_radix_butterflies(n_length, q_length, direction) {
                Ok(value) => match value {
                    None => {}
                    Some(executor) => return Ok(executor),
                },
                Err(err) => return Err(err),
            }
        }

        let p_fft = Zaft::strategy(n_length as usize, direction)?;
        let q_fft = Zaft::strategy(q_length as usize, direction)?;
        if num_integer::gcd(q_length, n_length) == 1 && q_length < 33 && n_length <= 33 {
            T::good_thomas(p_fft, q_fft)
        } else {
            T::mixed_radix(p_fft, q_fft)
        }
    }

    fn make_prime<
        T: AlgorithmFactory<T>
            + FftTrigonometry
            + Float
            + 'static
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + SpectrumOpsFactory<T>
            + TransposeFactory<T>
            + Copy
            + Display,
    >(
        n: usize,
        direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        let convolve_prime = PrimeFactors::from_number(n as u64 - 1);
        // n-1 may result in Cunningham chain, and we want to avoid compute multiple prime numbers FFT at once
        let big_factor = convolve_prime
            .factorization
            .iter()
            .any(|x| x.0 > 31 && x.1 == 1);
        if !big_factor {
            let convolve_fft = Zaft::strategy(n - 1, direction);
            T::raders(convolve_fft?, n, direction)
        } else {
            // we want to use bluestein's algorithm. we have a free choice of which inner FFT length to use
            // the only restriction is that it has to be (2 * len - 1) or larger. So we want the fastest FFT we can compute at or above that size.

            // the most obvious choice is the next-highest power of two, but there's one trick we can pull to get a smaller fft that we can be 100% certain will be faster
            let min_inner_len = 2 * n - 1;
            let inner_len_pow2 = min_inner_len.checked_next_power_of_two().unwrap();
            let inner_len_factor3 = inner_len_pow2 / 4 * 3;

            let inner_len = if inner_len_factor3 >= min_inner_len {
                inner_len_factor3
            } else {
                inner_len_pow2
            };
            let convolve_fft = Zaft::strategy(inner_len, direction)?;
            T::bluestein(convolve_fft, n, direction)
        }
    }

    pub(crate) fn strategy<
        T: AlgorithmFactory<T>
            + FftTrigonometry
            + Float
            + 'static
            + Send
            + Sync
            + MulAdd<T, Output = T>
            + SpectrumOpsFactory<T>
            + TransposeFactory<T>
            + Copy
            + Display,
    >(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        if n == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        if n == 1 {
            return T::butterfly1(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 2 {
            return T::butterfly2(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 3 {
            return T::butterfly3(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 4 {
            return T::butterfly4(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 5 {
            return T::butterfly5(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 6 {
            return T::butterfly6(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 7 {
            return T::butterfly7(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 8 {
            return T::butterfly8(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 9 {
            return T::butterfly9(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 10 {
            return T::butterfly10(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 11 {
            return T::butterfly11(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 12 {
            return T::butterfly12(fft_direction);
        } else if n == 13 {
            return T::butterfly13(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 14 {
            return T::butterfly14(fft_direction);
        } else if n == 15 {
            return T::butterfly15(fft_direction);
        } else if n == 16 {
            return T::butterfly16(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 17 {
            return T::butterfly17(fft_direction);
        } else if n == 18 {
            return T::butterfly18(fft_direction);
        } else if n == 19 {
            return T::butterfly19(fft_direction);
        } else if n == 20 {
            return T::butterfly20(fft_direction);
        } else if n == 23 {
            return T::butterfly23(fft_direction);
        } else if n == 25 {
            return T::butterfly25(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 27 {
            return T::butterfly27(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 29 {
            return T::butterfly29(fft_direction);
        } else if n == 31 {
            return T::butterfly31(fft_direction);
        } else if n == 32 {
            return T::butterfly32(fft_direction).map(|x| x.into_fft_executor());
        } else if n == 36 {
            if let Some(executor) = T::butterfly36(fft_direction) {
                return Ok(executor.into_fft_executor());
            }
        } else if n == 49 {
            if let Some(executor) = T::butterfly49(fft_direction) {
                return Ok(executor.into_fft_executor());
            }
        } else if n == 64 {
            if let Some(executor) = T::butterfly64(fft_direction) {
                return Ok(executor.into_fft_executor());
            }
        } else if n == 81 {
            if let Some(executor) = T::butterfly81(fft_direction) {
                return Ok(executor.into_fft_executor());
            }
        } else if n == 100 {
            if let Some(executor) = T::butterfly100(fft_direction) {
                return Ok(executor.into_fft_executor());
            }
        }
        let prime_factors = PrimeFactors::from_number(n as u64);
        if prime_factors.is_power_of_three {
            // Use Radix-3 if divisible by 3
            T::radix3(n, fft_direction)
        } else if prime_factors.is_power_of_five {
            // Use Radix-5 if power of 5
            T::radix5(n, fft_direction)
        } else if prime_factors.is_power_of_two {
            // Use Radix-4 if a power of 2
            T::radix4(n, fft_direction)
        } else if prime_factors.is_power_of_six {
            T::radix6(n, fft_direction)
        } else if prime_factors.is_power_of_seven {
            T::radix7(n, fft_direction)
        } else if prime_factors.is_power_of_ten {
            T::radix10(n, fft_direction)
        } else if prime_factors.is_power_of_eleven {
            T::radix11(n, fft_direction)
        } else if prime_factors.is_power_of_thirteen {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if Zaft::could_do_split_mixed_radix() {
                    let r = n / 13;
                    if r == 13 {
                        let right_fft = T::butterfly13(fft_direction)?;
                        if let Ok(Some(v)) = T::mixed_radix_butterfly13(right_fft) {
                            return Ok(v);
                        }
                    }
                    let right_fft = T::radix13(r, fft_direction)?;
                    if let Ok(Some(v)) = T::mixed_radix_butterfly13(right_fft) {
                        return Ok(v);
                    }
                }
            }
            T::radix13(n, fft_direction)
        } else if prime_factors.may_be_represented_in_mixed_radix() {
            Zaft::make_mixed_radix(fft_direction, prime_factors)
        } else if prime_factors.is_prime() {
            Zaft::make_prime(n, fft_direction)
        } else {
            T::dft(n, fft_direction)
        }
    }

    pub fn make_r2c_fft_f32(
        n: usize,
    ) -> Result<Box<dyn R2CFftExecutor<f32> + Send + Sync>, ZaftError> {
        if n == 1 {
            return Ok(Box::new(OneSizedRealFft {
                phantom_data: Default::default(),
            }));
        }
        if n.is_multiple_of(2) {
            R2CFftEvenInterceptor::install(n, Zaft::strategy(n / 2, FftDirection::Forward)?)
                .map(|x| Box::new(x) as Box<dyn R2CFftExecutor<f32> + Send + Sync>)
        } else {
            R2CFftOddInterceptor::install(n, Zaft::strategy(n, FftDirection::Forward)?)
                .map(|x| Box::new(x) as Box<dyn R2CFftExecutor<f32> + Send + Sync>)
        }
    }

    pub fn make_c2r_fft_f32(
        n: usize,
    ) -> Result<Box<dyn C2RFftExecutor<f32> + Send + Sync>, ZaftError> {
        if n == 1 {
            return Ok(Box::new(OneSizedRealFft {
                phantom_data: Default::default(),
            }));
        }
        if n.is_multiple_of(2) {
            C2RFftEvenInterceptor::install(n, Zaft::strategy(n / 2, FftDirection::Inverse)?)
                .map(|x| Box::new(x) as Box<dyn C2RFftExecutor<f32> + Send + Sync>)
        } else {
            C2RFftOddInterceptor::install(n, Zaft::strategy(n, FftDirection::Inverse)?)
                .map(|x| Box::new(x) as Box<dyn C2RFftExecutor<f32> + Send + Sync>)
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

    pub fn make_c2r_fft_f64(
        n: usize,
    ) -> Result<Box<dyn C2RFftExecutor<f64> + Send + Sync>, ZaftError> {
        if n == 1 {
            return Ok(Box::new(OneSizedRealFft {
                phantom_data: Default::default(),
            }));
        }
        if n.is_multiple_of(2) {
            C2RFftEvenInterceptor::install(n, Zaft::strategy(n / 2, FftDirection::Inverse)?)
                .map(|x| Box::new(x) as Box<dyn C2RFftExecutor<f64> + Send + Sync>)
        } else {
            C2RFftOddInterceptor::install(n, Zaft::strategy(n, FftDirection::Inverse)?)
                .map(|x| Box::new(x) as Box<dyn C2RFftExecutor<f64> + Send + Sync>)
        }
    }

    pub fn make_r2c_fft_f64(
        n: usize,
    ) -> Result<Box<dyn R2CFftExecutor<f64> + Send + Sync>, ZaftError> {
        if n == 1 {
            return Ok(Box::new(OneSizedRealFft {
                phantom_data: Default::default(),
            }));
        }
        if n.is_multiple_of(2) {
            R2CFftEvenInterceptor::install(n, Zaft::strategy(n / 2, FftDirection::Forward)?)
                .map(|x| Box::new(x) as Box<dyn R2CFftExecutor<f64> + Send + Sync>)
        } else {
            R2CFftOddInterceptor::install(n, Zaft::strategy(n, FftDirection::Forward)?)
                .map(|x| Box::new(x) as Box<dyn R2CFftExecutor<f64> + Send + Sync>)
        }
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

impl FftDirection {
    pub fn inverse(self) -> FftDirection {
        match self {
            FftDirection::Forward => FftDirection::Inverse,
            FftDirection::Inverse => FftDirection::Forward,
        }
    }
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
mod tests {
    use crate::Zaft;
    use num_complex::Complex;

    #[test]
    fn power_of_four() {
        fn is_power_of_four(n: u64) -> bool {
            n != 0 && (n & (n - 1)) == 0 && (n & 0x5555_5555_5555_5555) != 0
        }
        assert_eq!(is_power_of_four(4), true);
        assert_eq!(is_power_of_four(8), false);
        assert_eq!(is_power_of_four(16), true);
        assert_eq!(is_power_of_four(20), false);
    }

    #[test]
    fn test_everything_f32() {
        for i in 1..1150 {
            let mut data = vec![Complex::new(0.0019528865, 0.); i];
            for (i, chunk) in data.iter_mut().enumerate() {
                *chunk = Complex::new(
                    -0.19528865 + i as f32 * 0.001,
                    0.0019528865 - i as f32 * 0.001,
                );
            }
            let zaft_exec = Zaft::make_forward_fft_f32(data.len()).expect("Failed to make FFT!");
            let zaft_inverse = Zaft::make_inverse_fft_f32(data.len()).expect("Failed to make FFT!");
            let rust_fft_clone = data.clone();
            zaft_exec.execute(&mut data).unwrap();
            zaft_inverse.execute(&mut data).unwrap();
            let data_len = 1. / data.len() as f32;
            for i in data.iter_mut() {
                *i *= data_len;
            }
            data.iter()
                .zip(rust_fft_clone)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx}",
                        a.re,
                        b.re
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-2,
                        "a_im {}, b_im {} at {idx}",
                        a.im,
                        b.im
                    );
                });
        }
    }
}
