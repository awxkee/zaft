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
    clippy::assign_op_pattern,
    clippy::only_used_in_recursion,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::modulo_one
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
mod store;
mod td;
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
use std::collections::HashMap;

use crate::factory::AlgorithmFactory;
use crate::prime_factors::{
    PrimeFactors, can_be_two_factors, split_factors_closest, try_greedy_pure_power_split,
};
use crate::r2c::{
    C2RFftEvenInterceptor, C2RFftOddInterceptor, C2ROddExpanderFactory, OneSizedRealFft,
    R2CTwiddlesFactory, R2cAlgorithmFactory, strategy_r2c,
};
use crate::spectrum_arithmetic::ComplexArithFactory;
use crate::td::{TwoDimensionalC2C, TwoDimensionalC2R, TwoDimensionalR2C};
use crate::traits::FftTrigonometry;
use crate::transpose::TransposeFactory;
use crate::util::{
    ALWAYS_BLUESTEIN_1000, ALWAYS_BLUESTEIN_2000, ALWAYS_BLUESTEIN_3000, ALWAYS_BLUESTEIN_4000,
    ALWAYS_BLUESTEIN_5000, ALWAYS_BLUESTEIN_6000,
};
pub use err::ZaftError;
use num_complex::Complex;
use num_traits::{AsPrimitive, Float, MulAdd};
pub use r2c::{C2RFftExecutor, R2CFftExecutor};
use std::fmt::{Debug, Display, Formatter};
use std::sync::{Arc, OnceLock, RwLock};
pub use td::{TwoDimensionalExecutorC2R, TwoDimensionalExecutorR2C, TwoDimensionalFftExecutor};

pub(crate) trait FftSample:
    AlgorithmFactory<Self>
    + FftTrigonometry
    + Float
    + 'static
    + Send
    + Sync
    + MulAdd<Self, Output = Self>
    + ComplexArithFactory<Self>
    + TransposeFactory<Self>
    + Copy
    + Display
    + FftPrimeCache<Self>
    + Default
    + Debug
    + R2cAlgorithmFactory<Self>
    + R2CTwiddlesFactory<Self>
    + C2ROddExpanderFactory
{
    const HALF: Self;
    const SQRT_3_OVER_2: Self;
}

impl FftSample for f64 {
    const HALF: Self = 0.5;
    // from sage.all import *
    // import struct
    //
    // R = RealField(90)
    //
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // value = R(3).sqrt() / R(2)
    //
    // print(float_to_hex(value))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(value))
    const SQRT_3_OVER_2: Self = f64::from_bits(0x3febb67ae8584caa);
}
impl FftSample for f32 {
    const HALF: Self = 0.5;
    // from sage.all import *
    // import struct
    //
    // R = RealField(90)
    //
    // def float_to_hex(f):
    //     packed = struct.pack('>f', float(f))
    //     return '0x' + packed.hex()
    //
    // value = R(3).sqrt() / R(2)
    //
    // print(float_to_hex(value))
    //
    // def double_to_hex(f):
    //         packed = struct.pack('>d', float(f))
    //         return '0x' + packed.hex()
    //
    // print(double_to_hex(value))
    const SQRT_3_OVER_2: Self = f32::from_bits(0x3f5db3d7);
}

pub trait FftExecutor<T> {
    /// Executes the Complex-to-Complex FFT operation **in-place**.
    ///
    /// The input/output slice `in_place` must have a length equal to `self.length()`.
    /// The direction of the transform (Forward or Inverse) is determined by the executor's
    /// pre-configured state, accessible via `self.direction()`.
    ///
    /// # Parameters
    /// * `in_place`: The mutable slice containing the complex-valued input data. Upon completion,
    ///   it will contain the complex-valued frequency-domain result (for a Forward transform)
    ///   or the time-domain result (for an Inverse transform).
    ///
    /// # Errors
    /// Returns a `ZaftError` if the execution fails (e.g., due to an incorrect slice length).
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError>;
    /// Executes the FFT operation **in-place**, using caller-provided scratch memory.
    ///
    /// This variant avoids internal allocations and may improve performance
    /// in repeated executions.
    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Executes the FFT operation **out-of-place**.
    ///
    /// The input slice `src` is left unmodified. The result is written
    /// into `dst`.
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Executes the FFT operation **out-of-place**, using caller-provided scratch memory.
    ///
    /// This allows reuse of scratch memory across multiple FFT calls
    /// to avoid repeated allocations.
    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Executes the FFT operation using a **destructive input strategy**.
    ///
    /// The `src` buffer may be overwritten during computation and should
    /// not be assumed to retain its original contents after the call.
    /// The final transform result is written to `dst`.
    ///
    /// This variant may enable more memory-efficient algorithms.
    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError>;
    /// Returns the **direction** of the transform this executor is configured to perform.
    ///
    /// The direction is typically either `FftDirection::Forward` (Time to Frequency) or
    /// `FftDirection::Inverse` (Frequency to Time).
    fn direction(&self) -> FftDirection;
    /// Returns the **length** (size N) of the input and output complex vectors.
    ///
    /// This is the number of complex elements that the executor is designed to process.
    fn length(&self) -> usize;
    /// Returns the required scratch buffer length for
    /// [`execute_with_scratch`].
    /// The returned size is **not a stable constant** and may change
    /// between crate versions or algorithm implementations.
    /// Always query this value dynamically.
    fn scratch_length(&self) -> usize;
    /// Returns the required scratch buffer length for
    /// [`execute_out_of_place_with_scratch`].
    /// The returned size is **not a stable constant** and may change
    /// between crate versions or algorithm implementations.
    /// Always query this value dynamically.
    fn out_of_place_scratch_length(&self) -> usize;
    /// Returns the required scratch buffer length for
    /// [`execute_destructive_with_scratch`].
    /// The returned size is **not a stable constant** and may change
    /// between crate versions or algorithm implementations.
    /// Always query this value dynamically.
    fn destructive_scratch_length(&self) -> usize;
}

static PRIME_CACHE_F: OnceLock<RwLock<HashMap<usize, Arc<dyn FftExecutor<f32> + Send + Sync>>>> =
    OnceLock::new();

static PRIME_CACHE_B: OnceLock<RwLock<HashMap<usize, Arc<dyn FftExecutor<f32> + Send + Sync>>>> =
    OnceLock::new();

static PRIME_CACHE_DF: OnceLock<RwLock<HashMap<usize, Arc<dyn FftExecutor<f64> + Send + Sync>>>> =
    OnceLock::new();

static PRIME_CACHE_DB: OnceLock<RwLock<HashMap<usize, Arc<dyn FftExecutor<f64> + Send + Sync>>>> =
    OnceLock::new();

pub(crate) trait FftPrimeCache<T> {
    fn has_cached_prime(
        n: usize,
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn put_prime_to_cache(fft_direction: FftDirection, fft: Arc<dyn FftExecutor<T> + Send + Sync>);
}

impl FftPrimeCache<f32> for f32 {
    fn has_cached_prime(
        n: usize,
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        if n >= 4000 {
            return None;
        }
        let cache = (match fft_direction {
            FftDirection::Forward => &PRIME_CACHE_F,
            FftDirection::Inverse => &PRIME_CACHE_B,
        })
        .get_or_init(|| RwLock::new(HashMap::new()));
        cache.read().ok()?.get(&n).cloned()
    }

    fn put_prime_to_cache(
        fft_direction: FftDirection,
        fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) {
        let length = fft.length();
        if length > 4000 {
            return;
        }
        let cache = (match fft_direction {
            FftDirection::Forward => &PRIME_CACHE_F,
            FftDirection::Inverse => &PRIME_CACHE_B,
        })
        .get_or_init(|| RwLock::new(HashMap::new()));
        _ = cache.write().ok().and_then(|mut x| x.insert(length, fft));
    }
}

impl FftPrimeCache<f64> for f64 {
    fn has_cached_prime(
        n: usize,
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f64> + Send + Sync>> {
        if n >= 4000 {
            return None;
        }
        let cache = (match fft_direction {
            FftDirection::Forward => &PRIME_CACHE_DF,
            FftDirection::Inverse => &PRIME_CACHE_DB,
        })
        .get_or_init(|| RwLock::new(HashMap::new()));
        cache.read().ok()?.get(&n).cloned()
    }

    fn put_prime_to_cache(
        fft_direction: FftDirection,
        fft: Arc<dyn FftExecutor<f64> + Send + Sync>,
    ) {
        let length = fft.length();
        if length > 4000 {
            return;
        }
        let cache = (match fft_direction {
            FftDirection::Forward => &PRIME_CACHE_DF,
            FftDirection::Inverse => &PRIME_CACHE_DB,
        })
        .get_or_init(|| RwLock::new(HashMap::new()));
        _ = cache.write().ok().and_then(|mut x| x.insert(length, fft));
    }
}

pub struct Zaft {}

impl Zaft {
    fn could_do_split_mixed_radix() -> bool {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            return true;
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            true
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            false
        }
    }

    fn try_split_mixed_radix_butterflies<T: FftSample>(
        _n_length: u64,
        _q_length: u64,
        _direction: FftDirection,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            Ok(None)
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return Ok(None);
        }
        #[cfg(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        ))]
        {
            let min_length = _n_length.min(_q_length);
            let max_length = _n_length.max(_q_length);

            if !(2..=13).contains(&min_length) {
                // If no butterfly exists, return None
                return Ok(None);
            }

            // 1. Get the initial FFT strategy regardless of the butterfly size.
            let q_fft = Zaft::strategy(max_length as usize, _direction)?;

            let q_fft_opt = match min_length {
                2 => T::mixed_radix_butterfly2(q_fft),
                3 => T::mixed_radix_butterfly3(q_fft),
                4 => T::mixed_radix_butterfly4(q_fft),
                5 => T::mixed_radix_butterfly5(q_fft),
                6 => T::mixed_radix_butterfly6(q_fft),
                7 => T::mixed_radix_butterfly7(q_fft),
                8 => T::mixed_radix_butterfly8(q_fft),
                9 => T::mixed_radix_butterfly9(q_fft),
                10 => T::mixed_radix_butterfly10(q_fft),
                11 => T::mixed_radix_butterfly11(q_fft),
                12 => T::mixed_radix_butterfly12(q_fft),
                13 => T::mixed_radix_butterfly13(q_fft),
                // This arm is covered by the early exit, but is required if the early exit is removed.
                _ => unreachable!("min_length is outside the supported range [2, 13]."),
            };

            // 3. Handle the result once.
            if let Some(q_fft_opt) = q_fft_opt? {
                Ok(Some(q_fft_opt))
            } else {
                Ok(None)
            }
        }
    }

    fn make_mixed_radix<T: FftSample>(
        direction: FftDirection,
        prime_factors: PrimeFactors,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        let factorization = &prime_factors.factorization;
        let product = factorization.iter().map(|&x| x.0.pow(x.1)).product::<u64>();

        let (n_length, q_length) = if product <= 529 {
            match can_be_two_factors(factorization) {
                None => match try_greedy_pure_power_split(factorization) {
                    None => split_factors_closest(factorization),
                    Some(values) => values,
                },
                Some(factors) => factors,
            }
        } else {
            match try_greedy_pure_power_split(factorization) {
                None => split_factors_closest(factorization),
                Some(values) => values,
            }
        };

        macro_rules! try_mixed_radix {
            ($q: expr, $p: expr) => {{
                return if let Some(executor) =
                    Zaft::try_split_mixed_radix_butterflies($q as u64, $p as u64, direction)?
                {
                    Ok(executor)
                } else {
                    let p_fft = Zaft::strategy($q as usize, direction)?;
                    let q_fft = Zaft::strategy($p as usize, direction)?;
                    if $q < $p {
                        T::mixed_radix(p_fft, q_fft)
                    } else {
                        T::mixed_radix(q_fft, p_fft)
                    }
                };
            }};
        }

        if prime_factors.is_power_of_two_and_three() {
            let product2 = prime_factors
                .factorization
                .iter()
                .find(|x| x.0 == 2)
                .map(|x| x.0.pow(x.1))
                .expect("Factor of 2 must present in 2^n*3^m branch");
            let product3 = prime_factors
                .factorization
                .iter()
                .find(|x| x.0 == 3)
                .map(|x| x.0.pow(x.1))
                .expect("Factor of 3 must present in 2^n*3^m branch");

            let factor2 = prime_factors.factor_of_2();
            let factor3 = prime_factors.factor_of_3();

            if factor2 == 1 && factor3 > 3 && T::butterfly54(direction).is_some() {
                try_mixed_radix!(54, product / 54)
            }

            if product == 1536 {
                try_mixed_radix!(8, 192)
            }

            if factor3 >= 1 && factor2 >= 4 {
                if product.is_multiple_of(36)
                    && product / 36 > 1
                    && product / 36 <= 16
                    && T::butterfly36(direction).is_some()
                {
                    try_mixed_radix!(36, product / 36)
                }
                if factor2 > factor3 {
                    let mut factors_diff = 2u64.pow(factor2 - factor3);
                    let mut remainder_factor = product / factors_diff;
                    if remainder_factor <= 8 {
                        remainder_factor *= 2;
                        factors_diff /= 2;
                    }
                    try_mixed_radix!(factors_diff, remainder_factor)
                }
                if product.is_multiple_of(48)
                    && product / 48 > 1
                    && product / 48 <= 16
                    && T::butterfly48(direction).is_some()
                {
                    try_mixed_radix!(48, product / 48)
                }
            }

            return if let Some(executor) =
                Zaft::try_split_mixed_radix_butterflies(product2, product3, direction)?
            {
                Ok(executor)
            } else {
                let p_fft = Zaft::strategy(product2 as usize, direction)?;
                let q_fft = Zaft::strategy(product3 as usize, direction)?;
                T::mixed_radix(p_fft, q_fft)
            };
        } else if prime_factors.is_power_of_two_and_five() {
            let factor_of_5 = prime_factors.factor_of_5();
            let factor_of_2 = prime_factors.factor_of_2();
            if factor_of_5 == 1 {
                if factor_of_2 >= 8 && factor_of_2 != 9 {
                    try_mixed_radix!(5, product / 5)
                }
                if (2..=6).contains(&factor_of_2) {
                    try_mixed_radix!(20, product / 20)
                }
            } else if factor_of_5 == 2 {
                if factor_of_2 >= 3 && T::butterfly100(direction).is_some() {
                    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                    {
                        try_mixed_radix!(product / 100, 100)
                    }
                    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                    {
                        use crate::util::has_valid_avx;
                        if has_valid_avx() {
                            try_mixed_radix!(product / 100, 100)
                        }
                    }
                }
            } else if factor_of_5 == 3 {
                #[cfg(any(
                    all(target_arch = "aarch64", feature = "neon"),
                    all(target_arch = "x86_64", feature = "avx")
                ))]
                if product == 500 {
                    try_mixed_radix!(5, 100)
                }
            }
        } else if prime_factors.is_power_of_three_and_five() {
            let factor_of_5 = prime_factors.factor_of_5();
            let factor_of_3 = prime_factors.factor_of_3();
            if factor_of_5 == 1 && factor_of_3 > 1 {
                try_mixed_radix!(5, product / 5)
            } else if factor_of_5 == 2 && factor_of_5 > 1 && factor_of_3 > 2 && factor_of_3 < 10 {
                // 225 is more effective with mixed radix [9,25]
                try_mixed_radix!(25, product / 25)
            }
        } else if prime_factors.is_power_of_two_and_seven() {
            let factor_of_7 = prime_factors.factor_of_7();
            let factor_of_2 = prime_factors.factor_of_2();
            if factor_of_2 > 1 && factor_of_7 == 1 {
                try_mixed_radix!(14, product / 14)
            }
        } else if prime_factors.has_power_of_five_and_seven() {
            let factor_of_7 = prime_factors.factor_of_7();
            let factor_of_5 = prime_factors.factor_of_5();
            #[allow(clippy::collapsible_if)]
            if factor_of_7 == 1 || factor_of_5 == 1 {
                if product == 560 || product == 2240 {
                    if T::butterfly35(direction).is_some() {
                        try_mixed_radix!(35, product / 35)
                    }
                }
            }
            if (product == 210
                || product == 280
                || product == 315
                || product == 350
                || product == 420)
                && T::butterfly35(direction).is_some()
            {
                try_mixed_radix!(35, product / 35)
            }
            if prime_factors.is_power_of_five_and_seven() && (factor_of_7 > 2 && factor_of_5 > 2) {
                let product7 = prime_factors
                    .factorization
                    .iter()
                    .find(|x| x.0 == 7)
                    .map(|x| x.0.pow(x.1))
                    .expect("Power of 7 should exist if factor of 5^n*7^m branch");
                let product5 = prime_factors
                    .factorization
                    .iter()
                    .find(|x| x.0 == 5)
                    .map(|x| x.0.pow(x.1))
                    .expect("Power of 5 should exist if factor of 5^n*7^m branch");
                let p_fft = Zaft::strategy(product5 as usize, direction)?;
                let q_fft = Zaft::strategy(product7 as usize, direction)?;
                return if product5 < product7 {
                    T::mixed_radix(p_fft, q_fft)
                } else {
                    T::mixed_radix(q_fft, p_fft)
                };
            }
        } else if prime_factors.has_power_of_two_and_three() {
            #[allow(clippy::collapsible_if)]
            if (product == 84
                || product == 294
                || product == 252
                || product == 378
                || product == 504
                || product == 672
                || product == 756)
                && T::butterfly42(direction).is_some()
            {
                try_mixed_radix!(42, product / 42)
            }
            let factor_of_2 = prime_factors.factor_of_2();
            let factor_of_3 = prime_factors.factor_of_3();
            let factor_of_5 = prime_factors.factor_of_5();

            if (factor_of_2 == 1 && factor_of_3 == 1 && factor_of_5 > 1)
                && T::butterfly30(direction).is_some()
            {
                // factor out 30
                try_mixed_radix!(30, product / 30)
            }

            if ((product.is_multiple_of(144) && (product / 144) <= 16)
                || product == 5040
                || product == 4896
                || product == 8496
                || product == 8352)
                && T::butterfly144(direction).is_some()
            {
                // factor out 144
                try_mixed_radix!(product / 144, 144)
            }

            if ((product.is_multiple_of(72) && (product / 72) <= 16)
                || product == 2088
                || product == 3240
                || product == 3816
                || product == 4248)
                && T::butterfly72(direction).is_some()
            {
                // factor out 72
                try_mixed_radix!(product / 72, 72)
            }

            if product == 858 && T::butterfly66(direction).is_some() {
                // factor out 66
                try_mixed_radix!(66, product / 66)
            }
        }

        if product.is_multiple_of(63)
            && product / 63 <= 16
            && product != 126
            && T::butterfly64(direction).is_some()
        {
            // factor out 63
            try_mixed_radix!(63, product / 63)
        }

        if (product == 147 || product == 315 || product == 378 || product == 399)
            && T::butterfly21(direction).is_some()
        {
            try_mixed_radix!(21, product / 21)
        }

        #[cfg(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        ))]
        {
            macro_rules! get_mixed_butterflies {
                ($q: expr, $p: expr) => {{
                    if let Some(executor) =
                        Zaft::try_split_mixed_radix_butterflies($q as u64, $p as u64, direction)?
                    {
                        return Ok(executor);
                    }
                }};
            }
            let factor_2 = prime_factors.factor_of_2();
            let rem2_8 = factor_2 % 3;
            if product.is_multiple_of(10) {
                get_mixed_butterflies!(10, product / 10)
            }
            if factor_2 > 0 {
                if rem2_8 == 1 {
                    get_mixed_butterflies!(2, product / 2)
                }
                if rem2_8 == 2 {
                    get_mixed_butterflies!(4, product / 4)
                }
                get_mixed_butterflies!(8, product / 8)
            }
            if product.is_multiple_of(12) {
                get_mixed_butterflies!(12, product / 12)
            }
            if product.is_multiple_of(9) {
                get_mixed_butterflies!(9, product / 9)
            }
            if product.is_multiple_of(7) {
                get_mixed_butterflies!(7, product / 7)
            }
            if product.is_multiple_of(13) {
                get_mixed_butterflies!(13, product / 13)
            }
            if product.is_multiple_of(11) {
                get_mixed_butterflies!(11, product / 11)
            }
            if product.is_multiple_of(5) {
                get_mixed_butterflies!(5, product / 5)
            }
            if product.is_multiple_of(3) {
                get_mixed_butterflies!(3, product / 3)
            }
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

    fn make_prime<T: FftSample>(
        n: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        if let Some(cached) = T::has_cached_prime(n, direction) {
            return Ok(cached);
        }
        let convolve_prime = PrimeFactors::from_number(n as u64 - 1);
        if n <= 6000 {
            let bluesteins = [
                ALWAYS_BLUESTEIN_1000.as_slice(),
                ALWAYS_BLUESTEIN_2000.as_slice(),
                ALWAYS_BLUESTEIN_3000.as_slice(),
                ALWAYS_BLUESTEIN_4000.as_slice(),
                ALWAYS_BLUESTEIN_5000.as_slice(),
                ALWAYS_BLUESTEIN_6000.as_slice(),
            ];
            let subset = bluesteins[n / 1000];
            if subset.contains(&n) {
                return Zaft::make_bluestein(n, direction);
            }
            return Zaft::make_raders(n, direction);
        }
        // n-1 may result in Cunningham chain, and we want to avoid compute multiple prime numbers FFT at once
        let big_factor = convolve_prime.factorization.iter().any(|x| x.0 > 31);
        let new_prime = if !big_factor {
            Zaft::make_raders(n, direction)
        } else {
            Zaft::make_bluestein(n, direction)
        };
        let fft_executor = new_prime?;
        T::put_prime_to_cache(direction, fft_executor.clone());
        Ok(fft_executor)
    }

    fn make_raders<T: FftSample>(
        n: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        let convolve_fft = Zaft::strategy(n - 1, direction);
        T::raders(convolve_fft?, n, direction)
    }

    fn make_bluestein<T: FftSample>(
        n: usize,
        direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
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

    fn plan_butterfly<T: FftSample>(
        n: usize,
        fft_direction: FftDirection,
    ) -> Option<Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>> {
        match n {
            1 => return Some(T::butterfly1(fft_direction)),
            2 => return Some(T::butterfly2(fft_direction)),
            3 => return Some(T::butterfly3(fft_direction)),
            4 => return Some(T::butterfly4(fft_direction)),
            5 => return Some(T::butterfly5(fft_direction)),
            6 => return Some(T::butterfly6(fft_direction)),
            7 => return Some(T::butterfly7(fft_direction)),
            8 => return Some(T::butterfly8(fft_direction)),
            9 => return Some(T::butterfly9(fft_direction)),
            10 => return Some(T::butterfly10(fft_direction)),
            11 => return Some(T::butterfly11(fft_direction)),
            12 => return Some(T::butterfly12(fft_direction)),
            13 => return Some(T::butterfly13(fft_direction)),
            14 => return Some(T::butterfly14(fft_direction)),
            15 => return Some(T::butterfly15(fft_direction)),
            16 => return Some(T::butterfly16(fft_direction)),
            17 => return Some(T::butterfly17(fft_direction)),
            18 => return Some(T::butterfly18(fft_direction)),
            19 => return Some(T::butterfly19(fft_direction)),
            20 => return Some(T::butterfly20(fft_direction)),
            21 => {
                return T::butterfly21(fft_direction).map(Ok);
            }
            23 => return Some(T::butterfly23(fft_direction)),
            24 => {
                return T::butterfly24(fft_direction).map(Ok);
            }
            25 => return Some(T::butterfly25(fft_direction)),
            27 => return Some(T::butterfly27(fft_direction)),
            28 => return T::butterfly28(fft_direction).map(Ok),
            29 => return Some(T::butterfly29(fft_direction)),
            30 => {
                return T::butterfly30(fft_direction).map(Ok);
            }
            31 => return Some(T::butterfly31(fft_direction)),
            32 => return Some(T::butterfly32(fft_direction)),
            35 => {
                return T::butterfly35(fft_direction).map(Ok);
            }
            36 => {
                return T::butterfly36(fft_direction).map(Ok);
            }
            37 => return Some(T::butterfly37(fft_direction)),
            40 => {
                return T::butterfly40(fft_direction).map(Ok);
            }
            41 => return Some(T::butterfly41(fft_direction)),
            42 => {
                return T::butterfly42(fft_direction).map(Ok);
            }
            48 => {
                return T::butterfly48(fft_direction).map(Ok);
            }
            49 => {
                return T::butterfly49(fft_direction).map(Ok);
            }
            54 => {
                return T::butterfly54(fft_direction).map(Ok);
            }
            63 => {
                return T::butterfly63(fft_direction).map(Ok);
            }
            64 => {
                return T::butterfly64(fft_direction).map(Ok);
            }
            66 => {
                return T::butterfly66(fft_direction).map(Ok);
            }
            70 => {
                return T::butterfly70(fft_direction).map(Ok);
            }
            72 => {
                return T::butterfly72(fft_direction).map(Ok);
            }
            78 => {
                return T::butterfly78(fft_direction).map(Ok);
            }
            81 => {
                return T::butterfly81(fft_direction).map(Ok);
            }
            88 => {
                return T::butterfly88(fft_direction).map(Ok);
            }
            96 => {
                return T::butterfly96(fft_direction).map(Ok);
            }
            100 => {
                return T::butterfly100(fft_direction).map(Ok);
            }
            108 => {
                return T::butterfly108(fft_direction).map(Ok);
            }
            121 => {
                return T::butterfly121(fft_direction).map(Ok);
            }
            125 => {
                return T::butterfly125(fft_direction).map(Ok);
            }
            128 => {
                return T::butterfly128(fft_direction).map(Ok);
            }
            144 => {
                return T::butterfly144(fft_direction).map(Ok);
            }
            169 => {
                return T::butterfly169(fft_direction).map(Ok);
            }
            192 => {
                return T::butterfly192(fft_direction).map(Ok);
            }
            216 => {
                return T::butterfly216(fft_direction).map(Ok);
            }
            243 => {
                return T::butterfly243(fft_direction).map(Ok);
            }
            256 => {
                return T::butterfly256(fft_direction).map(Ok);
            }
            512 => {
                return T::butterfly512(fft_direction).map(Ok);
            }
            1024 => {
                return T::butterfly1024(fft_direction).map(Ok);
            }
            _ => {}
        }
        None
    }

    pub(crate) fn strategy<T: FftSample>(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>
    where
        f64: AsPrimitive<T>,
    {
        if n == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        if n <= 512 || n == 1024 {
            if let Some(bf) = Zaft::plan_butterfly(n, fft_direction) {
                return bf;
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
            if Zaft::could_do_split_mixed_radix() {
                if n == 2048 {
                    if let Some(bf) =
                        T::mixed_radix_butterfly8(Zaft::strategy(n / 8, fft_direction)?)?
                    {
                        return Ok(bf);
                    }
                }
                let rem3 = prime_factors.factor_of_2() % 3;
                if rem3 == 2 {
                    if let Some(bf) =
                        T::mixed_radix_butterfly4(Zaft::strategy(n / 4, fft_direction)?)?
                    {
                        return Ok(bf);
                    }
                } else if rem3 == 1 {
                    let has1024 = T::butterfly1024(fft_direction).is_some();
                    if has1024 {
                        if let Some(bf) =
                            T::mixed_radix_butterfly8(Zaft::strategy(n / 8, fft_direction)?)?
                        {
                            return Ok(bf);
                        }
                    }
                    if let Some(bf) =
                        T::mixed_radix_butterfly2(Zaft::strategy(n / 2, fft_direction)?)?
                    {
                        return Ok(bf);
                    }
                }
                if let Some(bf) = T::mixed_radix_butterfly8(Zaft::strategy(n / 8, fft_direction)?)?
                {
                    return Ok(bf);
                }
            }
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

    /// Creates a Real-to-Complex (R2C) FFT plan executor for single-precision floating-point numbers (`f32`).
    ///
    /// This plan transforms a real-valued input array of length `n` into a complex
    /// output array of length `n/2 + 1` (or `n` if odd, with a special handling for the last complex element).
    ///
    /// # Parameters
    /// * `n`: The **length** of the real-valued input vector.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `R2CFftExecutor<f32>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_r2c_fft_f32(
        n: usize,
    ) -> Result<Arc<dyn R2CFftExecutor<f32> + Send + Sync>, ZaftError> {
        strategy_r2c(n)
    }

    /// Creates a Complex-to-Real (C2R) Inverse FFT plan executor for single-precision floating-point numbers (`f32`).
    ///
    /// This plan transforms a complex input array (the result of an R2C FFT)
    /// back into a real-valued output array of length `n`.
    ///
    /// # Parameters
    /// * `n`: The **length** of the final real-valued output vector.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `C2RFftExecutor<f32>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_c2r_fft_f32(
        n: usize,
    ) -> Result<Arc<dyn C2RFftExecutor<f32> + Send + Sync>, ZaftError> {
        if n == 1 {
            return Ok(Arc::new(OneSizedRealFft {
                phantom_data: Default::default(),
            }));
        }
        if n.is_multiple_of(2) {
            C2RFftEvenInterceptor::install(n, Zaft::strategy(n / 2, FftDirection::Inverse)?)
                .map(|x| Arc::new(x) as Arc<dyn C2RFftExecutor<f32> + Send + Sync>)
        } else {
            C2RFftOddInterceptor::install(n, Zaft::strategy(n, FftDirection::Inverse)?)
                .map(|x| Arc::new(x) as Arc<dyn C2RFftExecutor<f32> + Send + Sync>)
        }
    }

    /// Creates a standard Complex-to-Complex Forward FFT plan executor for single-precision floating-point numbers (`f32`).
    ///
    /// This is used for a standard Discrete Fourier Transform (DFT) where both input and output are complex.
    ///
    /// # Parameters
    /// * `n`: The **length** of the input/output complex vectors.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `FftExecutor<f32>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_forward_fft_f32(
        n: usize,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Forward)
    }

    /// Creates a standard Complex-to-Complex Forward FFT plan executor for double-precision floating-point numbers (`f64`).
    ///
    /// # Parameters
    /// * `n`: The **length** of the input/output complex vectors.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `FftExecutor<f64>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_forward_fft_f64(
        n: usize,
    ) -> Result<Arc<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Forward)
    }

    /// Creates a Complex-to-Real (C2R) Inverse FFT plan executor for double-precision floating-point numbers (`f64`).
    ///
    /// This is the double-precision version of `make_c2r_fft_f32`.
    ///
    /// # Parameters
    /// * `n`: The **length** of the final real-valued output vector.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `C2RFftExecutor<f64>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_c2r_fft_f64(
        n: usize,
    ) -> Result<Arc<dyn C2RFftExecutor<f64> + Send + Sync>, ZaftError> {
        if n == 1 {
            return Ok(Arc::new(OneSizedRealFft {
                phantom_data: Default::default(),
            }));
        }
        if n.is_multiple_of(2) {
            C2RFftEvenInterceptor::install(n, Zaft::strategy(n / 2, FftDirection::Inverse)?)
                .map(|x| Arc::new(x) as Arc<dyn C2RFftExecutor<f64> + Send + Sync>)
        } else {
            C2RFftOddInterceptor::install(n, Zaft::strategy(n, FftDirection::Inverse)?)
                .map(|x| Arc::new(x) as Arc<dyn C2RFftExecutor<f64> + Send + Sync>)
        }
    }

    /// Creates a Real-to-Complex (R2C) FFT plan executor for double-precision floating-point numbers (`f64`).
    ///
    /// This is the double-precision version of `make_r2c_fft_f32`.
    ///
    /// # Parameters
    /// * `n`: The **length** of the real-valued input vector.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `R2CFftExecutor<f64>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_r2c_fft_f64(
        n: usize,
    ) -> Result<Arc<dyn R2CFftExecutor<f64> + Send + Sync>, ZaftError> {
        strategy_r2c(n)
    }

    /// Creates a standard Complex-to-Complex Inverse FFT plan executor for single-precision floating-point numbers (`f32`).
    ///
    /// This is the inverse transformation for the standard DFT, used to convert frequency-domain data
    /// back into the time domain.
    ///
    /// # Parameters
    /// * `n`: The **length** of the input/output complex vectors.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `FftExecutor<f32>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_inverse_fft_f32(
        n: usize,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Inverse)
    }

    /// Creates a standard Complex-to-Complex Inverse FFT plan executor for double-precision floating-point numbers (`f64`).
    ///
    /// # Parameters
    /// * `n`: The **length** of the input/output complex vectors.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `FftExecutor<f64>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    pub fn make_inverse_fft_f64(
        n: usize,
    ) -> Result<Arc<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        Zaft::strategy(n, FftDirection::Inverse)
    }

    /// Creates a high-performance, two-dimensional Real-to-Complex (R2C) FFT plan executor for single-precision floating-point numbers (`f32`).
    ///
    /// This function constructs a plan for transforming a **real-valued** 2D input array (with dimensions `height x width`)
    /// into its frequency-domain representation. The executor is optimized for parallel
    /// execution across the specified number of threads.
    ///
    /// **The R2C 2D FFT is typically performed in two steps:**
    /// 1. A 1D R2C FFT is performed across the **rows** (or columns) of the input.
    /// 2. A 1D Complex-to-Complex (C2C) FFT is performed across the **columns** (or rows) of the intermediate complex data.
    ///
    /// # Parameters
    /// * `width`: The number of elements in the X-dimension (columns) of the 2D input array.
    /// * `height`: The number of elements in the Y-dimension (rows) of the 2D input array.
    /// * `thread_count`: The maximum number of threads the executor is allowed to use during execution.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `TwoDimensionalExecutorR2C<f32>` plan,
    /// or a `ZaftError` if the plan cannot be generated (e.g., due to invalid dimensions).
    ///
    /// # Note on Output Size
    /// The resulting frequency-domain complex data will typically have dimensions of `height x (width/2 + 1)`
    /// complex elements, leveraging the Hermitian symmetry property of real-input FFTs.
    pub fn make_2d_r2c_fft_f32(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorR2C<f32> + Send + Sync>, ZaftError> {
        if width * height == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        let width_fft = Zaft::make_r2c_fft_f32(width)?;
        let height_fft = Zaft::make_forward_fft_f32(height)?;
        let width_scratch_length = width_fft.complex_scratch_length();
        let height_scratch_length = height_fft.scratch_length();
        Ok(Arc::new(TwoDimensionalR2C {
            width_r2c_executor: width_fft,
            height_c2c_executor: height_fft,
            thread_count: thread_count.max(1),
            width,
            height,
            transpose_width_to_height: f32::transpose_strategy((width / 2) + 1, height),
            width_scratch_length,
            height_scratch_length,
        }))
    }

    /// Creates a high-performance, two-dimensional Real-to-Complex (R2C) FFT plan executor for **double-precision floating-point numbers** (`f64`).
    ///
    /// This function constructs a plan for transforming a **real-valued** 2D input array (with dimensions `height x width`)
    /// into its frequency-domain representation. The executor is highly optimized for parallel
    /// execution across the specified number of threads.
    ///
    /// **The R2C 2D FFT typically follows a row-column or column-row decomposition:**
    /// 1. A 1D R2C FFT is performed across the elements of one axis.
    /// 2. A 1D Complex-to-Complex (C2C) FFT is performed across the elements of the other axis on the intermediate complex data.
    ///
    /// This process efficiently computes the 2D transform while leveraging the computational savings offered
    /// by the real-valued nature of the input data.
    ///
    ///
    /// # Parameters
    /// * `width`: The number of elements in the X-dimension (**columns**) of the 2D input array.
    /// * `height`: The number of elements in the Y-dimension (**rows**) of the 2D input array.
    /// * `thread_count`: The maximum number of threads the executor is allowed to use during the parallel computation.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `TwoDimensionalExecutorR2C<f64>` plan,
    /// which is safe to use across threads (`Send + Sync`), or a `ZaftError` if the plan cannot be generated.
    ///
    /// # Note on Output Size
    /// Due to the resulting frequency-domain complex data will only store approximately half
    /// the data required for a full C2C transform. Specifically, the output complex array will typically have dimensions
    /// of `height x (width/2 + 1)` complex elements.
    pub fn make_2d_r2c_fft_f64(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorR2C<f64> + Send + Sync>, ZaftError> {
        if width * height == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        let width_fft = Zaft::make_r2c_fft_f64(width)?;
        let height_fft = Zaft::make_forward_fft_f64(height)?;
        let width_scratch_length = width_fft.complex_scratch_length();
        let height_scratch_length = height_fft.scratch_length();
        Ok(Arc::new(TwoDimensionalR2C {
            width_r2c_executor: width_fft,
            height_c2c_executor: height_fft,
            thread_count: thread_count.max(1),
            width,
            height,
            transpose_width_to_height: f64::transpose_strategy((width / 2) + 1, height),
            width_scratch_length,
            height_scratch_length,
        }))
    }

    /// Creates a 2D complex-to-complex FFT executor for f32 inputs.
    ///
    /// This function constructs a two-dimensional FFT executor that can perform
    /// forward or inverse FFTs on a width × height grid of complex f32 values.
    /// The executor internally manages separate FFTs along the width and height dimensions,
    /// and uses a transpose strategy for efficient computation.
    ///
    /// # Parameters
    ///
    /// * width - The number of columns in the input data grid.
    /// * height - The number of rows in the input data grid.
    /// * fft_direction - The direction of the FFT (Forward or Inverse).
    /// * thread_count - Number of threads to use for parallel computation (minimum 1).
    ///
    /// # Returns
    ///
    /// Returns a Result containing an Arc to a type implementing
    /// [TwoDimensionalFftExecutor<f32>], or a [ZaftError] if FFT creation fails.
    ///
    /// # Notes
    ///
    /// The internal width and height of the FFT executors are swapped for inverse FFTs
    /// to correctly handle the transformation.
    pub fn make_2d_c2c_fft_f32(
        width: usize,
        height: usize,
        fft_direction: FftDirection,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalFftExecutor<f32> + Send + Sync>, ZaftError> {
        if width * height == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        let fft_width = match fft_direction {
            FftDirection::Forward => width,
            FftDirection::Inverse => height,
        };
        let fft_height = match fft_direction {
            FftDirection::Forward => height,
            FftDirection::Inverse => width,
        };
        let width_fft = match fft_direction {
            FftDirection::Forward => Zaft::make_forward_fft_f32(fft_width)?,
            FftDirection::Inverse => Zaft::make_inverse_fft_f32(fft_width)?,
        };
        let height_fft = match fft_direction {
            FftDirection::Forward => Zaft::make_forward_fft_f32(fft_height)?,
            FftDirection::Inverse => Zaft::make_inverse_fft_f32(fft_height)?,
        };
        let oof_scratch_width = width_fft.out_of_place_scratch_length();
        let inplace_scratch_height = height_fft.scratch_length();
        Ok(Arc::new(TwoDimensionalC2C {
            width_c2c_executor: width_fft,
            height_c2c_executor: height_fft,
            thread_count: thread_count.max(1),
            width: fft_width,
            height: fft_height,
            transpose_width_to_height: f32::transpose_strategy(fft_width, fft_height),
            oof_width_scratch_size: oof_scratch_width,
            height_scratch_size: inplace_scratch_height,
        }))
    }

    /// Creates a 2D complex-to-complex FFT executor for f64 inputs.
    ///
    /// This function constructs a two-dimensional FFT executor that can perform
    /// forward or inverse FFTs on a width × height grid of complex f64 values.
    /// The executor internally manages separate FFTs along the width and height dimensions,
    /// and uses a transpose strategy for efficient computation.
    ///
    /// # Parameters
    ///
    /// * width - The number of columns in the input data grid.
    /// * height - The number of rows in the input data grid.
    /// * fft_direction - The direction of the FFT (Forward or Inverse).
    /// * thread_count - Number of threads to use for parallel computation (minimum 1).
    ///
    /// # Returns
    ///
    /// Returns a Result containing an Arc to a type implementing
    /// [TwoDimensionalFftExecutor<f64>], or a [ZaftError] if FFT creation fails.
    ///
    /// # Notes
    ///
    /// The internal width and height of the FFT executors are swapped for inverse FFTs
    /// to correctly handle the transformation.
    pub fn make_2d_c2c_fft_f64(
        width: usize,
        height: usize,
        fft_direction: FftDirection,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalFftExecutor<f64> + Send + Sync>, ZaftError> {
        if width * height == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        let fft_width = match fft_direction {
            FftDirection::Forward => width,
            FftDirection::Inverse => height,
        };
        let fft_height = match fft_direction {
            FftDirection::Forward => height,
            FftDirection::Inverse => width,
        };
        let width_fft = match fft_direction {
            FftDirection::Forward => Zaft::make_forward_fft_f64(fft_width)?,
            FftDirection::Inverse => Zaft::make_inverse_fft_f64(fft_width)?,
        };
        let height_fft = match fft_direction {
            FftDirection::Forward => Zaft::make_forward_fft_f64(fft_height)?,
            FftDirection::Inverse => Zaft::make_inverse_fft_f64(fft_height)?,
        };
        let oof_scratch_width = width_fft.out_of_place_scratch_length();
        let inplace_scratch_height = height_fft.scratch_length();
        Ok(Arc::new(TwoDimensionalC2C {
            width_c2c_executor: width_fft,
            height_c2c_executor: height_fft,
            thread_count: thread_count.max(1),
            width: fft_width,
            height: fft_height,
            transpose_width_to_height: f64::transpose_strategy(fft_width, fft_height),
            oof_width_scratch_size: oof_scratch_width,
            height_scratch_size: inplace_scratch_height,
        }))
    }

    /// Creates two-dimensional Complex-to-Real (C2R) Inverse FFT plan executor
    /// for **single-precision floating-point numbers** (`f32`).
    ///
    /// This plan transforms a **Hermitian symmetric complex** frequency-domain input array (the output of an R2C FFT)
    /// back into a **real-valued** time-domain output array of size `height x width`. The executor is configured
    /// for parallel execution across the specified number of threads.
    ///
    /// # Parameters
    /// * `width`: The number of elements in the X-dimension (**columns**) of the final real output.
    /// * `height`: The number of elements in the Y-dimension (**rows**) of the final real output.
    /// * `thread_count`: The maximum number of threads the executor is allowed to use during the parallel computation.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `TwoDimensionalExecutorC2R<f32>` plan,
    /// or a `ZaftError` if the plan cannot be generated (e.g., `ZeroSizedFft`).
    ///
    /// # Errors
    /// Returns `ZaftError::ZeroSizedFft` if `width * height` is zero.
    pub fn make_2d_c2r_fft_f32(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorC2R<f32> + Send + Sync>, ZaftError> {
        if width * height == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        let width_fft = Zaft::make_c2r_fft_f32(width)?;
        let height_fft = Zaft::make_inverse_fft_f32(height)?;
        let width_scratch_length = width_fft.complex_scratch_length();
        let height_scratch_length = height_fft.scratch_length();
        Ok(Arc::new(TwoDimensionalC2R {
            width_c2r_executor: width_fft,
            height_c2c_executor: height_fft,
            thread_count: thread_count.max(1),
            width,
            height,
            transpose_height_to_width: f32::transpose_strategy(height, (width / 2) + 1),
            width_scratch_length,
            height_scratch_length,
        }))
    }

    /// Creates two-dimensional Complex-to-Real (C2R) Inverse FFT plan executor
    /// for **double-precision floating-point numbers** (`f64`).
    ///
    /// This is the double-precision equivalent of `make_2d_c2r_fft_f32`. It transforms the complex,
    /// frequency-domain input back into a real-valued array of size `height x width`.
    ///
    /// # Parameters
    /// * `width`: The number of elements in the X-dimension (**columns**) of the final real output.
    /// * `height`: The number of elements in the Y-dimension (**rows**) of the final real output.
    /// * `thread_count`: The maximum number of threads the executor is allowed to use during the parallel computation.
    ///
    /// # Returns
    /// A `Result` containing an `Arc` to a dynamically dispatched `TwoDimensionalExecutorC2R<f64>` plan,
    /// or a `ZaftError` if the plan cannot be generated.
    ///
    /// # Errors
    /// Returns `ZaftError::ZeroSizedFft` if `width * height` is zero.
    pub fn make_2d_c2r_fft_f64(
        width: usize,
        height: usize,
        thread_count: usize,
    ) -> Result<Arc<dyn TwoDimensionalExecutorC2R<f64> + Send + Sync>, ZaftError> {
        if width * height == 0 {
            return Err(ZaftError::ZeroSizedFft);
        }
        let width_fft = Zaft::make_c2r_fft_f64(width)?;
        let height_fft = Zaft::make_inverse_fft_f64(height)?;
        let width_scratch_length = width_fft.complex_scratch_length();
        let height_scratch_length = height_fft.scratch_length();
        Ok(Arc::new(TwoDimensionalC2R {
            width_c2r_executor: width_fft,
            height_c2c_executor: height_fft,
            thread_count: thread_count.max(1),
            width,
            height,
            transpose_height_to_width: f64::transpose_strategy(height, (width / 2) + 1),
            width_scratch_length,
            height_scratch_length,
        }))
    }
}

/// Specifies the direction of a Fast Fourier Transform (FFT) operation.
///
/// This enum is used to configure FFT executors, indicating whether the transform should
/// map from the time domain to the frequency domain, or vice versa.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub enum FftDirection {
    /// Represents the **Forward** transform, which typically maps data from the **time or spatial domain**
    /// into the **frequency domain**.
    Forward,
    /// Represents the **Inverse** transform, which maps data back from the **frequency domain**
    /// into the **time or spatial domain**.
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
    use num_traits::Zero;

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
        for i in 1..2010 {
            let mut data = vec![Complex::new(0.0019528865, 0.); i];
            for (i, chunk) in data.iter_mut().enumerate() {
                *chunk = Complex::new(
                    -0.19528865 + i as f32 * 0.001,
                    0.0019528865 - i as f32 * 0.001,
                );
            }
            let zaft_exec = Zaft::make_forward_fft_f32(data.len()).expect("Failed to make FFT!");
            let zaft_inverse = Zaft::make_inverse_fft_f32(data.len()).expect("Failed to make FFT!");
            let reference_clone = data.clone();
            zaft_exec
                .execute(&mut data)
                .expect(&format!("Failed to execute forward FFT for size {i}!"));
            zaft_inverse
                .execute(&mut data)
                .expect(&format!("Failed to execute inverse FFT for size {i}!"));
            let data_len = 1. / data.len() as f32;
            for i in data.iter_mut() {
                *i *= data_len;
            }
            data.iter()
                .zip(reference_clone)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.re,
                        b.re
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.im,
                        b.im
                    );
                });
        }
    }

    #[test]
    fn test_everything_oof_f32() {
        for i in 1..2010 {
            let mut data = vec![Complex::new(0.0019528865, 0.); i];
            let mut scratch = data.to_vec();
            for (i, chunk) in data.iter_mut().enumerate() {
                *chunk = Complex::new(
                    -0.19528865 + i as f32 * 0.001,
                    0.0019528865 - i as f32 * 0.001,
                );
            }
            let zaft_exec = Zaft::make_forward_fft_f32(data.len()).expect("Failed to make FFT!");
            let zaft_inverse = Zaft::make_inverse_fft_f32(data.len()).expect("Failed to make FFT!");
            let reference_clone = data.clone();
            zaft_exec
                .execute_out_of_place(&data, &mut scratch)
                .expect(&format!("Failed to execute forward FFT for size {i}!"));
            zaft_inverse
                .execute_out_of_place(&scratch, &mut data)
                .expect(&format!("Failed to execute inverse FFT for size {i}!"));
            let data_len = 1. / data.len() as f32;
            for i in data.iter_mut() {
                *i *= data_len;
            }
            data.iter()
                .zip(reference_clone)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.re,
                        b.re
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.im,
                        b.im
                    );
                });
        }
    }

    #[test]
    fn test_everything_f64() {
        for i in 1..1900 {
            let mut data = vec![Complex::new(0.0019528865, 0.); i];
            for (i, chunk) in data.iter_mut().enumerate() {
                *chunk = Complex::new(
                    -0.19528865 + i as f64 * 0.001,
                    0.0019528865 - i as f64 * 0.001,
                );
            }
            let zaft_exec = Zaft::make_forward_fft_f64(data.len()).expect("Failed to make FFT!");
            let zaft_inverse = Zaft::make_inverse_fft_f64(data.len()).expect("Failed to make FFT!");
            let rust_fft_clone = data.clone();
            zaft_exec
                .execute(&mut data)
                .expect(&format!("Failed to execute forward FFT for size {i}!"));
            zaft_inverse
                .execute(&mut data)
                .expect(&format!("Failed to execute inverse FFT for size {i}!"));
            let data_len = 1. / data.len() as f64;
            for i in data.iter_mut() {
                *i *= data_len;
            }
            data.iter()
                .zip(rust_fft_clone)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-6,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.re,
                        b.re
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-6,
                        "a_im {}, b_im {} at {idx}, for size {i}",
                        a.im,
                        b.im
                    );
                });
        }
    }

    #[test]
    fn test_everything_oof_f64() {
        for i in 1..1900 {
            let mut data = vec![Complex::new(0.0019528865, 0.); i];
            let mut scratch = data.clone();
            for (i, chunk) in data.iter_mut().enumerate() {
                *chunk = Complex::new(
                    -0.19528865 + i as f64 * 0.001,
                    0.0019528865 - i as f64 * 0.001,
                );
            }
            let zaft_exec = Zaft::make_forward_fft_f64(data.len()).expect("Failed to make FFT!");
            let zaft_inverse = Zaft::make_inverse_fft_f64(data.len()).expect("Failed to make FFT!");
            let rust_fft_clone = data.clone();
            zaft_exec
                .execute_out_of_place(&data, &mut scratch)
                .expect(&format!("Failed to execute forward FFT for size {i}!"));
            zaft_inverse
                .execute_out_of_place(&scratch, &mut data)
                .expect(&format!("Failed to execute inverse FFT for size {i}!"));
            let data_len = 1. / data.len() as f64;
            for i in data.iter_mut() {
                *i *= data_len;
            }
            data.iter()
                .zip(rust_fft_clone)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-6,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.re,
                        b.re
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-6,
                        "a_im {}, b_im {} at {idx}, for size {i}",
                        a.im,
                        b.im
                    );
                });
        }
    }

    #[test]
    fn test_destructive_everything_f64() {
        for i in 1..1900 {
            let mut data = vec![Complex::new(0.0019528865, 0.); i];
            for (i, chunk) in data.iter_mut().enumerate() {
                *chunk = Complex::new(
                    -0.19528865 + i as f64 * 0.001,
                    0.0019528865 - i as f64 * 0.001,
                );
            }
            let zaft_exec = Zaft::make_forward_fft_f64(data.len()).expect("Failed to make FFT!");
            let zaft_inverse = Zaft::make_inverse_fft_f64(data.len()).expect("Failed to make FFT!");
            let rust_fft_clone = data.clone();
            let mut fwd = vec![Complex::zero(); data.len()];
            let mut scratch = vec![Complex::zero(); zaft_exec.destructive_scratch_length()];
            zaft_exec
                .execute_destructive_with_scratch(&mut data, &mut fwd, &mut scratch)
                .expect(&format!("Failed to execute forward FFT for size {i}!"));
            zaft_inverse
                .execute_destructive_with_scratch(&mut fwd, &mut data, &mut scratch)
                .expect(&format!("Failed to execute inverse FFT for size {i}!"));
            let data_len = 1. / data.len() as f64;
            for i in data.iter_mut() {
                *i *= data_len;
            }
            data.iter()
                .zip(rust_fft_clone)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-6,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.re,
                        b.re
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-6,
                        "a_im {}, b_im {} at {idx}, for size {i}",
                        a.im,
                        b.im
                    );
                });
        }
    }

    #[test]
    fn test_destructive_everything_oof_f32() {
        for i in 1..2010 {
            let mut data = vec![Complex::new(0.0019528865, 0.); i];
            for (i, chunk) in data.iter_mut().enumerate() {
                *chunk = Complex::new(
                    -0.19528865 + i as f32 * 0.001,
                    0.0019528865 - i as f32 * 0.001,
                );
            }
            let zaft_exec = Zaft::make_forward_fft_f32(data.len()).expect("Failed to make FFT!");
            let zaft_inverse = Zaft::make_inverse_fft_f32(data.len()).expect("Failed to make FFT!");
            let reference_clone = data.clone();
            let mut scratch = vec![Complex::zero(); zaft_exec.destructive_scratch_length()];
            let mut target = vec![Complex::zero(); data.len()];
            zaft_exec
                .execute_destructive_with_scratch(&mut data, &mut target, &mut scratch)
                .expect(&format!("Failed to execute forward FFT for size {i}!"));
            zaft_inverse
                .execute_destructive_with_scratch(&mut target, &mut data, &mut scratch)
                .expect(&format!("Failed to execute inverse FFT for size {i}!"));
            let data_len = 1. / data.len() as f32;
            for i in data.iter_mut() {
                *i *= data_len;
            }
            data.iter()
                .zip(reference_clone)
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.re,
                        b.re
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-2,
                        "a_re {}, b_re {} at {idx}, for size {i}",
                        a.im,
                        b.im
                    );
                });
        }
    }
}
