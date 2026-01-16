/*
 * // Copyright (c) Radzivon Bartoshyk 1/2026. All rights reserved.
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
use crate::prime_factors::PrimeFactors;
use crate::r2c::R2CFftEvenInterceptor;
use crate::r2c::mixed_radix_r2c_odd::MixedRadixR2cOdd;
use crate::util::{
    ALWAYS_BLUESTEIN_1000, ALWAYS_BLUESTEIN_2000, ALWAYS_BLUESTEIN_3000, ALWAYS_BLUESTEIN_4000,
    ALWAYS_BLUESTEIN_5000, ALWAYS_BLUESTEIN_6000,
};
use crate::{FftDirection, FftSample, R2CFftExecutor, Zaft, ZaftError};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) fn r2c_butterflies<T: FftSample>(
    len: usize,
) -> Option<Arc<dyn R2CFftExecutor<T> + Send + Sync>> {
    match len {
        1 => Some(T::r2c_butterfly1()),
        2 => Some(T::r2c_butterfly2()),
        3 => Some(T::r2c_butterfly3()),
        4 => Some(T::r2c_butterfly4()),
        5 => Some(T::r2c_butterfly5()),
        6 => Some(T::r2c_butterfly6()),
        7 => Some(T::r2c_butterfly7()),
        8 => Some(T::r2c_butterfly8()),
        9 => Some(T::r2c_butterfly9()),
        11 => Some(T::r2c_butterfly11()),
        12 => Some(T::r2c_butterfly12()),
        13 => Some(T::r2c_butterfly13()),
        14 => Some(T::r2c_butterfly14()),
        15 => Some(T::r2c_butterfly15()),
        16 => Some(T::r2c_butterfly16()),
        17 => Some(T::r2c_butterfly17()),
        19 => Some(T::r2c_butterfly19()),
        23 => Some(T::r2c_butterfly23()),
        29 => Some(T::r2c_butterfly29()),
        31 => Some(T::r2c_butterfly31()),
        32 => Some(T::r2c_butterfly32()),
        _ => None,
    }
}

fn make_prime<T: FftSample>(n: usize) -> Result<Arc<dyn R2CFftExecutor<T> + Send + Sync>, ZaftError>
where
    f64: AsPrimitive<T>,
{
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
            return T::r2c_bluestein(n);
        }
        return T::r2c_raders(n);
    }
    // n-1 may result in Cunningham chain, and we want to avoid compute multiple prime numbers FFT at once
    let big_factor = convolve_prime.factorization.iter().any(|x| x.0 > 31);
    let new_prime = if !big_factor {
        T::r2c_raders(n)
    } else {
        T::r2c_bluestein(n)
    };
    let fft_executor = new_prime?;
    Ok(fft_executor)
}

pub(crate) fn strategy_r2c<T: FftSample>(
    len: usize,
) -> Result<Arc<dyn R2CFftExecutor<T> + Send + Sync>, ZaftError>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    if len == 0 {
        return Err(ZaftError::ZeroSizedFft);
    }
    if let Some(butterfly) = r2c_butterflies(len) {
        return Ok(butterfly);
    }

    let prime_factors = PrimeFactors::from_number(len as u64);

    fn pick_odd_factor(number: u64, factors: &[(u64, u32)]) -> u64 {
        // we want to pick some of smooth and very fast factors first, to perform fast first stage
        // and remove reasonable amount of work then, as well we want to consider that strided
        // copying of very small sizes as 2 is not effective so it's better to take
        // a bit bigger factors first, if there are none,
        // well, pick something we have
        static FACTORS: [u64; 30] = [
            16, 27, 8, 9, 10, 12, 25, 22, 15, 20, 24, 14, 21, 18, 28, 30, 26, 11, 13, 6, 7, 5, 4,
            3, 2, 17, 19, 23, 29, 31,
        ];
        for factor in FACTORS {
            if number.is_multiple_of(factor) {
                return factor;
            }
        }
        factors[0].0
    }

    if len.is_multiple_of(2) {
        R2CFftEvenInterceptor::install(len, Zaft::strategy(len / 2, FftDirection::Forward)?)
            .map(|x| Arc::new(x) as Arc<dyn R2CFftExecutor<T> + Send + Sync>)
    } else {
        if prime_factors.is_prime() {
            return make_prime(len);
        }

        let factor_out = pick_odd_factor(len as u64, &prime_factors.factorization);
        Ok(Arc::new(MixedRadixR2cOdd::new(
            Zaft::strategy((len as u64 / factor_out) as usize, FftDirection::Forward)?,
            Zaft::strategy(factor_out as usize, FftDirection::Forward)?,
        )?))

        // R2CFftOddInterceptor::install(len, Zaft::strategy(len, FftDirection::Forward)?)
        //     .map(|x| Arc::new(x) as Arc<dyn R2CFftExecutor<T> + Send + Sync>)
    }
}
