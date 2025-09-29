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
mod complex_fma;
mod dft;
mod err;
mod mla;
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
mod neon;
mod radix2;
mod radix3;
mod radix4;
mod radix5;
mod radix6;
mod traits;
mod util;

pub use err::ZaftError;

use crate::dft::Dft;
use crate::util::{is_power_of_five, is_power_of_six, is_power_of_three};
use num_complex::Complex;

/// Bit reversal permutation
fn bit_reverse_indices(n: usize) -> Vec<usize> {
    let bits = n.trailing_zeros();
    (0..n)
        .map(|i| i.reverse_bits() >> (usize::BITS - bits))
        .collect()
}

pub trait FftExecutor<T> {
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError>;
}

pub struct Zaft {}

impl Zaft {
    fn radix3_f32(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        // Use Radix-3 if divisible by 3
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix3;
                return NeonFcmaRadix3::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix3;
            NeonRadix3::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix3;
                    return AvxFmaRadix3::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
            use crate::radix3::Radix3;
            Radix3::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix5_f32(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix5;
                return NeonFcmaRadix5::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix5;
            NeonRadix5::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix5;
                    return AvxFmaRadix5::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
            use crate::radix5::Radix5;
            Radix5::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix4_f32(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix4;
                return NeonFcmaRadix4::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix4;
            NeonRadix4::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix4;
                    return AvxFmaRadix4::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
            use crate::radix4::Radix4;
            Radix4::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix2_f32(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix2;
                return NeonFcmaRadix2::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix2;
            NeonRadix2::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix2;
                    return AvxFmaRadix2::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
            use crate::radix2::Radix2;
            Radix2::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix6_f32(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix6;
                return NeonFcmaRadix6::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix6;
            NeonRadix6::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::radix6::Radix6;
            Radix6::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn strategy_f32(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        if is_power_of_three(n) {
            // Use Radix-3 if divisible by 3
            Zaft::radix3_f32(n, fft_direction)
        } else if is_power_of_five(n) {
            // Use Radix-5 if power of 5
            Zaft::radix5_f32(n, fft_direction)
        } else if n.is_power_of_two() && n.trailing_zeros() % 2 == 0 {
            // Use Radix-4 if a power of 4
            Zaft::radix4_f32(n, fft_direction)
        } else if n.is_power_of_two() {
            // Otherwise, fallback to Radix-2
            Zaft::radix2_f32(n, fft_direction)
        } else if is_power_of_six(n) {
            Zaft::radix6_f32(n, fft_direction)
        } else {
            Dft::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix3_f64(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix3;
                return NeonFcmaRadix3::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
            }
            use crate::neon::NeonRadix3;
            NeonRadix3::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix3;
                    return AvxFmaRadix3::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
                }
            }
            use crate::radix3::Radix3;
            Radix3::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
    }

    fn radix5_f64(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix5;
                return NeonFcmaRadix5::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
            }
            use crate::neon::NeonRadix5;
            NeonRadix5::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix5;
                    return AvxFmaRadix5::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
                }
            }
            use crate::radix5::Radix5;
            Radix5::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
    }

    fn radix4_f64(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix4;
                return NeonFcmaRadix4::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
            }
            use crate::neon::NeonRadix4;
            NeonRadix4::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix4;
                    return AvxFmaRadix4::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
                }
            }
            use crate::radix4::Radix4;
            Radix4::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
    }

    fn radix2_f64(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix2;
                return NeonFcmaRadix2::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
            }
            use crate::neon::NeonRadix2;
            NeonRadix2::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix2;
                    return AvxFmaRadix2::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
                }
            }
            use crate::radix2::Radix2;
            Radix2::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
    }

    fn radix6_f64(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix6;
                return NeonFcmaRadix6::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
            }
            use crate::neon::NeonRadix6;
            NeonRadix6::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::radix6::Radix6;
            Radix6::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
    }

    fn strategy_f64(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        if is_power_of_three(n) {
            // Use Radix-3 if divisible by 3
            Zaft::radix3_f64(n, fft_direction)
        } else if is_power_of_five(n) {
            // Use Radix-5 if power of 5
            Zaft::radix5_f64(n, fft_direction)
        } else if n.is_power_of_two() && n.trailing_zeros() % 2 == 0 {
            // Use Radix-4 if a power of 4
            Zaft::radix4_f64(n, fft_direction)
        } else if n.is_power_of_two() {
            // Otherwise, fallback to Radix-2
            Zaft::radix2_f64(n, fft_direction)
        } else if is_power_of_six(n) {
            Zaft::radix6_f64(n, fft_direction)
        } else {
            Dft::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
    }

    pub fn make_forward_fft_f32(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Zaft::strategy_f32(n, FftDirection::Forward)
    }

    pub fn make_forward_fft_f64(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        Zaft::strategy_f64(n, FftDirection::Forward)
    }

    pub fn make_inverse_fft_f32(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Zaft::strategy_f32(n, FftDirection::Inverse)
    }

    pub fn make_inverse_fft_f64(
        n: usize,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        Zaft::strategy_f64(n, FftDirection::Inverse)
    }
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash)]
pub(crate) enum FftDirection {
    Forward,
    Inverse,
}

#[cfg(test)]
mod tests {}
