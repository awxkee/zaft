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
use crate::bluestein::BluesteinFft;
use crate::butterflies::Butterfly1;
use crate::dft::Dft;
use crate::good_thomas::GoodThomasFft;
use crate::good_thomas_small::GoodThomasSmallFft;
use crate::mixed_radix::MixedRadix;
use crate::{FftDirection, FftExecutor, ZaftError};

pub(crate) trait AlgorithmFactory<T> {
    fn butterfly1(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly2(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly3(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly4(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly5(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly6(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly7(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly8(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly9(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly10(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly11(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly12(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly13(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly14(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly15(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly16(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly17(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly19(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly23(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix3(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix4(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix5(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix6(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix7(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix10(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix11(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix13(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn dft(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    fn raders(
        convolve_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    fn bluestein(
        convolve_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    fn mixed_radix(
        left_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    fn good_thomas(
        left_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
}

impl AlgorithmFactory<f32> for f32 {
    fn butterfly1(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Ok(Box::new(Butterfly1 {
            phantom_data: Default::default(),
            direction: fft_direction,
        }))
    }

    fn butterfly2(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly2;
            return Ok(Box::new(NeonButterfly2::new(fft_direction)));
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx::AvxButterfly2;
            return Ok(Box::new(AvxButterfly2::new(fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly2;
            Ok(Box::new(Butterfly2::new(fft_direction)))
        }
    }

    fn butterfly3(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            use crate::avx::AvxButterfly3;
            return Ok(Box::new(AvxButterfly3::new(fft_direction)));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly3;
            return Ok(Box::new(NeonButterfly3::new(fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly3;
            Ok(Box::new(Butterfly3::new(fft_direction)))
        }
    }

    fn butterfly4(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly4;
            return Ok(Box::new(NeonButterfly4::new(fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxButterfly4;
                return Ok(Box::new(AvxButterfly4::new(fft_direction)));
            }
            use crate::butterflies::Butterfly4;
            Ok(Box::new(Butterfly4::new(fft_direction)))
        }
    }

    fn butterfly5(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly5;
                return Ok(Box::new(NeonFcmaButterfly5::new(fft_direction)));
            }
            use crate::neon::NeonButterfly5;
            Ok(Box::new(NeonButterfly5::new(fft_direction)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            use crate::avx::AvxButterfly5;
            return Ok(Box::new(AvxButterfly5::new(fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly5;
            Ok(Box::new(Butterfly5::new(fft_direction)))
        }
    }

    fn butterfly6(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly6;
            Ok(Box::new(NeonButterfly6::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxButterfly6;
                return Ok(Box::new(AvxButterfly6::new(fft_direction)));
            }
            use crate::butterflies::Butterfly6;
            Ok(Box::new(Butterfly6::new(fft_direction)))
        }
    }

    fn butterfly7(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly7;
                return Ok(Box::new(NeonFcmaButterfly7::new(fft_direction)));
            }
            use crate::neon::NeonButterfly7;
            Ok(Box::new(NeonButterfly7::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly7;
            Ok(Box::new(Butterfly7::new(fft_direction)))
        }
    }

    fn butterfly8(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly8;
                return Ok(Box::new(NeonFcmaButterfly8::new(fft_direction)));
            }
            use crate::neon::NeonButterfly8;
            Ok(Box::new(NeonButterfly8::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxButterfly8;
                return Ok(Box::new(AvxButterfly8::new(fft_direction)));
            }
            use crate::butterflies::Butterfly8;
            Ok(Box::new(Butterfly8::new(fft_direction)))
        }
    }

    fn butterfly9(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly9;
                return Ok(Box::new(NeonFcmaButterfly9::new(fft_direction)));
            }
            use crate::neon::NeonButterfly9;
            Ok(Box::new(NeonButterfly9::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly9;
            Ok(Box::new(Butterfly9::new(fft_direction)))
        }
    }

    fn butterfly10(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly10;
                return Ok(Box::new(NeonFcmaButterfly10::new(fft_direction)));
            }
            use crate::neon::NeonButterfly10;
            Ok(Box::new(NeonButterfly10::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly10;
            Ok(Box::new(Butterfly10::new(fft_direction)))
        }
    }

    fn butterfly11(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly11;
                return Ok(Box::new(NeonFcmaButterfly11::new(fft_direction)));
            }
            use crate::neon::NeonButterfly11;
            Ok(Box::new(NeonButterfly11::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly11;
            Ok(Box::new(Butterfly11::new(fft_direction)))
        }
    }

    fn butterfly12(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly12;
            Ok(Box::new(NeonButterfly12::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly12;
            Ok(Box::new(Butterfly12::new(fft_direction)))
        }
    }

    fn butterfly13(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly13;
                return Ok(Box::new(NeonFcmaButterfly13::new(fft_direction)));
            }
            use crate::neon::NeonButterfly13;
            Ok(Box::new(NeonButterfly13::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly13;
            Ok(Box::new(Butterfly13::new(fft_direction)))
        }
    }

    fn butterfly14(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly14;
                return Ok(Box::new(NeonFcmaButterfly14::new(fft_direction)));
            }
            use crate::neon::NeonButterfly14;
            Ok(Box::new(NeonButterfly14::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly14;
            Ok(Box::new(Butterfly14::new(fft_direction)))
        }
    }

    fn butterfly15(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly15;
                return Ok(Box::new(NeonFcmaButterfly15::new(fft_direction)));
            }
            use crate::neon::NeonButterfly15;
            Ok(Box::new(NeonButterfly15::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly15;
            Ok(Box::new(Butterfly15::new(fft_direction)))
        }
    }

    fn butterfly16(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly16;
                return Ok(Box::new(NeonFcmaButterfly16::new(fft_direction)));
            }
            use crate::neon::NeonButterfly16;
            Ok(Box::new(NeonButterfly16::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly16;
            Ok(Box::new(Butterfly16::new(fft_direction)))
        }
    }

    fn butterfly17(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly17;
                return Ok(Box::new(NeonFcmaButterfly17::new(fft_direction)));
            }
            use crate::neon::NeonButterfly17;
            Ok(Box::new(NeonButterfly17::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly17;
            Ok(Box::new(Butterfly17::new(fft_direction)))
        }
    }

    fn butterfly19(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly19;
            Ok(Box::new(NeonButterfly19::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly19;
            Ok(Box::new(Butterfly19::new(fft_direction)))
        }
    }

    fn butterfly23(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly23;
            Ok(Box::new(NeonButterfly23::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly23;
            Ok(Box::new(Butterfly23::new(fft_direction)))
        }
    }

    fn radix3(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        if n == 3 {
            return Self::butterfly3(fft_direction);
        }
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

    fn radix4(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        if n == 4 {
            return Self::butterfly4(fft_direction);
        }
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

    fn radix5(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        if n == 5 {
            return Self::butterfly5(fft_direction);
        }
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

    fn radix6(
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
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    use crate::avx::AvxFmaRadix6;
                    return AvxFmaRadix6::new(n, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
            use crate::radix6::Radix6;
            Radix6::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix7(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix7;
                return NeonFcmaRadix7::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix7;
            NeonRadix7::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::radix7::Radix7;
            Radix7::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix10(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix10;
                return NeonFcmaRadix10::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix10;
            NeonRadix10::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::radix10::Radix10;
            Radix10::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix11(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix11;
                return NeonFcmaRadix11::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix11;
            NeonRadix11::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::radix11::Radix11;
            Radix11::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn radix13(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaRadix13;
                return NeonFcmaRadix13::new(n, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::NeonRadix13;
            NeonRadix13::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::radix13::Radix13;
            Radix13::new(n, fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }

    fn dft(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Dft::new(n, fft_direction).map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
    }

    fn raders(
        convolve_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                if n < (u32::MAX - 100_000u32) as usize {
                    use crate::avx::AvxRadersFft;
                    unsafe {
                        return AvxRadersFft::new(n, convolve_fft, fft_direction)
                            .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
                    }
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if n < (u32::MAX - 100_000u32) as usize {
                use crate::neon::NeonRadersFft;
                return NeonRadersFft::new(n, convolve_fft, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
        }
        use crate::raders::RadersFft;
        RadersFft::new(n, convolve_fft, fft_direction)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
    }

    fn bluestein(
        convolve_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        BluesteinFft::new(n, convolve_fft, fft_direction)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
    }

    fn mixed_radix(
        left_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        MixedRadix::new(left_fft, right_fft)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
    }

    fn good_thomas(
        left_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        let length = left_fft.length() * right_fft.length();
        if length < (u16::MAX - 100) as usize {
            return GoodThomasSmallFft::new(left_fft, right_fft)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
        }
        GoodThomasFft::new(left_fft, right_fft)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
    }
}
