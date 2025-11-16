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
use crate::factory::{AlgorithmFactory, make_default_butterfly};
use crate::good_thomas::GoodThomasFft;
use crate::good_thomas_small::GoodThomasSmallFft;
use crate::mixed_radix::MixedRadix;
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::util::has_valid_avx;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};

macro_rules! make_default_radix {
    ($n: expr, $fft_direction: expr, $scalar_name: ident, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::$fcma_name;
                return $fcma_name::new($n, $fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
            }
            use crate::neon::$neon_name;
            $neon_name::new($n, $fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return $avx_name::new($n, $fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
                }
            }
            use crate::$scalar_name;
            $scalar_name::new($n, $fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
        }
    }};
}

impl AlgorithmFactory<f64> for f64 {
    fn butterfly1(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        Ok(Box::new(Butterfly1 {
            phantom_data: Default::default(),
            direction: fft_direction,
        }))
    }

    fn butterfly2(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly2;
            Ok(Box::new(NeonButterfly2::new(fft_direction)))
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
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxButterfly3;
            return Ok(Box::new(AvxButterfly3::new(fft_direction)));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly3;
            Ok(Box::new(NeonButterfly3::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly3;
            Ok(Box::new(Butterfly3::new(fft_direction)))
        }
    }

    fn butterfly4(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly4,
            AvxButterfly4,
            NeonButterfly4,
            NeonFcmaButterfly4
        )
    }

    fn butterfly5(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly5,
            AvxButterfly5,
            NeonButterfly5,
            NeonFcmaButterfly5
        )
    }

    fn butterfly6(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly6,
            AvxButterfly6,
            NeonButterfly6,
            NeonFcmaButterfly6
        )
    }

    fn butterfly7(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly7,
            AvxButterfly7,
            NeonButterfly7,
            NeonFcmaButterfly7
        )
    }

    fn butterfly8(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly8,
            AvxButterfly8,
            NeonButterfly8,
            NeonFcmaButterfly8
        )
    }

    fn butterfly9(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly9,
            AvxButterfly9,
            NeonButterfly9,
            NeonFcmaButterfly9
        )
    }

    fn butterfly10(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly10,
            AvxButterfly10d,
            NeonButterfly10,
            NeonFcmaButterfly10
        )
    }

    fn butterfly11(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly11,
            AvxButterfly11,
            NeonButterfly11,
            NeonFcmaButterfly11
        )
    }

    fn butterfly12(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly12,
            AvxButterfly12,
            NeonButterfly12,
            NeonFcmaButterfly12
        )
    }

    fn butterfly13(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly13,
            AvxButterfly13,
            NeonButterfly13,
            NeonFcmaButterfly13
        )
    }

    fn butterfly14(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly14,
            AvxButterfly14,
            NeonButterfly14,
            NeonFcmaButterfly14
        )
    }

    fn butterfly15(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly15,
            AvxButterfly15d,
            NeonButterfly15,
            NeonFcmaButterfly15
        )
    }

    fn butterfly16(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly16,
            AvxButterfly16,
            NeonButterfly16,
            NeonFcmaButterfly16
        )
    }

    fn butterfly17(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly17,
            AvxButterfly17,
            NeonButterfly17,
            NeonFcmaButterfly17
        )
    }

    fn butterfly18(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly18,
            AvxButterfly18d,
            NeonButterfly18d,
            NeonFcmaButterfly18d
        )
    }

    fn butterfly19(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly19,
            AvxButterfly19,
            NeonButterfly19,
            NeonFcmaButterfly19
        )
    }

    fn butterfly20(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly20,
            AvxButterfly20d,
            NeonButterfly20,
            NeonFcmaButterfly20
        )
    }

    fn butterfly23(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly23,
            AvxButterfly23,
            NeonButterfly23,
            NeonFcmaButterfly23
        )
    }

    fn butterfly25(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly25,
            AvxButterfly25d,
            NeonButterfly25d,
            NeonFcmaButterfly25d
        )
    }

    fn butterfly27(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly27,
            AvxButterfly27d,
            NeonButterfly27d,
            NeonFcmaButterfly27d
        )
    }

    fn butterfly29(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly29,
            AvxButterfly29,
            NeonButterfly29,
            NeonFcmaButterfly29
        )
    }

    fn butterfly31(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly31,
            AvxButterfly31,
            NeonButterfly31d,
            NeonFcmaButterfly31d
        )
    }

    fn butterfly32(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly32,
            AvxButterfly32d,
            NeonButterfly32d,
            NeonFcmaButterfly32d
        )
    }

    fn butterfly36(
        _direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<f64> + Send + Sync>> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() {
                use crate::avx::AvxButterfly36d;
                return Some(Box::new(AvxButterfly36d::new(_direction)));
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly36d;
                return Some(Box::new(NeonFcmaButterfly36d::new(_direction)));
            }
            use crate::neon::NeonButterfly36d;
            Some(Box::new(NeonButterfly36d::new(_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            None
        }
    }

    fn butterfly49(
        _fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<f64> + Send + Sync>> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxButterfly49d;
            return Some(Box::new(AvxButterfly49d::new(_fft_direction)));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly49d;
                return Some(Box::new(NeonFcmaButterfly49d::new(_fft_direction)));
            }
            use crate::neon::NeonButterfly49d;
            Some(Box::new(NeonButterfly49d::new(_fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            None
        }
    }

    fn butterfly64(
        _fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<f64> + Send + Sync>> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly64d;
                return Some(Box::new(NeonFcmaButterfly64d::new(_fft_direction)));
            }
            use crate::neon::NeonButterfly64d;
            Some(Box::new(NeonButterfly64d::new(_fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            None
        }
    }

    fn butterfly81(
        _fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<f64> + Send + Sync>> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly81d;
                return Some(Box::new(NeonFcmaButterfly81d::new(_fft_direction)));
            }
            use crate::neon::NeonButterfly81d;
            Some(Box::new(NeonButterfly81d::new(_fft_direction)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxButterfly81d;
            return Some(Box::new(AvxButterfly81d::new(_fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            None
        }
    }

    fn butterfly100(
        _fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<f64> + Send + Sync>> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly100d;
                return Some(Box::new(NeonFcmaButterfly100d::new(_fft_direction)));
            }
            use crate::neon::NeonButterfly100d;
            Some(Box::new(NeonButterfly100d::new(_fft_direction)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxButterfly100d;
            return Some(Box::new(AvxButterfly100d::new(_fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            None
        }
    }

    fn radix3(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        if n == 3 {
            return Self::butterfly3(fft_direction).map(|x| x.into_fft_executor());
        }
        // Use Radix-3 if divisible by 3
        make_default_radix!(
            n,
            fft_direction,
            Radix3,
            AvxFmaRadix3,
            NeonRadix3,
            NeonFcmaRadix3
        )
    }

    fn radix4(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        if n == 4 {
            return Self::butterfly4(fft_direction).map(|x| x.into_fft_executor());
        }
        make_default_radix!(
            n,
            fft_direction,
            Radix4,
            AvxFmaRadix4,
            NeonRadix4,
            NeonFcmaRadix4
        )
    }

    fn radix5(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        if n == 5 {
            return Self::butterfly5(fft_direction).map(|x| x.into_fft_executor());
        }
        make_default_radix!(
            n,
            fft_direction,
            Radix5,
            AvxFmaRadix5,
            NeonRadix5,
            NeonFcmaRadix5
        )
    }

    fn radix6(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_radix!(
            n,
            fft_direction,
            Radix6,
            AvxFmaRadix6,
            NeonRadix6,
            NeonFcmaRadix6
        )
    }

    fn radix7(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_radix!(
            n,
            fft_direction,
            Radix7,
            AvxFmaRadix7,
            NeonRadix7,
            NeonFcmaRadix7
        )
    }

    fn radix10(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_radix!(
            n,
            fft_direction,
            Radix10,
            AvxFmaRadix10d,
            NeonRadix10,
            NeonFcmaRadix10
        )
    }

    fn radix11(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_radix!(
            n,
            fft_direction,
            Radix11,
            AvxFmaRadix11,
            NeonRadix11,
            NeonFcmaRadix11
        )
    }

    fn radix13(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        make_default_radix!(
            n,
            fft_direction,
            Radix13,
            AvxFmaRadix13,
            NeonRadix13,
            NeonFcmaRadix13
        )
    }

    fn dft(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        Dft::new(n, fft_direction).map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
    }

    fn raders(
        convolve_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() && n < (u32::MAX - 100_000u32) as usize {
                use crate::avx::AvxRadersFft;
                unsafe {
                    return AvxRadersFft::new(n, convolve_fft, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if n < (u32::MAX - 100_000u32) as usize {
                use crate::neon::NeonRadersFft;
                return NeonRadersFft::new(n, convolve_fft, fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
            }
        }
        use crate::raders::RadersFft;
        RadersFft::new(n, convolve_fft, fft_direction)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
    }

    fn bluestein(
        convolve_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        BluesteinFft::new(n, convolve_fft, fft_direction)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
    }

    #[allow(unused)]
    fn mixed_radix_butterfly2(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix2;
                    return NeonFcmaMixedRadix2::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix2;
            NeonMixedRadix2::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix2d;
                    return AvxMixedRadix2d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly3(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix3;
                    return NeonFcmaMixedRadix3::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix3;
            NeonMixedRadix3::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix3d;
                    return AvxMixedRadix3d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly4(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix4;
                    return NeonFcmaMixedRadix4::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix4;
            NeonMixedRadix4::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix4d;
                    return AvxMixedRadix4d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly5(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix5;
                    return NeonFcmaMixedRadix5::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix5;
            NeonMixedRadix5::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix5d;
                    return AvxMixedRadix5d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly6(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix6;
                    return NeonFcmaMixedRadix6::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix6;
            NeonMixedRadix6::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix6d;
                    return AvxMixedRadix6d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly7(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix7;
                    return NeonFcmaMixedRadix7::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix7;
            NeonMixedRadix7::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix7d;
                    return AvxMixedRadix7d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly8(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix8;
                    return NeonFcmaMixedRadix8::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix8;
            NeonMixedRadix8::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix8d;
                    return AvxMixedRadix8d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly9(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix9;
                    return NeonFcmaMixedRadix9::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix9;
            NeonMixedRadix9::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix9d;
                    return AvxMixedRadix9d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly10(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix10;
                    return NeonFcmaMixedRadix10::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix10;
            NeonMixedRadix10::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix10d;
                    return AvxMixedRadix10d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly11(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix11;
                    return NeonFcmaMixedRadix11::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix11;
            NeonMixedRadix11::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix11d;
                    return AvxMixedRadix11d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly12(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix12;
                    return NeonFcmaMixedRadix12::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix12;
            NeonMixedRadix12::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix12d;
                    return AvxMixedRadix12d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly13(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix13;
                    return NeonFcmaMixedRadix13::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix13;
            NeonMixedRadix13::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix13d;
                    return AvxMixedRadix13d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly16(
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f64> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix16;
                    return NeonFcmaMixedRadix16::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix16;
            NeonMixedRadix16::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix16d;
                    return AvxMixedRadix16d::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    fn mixed_radix(
        left_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        MixedRadix::new(left_fft, right_fft)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
    }

    fn good_thomas(
        left_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
        right_fft: Box<dyn FftExecutor<f64> + Send + Sync>,
    ) -> Result<Box<dyn FftExecutor<f64> + Send + Sync>, ZaftError> {
        let length = left_fft.length() * right_fft.length();
        if length < (u16::MAX - 100) as usize {
            return GoodThomasSmallFft::new(left_fft, right_fft)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>);
        }
        GoodThomasFft::new(left_fft, right_fft)
            .map(|x| Box::new(x) as Box<dyn FftExecutor<f64> + Send + Sync>)
    }
}
