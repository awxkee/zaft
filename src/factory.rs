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
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
use crate::util::has_valid_avx;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, ZaftError};

macro_rules! make_default_butterfly {
    ($fft_direction: expr, $scalar_name: ident, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::$fcma_name;
                return Ok(Box::new($fcma_name::new($fft_direction)));
            }
            use crate::neon::$neon_name;
            Ok(Box::new($neon_name::new($fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            if has_valid_avx() {
                use crate::avx::$avx_name;
                return Ok(Box::new($avx_name::new($fft_direction)));
            }
            use crate::butterflies::$scalar_name;
            Ok(Box::new($scalar_name::new($fft_direction)))
        }
    }};
}

pub(crate) use make_default_butterfly;

macro_rules! make_default_radix {
    ($n: expr, $fft_direction: expr, $scalar_name: ident, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::$fcma_name;
                return $fcma_name::new($n, $fft_direction)
                    .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::$neon_name;
            $neon_name::new($n, $fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return $avx_name::new($n, $fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
            use crate::$scalar_name;
            $scalar_name::new($n, $fft_direction)
                .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>)
        }
    }};
}

pub(crate) trait AlgorithmFactory<T> {
    fn butterfly1(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly2(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly3(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly4(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly5(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly6(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly7(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly8(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly9(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly10(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly11(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly12(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly13(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly14(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly15(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly16(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly17(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly18(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly19(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly20(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly23(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly27(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly29(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly31(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly32(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly36(
        fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly64(
        fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<T> + Send + Sync>>;
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

    #[allow(unused)]
    fn mixed_radix_butterfly2(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly3(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly4(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly5(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly6(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly7(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly8(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly9(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly10(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly11(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly12(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly13(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly16(
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    fn good_thomas(
        left_fft: Box<dyn FftExecutor<T> + Send + Sync>,
        right_fft: Box<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Box<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
}

impl AlgorithmFactory<f32> for f32 {
    fn butterfly1(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        Ok(Box::new(Butterfly1 {
            phantom_data: Default::default(),
            direction: fft_direction,
        }))
    }

    fn butterfly2(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly10,
            AvxButterfly10f,
            NeonButterfly10,
            NeonFcmaButterfly10
        )
    }

    fn butterfly11(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly15,
            AvxButterfly15f,
            NeonButterfly15,
            NeonFcmaButterfly15
        )
    }

    fn butterfly16(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly18,
            AvxButterfly18f,
            NeonButterfly18f,
            NeonFcmaButterfly18f
        )
    }

    fn butterfly19(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly20,
            AvxButterfly20f,
            NeonButterfly20,
            NeonFcmaButterfly20
        )
    }

    fn butterfly23(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly23,
            AvxButterfly23,
            NeonButterfly23,
            NeonFcmaButterfly23
        )
    }

    fn butterfly27(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly27,
            AvxButterfly27f,
            NeonButterfly27f,
            NeonFcmaButterfly27f
        )
    }

    fn butterfly29(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly31,
            AvxButterfly31,
            NeonButterfly31f,
            NeonFcmaButterfly31f
        )
    }

    fn butterfly32(
        fft_direction: FftDirection,
    ) -> Result<Box<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly32,
            AvxButterfly32f,
            NeonButterfly32f,
            NeonFcmaButterfly32f
        )
    }

    fn butterfly36(
        _fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly36f;
                return Some(Box::new(NeonFcmaButterfly36f::new(_fft_direction)));
            }
            use crate::neon::NeonButterfly36f;
            Some(Box::new(NeonButterfly36f::new(_fft_direction)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxButterfly36f;
            return Some(Box::new(AvxButterfly36f::new(_fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            None
        }
    }

    fn butterfly64(
        _fft_direction: FftDirection,
    ) -> Option<Box<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxButterfly64f;
            return Some(Box::new(AvxButterfly64f::new(_fft_direction)));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaButterfly64f;
                return Some(Box::new(NeonFcmaButterfly64f::new(_fft_direction)));
            }
            use crate::neon::NeonButterfly64f;
            Some(Box::new(NeonButterfly64f::new(_fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            None
        }
    }

    fn radix3(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_radix!(
            n,
            fft_direction,
            Radix10,
            AvxFmaRadix10f,
            NeonRadix10,
            NeonFcmaRadix10
        )
    }

    fn radix11(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Box<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
            if has_valid_avx() && n < (u32::MAX - 100_000u32) as usize {
                use crate::avx::AvxRadersFft;
                unsafe {
                    return AvxRadersFft::new(n, convolve_fft, fft_direction)
                        .map(|x| Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>);
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

    #[allow(unused)]
    fn mixed_radix_butterfly2(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix2f;
                    return NeonFcmaMixedRadix2f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix2f;
            NeonMixedRadix2f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix2f;
                    return AvxMixedRadix2f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly3(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix3f;
                    return NeonFcmaMixedRadix3f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix3f;
            NeonMixedRadix3f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix3f;
                    return AvxMixedRadix3f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly4(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix4f;
                    return NeonFcmaMixedRadix4f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix4f;
            NeonMixedRadix4f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix4f;
                    return AvxMixedRadix4f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly5(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix5f;
                    return NeonFcmaMixedRadix5f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix5f;
            NeonMixedRadix5f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix5f;
                    return AvxMixedRadix5f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly6(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix6f;
                    return NeonFcmaMixedRadix6f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix6f;
            NeonMixedRadix6f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix6f;
                    return AvxMixedRadix6f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly7(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix7f;
                    return NeonFcmaMixedRadix7f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix7f;
            NeonMixedRadix7f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix7f;
                    return AvxMixedRadix7f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly8(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix8f;
                    return NeonFcmaMixedRadix8f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix8f;
            NeonMixedRadix8f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix8f;
                    return AvxMixedRadix8f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly9(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix9f;
                    return NeonFcmaMixedRadix9f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix9f;
            NeonMixedRadix9f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix9f;
                    return AvxMixedRadix9f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly10(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix10f;
                    return NeonFcmaMixedRadix10f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix10f;
            NeonMixedRadix10f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix10f;
                    return AvxMixedRadix10f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly11(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix11f;
                    return NeonFcmaMixedRadix11f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix11f;
            NeonMixedRadix11f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix11f;
                    return AvxMixedRadix11f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly12(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix12f;
                    return NeonFcmaMixedRadix12f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix12f;
            NeonMixedRadix12f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix12f;
                    return AvxMixedRadix12f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly13(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix13f;
                    return NeonFcmaMixedRadix13f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix13f;
            NeonMixedRadix13f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix13f;
                    return AvxMixedRadix13f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }

    #[allow(unused)]
    fn mixed_radix_butterfly16(
        right_fft: Box<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Box<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::NeonFcmaMixedRadix16f;
                    return NeonFcmaMixedRadix16f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::NeonMixedRadix16f;
            NeonMixedRadix16f::new(right_fft)
                .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::AvxMixedRadix16f;
                    return AvxMixedRadix16f::new(right_fft)
                        .map(|x| Some(Box::new(x) as Box<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
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
