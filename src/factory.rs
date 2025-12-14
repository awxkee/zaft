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
use std::sync::{Arc, OnceLock};

macro_rules! make_default_butterfly {
    ($fft_direction: expr, $scalar_name: ident, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        static Q: OnceLock<Arc<dyn FftExecutor<f32> + Send + Sync>> = OnceLock::new();
        static B: OnceLock<Arc<dyn FftExecutor<f32> + Send + Sync>> = OnceLock::new();
        let selector = match $fft_direction {
            FftDirection::Forward => &Q,
            FftDirection::Inverse => &B,
        };
        Ok(selector
            .get_or_init(|| {
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                {
                    #[cfg(feature = "fcma")]
                    if std::arch::is_aarch64_feature_detected!("fcma") {
                        use crate::neon::$fcma_name;
                        return Arc::new($fcma_name::new($fft_direction));
                    }
                    use crate::neon::$neon_name;
                    Arc::new($neon_name::new($fft_direction))
                }
                #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
                {
                    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                    if has_valid_avx() {
                        use crate::avx::$avx_name;
                        return Arc::new($avx_name::new($fft_direction));
                    }
                    use crate::butterflies::$scalar_name;
                    Arc::new($scalar_name::new($fft_direction))
                }
            })
            .clone())
    }};
}

macro_rules! make_composite_butterfly {
    ($fft_direction: expr, $scalar_name: ident, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        static Q: OnceLock<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> = OnceLock::new();
        static B: OnceLock<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> = OnceLock::new();
        let selector = match $fft_direction {
            FftDirection::Forward => &Q,
            FftDirection::Inverse => &B,
        };
        Ok(selector
            .get_or_init(|| {
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                {
                    #[cfg(feature = "fcma")]
                    if std::arch::is_aarch64_feature_detected!("fcma") {
                        use crate::neon::$fcma_name;
                        return Arc::new($fcma_name::new($fft_direction));
                    }
                    use crate::neon::$neon_name;
                    Arc::new($neon_name::new($fft_direction))
                }
                #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
                {
                    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                    if has_valid_avx() {
                        use crate::avx::$avx_name;
                        return Arc::new($avx_name::new($fft_direction));
                    }
                    use crate::butterflies::$scalar_name;
                    Arc::new($scalar_name::new($fft_direction))
                }
            })
            .clone())
    }};
}

macro_rules! make_optional_butterfly {
    ($ftype: ident, $fft_direction: expr, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        static Q: OnceLock<Option<Arc<dyn $ftype<f32> + Send + Sync>>> = OnceLock::new();
        static B: OnceLock<Option<Arc<dyn $ftype<f32> + Send + Sync>>> = OnceLock::new();
        let selector = match $fft_direction {
            FftDirection::Forward => &Q,
            FftDirection::Inverse => &B,
        };
        selector
            .get_or_init(|| {
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return Some(Arc::new($avx_name::new($fft_direction)));
                }
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                {
                    #[cfg(feature = "fcma")]
                    if std::arch::is_aarch64_feature_detected!("fcma") {
                        use crate::neon::$fcma_name;
                        return Some(Arc::new($fcma_name::new($fft_direction)));
                    }
                    use crate::neon::$neon_name;
                    Some(Arc::new($neon_name::new($fft_direction)))
                }
                #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
                {
                    None
                }
            })
            .clone()
    }};
}

macro_rules! make_default_radix {
    ($n: expr, $fft_direction: expr, $scalar_name: ident, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::$fcma_name;
                return $fcma_name::new($n, $fft_direction)
                    .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>);
            }
            use crate::neon::$neon_name;
            $neon_name::new($n, $fft_direction)
                .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return $avx_name::new($n, $fft_direction)
                        .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
            use crate::$scalar_name;
            $scalar_name::new($n, $fft_direction)
                .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>)
        }
    }};
}

macro_rules! make_mixed_radix {
    ($right_fft: expr, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::$fcma_name;
                    return $fcma_name::new($right_fft)
                        .map(|x| Some(Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::$neon_name;
            $neon_name::new($right_fft)
                .map(|x| Some(Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return $avx_name::new($right_fft)
                        .map(|x| Some(Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }};
}

pub(crate) trait AlgorithmFactory<T> {
    fn butterfly1(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly2(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly3(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly4(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly5(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly6(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly7(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly8(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly9(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly10(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly11(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly12(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly13(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly14(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly15(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly16(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly17(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly18(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly19(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly20(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly23(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    // used only on NEON, or scalar
    fn butterfly21(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly24(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly25(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly27(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly29(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly30(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly31(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly32(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<T> + Send + Sync>, ZaftError>;
    fn butterfly35(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly36(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly40(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly42(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly48(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly49(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly54(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly63(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly64(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly66(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly70(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly72(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly81(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly96(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly100(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly121(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly125(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly128(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly144(fft_direction: FftDirection) -> Option<Arc<dyn FftExecutor<T> + Send + Sync>>;
    fn butterfly169(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly243(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly256(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn butterfly512(
        fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<T> + Send + Sync>>;
    fn radix3(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix4(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix5(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix6(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix7(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix10(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix11(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn radix13(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
    fn dft(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    fn raders(
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    fn bluestein(
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    fn mixed_radix(
        left_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly2(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly3(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly4(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly5(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly6(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly7(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly8(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly9(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly10(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly11(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly12(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly13(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly14(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;
    #[allow(unused)]
    fn mixed_radix_butterfly15(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    #[allow(unused)]
    fn mixed_radix_butterfly16(
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<T> + Send + Sync>>, ZaftError>;

    fn good_thomas(
        left_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        right_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<Arc<dyn FftExecutor<T> + Send + Sync>, ZaftError>;
}

impl AlgorithmFactory<f32> for f32 {
    fn butterfly1(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        Ok(Arc::new(Butterfly1 {
            phantom_data: Default::default(),
            direction: fft_direction,
        }))
    }

    fn butterfly2(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly2;
            Ok(Arc::new(NeonButterfly2::new(fft_direction)))
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx::AvxButterfly2;
            return Ok(Arc::new(AvxButterfly2::new(fft_direction)));
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly2;
            Ok(Arc::new(Butterfly2::new(fft_direction)))
        }
    }

    fn butterfly3(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if has_valid_avx() {
            use crate::avx::AvxButterfly3;
            return Ok(Arc::new(AvxButterfly3::new(fft_direction)));
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonButterfly3;
            Ok(Arc::new(NeonButterfly3::new(fft_direction)))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            use crate::butterflies::Butterfly3;
            Ok(Arc::new(Butterfly3::new(fft_direction)))
        }
    }

    fn butterfly4(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly4,
            AvxButterfly4,
            NeonButterfly4,
            NeonFcmaButterfly4
        )
    }

    fn butterfly5(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly5,
            AvxButterfly5,
            NeonButterfly5,
            NeonFcmaButterfly5
        )
    }

    fn butterfly6(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly6,
            AvxButterfly6,
            NeonButterfly6,
            NeonFcmaButterfly6
        )
    }

    fn butterfly7(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly7,
            AvxButterfly7,
            NeonButterfly7,
            NeonFcmaButterfly7
        )
    }

    fn butterfly8(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly8,
            AvxButterfly8,
            NeonButterfly8,
            NeonFcmaButterfly8
        )
    }

    fn butterfly9(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly9,
            AvxButterfly9f,
            NeonButterfly9f,
            NeonFcmaButterfly9f
        )
    }

    fn butterfly10(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly10,
            AvxButterfly10f,
            NeonButterfly10f,
            NeonFcmaButterfly10f
        )
    }

    fn butterfly11(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly11,
            AvxButterfly11,
            NeonButterfly11f,
            NeonFcmaButterfly11f
        )
    }

    fn butterfly12(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly12,
            AvxButterfly12f,
            NeonButterfly12f,
            NeonFcmaButterfly12f
        )
    }

    fn butterfly13(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly13,
            AvxButterfly13,
            NeonButterfly13f,
            NeonFcmaButterfly13f
        )
    }

    fn butterfly14(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly14,
            AvxButterfly14f,
            NeonButterfly14f,
            NeonFcmaButterfly14f
        )
    }

    fn butterfly15(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly15,
            AvxButterfly15f,
            NeonButterfly15f,
            NeonFcmaButterfly15f
        )
    }

    fn butterfly16(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly16,
            AvxButterfly16f,
            NeonButterfly16f,
            NeonFcmaButterfly16f
        )
    }

    fn butterfly17(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly20,
            AvxButterfly20f,
            NeonButterfly20f,
            NeonFcmaButterfly20f
        )
    }

    fn butterfly21(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly21f,
            NeonButterfly21f,
            NeonFcmaButterfly21f
        )
    }

    fn butterfly23(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly23,
            AvxButterfly23,
            NeonButterfly23,
            NeonFcmaButterfly23
        )
    }

    fn butterfly24(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly24f,
            NeonButterfly24f,
            NeonFcmaButterfly24f
        )
    }

    fn butterfly25(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly25,
            AvxButterfly25f,
            NeonButterfly25f,
            NeonFcmaButterfly25f
        )
    }

    fn butterfly27(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly27,
            AvxButterfly27f,
            NeonButterfly27f,
            NeonFcmaButterfly27f
        )
    }

    fn butterfly29(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        make_default_butterfly!(
            fft_direction,
            Butterfly29,
            AvxButterfly29,
            NeonButterfly29f,
            NeonFcmaButterfly29f
        )
    }

    fn butterfly30(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly30f,
            NeonButterfly30f,
            NeonFcmaButterfly30f
        )
    }

    fn butterfly31(
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>, ZaftError> {
        make_composite_butterfly!(
            fft_direction,
            Butterfly32,
            AvxButterfly32f,
            NeonButterfly32f,
            NeonFcmaButterfly32f
        )
    }

    fn butterfly35(_direction: FftDirection) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _direction,
            AvxButterfly35f,
            NeonButterfly35f,
            NeonFcmaButterfly35f
        )
    }

    fn butterfly36(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly36f,
            NeonButterfly36f,
            NeonFcmaButterfly36f
        )
    }

    fn butterfly40(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly40f,
            NeonButterfly40f,
            NeonFcmaButterfly40f
        )
    }

    fn butterfly42(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly42f,
            NeonButterfly42f,
            NeonFcmaButterfly42f
        )
    }

    fn butterfly48(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly48f,
            NeonButterfly48f,
            NeonFcmaButterfly48f
        )
    }

    fn butterfly49(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly49f,
            NeonButterfly49f,
            NeonFcmaButterfly49f
        )
    }

    fn butterfly54(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly54f,
            NeonButterfly54f,
            NeonFcmaButterfly54f
        )
    }

    fn butterfly63(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly63f,
            NeonButterfly63f,
            NeonFcmaButterfly63f
        )
    }

    fn butterfly64(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly64f,
            NeonButterfly64f,
            NeonFcmaButterfly64f
        )
    }

    fn butterfly66(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly66f,
            NeonButterfly66f,
            NeonFcmaButterfly66f
        )
    }

    fn butterfly70(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly70f,
            NeonButterfly70f,
            NeonFcmaButterfly70f
        )
    }

    fn butterfly72(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly72f,
            NeonButterfly72f,
            NeonFcmaButterfly72f
        )
    }

    fn butterfly81(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly81f,
            NeonButterfly81f,
            NeonFcmaButterfly81f
        )
    }

    fn butterfly96(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly96f,
            NeonButterfly96f,
            NeonFcmaButterfly96f
        )
    }

    fn butterfly100(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly100f,
            NeonButterfly100f,
            NeonFcmaButterfly100f
        )
    }

    fn butterfly121(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly121f,
            NeonButterfly121f,
            NeonFcmaButterfly121f
        )
    }

    fn butterfly125(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly125f,
            NeonButterfly125f,
            NeonFcmaButterfly125f
        )
    }

    fn butterfly128(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly128f,
            NeonButterfly128f,
            NeonFcmaButterfly128f
        )
    }

    fn butterfly144(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn FftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            FftExecutor,
            _fft_direction,
            AvxButterfly144f,
            NeonButterfly144f,
            NeonFcmaButterfly144f
        )
    }

    fn butterfly169(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly169f,
            NeonButterfly169f,
            NeonFcmaButterfly169f
        )
    }

    fn butterfly243(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly243f,
            NeonButterfly243f,
            NeonFcmaButterfly243f
        )
    }

    fn butterfly256(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly256f,
            NeonButterfly256f,
            NeonFcmaButterfly256f
        )
    }

    fn butterfly512(
        _fft_direction: FftDirection,
    ) -> Option<Arc<dyn CompositeFftExecutor<f32> + Send + Sync>> {
        make_optional_butterfly!(
            CompositeFftExecutor,
            _fft_direction,
            AvxButterfly512f,
            NeonButterfly512f,
            NeonFcmaButterfly512f
        )
    }

    fn radix3(
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
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
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        Dft::new(n, fft_direction).map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>)
    }

    fn raders(
        convolve_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if has_valid_avx() && n < (u32::MAX - 100_000u32) as usize {
                use crate::avx::AvxRadersFft;
                unsafe {
                    return AvxRadersFft::new(n, convolve_fft, fft_direction)
                        .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>);
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if n < (u32::MAX - 100_000u32) as usize {
                use crate::neon::NeonRadersFft;
                return NeonRadersFft::new(n, convolve_fft, fft_direction)
                    .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>);
            }
        }
        use crate::raders::RadersFft;
        RadersFft::new(n, convolve_fft, fft_direction)
            .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>)
    }

    fn bluestein(
        convolve_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
        n: usize,
        fft_direction: FftDirection,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        BluesteinFft::new(n, convolve_fft, fft_direction)
            .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>)
    }

    fn mixed_radix(
        left_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        MixedRadix::new(left_fft, right_fft)
            .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>)
    }

    #[allow(unused)]
    fn mixed_radix_butterfly2(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix2f,
            NeonMixedRadix2f,
            NeonFcmaMixedRadix2f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly3(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix3f,
            NeonMixedRadix3f,
            NeonFcmaMixedRadix3f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly4(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix4f,
            NeonMixedRadix4f,
            NeonFcmaMixedRadix4f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly5(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix5f,
            NeonMixedRadix5f,
            NeonFcmaMixedRadix5f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly6(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix6f,
            NeonMixedRadix6f,
            NeonFcmaMixedRadix6f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly7(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix7f,
            NeonMixedRadix7f,
            NeonFcmaMixedRadix7f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly8(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix8f,
            NeonMixedRadix8f,
            NeonFcmaMixedRadix8f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly9(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix9f,
            NeonMixedRadix9f,
            NeonFcmaMixedRadix9f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly10(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix10f,
            NeonMixedRadix10f,
            NeonFcmaMixedRadix10f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly11(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix11f,
            NeonMixedRadix11f,
            NeonFcmaMixedRadix11f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly12(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix12f,
            NeonMixedRadix12f,
            NeonFcmaMixedRadix12f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly13(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix13f,
            NeonMixedRadix13f,
            NeonFcmaMixedRadix13f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly14(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix14f,
            NeonMixedRadix14f,
            NeonFcmaMixedRadix14f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly15(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix15f,
            NeonMixedRadix15f,
            NeonFcmaMixedRadix15f
        )
    }

    #[allow(unused)]
    fn mixed_radix_butterfly16(
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn FftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            right_fft,
            AvxMixedRadix16f,
            NeonMixedRadix16f,
            NeonFcmaMixedRadix16f
        )
    }

    fn good_thomas(
        left_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
        right_fft: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Arc<dyn FftExecutor<f32> + Send + Sync>, ZaftError> {
        let length = left_fft.length() * right_fft.length();
        if length < (u16::MAX - 100) as usize {
            return GoodThomasSmallFft::new(left_fft, right_fft)
                .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>);
        }
        GoodThomasFft::new(left_fft, right_fft)
            .map(|x| Arc::new(x) as Arc<dyn FftExecutor<f32> + Send + Sync>)
    }
}
