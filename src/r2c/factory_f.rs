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
use crate::r2c::rfft_bluestein::BluesteinRfft;
use crate::r2c::rfft_raders::RadersRfft;
use crate::r2c::{OneSizedRealFft, R2cAlgorithmFactory};
use crate::{FftDirection, FftExecutor, R2CFftExecutor, Zaft, ZaftError};
use std::sync::Arc;

macro_rules! make_default_butterfly {
    ($scalar_name: ident) => {{
        use crate::FftDirection;
        use std::sync::OnceLock;
        static Q: OnceLock<Arc<dyn R2CFftExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            use crate::butterflies::$scalar_name;
            Arc::new($scalar_name::new(FftDirection::Forward))
        })
        .clone()
    }};
}

macro_rules! make_vec_default_butterfly {
    ($scalar_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        use crate::FftDirection;
        use std::sync::OnceLock;
        static Q: OnceLock<Arc<dyn R2CFftExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                #[cfg(feature = "fcma")]
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::$fcma_name;
                    return Arc::new($fcma_name::new(FftDirection::Forward));
                }
                use crate::neon::$neon_name;
                Arc::new($neon_name::new(FftDirection::Forward))
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::$scalar_name;
                Arc::new($scalar_name::new(FftDirection::Forward))
            }
        })
        .clone()
    }};
}

macro_rules! make_vec_default_butterfly2 {
    ($scalar_name: ident, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        use crate::FftDirection;
        use std::sync::OnceLock;
        static Q: OnceLock<Arc<dyn R2CFftExecutor<f32> + Send + Sync>> = OnceLock::new();
        Q.get_or_init(|| {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                use crate::util::has_valid_avx;
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return Arc::new($avx_name::new(FftDirection::Forward));
                }
            }
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                #[cfg(feature = "fcma")]
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::$fcma_name;
                    return Arc::new($fcma_name::new(FftDirection::Forward));
                }
                use crate::neon::$neon_name;
                Arc::new($neon_name::new(FftDirection::Forward))
            }
            #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
            {
                use crate::butterflies::$scalar_name;
                Arc::new($scalar_name::new(FftDirection::Forward))
            }
        })
        .clone()
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
                        .map(|x| Some(Arc::new(x) as Arc<dyn R2CFftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::$neon_name;
            $neon_name::new($right_fft)
                .map(|x| Some(Arc::new(x) as Arc<dyn R2CFftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                use crate::util::has_valid_avx;
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return $avx_name::new($right_fft)
                        .map(|x| Some(Arc::new(x) as Arc<dyn R2CFftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }};
}

impl R2cAlgorithmFactory<f32> for f32 {
    fn r2c_butterfly1() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        Arc::new(OneSizedRealFft {
            phantom_data: Default::default(),
        })
    }

    fn r2c_butterfly2() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly2)
    }

    fn r2c_butterfly3() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly3)
    }

    fn r2c_butterfly4() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly4)
    }

    fn r2c_butterfly5() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly5)
    }

    fn r2c_butterfly6() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_vec_default_butterfly!(Butterfly6, NeonButterfly6f, NeonFcmaButterfly6f)
    }

    fn r2c_butterfly7() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly7)
    }

    fn r2c_butterfly8() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_vec_default_butterfly2!(
            Butterfly8,
            AvxButterfly8f,
            NeonButterfly8f,
            NeonFcmaButterfly8f
        )
    }

    fn r2c_butterfly9() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_vec_default_butterfly!(Butterfly9, NeonButterfly9f, NeonFcmaButterfly9f)
    }

    fn r2c_butterfly11() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly11)
    }

    fn r2c_butterfly12() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_vec_default_butterfly!(Butterfly12, NeonButterfly12f, NeonFcmaButterfly12f)
    }

    fn r2c_butterfly13() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly13)
    }

    fn r2c_butterfly14() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly14)
    }

    fn r2c_butterfly15() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_vec_default_butterfly!(Butterfly15, NeonButterfly15f, NeonFcmaButterfly15f)
    }

    fn r2c_butterfly16() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_vec_default_butterfly2!(
            Butterfly16,
            AvxButterfly16f,
            NeonButterfly16f,
            NeonFcmaButterfly16f
        )
    }

    fn r2c_butterfly17() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly17)
    }

    fn r2c_butterfly19() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly19)
    }

    fn r2c_butterfly23() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly23)
    }

    fn r2c_butterfly29() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(RfftButterfly29)
    }

    fn r2c_butterfly31() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_default_butterfly!(Butterfly31)
    }

    fn r2c_butterfly32() -> Arc<dyn R2CFftExecutor<f32> + Send + Sync> {
        make_vec_default_butterfly2!(
            Butterfly32,
            AvxButterfly32f,
            NeonButterfly32f,
            NeonFcmaButterfly32f
        )
    }

    fn r2c_raders(n: usize) -> Result<Arc<dyn R2CFftExecutor<f32> + Send + Sync>, ZaftError> {
        let convolve_fft = Zaft::strategy(n - 1, FftDirection::Forward)?;
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            use crate::util::has_valid_avx;
            if has_valid_avx() && n < (u32::MAX - 100_000u32) as usize {
                use crate::avx::AvxRadersFft;
                unsafe {
                    return Ok(Arc::new(AvxRadersFft::new(
                        n,
                        convolve_fft,
                        FftDirection::Forward,
                    )?));
                }
            }
        }
        Ok(Arc::new(RadersRfft::new(
            n,
            convolve_fft,
            FftDirection::Forward,
        )?))
    }

    fn r2c_bluestein(n: usize) -> Result<Arc<dyn R2CFftExecutor<f32> + Send + Sync>, ZaftError> {
        let min_inner_len = 2 * n - 1;
        let inner_len_pow2 = min_inner_len.checked_next_power_of_two().unwrap();
        let inner_len_factor3 = inner_len_pow2 / 4 * 3;

        let inner_len = if inner_len_factor3 >= min_inner_len {
            inner_len_factor3
        } else {
            inner_len_pow2
        };
        let convolve_fft = Zaft::strategy(inner_len, FftDirection::Forward)?;
        Ok(Arc::new(BluesteinRfft::new(
            n,
            convolve_fft,
            FftDirection::Forward,
        )?))
    }

    fn r2c_mixed_radix3(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn R2CFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxR2CMixedRadix3f,
            NeonR2CMixedRadix3f,
            NeonFcmaR2CMixedRadix3f
        )
    }

    fn r2c_mixed_radix5(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn R2CFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxR2CMixedRadix5f,
            NeonR2CMixedRadix5f,
            NeonFcmaR2CMixedRadix5f
        )
    }

    fn r2c_mixed_radix7(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn R2CFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxR2CMixedRadix7f,
            NeonR2CMixedRadix7f,
            NeonFcmaR2CMixedRadix7f
        )
    }

    fn r2c_mixed_radix9(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn R2CFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxR2CMixedRadix9f,
            NeonR2CMixedRadix9f,
            NeonFcmaR2CMixedRadix9f
        )
    }

    fn r2c_mixed_radix11(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn R2CFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxR2CMixedRadix11f,
            NeonR2CMixedRadix11f,
            NeonFcmaR2CMixedRadix11f
        )
    }

    fn r2c_mixed_radix13(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn R2CFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxR2CMixedRadix13f,
            NeonR2CMixedRadix13f,
            NeonFcmaR2CMixedRadix13f
        )
    }
}
