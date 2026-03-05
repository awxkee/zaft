/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::r2c::c2r_factory::C2RAlgorithmFactory;
use crate::{C2RFftExecutor, FftExecutor, ZaftError};
use std::sync::Arc;

macro_rules! make_mixed_radix {
    ($right_fft: expr, $avx_name: ident, $neon_name: ident, $fcma_name: ident) => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::$fcma_name;
                    return $fcma_name::new($right_fft)
                        .map(|x| Some(Arc::new(x) as Arc<dyn C2RFftExecutor<f32> + Send + Sync>));
                }
            }
            use crate::neon::$neon_name;
            $neon_name::new($right_fft)
                .map(|x| Some(Arc::new(x) as Arc<dyn C2RFftExecutor<f32> + Send + Sync>))
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                use crate::util::has_valid_avx;
                if has_valid_avx() {
                    use crate::avx::$avx_name;
                    return $avx_name::new($right_fft)
                        .map(|x| Some(Arc::new(x) as Arc<dyn C2RFftExecutor<f32> + Send + Sync>));
                }
            }
            Ok(None)
        }
    }};
}

impl C2RAlgorithmFactory<f32> for f32 {
    fn c2r_mixed_radix3(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn C2RFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxC2RMixedRadix3f,
            NeonC2RMixedRadix3f,
            NeonFcmaC2RMixedRadix3f
        )
    }

    fn c2r_mixed_radix5(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn C2RFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxC2RMixedRadix5f,
            NeonC2RMixedRadix5f,
            NeonFcmaC2RMixedRadix5f
        )
    }

    fn c2r_mixed_radix7(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn C2RFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxC2RMixedRadix7f,
            NeonC2RMixedRadix7f,
            NeonFcmaC2RMixedRadix7f
        )
    }

    fn c2r_mixed_radix9(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn C2RFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxC2RMixedRadix9f,
            NeonC2RMixedRadix9f,
            NeonFcmaC2RMixedRadix9f
        )
    }

    fn c2r_mixed_radix11(
        _width_executor: Arc<dyn FftExecutor<f32> + Send + Sync>,
    ) -> Result<Option<Arc<dyn C2RFftExecutor<f32> + Send + Sync>>, ZaftError> {
        make_mixed_radix!(
            _width_executor,
            AvxC2RMixedRadix11f,
            NeonC2RMixedRadix11f,
            NeonFcmaC2RMixedRadix11f
        )
    }
}
