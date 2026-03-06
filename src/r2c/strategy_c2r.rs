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
use crate::r2c::c2r_twiddles::C2RTwiddlesFactory;
use crate::r2c::{C2RFftEvenInterceptor, C2RFftOddInterceptor, OneSizedRealFft};
use crate::{C2RFftExecutor, FftDirection, FftSample, Zaft, ZaftError};
use num_traits::AsPrimitive;
use std::sync::Arc;

pub(crate) fn strategy_c2r<T: FftSample + C2RTwiddlesFactory<T>>(
    len: usize,
) -> Result<Arc<dyn C2RFftExecutor<T> + Send + Sync>, ZaftError>
where
    f64: AsPrimitive<T>,
    usize: AsPrimitive<T>,
{
    if len == 0 {
        return Err(ZaftError::ZeroSizedFft);
    }
    if len == 1 {
        return Ok(Arc::new(OneSizedRealFft {
            phantom_data: Default::default(),
        }));
    }
    if len.is_multiple_of(2) {
        C2RFftEvenInterceptor::install(len, Zaft::strategy(len / 2, FftDirection::Inverse)?)
            .map(|x| Arc::new(x) as Arc<dyn C2RFftExecutor<T> + Send + Sync>)
    } else {
        if Zaft::could_do_split_mixed_radix() {
            if len.is_multiple_of(9) {
                if let Some(mx9) =
                    T::c2r_mixed_radix9(Zaft::strategy(len / 9, FftDirection::Inverse)?)?
                {
                    return Ok(mx9);
                }
            }
            if len.is_multiple_of(5) {
                if let Some(mx5) =
                    T::c2r_mixed_radix5(Zaft::strategy(len / 5, FftDirection::Inverse)?)?
                {
                    return Ok(mx5);
                }
            }
            if len.is_multiple_of(7) {
                if let Some(mx7) =
                    T::c2r_mixed_radix7(Zaft::strategy(len / 7, FftDirection::Inverse)?)?
                {
                    return Ok(mx7);
                }
            }
            if len.is_multiple_of(11) {
                if let Some(mx11) =
                    T::c2r_mixed_radix11(Zaft::strategy(len / 11, FftDirection::Inverse)?)?
                {
                    return Ok(mx11);
                }
            }
            if len.is_multiple_of(3) {
                if let Some(mx3) =
                    T::c2r_mixed_radix3(Zaft::strategy(len / 3, FftDirection::Inverse)?)?
                {
                    return Ok(mx3);
                }
            }
        }

        C2RFftOddInterceptor::install(len, Zaft::strategy(len, FftDirection::Inverse)?)
            .map(|x| Arc::new(x) as Arc<dyn C2RFftExecutor<T> + Send + Sync>)
    }
}
