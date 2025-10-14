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
mod butterflies;
mod c2r;
#[cfg(feature = "fcma")]
mod c2r_fcma;
mod f32x2_2x2;
mod f32x2_4x4;
mod r2c;
mod raders;
mod radix10;
#[cfg(feature = "fcma")]
mod radix10_fcma;
mod radix11;
#[cfg(feature = "fcma")]
mod radix11_fcma;
mod radix13;
#[cfg(feature = "fcma")]
mod radix13_fcma;
mod radix3;
#[cfg(feature = "fcma")]
mod radix3_fcma;
mod radix4;
#[cfg(feature = "fcma")]
mod radix4_fcma;
mod radix5;
#[cfg(feature = "fcma")]
mod radix5_fcma;
mod radix6;
#[cfg(feature = "fcma")]
mod radix6_fcma;
mod radix7;
#[cfg(feature = "fcma")]
mod radix7_fcma;
mod spectrum_arithmetic;
#[cfg(feature = "fcma")]
mod spectrum_arithmetic_fcma;
mod util;

pub(crate) use butterflies::{
    NeonButterfly2, NeonButterfly3, NeonButterfly4, NeonButterfly5, NeonButterfly6, NeonButterfly7,
    NeonButterfly8, NeonButterfly9, NeonButterfly10, NeonButterfly11, NeonButterfly12,
    NeonButterfly13, NeonButterfly14, NeonButterfly15, NeonButterfly16, NeonButterfly17,
    NeonButterfly19, NeonButterfly23, NeonButterfly29,
};
#[cfg(feature = "fcma")]
pub(crate) use butterflies::{
    NeonFcmaButterfly4, NeonFcmaButterfly5, NeonFcmaButterfly7, NeonFcmaButterfly8,
    NeonFcmaButterfly9, NeonFcmaButterfly10, NeonFcmaButterfly11, NeonFcmaButterfly12,
    NeonFcmaButterfly13, NeonFcmaButterfly14, NeonFcmaButterfly15, NeonFcmaButterfly16,
    NeonFcmaButterfly17, NeonFcmaButterfly19, NeonFcmaButterfly29,
};
pub(crate) use c2r::C2RNeonTwiddles;
#[cfg(feature = "fcma")]
pub(crate) use c2r_fcma::C2RNeonFcmaTwiddles;
pub(crate) use f32x2_2x2::neon_transpose_f32x2_2x2;
pub(crate) use f32x2_4x4::neon_transpose_f32x2_4x4;
pub(crate) use r2c::R2CNeonTwiddles;
pub(crate) use raders::NeonRadersFft;
pub(crate) use radix3::NeonRadix3;
#[cfg(feature = "fcma")]
pub(crate) use radix3_fcma::NeonFcmaRadix3;
pub(crate) use radix4::NeonRadix4;
#[cfg(feature = "fcma")]
pub(crate) use radix4_fcma::NeonFcmaRadix4;
pub(crate) use radix5::NeonRadix5;
#[cfg(feature = "fcma")]
pub(crate) use radix5_fcma::NeonFcmaRadix5;
pub(crate) use radix6::NeonRadix6;
#[cfg(feature = "fcma")]
pub(crate) use radix6_fcma::NeonFcmaRadix6;
pub(crate) use radix7::NeonRadix7;
#[cfg(feature = "fcma")]
pub(crate) use radix7_fcma::NeonFcmaRadix7;
pub(crate) use radix10::NeonRadix10;
#[cfg(feature = "fcma")]
pub(crate) use radix10_fcma::NeonFcmaRadix10;
pub(crate) use radix11::NeonRadix11;
#[cfg(feature = "fcma")]
pub(crate) use radix11_fcma::NeonFcmaRadix11;
pub(crate) use radix13::NeonRadix13;
#[cfg(feature = "fcma")]
pub(crate) use radix13_fcma::NeonFcmaRadix13;
pub(crate) use spectrum_arithmetic::NeonSpectrumArithmetic;
#[cfg(feature = "fcma")]
pub(crate) use spectrum_arithmetic_fcma::NeonFcmaSpectrumArithmetic;
