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
mod f32x2_2x2;
mod f32x2_4x4;
mod f32x2_8x4;
mod f64x2_2x2;
mod f64x2_4x4;
mod mixed;
mod r2c;
mod raders;
mod radix10;
mod radix11;
mod radix13;
mod radix3;
mod radix4;
mod radix5;
mod radix6;
mod radix7;
mod rotate;
mod spectrum_arithmetic;
mod util;

pub(crate) use butterflies::{
    AvxButterfly2, AvxButterfly3, AvxButterfly4, AvxButterfly5, AvxButterfly6, AvxButterfly7,
    AvxButterfly8, AvxButterfly9, AvxButterfly10d, AvxButterfly10f, AvxButterfly11, AvxButterfly12,
    AvxButterfly13, AvxButterfly14, AvxButterfly15d, AvxButterfly15f, AvxButterfly16,
    AvxButterfly17, AvxButterfly19, AvxButterfly27d, AvxButterfly27f,
};
pub(crate) use f32x2_2x2::avx_transpose_f32x2_2x2;
pub(crate) use f32x2_4x4::avx2_transpose_f32x2_4x4;
pub(crate) use f32x2_8x4::avx2_transpose_f32x2_8x4;
pub(crate) use f64x2_2x2::avx_transpose_f64x2_2x2;
pub(crate) use f64x2_4x4::avx_transpose_f64x2_4x4;
pub(crate) use mixed::{
    AvxMixedRadix2d, AvxMixedRadix3d, AvxMixedRadix4d, AvxMixedRadix5d, AvxMixedRadix6d,
    AvxMixedRadix7d, AvxMixedRadix8d, AvxMixedRadix9d, AvxMixedRadix10d, AvxMixedRadix11d,
    AvxMixedRadix12d, AvxMixedRadix13d, AvxMixedRadix16d,
};
pub(crate) use mixed::{
    AvxMixedRadix2f, AvxMixedRadix3f, AvxMixedRadix4f, AvxMixedRadix5f, AvxMixedRadix6f,
    AvxMixedRadix7f, AvxMixedRadix8f, AvxMixedRadix9f, AvxMixedRadix10f, AvxMixedRadix11f,
    AvxMixedRadix12f, AvxMixedRadix13f, AvxMixedRadix16f,
};
pub(crate) use r2c::R2CAvxTwiddles;
pub(crate) use raders::AvxRadersFft;
pub(crate) use radix3::AvxFmaRadix3;
pub(crate) use radix4::AvxFmaRadix4;
pub(crate) use radix5::AvxFmaRadix5;
pub(crate) use radix6::AvxFmaRadix6;
pub(crate) use radix7::AvxFmaRadix7;
pub(crate) use radix10::{AvxFmaRadix10d, AvxFmaRadix10f};
pub(crate) use radix11::AvxFmaRadix11;
pub(crate) use radix13::AvxFmaRadix13;
pub(crate) use spectrum_arithmetic::AvxSpectrumArithmetic;
