/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
mod bf10;
#[cfg(feature = "fcma")]
mod bf10_fcma;
mod bf11;
#[cfg(feature = "fcma")]
mod bf11_fcma;
mod bf12;
#[cfg(feature = "fcma")]
mod bf12_fcma;
mod bf13;
#[cfg(feature = "fcma")]
mod bf13_fcma;
mod bf14;
#[cfg(feature = "fcma")]
mod bf14_fcma;
mod bf15;
#[cfg(feature = "fcma")]
mod bf15_fcma;
mod bf16;
#[cfg(feature = "fcma")]
mod bf16_fcma;
mod bf17;
#[cfg(feature = "fcma")]
mod bf17_fcma;
mod bf19;
#[cfg(feature = "fcma")]
mod bf19_fcma;
mod bf2;
mod bf23;
#[cfg(feature = "fcma")]
mod bf23_fcma;
mod bf29;
#[cfg(feature = "fcma")]
mod bf29_fcma;
mod bf3;
mod bf4;
#[cfg(feature = "fcma")]
mod bf4_fcma;
mod bf5;
#[cfg(feature = "fcma")]
mod bf5_fcma;
mod bf6;
mod bf7;
#[cfg(feature = "fcma")]
mod bf7_fcma;
mod bf8;
#[cfg(feature = "fcma")]
mod bf8_fcma;
mod bf9;
#[cfg(feature = "fcma")]
mod bf9_fcma;
mod fast_bf5;
mod fast_bf7;
mod fast_bf8;
mod shared;

pub(crate) use bf2::NeonButterfly2;
pub(crate) use bf3::NeonButterfly3;
pub(crate) use bf4::NeonButterfly4;
#[cfg(feature = "fcma")]
pub(crate) use bf4_fcma::NeonFcmaButterfly4;
pub(crate) use bf5::NeonButterfly5;
#[cfg(feature = "fcma")]
pub(crate) use bf5_fcma::NeonFcmaButterfly5;
pub(crate) use bf6::NeonButterfly6;
pub(crate) use bf7::NeonButterfly7;
#[cfg(feature = "fcma")]
pub(crate) use bf7_fcma::NeonFcmaButterfly7;
pub(crate) use bf8::NeonButterfly8;
#[cfg(feature = "fcma")]
pub(crate) use bf8_fcma::NeonFcmaButterfly8;
pub(crate) use bf9::NeonButterfly9;
#[cfg(feature = "fcma")]
pub(crate) use bf9_fcma::NeonFcmaButterfly9;
pub(crate) use bf10::NeonButterfly10;
#[cfg(feature = "fcma")]
pub(crate) use bf10_fcma::NeonFcmaButterfly10;
pub(crate) use bf11::NeonButterfly11;
#[cfg(feature = "fcma")]
pub(crate) use bf11_fcma::NeonFcmaButterfly11;
pub(crate) use bf12::NeonButterfly12;
#[cfg(feature = "fcma")]
pub(crate) use bf12_fcma::NeonFcmaButterfly12;
pub(crate) use bf13::NeonButterfly13;
#[cfg(feature = "fcma")]
pub(crate) use bf13_fcma::NeonFcmaButterfly13;
pub(crate) use bf14::NeonButterfly14;
#[cfg(feature = "fcma")]
pub(crate) use bf14_fcma::NeonFcmaButterfly14;
pub(crate) use bf15::NeonButterfly15;
#[cfg(feature = "fcma")]
pub(crate) use bf15_fcma::NeonFcmaButterfly15;
pub(crate) use bf16::NeonButterfly16;
#[cfg(feature = "fcma")]
pub(crate) use bf16_fcma::NeonFcmaButterfly16;
pub(crate) use bf17::NeonButterfly17;
#[cfg(feature = "fcma")]
pub(crate) use bf17_fcma::NeonFcmaButterfly17;
pub(crate) use bf19::NeonButterfly19;
#[cfg(feature = "fcma")]
pub(crate) use bf19_fcma::NeonFcmaButterfly19;
pub(crate) use bf23::NeonButterfly23;
#[cfg(feature = "fcma")]
pub(crate) use bf23_fcma::NeonFcmaButterfly23;
pub(crate) use bf29::NeonButterfly29;
#[cfg(feature = "fcma")]
pub(crate) use bf29_fcma::NeonFcmaButterfly29;
pub(crate) use fast_bf5::NeonFastButterfly5;
pub(crate) use shared::NeonButterfly;
