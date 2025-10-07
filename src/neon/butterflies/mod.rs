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
mod bf11;
mod bf13;
mod bf15;
mod bf16;
#[cfg(feature = "fcma")]
mod bf16_fcma;
mod bf17;
mod bf19;
mod bf2;
mod bf3;
mod bf4;
mod bf5;
mod bf6;
mod bf7;
mod bf8;
mod bf9;
#[cfg(feature = "fcma")]
mod bf9_fcma;
mod fast_bf5;
mod fast_bf8;
mod shared;

pub(crate) use bf2::NeonButterfly2;
pub(crate) use bf3::NeonButterfly3;
pub(crate) use bf4::NeonButterfly4;
pub(crate) use bf5::NeonButterfly5;
pub(crate) use bf6::NeonButterfly6;
pub(crate) use bf7::NeonButterfly7;
pub(crate) use bf8::NeonButterfly8;
pub(crate) use bf9::NeonButterfly9;
#[cfg(feature = "fcma")]
pub(crate) use bf9_fcma::NeonFcmaButterfly9;
pub(crate) use bf10::NeonButterfly10;
pub(crate) use bf11::NeonButterfly11;
pub(crate) use bf13::NeonButterfly13;
pub(crate) use bf15::NeonButterfly15;
pub(crate) use bf16::NeonButterfly16;
#[cfg(feature = "fcma")]
pub(crate) use bf16_fcma::NeonFcmaButterfly16;
pub(crate) use bf17::NeonButterfly17;
pub(crate) use bf19::NeonButterfly19;
pub(crate) use shared::NeonButterfly;
