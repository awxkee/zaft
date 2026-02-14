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
mod bf12;
mod bf13;
mod bf16;
mod bf18;
mod bf2;
mod bf3;
mod bf4;
mod bf5;
mod bf6;
mod bf7;
mod bf8;
mod bf9;
mod mixed_radix;
mod neon_store;

pub(crate) use bf2::{ColumnButterfly2d, ColumnButterfly2f};
pub(crate) use bf3::{ColumnButterfly3d, ColumnButterfly3f};
#[cfg(feature = "fcma")]
pub(crate) use bf3::{ColumnFcmaButterfly3d, ColumnFcmaButterfly3f};
pub(crate) use bf4::{ColumnButterfly4d, ColumnButterfly4f};
#[cfg(feature = "fcma")]
pub(crate) use bf4::{ColumnFcmaButterfly4d, ColumnFcmaButterfly4f};
pub(crate) use bf5::{ColumnButterfly5d, ColumnButterfly5f};
#[cfg(feature = "fcma")]
pub(crate) use bf5::{ColumnFcmaButterfly5d, ColumnFcmaButterfly5f};
pub(crate) use bf6::{ColumnButterfly6d, ColumnButterfly6f};
#[cfg(feature = "fcma")]
pub(crate) use bf6::{ColumnFcmaButterfly6d, ColumnFcmaButterfly6f};
pub(crate) use bf7::{ColumnButterfly7d, ColumnButterfly7f};
#[cfg(feature = "fcma")]
pub(crate) use bf7::{ColumnFcmaButterfly7d, ColumnFcmaButterfly7f};
pub(crate) use bf8::{ColumnButterfly8d, ColumnButterfly8f};
#[cfg(feature = "fcma")]
pub(crate) use bf8::{
    ColumnFcmaButterfly8d, ColumnFcmaButterfly8f, ColumnFcmaForwardButterfly8f,
    ColumnFcmaInverseButterfly8f,
};
pub(crate) use bf9::{ColumnButterfly9d, ColumnButterfly9f};
#[cfg(feature = "fcma")]
pub(crate) use bf9::{ColumnFcmaButterfly9d, ColumnFcmaButterfly9f};
pub(crate) use bf10::{ColumnButterfly10d, ColumnButterfly10f};
#[cfg(feature = "fcma")]
pub(crate) use bf10::{ColumnFcmaButterfly10d, ColumnFcmaButterfly10f};
pub(crate) use bf11::{ColumnButterfly11d, ColumnButterfly11f};
#[cfg(feature = "fcma")]
pub(crate) use bf11::{ColumnFcmaButterfly11d, ColumnFcmaButterfly11f};
pub(crate) use bf12::{ColumnButterfly12d, ColumnButterfly12f};
#[cfg(feature = "fcma")]
pub(crate) use bf12::{ColumnFcmaButterfly12d, ColumnFcmaButterfly12f};
pub(crate) use bf13::{ColumnButterfly13d, ColumnButterfly13f};
#[cfg(feature = "fcma")]
pub(crate) use bf13::{ColumnFcmaButterfly13d, ColumnFcmaButterfly13f};
pub(crate) use bf16::{ColumnButterfly16d, ColumnButterfly16f};
#[cfg(feature = "fcma")]
pub(crate) use bf16::{ColumnFcmaButterfly16d, ColumnFcmaButterfly16f};
pub(crate) use bf18::{ColumnButterfly18d, ColumnButterfly18f};
#[cfg(feature = "fcma")]
pub(crate) use bf18::{ColumnFcmaButterfly18d, ColumnFcmaButterfly18f};
#[cfg(feature = "fcma")]
pub(crate) use mixed_radix::{
    NeonFcmaForwardMixedRadix4f, NeonFcmaForwardMixedRadix8f, NeonFcmaInverseMixedRadix4f,
    NeonFcmaInverseMixedRadix8f, NeonFcmaMixedRadix2, NeonFcmaMixedRadix2f, NeonFcmaMixedRadix3,
    NeonFcmaMixedRadix3f, NeonFcmaMixedRadix4, NeonFcmaMixedRadix5, NeonFcmaMixedRadix5f,
    NeonFcmaMixedRadix6, NeonFcmaMixedRadix6f, NeonFcmaMixedRadix7, NeonFcmaMixedRadix7f,
    NeonFcmaMixedRadix8, NeonFcmaMixedRadix9, NeonFcmaMixedRadix9f, NeonFcmaMixedRadix10,
    NeonFcmaMixedRadix10f, NeonFcmaMixedRadix11, NeonFcmaMixedRadix11f, NeonFcmaMixedRadix12,
    NeonFcmaMixedRadix12f, NeonFcmaMixedRadix13, NeonFcmaMixedRadix13f,
};
pub(crate) use mixed_radix::{
    NeonMixedRadix2, NeonMixedRadix2f, NeonMixedRadix3, NeonMixedRadix3f, NeonMixedRadix4,
    NeonMixedRadix4f, NeonMixedRadix5, NeonMixedRadix5f, NeonMixedRadix6, NeonMixedRadix6f,
    NeonMixedRadix7, NeonMixedRadix7f, NeonMixedRadix8, NeonMixedRadix8f, NeonMixedRadix9,
    NeonMixedRadix9f, NeonMixedRadix10, NeonMixedRadix10f, NeonMixedRadix11, NeonMixedRadix11f,
    NeonMixedRadix12, NeonMixedRadix12f, NeonMixedRadix13, NeonMixedRadix13f,
};
pub(crate) use neon_store::{NeonStoreD, NeonStoreF};
