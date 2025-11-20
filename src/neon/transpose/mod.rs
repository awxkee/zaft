/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
mod blocks;
mod f32x2_11x2;
mod f32x2_2x10;
mod f32x2_2x11;
mod f32x2_2x12;
mod f32x2_2x2;
mod f32x2_2x9;
mod f32x2_4x4;
mod f32x2_6x4;
mod f32x2_6x5;
mod f32x2_6x6;
mod f32x2_7x5;
mod f32x2_7x7;
mod f32x2_8x8;
mod f64x2_2x2;
mod f64x2_4x4;
mod f64x2_4x3;

pub(crate) use blocks::{
    NeonTranspose2x2F32, NeonTranspose2x2F64, NeonTranspose2x9F32, NeonTranspose2x10F32,
    NeonTranspose2x11F32, NeonTranspose2x12F32, NeonTranspose4x4F32, NeonTranspose4x4F64,
    NeonTranspose6x4F32, NeonTranspose6x5F32, NeonTranspose7x5F32, NeonTranspose7x7F32,
    NeonTranspose11x2F32,NeonTranspose4x3F64
};
pub(crate) use f32x2_2x2::{block_transpose_f32x2_2x2, neon_transpose_f32x2_2x2_impl};
pub(crate) use f32x2_2x9::{block_transpose_f32x2_2x9, transpose_2x9};
pub(crate) use f32x2_2x10::{block_transpose_f32x2_2x10, transpose_2x10};
pub(crate) use f32x2_2x11::{block_transpose_f32x2_2x11, transpose_2x11};
pub(crate) use f32x2_2x12::block_transpose_f32x2_2x12;
pub(crate) use f32x2_4x4::{neon_transpose_f32x2_4x4, transpose_f32x2_4x4};
pub(crate) use f32x2_6x4::neon_transpose_f32x2_6x4;
pub(crate) use f32x2_6x5::{block_transpose_f32x2_6x5, transpose_6x5};
pub(crate) use f32x2_6x6::{neon_transpose_f32x2_6x6, neon_transpose_f32x2_6x6_aos};
pub(crate) use f32x2_7x5::block_transpose_f32x2_7x5;
pub(crate) use f32x2_7x7::{block_transpose_f32x2_7x7, neon_transpose_f32x2_7x7_aos};
pub(crate) use f32x2_8x8::transpose_8x8_f32;
pub(crate) use f32x2_11x2::block_transpose_f32x2_11x2;
pub(crate) use f64x2_2x2::{neon_transpose_f64x2_2x2, neon_transpose_f64x2_4x4_impl};
pub(crate) use f64x2_4x4::block_transpose_f64x2_4x4;
pub(crate) use f64x2_4x3::block_transpose_f64x2_4x3;