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
#![allow(clippy::needless_range_loop)]
mod blocks;
mod f32x2_11x2;
mod f32x2_2x10;
mod f32x2_2x11;
mod f32x2_2x2;
mod f32x2_2x9;
mod f32x2_2xn;
mod f32x2_4x4;
mod f32x2_6x4;
mod f32x2_6x5;
mod f32x2_6x6;
mod f32x2_7x5;
mod f32x2_7x7;
mod f32x2_8x3;
mod f32x2_8x8;
mod f32x2_9x2;
mod f64x2_2x2;
mod f64x2_4x3;
mod f64x2_4x4;
mod transpose_real_d;
mod transpose_real_s;

pub(crate) use blocks::{
    NeonTranspose2x2F32, NeonTranspose2x2F64, NeonTranspose4x3F64, NeonTranspose4x4F32,
    NeonTranspose4x4F64, NeonTranspose4x7F32, NeonTranspose5x7F32, NeonTranspose6x4F32,
    NeonTranspose6x5F32, NeonTranspose7x2F32, NeonTranspose7x3F32, NeonTranspose7x5F32,
    NeonTranspose7x6F32, NeonTranspose7x7F32, NeonTranspose8x3F32, NeonTranspose9x2F32,
    NeonTranspose11x2F32, NeonTransposeNx2F32, NeonTransposeNx2F64, NeonTransposeNx3F32,
    NeonTransposeNx3F64, NeonTransposeNx4F32, NeonTransposeNx4F64, NeonTransposeNx5F32,
    NeonTransposeNx5F64, NeonTransposeNx6F32, NeonTransposeNx6F64, NeonTransposeNx7F32,
    NeonTransposeNx7F64, NeonTransposeNx8F32, NeonTransposeNx8F64, NeonTransposeNx9F32,
    NeonTransposeNx9F64, NeonTransposeNx10F32, NeonTransposeNx10F64, NeonTransposeNx11F32,
    NeonTransposeNx11F64, NeonTransposeNx12F32, NeonTransposeNx12F64, NeonTransposeNx13F32,
    NeonTransposeNx13F64, NeonTransposeNx14F32, NeonTransposeNx14F64, NeonTransposeNx15F32,
    NeonTransposeNx15F64, NeonTransposeNx16F32, NeonTransposeNx16F64, NeonTransposeNx17F32,
    NeonTransposeNx17F64, NeonTransposeNx18F32, NeonTransposeNx18F64, NeonTransposeNx19F32,
    NeonTransposeNx19F64, NeonTransposeNx20F32,
};
pub(crate) use f32x2_2x2::{block_transpose_f32x2_2x2, neon_transpose_f32x2_2x2_impl};
pub(crate) use f32x2_2x9::{transpose_2x8, transpose_2x9};
pub(crate) use f32x2_2x10::transpose_2x10;
pub(crate) use f32x2_2x11::transpose_2x11;
pub(crate) use f32x2_2xn::{
    transpose_2x2, transpose_2x3, transpose_2x4, transpose_2x12, transpose_2x13, transpose_2x14,
    transpose_2x15, transpose_2x16, transpose_2x17, transpose_2x18, transpose_2x19, transpose_2x20,
};
pub(crate) use f32x2_4x4::{neon_transpose_f32x2_4x4, transpose_f32x2_4x4};
pub(crate) use f32x2_6x4::neon_transpose_f32x2_6x4;
pub(crate) use f32x2_6x5::{block_transpose_f32x2_6x5, transpose_2x5, transpose_6x5};
pub(crate) use f32x2_6x6::{neon_transpose_f32x2_6x6, neon_transpose_f32x2_6x6_aos, transpose_2x6};
pub(crate) use f32x2_7x5::{
    block_transpose_f32x2_4x7, block_transpose_f32x2_5x7, block_transpose_f32x2_7x2,
    block_transpose_f32x2_7x3, block_transpose_f32x2_7x5, block_transpose_f32x2_7x6,
    neon_transpose_f32x2_7x6_aos, transpose_2x7,
};
pub(crate) use f32x2_7x7::{block_transpose_f32x2_7x7, neon_transpose_f32x2_7x7_aos};
pub(crate) use f32x2_8x3::block_transpose_f32x2_8x3;
pub(crate) use f32x2_8x8::{neon_transpose_f32x2_8x5_aos, transpose_8x8_f32};
pub(crate) use f32x2_9x2::block_transpose_f32x2_9x2;
pub(crate) use f32x2_11x2::block_transpose_f32x2_11x2;
pub(crate) use f64x2_2x2::{neon_transpose_f64x2_2x2, neon_transpose_f64x2_4x4_impl};
pub(crate) use f64x2_4x3::block_transpose_f64x2_4x3;
pub(crate) use f64x2_4x4::block_transpose_f64x2_4x4;
pub(crate) use transpose_real_d::NeonTransposeDReal4x4;
pub(crate) use transpose_real_s::NeonTransposeReal4x4;
