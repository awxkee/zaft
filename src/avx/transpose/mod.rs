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
mod block;
mod f32x2_2x2;
mod f32x2_4x4;
mod f32x2_4xn;
mod f32x2_6x6;
mod f32x2_7x7;
mod f32x2_8x3;
mod f32x2_8xn;
mod f64x2_2x2;
mod f64x2_2xn;
mod f64x2_4x4;
mod f64x2_6x6;
mod transpose_5x5;
mod transpose_real_d;
mod transpose_real_s;

pub(crate) use block::{
    AvxTransposeF323x8, AvxTransposeF325x5, AvxTransposeF327x2, AvxTransposeF327x3,
    AvxTransposeF327x5, AvxTransposeF327x6, AvxTransposeF327x7, AvxTransposeF328x3,
    AvxTransposeF644x4, AvxTransposeNx2F32, AvxTransposeNx2F64, AvxTransposeNx3F32,
    AvxTransposeNx3F64, AvxTransposeNx4F32, AvxTransposeNx4F64, AvxTransposeNx5F32,
    AvxTransposeNx5F64, AvxTransposeNx6F32, AvxTransposeNx6F64, AvxTransposeNx7F32,
    AvxTransposeNx7F64, AvxTransposeNx8F32, AvxTransposeNx8F64, AvxTransposeNx9F32,
    AvxTransposeNx9F64, AvxTransposeNx10F32, AvxTransposeNx10F64, AvxTransposeNx11F32,
    AvxTransposeNx11F64, AvxTransposeNx12F32, AvxTransposeNx12F64, AvxTransposeNx13F32,
    AvxTransposeNx13F64, AvxTransposeNx14F32, AvxTransposeNx14F64, AvxTransposeNx15F32,
    AvxTransposeNx15F64, AvxTransposeNx16F32, AvxTransposeNx16F64,
};
pub(crate) use f32x2_2x2::{avx_transpose_f32x2_2x2, transpose_f32_2x2_impl};
pub(crate) use f32x2_4x4::{
    avx_transpose_f32x2_4x4_impl, avx_transpose_u64_4x4_impl, avx2_transpose_f32x2_4x4,
    transpose_f32x2_4x2, transpose_f32x2_4x4_aos,
};
pub(crate) use f32x2_4xn::{
    transpose_4x2, transpose_4x3, transpose_4x4, transpose_4x5, transpose_4x6, transpose_4x7,
    transpose_4x8, transpose_4x9, transpose_4x10, transpose_4x11, transpose_4x12, transpose_4x13,
    transpose_4x14, transpose_4x15, transpose_4x16,
};
pub(crate) use f32x2_6x6::transpose_6x6_f32;
pub(crate) use f32x2_7x7::{
    block_transpose_f32x2_7x2, block_transpose_f32x2_7x3, block_transpose_f32x2_7x5,
    block_transpose_f32x2_7x6, block_transpose_f32x2_7x7, store_transpose_7x7_f32,
    transpose_7x7_f32, transpose_7x7_f64,
};
pub(crate) use f32x2_8x3::{block_transpose_f32x2_3x8, block_transpose_f32x2_8x3};
pub(crate) use f32x2_8xn::{avx2_transpose_f32x2_8x4, transpose_f32x2_8x5};
pub(crate) use f64x2_2x2::{avx_transpose_f64x2_2x2, transpose_f64x2_2x2};
pub(crate) use f64x2_2xn::{
    transpose_2x3d, transpose_2x4d, transpose_f64x2_2x2d, transpose_f64x2_2x5, transpose_f64x2_2x6,
    transpose_f64x2_2x7, transpose_f64x2_2x8, transpose_f64x2_2x9, transpose_f64x2_2x10,
    transpose_f64x2_2x11, transpose_f64x2_2x12, transpose_f64x2_2x13, transpose_f64x2_2x14,
    transpose_f64x2_2x15, transpose_f64x2_2x16,
};
pub(crate) use f64x2_4x4::{avx_transpose_f64x2_4x4, avx_transpose_f64x2_4x4_impl};
pub(crate) use f64x2_6x6::avx_transpose_f64x2_6x6_impl;
pub(crate) use transpose_5x5::{block_transpose_f32x2_5x5, transpose_5x5_f32, transpose_5x5_f64};
pub(crate) use transpose_real_d::AvxTransposeDReal4x4;
pub(crate) use transpose_real_s::AvxTransposeFReal4x4;
