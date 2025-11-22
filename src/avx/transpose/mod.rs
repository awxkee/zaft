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
mod block;
mod f32x2_2x2;
mod f32x2_4x11;
mod f32x2_4x4;
mod f32x2_6x6;
mod f32x2_7x7;
mod f32x2_8x3;
mod f32x2_8x4;
mod f64x2_2x2;
mod f64x2_2xn;
mod f64x2_4x4;
mod f64x2_6x6;
mod transpose_5x5;

pub(crate) use block::{
    AvxTransposeF322x2, AvxTransposeF322x11, AvxTransposeF323x8, AvxTransposeF324x3,
    AvxTransposeF324x4, AvxTransposeF324x11, AvxTransposeF325x2, AvxTransposeF325x3,
    AvxTransposeF325x5, AvxTransposeF327x2, AvxTransposeF327x3, AvxTransposeF327x5,
    AvxTransposeF327x6, AvxTransposeF327x7, AvxTransposeF328x3, AvxTransposeF642x2,
    AvxTransposeF644x4, AvxTransposeNx5F64, AvxTransposeNx6F64, AvxTransposeNx7F64,
    AvxTransposeNx8F64, AvxTransposeNx9F64, AvxTransposeNx10F64, AvxTransposeNx11F64,
    AvxTransposeNx12F64, AvxTransposeNx13F64, AvxTransposeNx14F64, AvxTransposeNx15F64,
    AvxTransposeNx16F64
};
pub(crate) use f32x2_2x2::{avx_transpose_f32x2_2x2, transpose_f32_2x2_impl};
pub(crate) use f32x2_4x4::avx2_transpose_f32x2_4x3;
pub(crate) use f32x2_4x4::{
    avx_transpose_f32x2_4x4_impl, avx_transpose_u64_4x4_impl, avx2_transpose_f32x2_4x4,
};
pub(crate) use f32x2_4x11::{
    block_transpose_f32x2_2x11, block_transpose_f32x2_4x11, transpose_4x11,
};
pub(crate) use f32x2_6x6::transpose_6x6_f32;
pub(crate) use f32x2_7x7::{
    block_transpose_f32x2_7x2, block_transpose_f32x2_7x3, block_transpose_f32x2_7x5,
    block_transpose_f32x2_7x6, block_transpose_f32x2_7x7, store_transpose_7x7_f32,
    transpose_7x7_f32, transpose_7x7_f64,
};
pub(crate) use f32x2_8x3::{block_transpose_f32x2_3x8, block_transpose_f32x2_8x3};
pub(crate) use f32x2_8x4::avx2_transpose_f32x2_8x4;
pub(crate) use f64x2_2x2::{avx_transpose_f64x2_2x2, transpose_f64x2_2x2};
pub(crate) use f64x2_2xn::{
    transpose_2x5d, transpose_f64x2_2x6, transpose_f64x2_2x7, transpose_f64x2_2x8,
    transpose_f64x2_2x9, transpose_f64x2_2x10, transpose_f64x2_2x11, transpose_f64x2_2x12,
    transpose_f64x2_2x13, transpose_f64x2_2x14, transpose_f64x2_2x15, transpose_f64x2_2x16
};
pub(crate) use f64x2_4x4::{avx_transpose_f64x2_4x4, avx_transpose_f64x2_4x4_impl};
pub(crate) use f64x2_6x6::avx_transpose_f64x2_6x6_impl;
pub(crate) use transpose_5x5::{
    block_transpose_f32x2_5x2, block_transpose_f32x2_5x3, block_transpose_f32x2_5x5,
    transpose_5x5_f32, transpose_5x5_f64,
};
