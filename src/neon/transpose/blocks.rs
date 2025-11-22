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
use crate::neon::mixed::NeonStoreF;
use crate::transpose::TransposeExecutor;
use num_complex::Complex;

// assumes that execution block is exactly divisible by executor
pub(crate) fn transpose_fixed_block_executor2d<
    V: Copy + Default,
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    E: Fn(&[Complex<V>], usize, &mut [Complex<V>], usize),
>(
    input: &[Complex<V>],
    input_stride: usize,
    output: &mut [Complex<V>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: E,
) -> usize {
    let mut y = start_y;
    unsafe {
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                exec(src, input_stride, dst, output_stride);

                x += X_BLOCK_SIZE;
            }

            y += Y_BLOCK_SIZE;
        }
    }

    y
}

// assumes that execution block is exactly divisible by executor
pub(crate) fn transpose_height_block_executor2_f32<
    V: Copy + Default,
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    E: Fn([NeonStoreF; Y_BLOCK_SIZE]) -> [NeonStoreF; Y_BLOCK_SIZE],
>(
    input: &[Complex<f32>],
    input_stride: usize,
    output: &mut [Complex<f32>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: E,
) -> usize {
    let mut y = start_y;
    unsafe {
        let mut store = [NeonStoreF::default(); Y_BLOCK_SIZE];
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * input_stride..));
                }

                let q = exec(store);

                for i in 0..Y_BLOCK_SIZE / 2 {
                    q[i * 2].write(dst.get_unchecked_mut(i * 2..));
                    q[i * 2 + 1].write(dst.get_unchecked_mut(i * 2 + output_stride..));
                }

                x += X_BLOCK_SIZE;
            }

            if x < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * input_stride..));
                }

                let q = exec(store);

                for i in 0..Y_BLOCK_SIZE / 2 {
                    q[i * 2].write(dst.get_unchecked_mut(i * 2..));
                }
            }

            y += Y_BLOCK_SIZE;
        }
    }

    y
}

// assumes that execution block is exactly divisible by executor
pub(crate) fn transpose_height_block_executor2_f32_odd<
    V: Copy + Default,
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    const Y_ODD_BLOCK_SIZE: usize,
    E: Fn([NeonStoreF; Y_BLOCK_SIZE]) -> [NeonStoreF; Y_ODD_BLOCK_SIZE],
>(
    input: &[Complex<f32>],
    input_stride: usize,
    output: &mut [Complex<f32>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: E,
) -> usize {
    let mut y = start_y;
    unsafe {
        let mut store = [NeonStoreF::default(); Y_BLOCK_SIZE];
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * input_stride..));
                }

                let q = exec(store);

                for i in 0..Y_BLOCK_SIZE / 2 {
                    q[i * 2].write(dst.get_unchecked_mut(i * 2..));
                    q[i * 2 + 1].write(dst.get_unchecked_mut(i * 2 + output_stride..));
                }

                q[(Y_BLOCK_SIZE / 2) * 2].write_lo(dst.get_unchecked_mut((Y_BLOCK_SIZE / 2) * 2..));
                q[(Y_BLOCK_SIZE / 2) * 2 + 1]
                    .write_lo(dst.get_unchecked_mut((Y_BLOCK_SIZE / 2) * 2 + output_stride..));

                x += X_BLOCK_SIZE;
            }

            if x < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = NeonStoreF::from_complex_ref(src.get_unchecked(i * input_stride..));
                }

                let q = exec(store);

                for i in 0..Y_BLOCK_SIZE / 2 {
                    q[i * 2].write(dst.get_unchecked_mut(i * 2..));
                }

                q[(Y_BLOCK_SIZE / 2) * 2].write_lo(dst.get_unchecked_mut((Y_BLOCK_SIZE / 2) * 2..));
            }

            y += Y_BLOCK_SIZE;
        }
    }

    y
}

type Function<V> = fn(&[Complex<V>], usize, &mut [Complex<V>], usize);
type FunctionEvenF<const N: usize> = fn([NeonStoreF; N]) -> [NeonStoreF; N];
type FunctionOddF<const N: usize, const ODD: usize> = fn([NeonStoreF; N]) -> [NeonStoreF; ODD];

macro_rules! define_transpose {
    ($rule_name: ident, $complex_type: ident, $rot_name: ident, $block_width: expr, $block_height: expr) => {
        #[derive(Default)]
        pub(crate) struct $rule_name {}

        impl TransposeExecutor<$complex_type> for $rule_name {
            fn transpose(
                &self,
                input: &[Complex<$complex_type>],
                output: &mut [Complex<$complex_type>],
                width: usize,
                height: usize,
            ) {
                use crate::neon::transpose::$rot_name;
                transpose_fixed_block_executor2d::<
                    $complex_type,
                    $block_width,
                    $block_height,
                    Function<$complex_type>,
                >(input, width, output, height, width, height, 0, $rot_name);
            }
        }
    };
}

macro_rules! define_transpose_evenf {
    ($rule_name: ident, $complex_type: ident, $rot_name: ident, $block_width: expr, $block_height: expr) => {
        #[derive(Default)]
        pub(crate) struct $rule_name {}

        impl TransposeExecutor<$complex_type> for $rule_name {
            fn transpose(
                &self,
                input: &[Complex<$complex_type>],
                output: &mut [Complex<$complex_type>],
                width: usize,
                height: usize,
            ) {
                use crate::neon::transpose::$rot_name;
                transpose_height_block_executor2_f32::<
                    $complex_type,
                    $block_width,
                    $block_height,
                    FunctionEvenF<$block_height>,
                >(input, width, output, height, width, height, 0, $rot_name);
            }
        }
    };
}

macro_rules! define_transpose_oddf {
    ($rule_name: ident, $complex_type: ident, $rot_name: ident, $block_width: expr, $block_height: expr) => {
        #[derive(Default)]
        pub(crate) struct $rule_name {}

        impl TransposeExecutor<$complex_type> for $rule_name {
            fn transpose(
                &self,
                input: &[Complex<$complex_type>],
                output: &mut [Complex<$complex_type>],
                width: usize,
                height: usize,
            ) {
                use crate::neon::transpose::$rot_name;
                const R: usize = $block_height + 1;
                transpose_height_block_executor2_f32_odd::<
                    $complex_type,
                    $block_width,
                    $block_height,
                    R,
                    FunctionOddF<$block_height, R>,
                >(input, width, output, height, width, height, 0, $rot_name);
            }
        }
    };
}

define_transpose!(NeonTranspose7x7F32, f32, block_transpose_f32x2_7x7, 7, 7);
define_transpose!(NeonTranspose7x6F32, f32, block_transpose_f32x2_7x6, 7, 6);
define_transpose!(NeonTranspose7x5F32, f32, block_transpose_f32x2_7x5, 7, 5);
define_transpose!(NeonTranspose7x3F32, f32, block_transpose_f32x2_7x3, 7, 3);
define_transpose!(NeonTranspose7x2F32, f32, block_transpose_f32x2_7x2, 7, 2);
define_transpose!(NeonTranspose6x4F32, f32, neon_transpose_f32x2_6x4, 6, 4);
define_transpose!(NeonTranspose6x5F32, f32, block_transpose_f32x2_6x5, 6, 5);
define_transpose_evenf!(NeonTransposeNx16F32, f32, transpose_2x16, 2, 16);
define_transpose_oddf!(NeonTransposeNx15F32, f32, transpose_2x15, 2, 15);
define_transpose_evenf!(NeonTransposeNx14F32, f32, transpose_2x14, 2, 14);
define_transpose_oddf!(NeonTransposeNx13F32, f32, transpose_2x13, 2, 13);
define_transpose_evenf!(NeonTransposeNx12F32, f32, transpose_2x12, 2, 12);
define_transpose_oddf!(NeonTransposeNx11F32, f32, transpose_2x11, 2, 11);
define_transpose_evenf!(NeonTransposeNx10F32, f32, transpose_2x10, 2, 10);
define_transpose_oddf!(NeonTransposeNx9F32, f32, transpose_2x9, 2, 9);
define_transpose_oddf!(NeonTransposeNx7F32, f32, transpose_2x7, 2, 7);
define_transpose_evenf!(NeonTransposeNx8F32, f32, transpose_2x8, 2, 8);
define_transpose_evenf!(NeonTransposeNx6F32, f32, transpose_2x6, 2, 6);
define_transpose_oddf!(NeonTransposeNx5F32, f32, transpose_2x5, 2, 5);
define_transpose!(NeonTranspose11x2F32, f32, block_transpose_f32x2_11x2, 11, 2);
define_transpose!(NeonTranspose4x4F32, f32, neon_transpose_f32x2_4x4, 4, 4);
define_transpose!(NeonTranspose2x2F32, f32, block_transpose_f32x2_2x2, 2, 2);
define_transpose!(NeonTranspose2x2F64, f64, neon_transpose_f64x2_2x2, 2, 2);
define_transpose!(NeonTranspose4x4F64, f64, block_transpose_f64x2_4x4, 4, 4);
define_transpose!(NeonTranspose4x3F64, f64, block_transpose_f64x2_4x3, 4, 3);
define_transpose!(NeonTranspose4x3F32, f32, block_transpose_f32x2_4x3, 4, 3);
define_transpose!(NeonTranspose5x4F32, f32, block_transpose_f32x2_5x4, 5, 4);
define_transpose!(NeonTranspose5x3F32, f32, block_transpose_f32x2_5x3, 5, 3);
define_transpose!(NeonTranspose5x2F32, f32, block_transpose_f32x2_5x2, 5, 2);
define_transpose!(NeonTranspose8x3F32, f32, block_transpose_f32x2_8x3, 8, 3);
define_transpose!(NeonTranspose3x8F32, f32, block_transpose_f32x2_3x8, 3, 8);
define_transpose!(NeonTranspose4x7F32, f32, block_transpose_f32x2_4x7, 4, 7);
define_transpose!(NeonTranspose5x7F32, f32, block_transpose_f32x2_5x7, 5, 7);
define_transpose!(NeonTranspose9x2F32, f32, block_transpose_f32x2_9x2, 9, 2);
