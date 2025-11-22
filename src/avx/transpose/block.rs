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
use crate::avx::mixed::AvxStoreD;
use crate::transpose::TransposeExecutor;
use num_complex::Complex;

// assumes that execution block is exactly divisible by executor
#[target_feature(enable = "avx2")]
fn transpose_fixed_block_executor2d<
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

type Function<V> = fn(&[Complex<V>], usize, &mut [Complex<V>], usize);

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
                use crate::avx::transpose::$rot_name;
                unsafe {
                    transpose_fixed_block_executor2d::<
                        $complex_type,
                        $block_width,
                        $block_height,
                        Function<$complex_type>,
                    >(
                        input,
                        width,
                        output,
                        height,
                        width,
                        height,
                        0,
                        |src, ss, dst, ds| $rot_name(src, ss, dst, ds),
                    );
                }
            }
        }
    };
}

type FunctionEvenD<const N: usize> = fn([AvxStoreD; N]) -> [AvxStoreD; N];
type FunctionOddD<const N: usize, const ODD: usize> = fn([AvxStoreD; N]) -> [AvxStoreD; ODD];

// assumes that execution block is exactly divisible by executor
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_height_block_executor2_f64<
    V: Copy + Default,
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    E: Fn([AvxStoreD; Y_BLOCK_SIZE]) -> [AvxStoreD; Y_BLOCK_SIZE],
>(
    input: &[Complex<f64>],
    input_stride: usize,
    output: &mut [Complex<f64>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: E,
) -> usize {
    let mut y = start_y;
    unsafe {
        let mut store = [AvxStoreD::zero(); Y_BLOCK_SIZE];
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * input_stride..));
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
                    store[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * input_stride..));
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
#[target_feature(enable = "avx2")]
pub(crate) fn transpose_height_block_executor2_f64_odd<
    V: Copy + Default,
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    const Y_ODD_BLOCK_SIZE: usize,
    E: Fn([AvxStoreD; Y_BLOCK_SIZE]) -> [AvxStoreD; Y_ODD_BLOCK_SIZE],
>(
    input: &[Complex<f64>],
    input_stride: usize,
    output: &mut [Complex<f64>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: E,
) -> usize {
    let mut y = start_y;
    unsafe {
        let mut store = [AvxStoreD::zero(); Y_BLOCK_SIZE];
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * input_stride..));
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
                    store[i] = AvxStoreD::from_complex_ref(src.get_unchecked(i * input_stride..));
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

macro_rules! define_transpose_evend {
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
                use crate::avx::transpose::$rot_name;
                unsafe {
                    transpose_height_block_executor2_f64::<
                        $complex_type,
                        $block_width,
                        $block_height,
                        FunctionEvenD<$block_height>,
                    >(input, width, output, height, width, height, 0, $rot_name);
                }
            }
        }
    };
}

macro_rules! define_transpose_oddd {
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
                use crate::avx::transpose::$rot_name;
                const R: usize = $block_height + 1;
                unsafe {
                    transpose_height_block_executor2_f64_odd::<
                        $complex_type,
                        $block_width,
                        $block_height,
                        R,
                        FunctionOddD<$block_height, R>,
                    >(input, width, output, height, width, height, 0, $rot_name);
                }
            }
        }
    };
}

define_transpose!(AvxTransposeF324x4, f32, avx2_transpose_f32x2_4x4, 4, 4);
define_transpose!(AvxTransposeF324x3, f32, avx2_transpose_f32x2_4x3, 4, 3);
define_transpose!(AvxTransposeF642x2, f64, avx_transpose_f64x2_2x2, 2, 2);
define_transpose!(AvxTransposeF644x4, f64, avx_transpose_f64x2_4x4, 4, 4);
define_transpose!(AvxTransposeF327x7, f32, block_transpose_f32x2_7x7, 7, 7);
define_transpose!(AvxTransposeF322x2, f32, avx_transpose_f32x2_2x2, 2, 2);
define_transpose!(AvxTransposeF325x5, f32, block_transpose_f32x2_5x5, 5, 5);
define_transpose!(AvxTransposeF325x3, f32, block_transpose_f32x2_5x3, 5, 3);
define_transpose!(AvxTransposeF325x2, f32, block_transpose_f32x2_5x2, 5, 2);
define_transpose!(AvxTransposeF327x6, f32, block_transpose_f32x2_7x6, 7, 6);
define_transpose!(AvxTransposeF327x5, f32, block_transpose_f32x2_7x5, 7, 5);
define_transpose!(AvxTransposeF327x3, f32, block_transpose_f32x2_7x3, 7, 3);
define_transpose!(AvxTransposeF327x2, f32, block_transpose_f32x2_7x2, 7, 2);
define_transpose!(AvxTransposeF324x11, f32, block_transpose_f32x2_4x11, 4, 11);
define_transpose!(AvxTransposeF322x11, f32, block_transpose_f32x2_2x11, 2, 11);
define_transpose!(AvxTransposeF328x3, f32, block_transpose_f32x2_8x3, 8, 3);
define_transpose!(AvxTransposeF323x8, f32, block_transpose_f32x2_3x8, 3, 8);

define_transpose_evend!(AvxTransposeNx16F64, f64, transpose_f64x2_2x16, 2, 16);
define_transpose_oddd!(AvxTransposeNx15F64, f64, transpose_f64x2_2x15, 2, 15);
define_transpose_evend!(AvxTransposeNx14F64, f64, transpose_f64x2_2x14, 2, 14);
define_transpose_oddd!(AvxTransposeNx13F64, f64, transpose_f64x2_2x13, 2, 13);
define_transpose_evend!(AvxTransposeNx12F64, f64, transpose_f64x2_2x12, 2, 12);
define_transpose_oddd!(AvxTransposeNx11F64, f64, transpose_f64x2_2x11, 2, 11);
define_transpose_evend!(AvxTransposeNx10F64, f64, transpose_f64x2_2x10, 2, 10);
define_transpose_oddd!(AvxTransposeNx9F64, f64, transpose_f64x2_2x9, 2, 9);
define_transpose_evend!(AvxTransposeNx8F64, f64, transpose_f64x2_2x8, 2, 8);
define_transpose_oddd!(AvxTransposeNx7F64, f64, transpose_f64x2_2x7, 2, 7);
define_transpose_evend!(AvxTransposeNx6F64, f64, transpose_f64x2_2x6, 2, 6);
define_transpose_oddd!(AvxTransposeNx5F64, f64, transpose_2x5d, 2, 5);
