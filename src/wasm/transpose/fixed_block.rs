/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::transpose::TransposeExecutor;
use crate::wasm::store::WasmStoreF;
use num_complex::Complex;

type FunctionEvenF<const N: usize> = fn([WasmStoreF; N]) -> [WasmStoreF; N];
type FunctionOddF<const N: usize, const ODD: usize> = fn([WasmStoreF; N]) -> [WasmStoreF; ODD];

// assumes that execution block is exactly divisible by executor
pub(crate) fn transpose_height_block_executor2_f32<
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    E: Fn([WasmStoreF; Y_BLOCK_SIZE]) -> [WasmStoreF; Y_BLOCK_SIZE],
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
        let mut store = [WasmStoreF::default(); Y_BLOCK_SIZE];
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = WasmStoreF::from_complex_ref(src.get_unchecked(i * input_stride..));
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
                    store[i] = WasmStoreF::load_complex(src.get_unchecked(i * input_stride));
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

// assumes that execution block is not exactly divisible by executor
pub(crate) fn transpose_height_block_executor2_f32_odd<
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    const Y_ODD_BLOCK_SIZE: usize,
    E: Fn([WasmStoreF; Y_BLOCK_SIZE]) -> [WasmStoreF; Y_ODD_BLOCK_SIZE],
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
        let mut store = [WasmStoreF::default(); Y_BLOCK_SIZE];
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = WasmStoreF::from_complex_ref(src.get_unchecked(i * input_stride..));
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
                    store[i] = WasmStoreF::load_complex(src.get_unchecked(i * input_stride));
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
                self.transpose_strided(input, width, output, height, width, height);
            }

            fn transpose_strided(
                &self,
                input: &[Complex<$complex_type>],
                input_stride: usize,
                output: &mut [Complex<$complex_type>],
                output_stride: usize,
                width: usize,
                height: usize,
            ) {
                use crate::wasm::transpose::$rot_name;
                transpose_height_block_executor2_f32::<
                    $block_width,
                    $block_height,
                    FunctionEvenF<$block_height>,
                >(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    width,
                    height,
                    0,
                    $rot_name,
                );
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
                self.transpose_strided(input, width, output, height, width, height);
            }

            fn transpose_strided(
                &self,
                input: &[Complex<$complex_type>],
                input_stride: usize,
                output: &mut [Complex<$complex_type>],
                output_stride: usize,
                width: usize,
                height: usize,
            ) {
                use crate::wasm::transpose::$rot_name;
                const R: usize = $block_height + 1;
                transpose_height_block_executor2_f32_odd::<
                    $block_width,
                    $block_height,
                    R,
                    FunctionOddF<$block_height, R>,
                >(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    width,
                    height,
                    0,
                    $rot_name,
                );
            }
        }
    };
}

define_transpose_oddf!(WasmTransposeNx9F32, f32, transpose_f32x2_2x9, 2, 9);
define_transpose_evenf!(WasmTransposeNx8F32, f32, transpose_f32x2_2x8, 2, 8);
define_transpose_oddf!(WasmTransposeNx7F32, f32, transpose_f32x2_2x7, 2, 7);
define_transpose_oddf!(WasmTransposeNx5F32, f32, transpose_f32x2_2x5, 2, 5);
define_transpose_evenf!(WasmTransposeNx4F32, f32, transpose_f32x2_2x4, 2, 4);
define_transpose_oddf!(WasmTransposeNx3F32, f32, transpose_f32x2_2x3, 2, 3);
define_transpose_evenf!(WasmTransposeNx2F32, f32, transpose_f32x2_2x2, 2, 2);
