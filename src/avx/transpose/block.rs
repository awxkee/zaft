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

type FunctionF32 = fn(&[Complex<f32>], usize, &mut [Complex<f32>], usize);
type FunctionF64 = fn(&[Complex<f64>], usize, &mut [Complex<f64>], usize);

macro_rules! define_transpose {
    ($rule_name: ident, $complex_type: ident, $rot_name: ident, $block_width: expr, $block_height: expr, $func: ident) => {
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
                use crate::avx::$rot_name;
                unsafe {
                    transpose_fixed_block_executor2d::<
                        $complex_type,
                        $block_width,
                        $block_height,
                        $func,
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

define_transpose!(
    AvxTransposeF324x4,
    f32,
    avx2_transpose_f32x2_4x4,
    4,
    4,
    FunctionF32
);
define_transpose!(
    AvxTransposeF642x2,
    f64,
    avx_transpose_f64x2_2x2,
    2,
    2,
    FunctionF64
);
define_transpose!(
    AvxTransposeF644x4,
    f64,
    avx_transpose_f64x2_4x4,
    4,
    4,
    FunctionF64
);
define_transpose!(
    AvxTransposeF647x7,
    f64,
    block_transpose_f64x2_7x7,
    7,
    7,
    FunctionF64
);
define_transpose!(
    AvxTransposeF327x7,
    f32,
    block_transpose_f32x2_7x7,
    7,
    7,
    FunctionF32
);
define_transpose!(
    AvxTransposeF322x2,
    f32,
    avx_transpose_f32x2_2x2,
    2,
    2,
    FunctionF32
);
