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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::transpose::{TransposeExecutor, TransposeExecutorRealInv};
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
                    store[i] = NeonStoreF::from_complex(src.get_unchecked(i * input_stride));
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
                    store[i] = NeonStoreF::from_complex(src.get_unchecked(i * input_stride));
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

pub(crate) fn transpose_f64x2_complex_to_real<
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
    const Y_ODD_BLOCK_SIZE: usize,
    E: Fn([NeonStoreD; Y_BLOCK_SIZE]) -> [NeonStoreD; Y_ODD_BLOCK_SIZE],
>(
    input: &[Complex<f64>],
    input_stride: usize,
    output: &mut [f64],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: E,
) -> usize {
    let mut y = start_y;
    unsafe {
        let mut store = [NeonStoreD::default(); Y_BLOCK_SIZE];
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE <= width {
                let output_x = x;

                let src = src.get_unchecked(x..);

                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    let r0 = NeonStoreD::from_complex_ref(src.get_unchecked(i * input_stride..));
                    let r1 =
                        NeonStoreD::from_complex_ref(src.get_unchecked(i * input_stride + 1..));
                    store[i] = r0.unpack_evens(r1);
                }

                let q = exec(store);

                for i in 0..Y_BLOCK_SIZE / 2 {
                    q[i * 2].write_real(dst.get_unchecked_mut(i * 2..));
                    q[i * 2 + 1].write_real(dst.get_unchecked_mut(i * 2 + output_stride..));
                }

                q[(Y_BLOCK_SIZE / 2) * 2]
                    .write_real_lo(dst.get_unchecked_mut((Y_BLOCK_SIZE / 2) * 2..));
                q[(Y_BLOCK_SIZE / 2) * 2 + 1]
                    .write_real_lo(dst.get_unchecked_mut((Y_BLOCK_SIZE / 2) * 2 + output_stride..));

                x += X_BLOCK_SIZE;
            }

            if x < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = NeonStoreD::from_complex(src.get_unchecked(i * input_stride));
                }

                let q = exec(store);

                for i in 0..Y_BLOCK_SIZE / 2 {
                    q[i * 2].write_real(dst.get_unchecked_mut(i * 2..));
                }

                q[(Y_BLOCK_SIZE / 2) * 2]
                    .write_real_lo(dst.get_unchecked_mut((Y_BLOCK_SIZE / 2) * 2..));
            }

            y += Y_BLOCK_SIZE;
        }
    }

    y
}

// assumes that execution block is exactly divisible by executor
pub(crate) fn transpose_height_block_executor2_f64<
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
>(
    input: &[Complex<f64>],
    input_stride: usize,
    output: &mut [Complex<f64>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
) -> usize {
    let mut y = start_y;
    unsafe {
        while y + Y_BLOCK_SIZE <= height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                let mut store = [NeonStoreD::default(); Y_BLOCK_SIZE];

                for i in 0..Y_BLOCK_SIZE {
                    store[i] = NeonStoreD::from_complex_ref(src.get_unchecked(i * input_stride..));
                }

                for i in 0..Y_BLOCK_SIZE {
                    store[i].write(dst.get_unchecked_mut(i..));
                }

                x += 1;
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
                use crate::neon::transpose::$rot_name;
                transpose_fixed_block_executor2d::<
                    $complex_type,
                    $block_width,
                    $block_height,
                    Function<$complex_type>,
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
                use crate::neon::transpose::$rot_name;
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
                use crate::neon::transpose::$rot_name;
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

macro_rules! define_transposed {
    ($rule_name: ident, $complex_type: ident, $block_width: expr, $block_height: expr) => {
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
                transpose_height_block_executor2_f64::<$block_width, $block_height>(
                    input,
                    input_stride,
                    output,
                    output_stride,
                    width,
                    height,
                    0,
                );
            }
        }
    };
}

macro_rules! define_transpose_inv_d {
    ($rule_name: ident, $block_exec: ident, $complex_type: ident, $block_width: expr, $block_height: expr) => {
        #[derive(Default)]
        pub(crate) struct $rule_name {}

        impl TransposeExecutorRealInv<$complex_type> for $rule_name {
            fn transpose(
                &self,
                input: &[Complex<$complex_type>],
                output: &mut [$complex_type],
                width: usize,
                height: usize,
            ) {
                use crate::neon::transpose::$block_exec;
                const R: usize = $block_height + 1;
                transpose_f64x2_complex_to_real::<
                    $block_width,
                    $block_height,
                    R,
                    fn([NeonStoreD; $block_height]) -> [NeonStoreD; $block_height + 1],
                >(input, width, output, height, width, height, 0, $block_exec);
            }
        }
    };
}

macro_rules! define_transpose_to_real_oddf {
    ($rule_name: ident, $method_name: ident, $rot_name: ident, $block_width: expr, $block_height: expr, $odd_block_height: expr) => {
        #[allow(clippy::out_of_bounds_indexing, clippy::reversed_empty_ranges)]
        pub(crate) fn $method_name(
            input: &[Complex<f32>],
            input_stride: usize,
            output: &mut [f32],
            output_stride: usize,
            width: usize,
            height: usize,
            start_y: usize,
        ) -> usize {
            use crate::neon::transpose::$rot_name;
            let mut y = start_y;
            const X_BLOCK_SIZE: usize = $block_width;
            const Y_BLOCK_SIZE: usize = $block_height;
            unsafe {
                let mut store = [NeonStoreF::default(); Y_BLOCK_SIZE];

                const REM: usize = $block_height % 4;
                const QUO: usize = $block_height / 4;

                while y + Y_BLOCK_SIZE <= height {
                    let input_y = y;

                    let src = input.get_unchecked(input_stride * input_y..);

                    let mut x = 0usize;

                    if REM == 0 {
                        while x + X_BLOCK_SIZE <= width {
                            let output_x = x;

                            let src = src.get_unchecked(x..);
                            let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                            for i in 0..Y_BLOCK_SIZE {
                                let q0 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                                let q1 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride + 2..),
                                );
                                let v = q0.unpack_evens(q1);
                                store[i] = v;
                            }

                            let q = $rot_name(store);

                            match QUO {
                                0 => {
                                    // nothing
                                }

                                1 => {
                                    q[0].write_real(dst.get_unchecked_mut(0..));
                                    q[1].write_real(dst.get_unchecked_mut(output_stride..));
                                    q[2].write_real(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write_real(dst.get_unchecked_mut(output_stride * 3..));
                                }

                                2 => {
                                    // i = 0
                                    q[0].write_real(dst.get_unchecked_mut(0..));
                                    q[1].write_real(dst.get_unchecked_mut(output_stride..));
                                    q[2].write_real(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write_real(dst.get_unchecked_mut(output_stride * 3..));

                                    // i = 1
                                    q[4].write_real(dst.get_unchecked_mut(4..));
                                    q[5].write_real(dst.get_unchecked_mut(4 + output_stride..));
                                    q[6].write_real(dst.get_unchecked_mut(4 + output_stride * 2..));
                                    q[7].write_real(dst.get_unchecked_mut(4 + output_stride * 3..));
                                }

                                3 => {
                                    // i = 0
                                    q[0].write_real(dst.get_unchecked_mut(0..));
                                    q[1].write_real(dst.get_unchecked_mut(output_stride..));
                                    q[2].write_real(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write_real(dst.get_unchecked_mut(output_stride * 3..));

                                    // i = 1
                                    q[4].write_real(dst.get_unchecked_mut(4..));
                                    q[5].write_real(dst.get_unchecked_mut(4 + output_stride..));
                                    q[6].write_real(dst.get_unchecked_mut(4 + output_stride * 2..));
                                    q[7].write_real(dst.get_unchecked_mut(4 + output_stride * 3..));

                                    // i = 2
                                    q[8].write_real(dst.get_unchecked_mut(8..));
                                    q[9].write_real(dst.get_unchecked_mut(8 + output_stride..));
                                    q[10]
                                        .write_real(dst.get_unchecked_mut(8 + output_stride * 2..));
                                    q[11]
                                        .write_real(dst.get_unchecked_mut(8 + output_stride * 3..));
                                }

                                4 => {
                                    // i = 0
                                    q[0].write_real(dst.get_unchecked_mut(0..));
                                    q[1].write_real(dst.get_unchecked_mut(output_stride..));
                                    q[2].write_real(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write_real(dst.get_unchecked_mut(output_stride * 3..));

                                    // i = 1
                                    q[4].write_real(dst.get_unchecked_mut(4..));
                                    q[5].write_real(dst.get_unchecked_mut(4 + output_stride..));
                                    q[6].write_real(dst.get_unchecked_mut(4 + output_stride * 2..));
                                    q[7].write_real(dst.get_unchecked_mut(4 + output_stride * 3..));

                                    // i = 2
                                    q[8].write_real(dst.get_unchecked_mut(8..));
                                    q[9].write_real(dst.get_unchecked_mut(8 + output_stride..));
                                    q[10]
                                        .write_real(dst.get_unchecked_mut(8 + output_stride * 2..));
                                    q[11]
                                        .write_real(dst.get_unchecked_mut(8 + output_stride * 3..));

                                    // i = 3
                                    q[12].write_real(dst.get_unchecked_mut(12..));
                                    q[13].write_real(dst.get_unchecked_mut(12 + output_stride..));
                                    q[14].write_real(
                                        dst.get_unchecked_mut(12 + output_stride * 2..),
                                    );
                                    q[15].write_real(
                                        dst.get_unchecked_mut(12 + output_stride * 3..),
                                    );
                                }
                                _ => {
                                    unreachable!()
                                }
                            }

                            x += X_BLOCK_SIZE;
                        }
                    } else if REM == 1 {
                        while x + X_BLOCK_SIZE <= width {
                            let output_x = x;

                            let src = src.get_unchecked(x..);
                            let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                            for i in 0..Y_BLOCK_SIZE {
                                let q0 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                                let q1 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride + 2..),
                                );
                                let v = q0.unpack_evens(q1);
                                store[i] = v;
                            }

                            let q = $rot_name(store);

                            for i in 0..QUO {
                                q[i * 4].write_real(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                                q[i * 4 + 3]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                            }

                            q[QUO * 4].write_real_lo1(dst.get_unchecked_mut(QUO * 4..));
                            q[QUO * 4 + 1]
                                .write_real_lo1(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            q[QUO * 4 + 2].write_real_lo1(
                                dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                            );
                            q[QUO * 4 + 3].write_real_lo1(
                                dst.get_unchecked_mut(QUO * 4 + output_stride * 3..),
                            );

                            x += X_BLOCK_SIZE;
                        }
                    } else if REM == 2 {
                        while x + X_BLOCK_SIZE <= width {
                            let output_x = x;

                            let src = src.get_unchecked(x..);
                            let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                            for i in 0..Y_BLOCK_SIZE {
                                let q0 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                                let q1 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride + 2..),
                                );
                                let v = q0.unpack_evens(q1);
                                store[i] = v;
                            }

                            let q = $rot_name(store);

                            for i in 0..QUO {
                                q[i * 4].write_real(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                                q[i * 4 + 3]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                            }

                            q[QUO * 4].write_real_lo2(dst.get_unchecked_mut(QUO * 4..));
                            q[QUO * 4 + 1]
                                .write_real_lo2(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            q[QUO * 4 + 2].write_real_lo2(
                                dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                            );
                            q[QUO * 4 + 3].write_real_lo2(
                                dst.get_unchecked_mut(QUO * 4 + output_stride * 3..),
                            );

                            x += X_BLOCK_SIZE;
                        }
                    } else if REM == 3 {
                        while x + X_BLOCK_SIZE <= width {
                            let output_x = x;

                            let src = src.get_unchecked(x..);
                            let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                            for i in 0..Y_BLOCK_SIZE {
                                let q0 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                                let q1 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride + 2..),
                                );
                                let v = q0.unpack_evens(q1);
                                store[i] = v;
                            }

                            let q = $rot_name(store);

                            for i in 0..QUO {
                                q[i * 4].write_real(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                                q[i * 4 + 3]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                            }

                            q[QUO * 4].write_real_lo3(dst.get_unchecked_mut(QUO * 4..));
                            q[QUO * 4 + 1]
                                .write_real_lo3(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            q[QUO * 4 + 2].write_real_lo3(
                                dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                            );
                            q[QUO * 4 + 3].write_real_lo3(
                                dst.get_unchecked_mut(QUO * 4 + output_stride * 3..),
                            );

                            x += X_BLOCK_SIZE;
                        }
                    }

                    if x < width {
                        let rem_x = width - x;
                        let output_x = x;

                        let src = src.get_unchecked(x..);
                        let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                        if rem_x == 1 {
                            for i in 0..Y_BLOCK_SIZE {
                                store[i] =
                                    NeonStoreF::from_complex(src.get_unchecked(i * input_stride));
                            }
                        } else if rem_x == 2 {
                            for i in 0..Y_BLOCK_SIZE {
                                store[i] = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                )
                                .dup_even_odds()[0];
                            }
                        } else if rem_x == 3 {
                            for i in 0..Y_BLOCK_SIZE {
                                let q0 = NeonStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                                let q1 = NeonStoreF::from_complex(
                                    src.get_unchecked(i * input_stride + 2),
                                );
                                let v = q0.unpack_evens(q1);
                                store[i] = v;
                            }
                        }

                        let q = $rot_name(store);

                        if rem_x == 1 {
                            for i in 0..QUO {
                                q[i * 4].write_real(dst.get_unchecked_mut(i * 4..));
                            }

                            if REM == 1 {
                                q[QUO * 4].write_real_lo1(dst.get_unchecked_mut(QUO * 4..));
                            } else if REM == 2 {
                                q[QUO * 4].write_real_lo2(dst.get_unchecked_mut(QUO * 4..));
                            } else if REM == 3 {
                                q[QUO * 4].write_real_lo3(dst.get_unchecked_mut(QUO * 4..));
                            }
                        } else if rem_x == 2 {
                            for i in 0..QUO {
                                q[i * 4].write_real(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride..));
                            }

                            if REM == 1 {
                                q[QUO * 4].write_real_lo1(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1].write_real_lo1(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride..),
                                );
                            } else if REM == 2 {
                                q[QUO * 4].write_real_lo2(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1].write_real_lo2(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride..),
                                );
                            } else if REM == 3 {
                                q[QUO * 4].write_real_lo3(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1].write_real_lo3(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride..),
                                );
                            }
                        } else if rem_x == 3 {
                            for i in 0..QUO {
                                q[i * 4].write_real(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write_real(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                            }

                            if REM == 1 {
                                q[QUO * 4].write_real_lo1(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1].write_real_lo1(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride..),
                                );
                                q[QUO * 4 + 2].write_real_lo1(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                                );
                            } else if REM == 2 {
                                q[QUO * 4].write_real_lo2(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1].write_real_lo2(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride..),
                                );
                                q[QUO * 4 + 2].write_real_lo2(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                                );
                            } else if REM == 3 {
                                q[QUO * 4].write_real_lo3(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1].write_real_lo3(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride..),
                                );
                                q[QUO * 4 + 2].write_real_lo3(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                                );
                            }
                        }
                    }

                    y += Y_BLOCK_SIZE;
                }
            }

            y
        }

        #[derive(Default)]
        pub(crate) struct $rule_name {}

        impl TransposeExecutorRealInv<f32> for $rule_name {
            fn transpose(
                &self,
                input: &[Complex<f32>],
                output: &mut [f32],
                width: usize,
                height: usize,
            ) {
                $method_name(input, width, output, height, width, height, 0);
            }
        }
    };
}

define_transpose!(NeonTranspose7x7F32, f32, block_transpose_f32x2_7x7, 7, 7);
define_transpose!(NeonTranspose7x6F32, f32, block_transpose_f32x2_7x6, 7, 6);
define_transpose!(NeonTranspose7x5F32, f32, block_transpose_f32x2_7x5, 7, 5);
define_transpose!(NeonTranspose6x4F32, f32, neon_transpose_f32x2_6x4, 6, 4);
define_transpose!(NeonTranspose6x5F32, f32, block_transpose_f32x2_6x5, 6, 5);
define_transpose_evenf!(NeonTransposeNx20F32, f32, transpose_2x20, 2, 20);
define_transpose_oddf!(NeonTransposeNx19F32, f32, transpose_2x19, 2, 19);
define_transpose_evenf!(NeonTransposeNx18F32, f32, transpose_2x18, 2, 18);
define_transpose_oddf!(NeonTransposeNx17F32, f32, transpose_2x17, 2, 17);
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
define_transpose_evenf!(NeonTransposeNx4F32, f32, transpose_2x4, 2, 4);
define_transpose_oddf!(NeonTransposeNx3F32, f32, transpose_2x3, 2, 3);
define_transpose_evenf!(NeonTransposeNx2F32, f32, transpose_2x2, 2, 2);
define_transpose!(NeonTranspose11x2F32, f32, block_transpose_f32x2_11x2, 11, 2);
define_transpose!(NeonTranspose4x4F32, f32, neon_transpose_f32x2_4x4, 4, 4);
define_transpose!(NeonTranspose2x2F32, f32, block_transpose_f32x2_2x2, 2, 2);
define_transpose!(NeonTranspose2x2F64, f64, neon_transpose_f64x2_2x2, 2, 2);
define_transpose!(NeonTranspose4x4F64, f64, block_transpose_f64x2_4x4, 4, 4);
define_transpose!(NeonTranspose4x3F64, f64, block_transpose_f64x2_4x3, 4, 3);
define_transpose!(NeonTranspose5x7F32, f32, block_transpose_f32x2_5x7, 5, 7);
define_transpose!(NeonTranspose9x2F32, f32, block_transpose_f32x2_9x2, 9, 2);
define_transposed!(NeonTransposeNx2F64, f64, 1, 2);
define_transposed!(NeonTransposeNx3F64, f64, 1, 3);
define_transposed!(NeonTransposeNx4F64, f64, 1, 4);
define_transposed!(NeonTransposeNx5F64, f64, 1, 5);
define_transposed!(NeonTransposeNx6F64, f64, 1, 6);
define_transposed!(NeonTransposeNx7F64, f64, 1, 7);
define_transposed!(NeonTransposeNx8F64, f64, 1, 8);
define_transposed!(NeonTransposeNx9F64, f64, 1, 9);
define_transposed!(NeonTransposeNx10F64, f64, 1, 10);
define_transposed!(NeonTransposeNx11F64, f64, 1, 11);
define_transposed!(NeonTransposeNx12F64, f64, 1, 12);
define_transposed!(NeonTransposeNx13F64, f64, 1, 13);
define_transposed!(NeonTransposeNx14F64, f64, 1, 14);
define_transposed!(NeonTransposeNx15F64, f64, 1, 15);
define_transposed!(NeonTransposeNx16F64, f64, 1, 16);
define_transposed!(NeonTransposeNx17F64, f64, 1, 17);
define_transposed!(NeonTransposeNx18F64, f64, 1, 18);
define_transposed!(NeonTransposeNx19F64, f64, 1, 19);

define_transpose_inv_d!(NeonTransposeRealInvNx3F64, transpose_f64x2_2x3, f64, 2, 3);
define_transpose_inv_d!(NeonTransposeRealInvNx5F64, transpose_f64x2_2x5, f64, 2, 5);
define_transpose_inv_d!(NeonTransposeRealInvNx7F64, transpose_f64x2_2x7, f64, 2, 7);
define_transpose_inv_d!(NeonTransposeRealInvNx9F64, transpose_f64x2_2x9, f64, 2, 9);
define_transpose_inv_d!(
    NeonTransposeRealInvNx11F64,
    transpose_f64x2_2x11,
    f64,
    2,
    11
);

define_transpose_to_real_oddf!(
    NeonTransposeRealInvNx3F32,
    neon_f32x4_trns_4x3,
    transpose_f32x4_4x3,
    4,
    3,
    4
);
define_transpose_to_real_oddf!(
    NeonTransposeRealInvNx5F32,
    neon_f32x4_trns_4x5,
    transpose_f32x4_4x5,
    4,
    5,
    8
);
define_transpose_to_real_oddf!(
    NeonTransposeRealInvNx7F32,
    neon_f32x4_trns_4x7,
    transpose_f32x4_4x7,
    4,
    7,
    8
);
define_transpose_to_real_oddf!(
    NeonTransposeRealInvNx9F32,
    neon_f32x4_trns_4x9,
    transpose_f32x4_4x9,
    4,
    9,
    12
);
define_transpose_to_real_oddf!(
    NeonTransposeRealInvNx11F32,
    neon_f32x4_trns_4x11,
    transpose_f32x4_4x11,
    4,
    11,
    12
);
