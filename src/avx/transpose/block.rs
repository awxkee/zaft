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
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
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
                use crate::avx::transpose::$rot_name;
                unsafe {
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
                    store[i] = AvxStoreD::from_complex(src.get_unchecked(i * input_stride));
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
                    store[i] = AvxStoreD::from_complex(src.get_unchecked(i * input_stride));
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
                use crate::avx::transpose::$rot_name;
                unsafe {
                    transpose_height_block_executor2_f64::<
                        $block_width,
                        $block_height,
                        FunctionEvenD<$block_height>,
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
                use crate::avx::transpose::$rot_name;
                const R: usize = $block_height + 1;
                unsafe {
                    transpose_height_block_executor2_f64_odd::<
                        $block_width,
                        $block_height,
                        R,
                        FunctionOddD<$block_height, R>,
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
        }
    };
}

macro_rules! define_transpose_oddf {
    ($rule_name: ident, $method_name: ident, $complex_type: ident, $rot_name: ident, $block_width: expr, $block_height: expr, $odd_block_height: expr) => {
        #[target_feature(enable = "avx2")]
        #[allow(clippy::out_of_bounds_indexing, clippy::reversed_empty_ranges)]
        pub(crate) fn $method_name(
            input: &[Complex<f32>],
            input_stride: usize,
            output: &mut [Complex<f32>],
            output_stride: usize,
            width: usize,
            height: usize,
            start_y: usize,
        ) -> usize {
            use crate::avx::transpose::$rot_name;
            let mut y = start_y;
            const X_BLOCK_SIZE: usize = $block_width;
            const Y_BLOCK_SIZE: usize = $block_height;
            unsafe {
                let mut store = [AvxStoreF::zero(); Y_BLOCK_SIZE];

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
                                store[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                            }

                            let q = $rot_name(store);

                            match QUO {
                                0 => {
                                    // nothing
                                }

                                1 => {
                                    q[0].write(dst.get_unchecked_mut(0..));
                                    q[1].write(dst.get_unchecked_mut(output_stride..));
                                    q[2].write(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write(dst.get_unchecked_mut(output_stride * 3..));
                                }

                                2 => {
                                    // i = 0
                                    q[0].write(dst.get_unchecked_mut(0..));
                                    q[1].write(dst.get_unchecked_mut(output_stride..));
                                    q[2].write(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write(dst.get_unchecked_mut(output_stride * 3..));

                                    // i = 1
                                    q[4].write(dst.get_unchecked_mut(4..));
                                    q[5].write(dst.get_unchecked_mut(4 + output_stride..));
                                    q[6].write(dst.get_unchecked_mut(4 + output_stride * 2..));
                                    q[7].write(dst.get_unchecked_mut(4 + output_stride * 3..));
                                }

                                3 => {
                                    // i = 0
                                    q[0].write(dst.get_unchecked_mut(0..));
                                    q[1].write(dst.get_unchecked_mut(output_stride..));
                                    q[2].write(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write(dst.get_unchecked_mut(output_stride * 3..));

                                    // i = 1
                                    q[4].write(dst.get_unchecked_mut(4..));
                                    q[5].write(dst.get_unchecked_mut(4 + output_stride..));
                                    q[6].write(dst.get_unchecked_mut(4 + output_stride * 2..));
                                    q[7].write(dst.get_unchecked_mut(4 + output_stride * 3..));

                                    // i = 2
                                    q[8].write(dst.get_unchecked_mut(8..));
                                    q[9].write(dst.get_unchecked_mut(8 + output_stride..));
                                    q[10].write(dst.get_unchecked_mut(8 + output_stride * 2..));
                                    q[11].write(dst.get_unchecked_mut(8 + output_stride * 3..));
                                }

                                4 => {
                                    // i = 0
                                    q[0].write(dst.get_unchecked_mut(0..));
                                    q[1].write(dst.get_unchecked_mut(output_stride..));
                                    q[2].write(dst.get_unchecked_mut(output_stride * 2..));
                                    q[3].write(dst.get_unchecked_mut(output_stride * 3..));

                                    // i = 1
                                    q[4].write(dst.get_unchecked_mut(4..));
                                    q[5].write(dst.get_unchecked_mut(4 + output_stride..));
                                    q[6].write(dst.get_unchecked_mut(4 + output_stride * 2..));
                                    q[7].write(dst.get_unchecked_mut(4 + output_stride * 3..));

                                    // i = 2
                                    q[8].write(dst.get_unchecked_mut(8..));
                                    q[9].write(dst.get_unchecked_mut(8 + output_stride..));
                                    q[10].write(dst.get_unchecked_mut(8 + output_stride * 2..));
                                    q[11].write(dst.get_unchecked_mut(8 + output_stride * 3..));

                                    // i = 3
                                    q[12].write(dst.get_unchecked_mut(12..));
                                    q[13].write(dst.get_unchecked_mut(12 + output_stride..));
                                    q[14].write(dst.get_unchecked_mut(12 + output_stride * 2..));
                                    q[15].write(dst.get_unchecked_mut(12 + output_stride * 3..));
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
                                store[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                            }

                            let q = $rot_name(store);

                            for i in 0..QUO {
                                q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                                q[i * 4 + 3]
                                    .write(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                            }

                            q[QUO * 4].write_lo1(dst.get_unchecked_mut(QUO * 4..));
                            q[QUO * 4 + 1]
                                .write_lo1(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            q[QUO * 4 + 2]
                                .write_lo1(dst.get_unchecked_mut(QUO * 4 + output_stride * 2..));
                            q[QUO * 4 + 3]
                                .write_lo1(dst.get_unchecked_mut(QUO * 4 + output_stride * 3..));

                            x += X_BLOCK_SIZE;
                        }
                    } else if REM == 2 {
                        while x + X_BLOCK_SIZE <= width {
                            let output_x = x;

                            let src = src.get_unchecked(x..);
                            let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                            for i in 0..Y_BLOCK_SIZE {
                                store[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                            }

                            let q = $rot_name(store);

                            for i in 0..QUO {
                                q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                                q[i * 4 + 3]
                                    .write(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                            }

                            q[QUO * 4].write_lo2(dst.get_unchecked_mut(QUO * 4..));
                            q[QUO * 4 + 1]
                                .write_lo2(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            q[QUO * 4 + 2]
                                .write_lo2(dst.get_unchecked_mut(QUO * 4 + output_stride * 2..));
                            q[QUO * 4 + 3]
                                .write_lo2(dst.get_unchecked_mut(QUO * 4 + output_stride * 3..));

                            x += X_BLOCK_SIZE;
                        }
                    } else if REM == 3 {
                        while x + X_BLOCK_SIZE <= width {
                            let output_x = x;

                            let src = src.get_unchecked(x..);
                            let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                            for i in 0..Y_BLOCK_SIZE {
                                store[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                );
                            }

                            let q = $rot_name(store);

                            for i in 0..QUO {
                                q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                                q[i * 4 + 3]
                                    .write(dst.get_unchecked_mut(i * 4 + output_stride * 3..));
                            }

                            q[QUO * 4].write_lo3(dst.get_unchecked_mut(QUO * 4..));
                            q[QUO * 4 + 1]
                                .write_lo3(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            q[QUO * 4 + 2]
                                .write_lo3(dst.get_unchecked_mut(QUO * 4 + output_stride * 2..));
                            q[QUO * 4 + 3]
                                .write_lo3(dst.get_unchecked_mut(QUO * 4 + output_stride * 3..));

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
                                    AvxStoreF::from_complex(src.get_unchecked(i * input_stride));
                            }
                        } else if rem_x == 2 {
                            for i in 0..Y_BLOCK_SIZE {
                                store[i] =
                                    AvxStoreF::from_complex2(src.get_unchecked(i * input_stride..));
                            }
                        } else if rem_x == 3 {
                            for i in 0..Y_BLOCK_SIZE {
                                store[i] =
                                    AvxStoreF::from_complex3(src.get_unchecked(i * input_stride..));
                            }
                        }

                        let q = $rot_name(store);

                        if rem_x == 1 {
                            for i in 0..QUO {
                                q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                            }

                            if REM == 1 {
                                q[QUO * 4].write_lo1(dst.get_unchecked_mut(QUO * 4..));
                            } else if REM == 2 {
                                q[QUO * 4].write_lo2(dst.get_unchecked_mut(QUO * 4..));
                            } else if REM == 3 {
                                q[QUO * 4].write_lo3(dst.get_unchecked_mut(QUO * 4..));
                            }
                        } else if rem_x == 2 {
                            for i in 0..QUO {
                                q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                            }

                            if REM == 1 {
                                q[QUO * 4].write_lo1(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1]
                                    .write_lo1(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            } else if REM == 2 {
                                q[QUO * 4].write_lo2(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1]
                                    .write_lo2(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            } else if REM == 3 {
                                q[QUO * 4].write_lo3(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1]
                                    .write_lo3(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                            }
                        } else if rem_x == 3 {
                            for i in 0..QUO {
                                q[i * 4].write(dst.get_unchecked_mut(i * 4..));
                                q[i * 4 + 1].write(dst.get_unchecked_mut(i * 4 + output_stride..));
                                q[i * 4 + 2]
                                    .write(dst.get_unchecked_mut(i * 4 + output_stride * 2..));
                            }

                            if REM == 1 {
                                q[QUO * 4].write_lo1(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1]
                                    .write_lo1(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                                q[QUO * 4 + 2].write_lo1(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                                );
                            } else if REM == 2 {
                                q[QUO * 4].write_lo2(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1]
                                    .write_lo2(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                                q[QUO * 4 + 2].write_lo2(
                                    dst.get_unchecked_mut(QUO * 4 + output_stride * 2..),
                                );
                            } else if REM == 3 {
                                q[QUO * 4].write_lo3(dst.get_unchecked_mut(QUO * 4..));
                                q[QUO * 4 + 1]
                                    .write_lo3(dst.get_unchecked_mut(QUO * 4 + output_stride..));
                                q[QUO * 4 + 2].write_lo3(
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
                unsafe {
                    $method_name(input, input_stride, output, output_stride, width, height, 0);
                }
            }
        }
    };
}

define_transpose!(AvxTransposeF644x4, f64, avx_transpose_f64x2_4x4, 4, 4);
define_transpose!(AvxTransposeF327x7, f32, block_transpose_f32x2_7x7, 7, 7);
define_transpose!(AvxTransposeF325x5, f32, block_transpose_f32x2_5x5, 5, 5);
define_transpose!(AvxTransposeF327x6, f32, block_transpose_f32x2_7x6, 7, 6);
define_transpose!(AvxTransposeF327x5, f32, block_transpose_f32x2_7x5, 7, 5);
define_transpose!(AvxTransposeF327x3, f32, block_transpose_f32x2_7x3, 7, 3);
define_transpose!(AvxTransposeF327x2, f32, block_transpose_f32x2_7x2, 7, 2);
define_transpose!(AvxTransposeF328x3, f32, block_transpose_f32x2_8x3, 8, 3);
define_transpose!(AvxTransposeF323x8, f32, block_transpose_f32x2_3x8, 3, 8);

define_transpose_oddf!(AvxTransposeNx2F32, trs_nf32x2, f32, transpose_4x2, 4, 2, 4);
define_transpose_oddf!(AvxTransposeNx3F32, trs_nf32x3, f32, transpose_4x3, 4, 3, 4);
define_transpose_oddf!(AvxTransposeNx4F32, trs_nf32x4, f32, transpose_4x4, 4, 4, 4);
define_transpose_oddf!(AvxTransposeNx5F32, trs_nf32x5, f32, transpose_4x5, 4, 5, 8);
define_transpose_oddf!(AvxTransposeNx6F32, trs_nf32x6, f32, transpose_4x6, 4, 6, 8);
define_transpose_oddf!(AvxTransposeNx7F32, trs_nf32x7, f32, transpose_4x7, 4, 7, 8);
define_transpose_oddf!(AvxTransposeNx8F32, trs_nf32x8, f32, transpose_4x8, 4, 8, 8);
define_transpose_oddf!(AvxTransposeNx9F32, trs_nf32x9, f32, transpose_4x9, 4, 9, 12);
define_transpose_oddf!(
    AvxTransposeNx10F32,
    trs_nf32x10,
    f32,
    transpose_4x10,
    4,
    10,
    12
);
define_transpose_oddf!(
    AvxTransposeNx11F32,
    trs_nf32x11,
    f32,
    transpose_4x11,
    4,
    11,
    12
);
define_transpose_oddf!(
    AvxTransposeNx12F32,
    trs_nf32x12,
    f32,
    transpose_4x12,
    4,
    12,
    12
);
define_transpose_oddf!(
    AvxTransposeNx13F32,
    trs_nf32x13,
    f32,
    transpose_4x13,
    4,
    13,
    16
);
define_transpose_oddf!(
    AvxTransposeNx14F32,
    trs_nf32x14,
    f32,
    transpose_4x14,
    4,
    14,
    16
);
define_transpose_oddf!(
    AvxTransposeNx15F32,
    trs_nf32x15,
    f32,
    transpose_4x15,
    4,
    15,
    16
);
define_transpose_oddf!(
    AvxTransposeNx16F32,
    trs_nf32x16,
    f32,
    transpose_4x16,
    4,
    16,
    16
);

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
define_transpose_oddd!(AvxTransposeNx5F64, f64, transpose_f64x2_2x5, 2, 5);
define_transpose_evend!(AvxTransposeNx4F64, f64, transpose_2x4d, 2, 4);
define_transpose_oddd!(AvxTransposeNx3F64, f64, transpose_2x3d, 2, 3);
define_transpose_evend!(AvxTransposeNx2F64, f64, transpose_f64x2_2x2d, 2, 2);
