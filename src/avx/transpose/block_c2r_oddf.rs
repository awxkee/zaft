/*
 * // Copyright (c) Radzivon Bartoshyk 3/2026. All rights reserved.
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
use crate::avx::mixed::{AvxStoreF, SseStoreF};
use crate::transpose::TransposeExecutorRealInv;
use num_complex::Complex;

macro_rules! define_transpose_oddf {
    ($rule_name: ident, $method_name: ident, $rot_name: ident, $block_width: expr, $block_height: expr, $odd_block_height: expr) => {
        #[target_feature(enable = "avx2")]
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
            use crate::avx::transpose::$rot_name;
            let mut y = start_y;
            const X_BLOCK_SIZE: usize = $block_width;
            const Y_BLOCK_SIZE: usize = $block_height;
            unsafe {
                let mut store = [SseStoreF::zero(); Y_BLOCK_SIZE];

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
                                )
                                .pack_evens();
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
                                store[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                )
                                .pack_evens();
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
                                store[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                )
                                .pack_evens();
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
                                store[i] = AvxStoreF::from_complex_ref(
                                    src.get_unchecked(i * input_stride..),
                                )
                                .pack_evens();
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
                                    AvxStoreF::from_complex(src.get_unchecked(i * input_stride))
                                        .pack_evens();
                            }
                        } else if rem_x == 2 {
                            for i in 0..Y_BLOCK_SIZE {
                                store[i] =
                                    AvxStoreF::from_complex2(src.get_unchecked(i * input_stride..))
                                        .pack_evens();
                            }
                        } else if rem_x == 3 {
                            for i in 0..Y_BLOCK_SIZE {
                                store[i] =
                                    AvxStoreF::from_complex3(src.get_unchecked(i * input_stride..))
                                        .pack_evens();
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
                unsafe {
                    $method_name(input, width, output, height, width, height, 0);
                }
            }
        }
    };
}

define_transpose_oddf!(
    AvxTransposeRealInvNx3F32,
    avx_f32x4_trns_4x3,
    transpose_sse_f32x4_4x3,
    4,
    3,
    4
);
define_transpose_oddf!(
    AvxTransposeRealInvNx5F32,
    avx_f32x4_trns_4x5,
    transpose_sse_f32x4_4x5,
    4,
    5,
    8
);
define_transpose_oddf!(
    AvxTransposeRealInvNx7F32,
    avx_f32x4_trns_4x7,
    transpose_sse_f32x4_4x7,
    4,
    7,
    8
);
define_transpose_oddf!(
    AvxTransposeRealInvNx9F32,
    avx_f32x4_trns_4x9,
    transpose_sse_f32x4_4x9,
    4,
    9,
    12
);
define_transpose_oddf!(
    AvxTransposeRealInvNx11F32,
    avx_f32x4_trns_4x11,
    transpose_sse_f32x4_4x11,
    4,
    11,
    12
);
