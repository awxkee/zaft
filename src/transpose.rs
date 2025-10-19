/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
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
use num_complex::Complex;
use std::marker::PhantomData;

pub(crate) trait TransposeExecutor<T> {
    fn transpose(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        width: usize,
        height: usize,
    );
}

pub(crate) trait TransposeFactory<T> {
    fn transpose_strategy(
        width: usize,
        height: usize,
    ) -> Box<dyn TransposeExecutor<T> + Send + Sync>;
}

impl TransposeFactory<f32> for f32 {
    fn transpose_strategy(
        _width: usize,
        _height: usize,
    ) -> Box<dyn TransposeExecutor<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if _width > 2 && _height > 2 {
                return Box::new(NeonDefaultExecutorSingle {});
            }
            Box::new(TransposeTiny {
                phantom_data: Default::default(),
            })
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::arch::is_x86_feature_detected!("avx2") {
                    return Box::new(AvxDefaultExecutorSingle {});
                }
            }
            if _width > 31 && _height > 31 {
                use crate::transpose_arbitrary::TransposeArbitrary;
                return Box::new(TransposeArbitrary {
                    phantom_data: Default::default(),
                });
            }
            Box::new(TransposeTiny {
                phantom_data: Default::default(),
            })
        }
    }
}

impl TransposeFactory<f64> for f64 {
    fn transpose_strategy(
        _width: usize,
        _height: usize,
    ) -> Box<dyn TransposeExecutor<f64> + Send + Sync> {
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx") {
                return Box::new(AvxDefaultExecutorDouble {});
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            if _width > 2 && _height > 2 {
                return Box::new(NeonDefaultExecutorDouble {});
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        if _width > 31 && _height > 31 {
            use crate::transpose_arbitrary::TransposeArbitrary;
            return Box::new(TransposeArbitrary {
                phantom_data: Default::default(),
            });
        }
        Box::new(TransposeTiny {
            phantom_data: Default::default(),
        })
    }
}

struct TransposeTiny<T> {
    phantom_data: PhantomData<T>,
}

impl<T: Copy> TransposeExecutor<T> for TransposeTiny<T> {
    fn transpose(
        &self,
        input: &[Complex<T>],
        output: &mut [Complex<T>],
        width: usize,
        height: usize,
    ) {
        for x in 0..width {
            for y in 0..height {
                let input_index = x + y * width;
                let output_index = y + x * height;

                unsafe {
                    *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
                }
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) trait TransposeBlock<V> {
    unsafe fn transpose_block(
        &self,
        src: &[Complex<V>],
        src_stride: usize,
        dst: &mut [Complex<V>],
        dst_stride: usize,
    );
}

#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn transpose_executor<V: Copy + Default, const BLOCK_SIZE: usize>(
    input: &[Complex<V>],
    input_stride: usize,
    output: &mut [Complex<V>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: impl TransposeBlock<V>,
) -> usize {
    let mut y = start_y;

    let mut src_buffer = vec![Complex::<V>::default(); BLOCK_SIZE * BLOCK_SIZE];
    let mut dst_buffer = vec![Complex::<V>::default(); BLOCK_SIZE * BLOCK_SIZE];

    unsafe {
        while y + BLOCK_SIZE < height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + BLOCK_SIZE < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                exec.transpose_block(src, input_stride, dst, output_stride);

                x += BLOCK_SIZE;
            }

            if x < width {
                let rem_x = width - x;
                assert!(
                    rem_x <= BLOCK_SIZE,
                    "Remainder is expected to be less than {BLOCK_SIZE}, but got {rem_x}"
                );

                let output_x = x;
                let src = src.get_unchecked(x..);

                for j in 0..BLOCK_SIZE {
                    std::ptr::copy_nonoverlapping(
                        src.get_unchecked(j * input_stride..).as_ptr(),
                        src_buffer
                            .get_unchecked_mut(j * (BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        rem_x,
                    );
                }

                exec.transpose_block(
                    src_buffer.as_slice(),
                    BLOCK_SIZE,
                    dst_buffer.as_mut_slice(),
                    BLOCK_SIZE,
                );

                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for j in 0..rem_x {
                    std::ptr::copy_nonoverlapping(
                        dst_buffer
                            .get_unchecked_mut(j * (BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        dst.get_unchecked_mut(j * output_stride..).as_mut_ptr(),
                        BLOCK_SIZE,
                    );
                }
            }

            y += BLOCK_SIZE;
        }
    }

    y
}

#[allow(dead_code)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn transpose_executor2d<
    V: Copy + Default,
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
>(
    input: &[Complex<V>],
    input_stride: usize,
    output: &mut [Complex<V>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: impl TransposeBlock<V>,
) -> usize {
    let mut y = start_y;

    let mut src_buffer = vec![Complex::<V>::default(); X_BLOCK_SIZE * Y_BLOCK_SIZE];
    let mut dst_buffer = vec![Complex::<V>::default(); X_BLOCK_SIZE * Y_BLOCK_SIZE];

    unsafe {
        while y + Y_BLOCK_SIZE < height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                exec.transpose_block(src, input_stride, dst, output_stride);

                x += X_BLOCK_SIZE;
            }

            if x < width {
                let rem_x = width - x;
                assert!(
                    rem_x <= X_BLOCK_SIZE,
                    "Remainder is expected to be less than {X_BLOCK_SIZE}, but got {rem_x}"
                );

                let output_x = x;
                let src = src.get_unchecked(x..);

                for j in 0..Y_BLOCK_SIZE {
                    std::ptr::copy_nonoverlapping(
                        src.get_unchecked(j * input_stride..).as_ptr(),
                        src_buffer
                            .get_unchecked_mut(j * (X_BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        rem_x,
                    );
                }

                exec.transpose_block(
                    src_buffer.as_slice(),
                    X_BLOCK_SIZE,
                    dst_buffer.as_mut_slice(),
                    Y_BLOCK_SIZE,
                );

                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for j in 0..rem_x {
                    std::ptr::copy_nonoverlapping(
                        dst_buffer
                            .get_unchecked_mut(j * (Y_BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        dst.get_unchecked_mut(j * output_stride..).as_mut_ptr(),
                        Y_BLOCK_SIZE,
                    );
                }
            }

            y += Y_BLOCK_SIZE;
        }
    }

    y
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
pub(crate) unsafe fn transpose_executor2d<
    V: Copy + Default,
    const X_BLOCK_SIZE: usize,
    const Y_BLOCK_SIZE: usize,
>(
    input: &[Complex<V>],
    input_stride: usize,
    output: &mut [Complex<V>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: impl TransposeBlock<V>,
) -> usize {
    let mut y = start_y;

    let mut src_buffer = vec![Complex::<V>::default(); X_BLOCK_SIZE * Y_BLOCK_SIZE];
    let mut dst_buffer = vec![Complex::<V>::default(); X_BLOCK_SIZE * Y_BLOCK_SIZE];

    unsafe {
        while y + Y_BLOCK_SIZE < height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + X_BLOCK_SIZE < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                exec.transpose_block(src, input_stride, dst, output_stride);

                x += X_BLOCK_SIZE;
            }

            if x < width {
                let rem_x = width - x;
                assert!(
                    rem_x <= X_BLOCK_SIZE,
                    "Remainder is expected to be less than {X_BLOCK_SIZE}, but got {rem_x}"
                );

                let output_x = x;
                let src = src.get_unchecked(x..);

                for j in 0..Y_BLOCK_SIZE {
                    std::ptr::copy_nonoverlapping(
                        src.get_unchecked(j * input_stride..).as_ptr(),
                        src_buffer
                            .get_unchecked_mut(j * (X_BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        rem_x,
                    );
                }

                exec.transpose_block(
                    src_buffer.as_slice(),
                    X_BLOCK_SIZE,
                    dst_buffer.as_mut_slice(),
                    Y_BLOCK_SIZE,
                );

                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for j in 0..rem_x {
                    std::ptr::copy_nonoverlapping(
                        dst_buffer
                            .get_unchecked_mut(j * (Y_BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        dst.get_unchecked_mut(j * output_stride..).as_mut_ptr(),
                        Y_BLOCK_SIZE,
                    );
                }
            }

            y += Y_BLOCK_SIZE;
        }
    }

    y
}

#[allow(dead_code)]
#[cfg(not(all(target_arch = "x86_64", feature = "avx")))]
pub(crate) unsafe fn transpose_executor<V: Copy + Default, const BLOCK_SIZE: usize>(
    input: &[Complex<V>],
    input_stride: usize,
    output: &mut [Complex<V>],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: impl TransposeBlock<V>,
) -> usize {
    let mut y = start_y;

    let mut src_buffer = [Complex::<V>::default(); 16];
    let mut dst_buffer = [Complex::<V>::default(); 16];

    unsafe {
        while y + BLOCK_SIZE < height {
            let input_y = y;

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + BLOCK_SIZE < width {
                let output_x = x;

                let src = src.get_unchecked(x..);
                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                exec.transpose_block(src, input_stride, dst, output_stride);

                x += BLOCK_SIZE;
            }

            if x < width {
                let rem_x = width - x;
                assert!(
                    rem_x <= BLOCK_SIZE,
                    "Remainder is expected to be less than {BLOCK_SIZE}, but got {rem_x}"
                );

                let output_x = x;
                let src = src.get_unchecked(x..);

                for j in 0..BLOCK_SIZE {
                    std::ptr::copy_nonoverlapping(
                        src.get_unchecked(j * input_stride..).as_ptr(),
                        src_buffer
                            .get_unchecked_mut(j * (BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        rem_x,
                    );
                }

                exec.transpose_block(
                    src_buffer.as_slice(),
                    BLOCK_SIZE,
                    dst_buffer.as_mut_slice(),
                    BLOCK_SIZE,
                );

                let dst = output.get_unchecked_mut(y + output_stride * output_x..);

                for j in 0..rem_x {
                    std::ptr::copy_nonoverlapping(
                        dst_buffer
                            .get_unchecked_mut(j * (BLOCK_SIZE)..)
                            .as_mut_ptr(),
                        dst.get_unchecked_mut(j * output_stride..).as_mut_ptr(),
                        BLOCK_SIZE,
                    );
                }
            }

            y += BLOCK_SIZE;
        }
    }

    y
}

#[allow(dead_code)]
pub(crate) fn transpose_section<V: Copy>(
    input: &[V],
    input_stride: usize,
    output: &mut [V],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
) {
    for y in start_y..height {
        for x in 0..width {
            let input_index = x + y * input_stride;
            let output_index = y + x * output_stride;

            unsafe {
                *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
struct TransposeBlockAvx2x2F32x2 {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl TransposeBlock<f32> for TransposeBlockAvx2x2F32x2 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f32>],
        src_stride: usize,
        dst: &mut [Complex<f32>],
        dst_stride: usize,
    ) {
        use crate::avx::avx_transpose_f32x2_2x2;
        unsafe {
            avx_transpose_f32x2_2x2(src, src_stride, dst, dst_stride);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
struct TransposeBlockAvx4x4F32x4 {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl TransposeBlock<f32> for TransposeBlockAvx4x4F32x4 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f32>],
        src_stride: usize,
        dst: &mut [Complex<f32>],
        dst_stride: usize,
    ) {
        use crate::avx::avx2_transpose_f32x2_4x4;
        unsafe {
            avx2_transpose_f32x2_4x4(src, src_stride, dst, dst_stride);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
struct TransposeBlockAvx8x4F32x2 {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl TransposeBlock<f32> for TransposeBlockAvx8x4F32x2 {
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f32>],
        src_stride: usize,
        dst: &mut [Complex<f32>],
        dst_stride: usize,
    ) {
        use crate::avx::avx2_transpose_f32x2_8x4;
        unsafe {
            avx2_transpose_f32x2_8x4(src, src_stride, dst, dst_stride);
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
struct TransposeBlockNeon6x4F32x2 {}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
impl TransposeBlock<f32> for TransposeBlockNeon6x4F32x2 {
    #[inline(always)]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f32>],
        src_stride: usize,
        dst: &mut [Complex<f32>],
        dst_stride: usize,
    ) {
        use crate::neon::neon_transpose_f32x2_6x4;
        neon_transpose_f32x2_6x4(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
struct TransposeBlockNeon4x4F32x2 {}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
impl TransposeBlock<f32> for TransposeBlockNeon4x4F32x2 {
    #[inline(always)]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f32>],
        src_stride: usize,
        dst: &mut [Complex<f32>],
        dst_stride: usize,
    ) {
        use crate::neon::neon_transpose_f32x2_4x4;
        neon_transpose_f32x2_4x4(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
struct TransposeBlockNeon2x2F32x2 {}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
impl TransposeBlock<f32> for TransposeBlockNeon2x2F32x2 {
    #[inline(always)]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f32>],
        src_stride: usize,
        dst: &mut [Complex<f32>],
        dst_stride: usize,
    ) {
        use crate::neon::neon_transpose_f32x2_2x2;
        neon_transpose_f32x2_2x2(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
struct TransposeBlockNeon2x2F64x2 {}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
impl TransposeBlock<f64> for TransposeBlockNeon2x2F64x2 {
    #[inline(always)]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f64>],
        src_stride: usize,
        dst: &mut [Complex<f64>],
        dst_stride: usize,
    ) {
        use crate::neon::neon_transpose_f64x2_2x2;
        neon_transpose_f64x2_2x2(src, src_stride, dst, dst_stride);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
struct NeonDefaultExecutorSingle {}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
impl TransposeExecutor<f32> for NeonDefaultExecutorSingle {
    fn transpose(
        &self,
        input: &[Complex<f32>],
        output: &mut [Complex<f32>],
        width: usize,
        height: usize,
    ) {
        let mut y = 0usize;

        let input_stride = width;
        let output_stride = height;

        unsafe {
            y = transpose_executor2d::<f32, 6, 4>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockNeon6x4F32x2 {},
            );
        }

        unsafe {
            y = transpose_executor::<f32, 4>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockNeon4x4F32x2 {},
            );
        }

        unsafe {
            y = transpose_executor::<f32, 2>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockNeon2x2F32x2 {},
            );
        }

        transpose_section::<Complex<f32>>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
        );
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
struct NeonDefaultExecutorDouble {}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
impl TransposeExecutor<f64> for NeonDefaultExecutorDouble {
    fn transpose(
        &self,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
        width: usize,
        height: usize,
    ) {
        let mut y = 0usize;

        let input_stride = width;
        let output_stride = height;

        unsafe {
            y = transpose_executor::<f64, 2>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockNeon2x2F64x2 {},
            );
        }

        transpose_section::<Complex<f64>>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
        );
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
struct AvxDefaultExecutorSingle {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl TransposeExecutor<f32> for AvxDefaultExecutorSingle {
    fn transpose(
        &self,
        input: &[Complex<f32>],
        output: &mut [Complex<f32>],
        width: usize,
        height: usize,
    ) {
        let mut y = 0usize;

        let input_stride = width;
        let output_stride = height;

        unsafe {
            y = transpose_executor2d::<f32, 8, 4>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockAvx8x4F32x2 {},
            );
        }

        unsafe {
            y = transpose_executor::<f32, 4>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockAvx4x4F32x4 {},
            );
        }

        unsafe {
            y = transpose_executor::<f32, 2>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockAvx2x2F32x2 {},
            );
        }

        transpose_section::<Complex<f32>>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
        );
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
struct TransposeBlockAvx2x2F64x2 {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl TransposeBlock<f64> for TransposeBlockAvx2x2F64x2 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f64>],
        src_stride: usize,
        dst: &mut [Complex<f64>],
        dst_stride: usize,
    ) {
        use crate::avx::avx_transpose_f64x2_2x2;
        unsafe {
            avx_transpose_f64x2_2x2(src, src_stride, dst, dst_stride);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
struct TransposeBlockAvx4x4F64x2 {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl TransposeBlock<f64> for TransposeBlockAvx4x4F64x2 {
    #[inline]
    #[target_feature(enable = "avx")]
    unsafe fn transpose_block(
        &self,
        src: &[Complex<f64>],
        src_stride: usize,
        dst: &mut [Complex<f64>],
        dst_stride: usize,
    ) {
        use crate::avx::avx_transpose_f64x2_4x4;
        unsafe {
            avx_transpose_f64x2_4x4(src, src_stride, dst, dst_stride);
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
struct AvxDefaultExecutorDouble {}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
impl TransposeExecutor<f64> for AvxDefaultExecutorDouble {
    fn transpose(
        &self,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
        width: usize,
        height: usize,
    ) {
        let mut y = 0usize;

        let input_stride = width;
        let output_stride = height;

        unsafe {
            y = transpose_executor::<f64, 4>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockAvx4x4F64x2 {},
            );
        }

        unsafe {
            y = transpose_executor::<f64, 2>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
                y,
                TransposeBlockAvx2x2F64x2 {},
            );
        }

        transpose_section::<Complex<f64>>(
            input,
            input_stride,
            output,
            output_stride,
            width,
            height,
            y,
        );
    }
}
