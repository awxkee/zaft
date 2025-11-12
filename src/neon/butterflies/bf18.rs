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
use crate::neon::butterflies::NeonButterfly;
use crate::neon::butterflies::fast_bf9d::NeonFastButterfly9d;
use crate::neon::butterflies::fast_bf9f::NeonFastButterfly9f;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;

macro_rules! butterfly18d {
    ($name: ident, $bf_name: ident) => {
        pub(crate) struct $name {
            direction: FftDirection,
            bf9: $bf_name,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf9: $bf_name::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f64> for $name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if in_place.len() % 18 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(18) {
                        let u0 = vld1q_f64(chunk.as_ptr().cast());
                        let u3 = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                        let u4 = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                        let u7 = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());

                        let u8 = vld1q_f64(chunk.get_unchecked(4..).as_ptr().cast());
                        let u11 = vld1q_f64(chunk.get_unchecked(5..).as_ptr().cast());
                        let u12 = vld1q_f64(chunk.get_unchecked(6..).as_ptr().cast());
                        let u15 = vld1q_f64(chunk.get_unchecked(7..).as_ptr().cast());

                        let u16 = vld1q_f64(chunk.get_unchecked(8..).as_ptr().cast());
                        let u1 = vld1q_f64(chunk.get_unchecked(9..).as_ptr().cast());
                        let u2 = vld1q_f64(chunk.get_unchecked(10..).as_ptr().cast());
                        let u5 = vld1q_f64(chunk.get_unchecked(11..).as_ptr().cast());

                        let u6 = vld1q_f64(chunk.get_unchecked(12..).as_ptr().cast());
                        let u9 = vld1q_f64(chunk.get_unchecked(13..).as_ptr().cast());
                        let u10 = vld1q_f64(chunk.get_unchecked(14..).as_ptr().cast());
                        let u13 = vld1q_f64(chunk.get_unchecked(15..).as_ptr().cast());

                        let u14 = vld1q_f64(chunk.get_unchecked(16..).as_ptr().cast());
                        let u17 = vld1q_f64(chunk.get_unchecked(17..).as_ptr().cast());

                        let (t0, t1) = NeonButterfly::butterfly2_f64(u0, u1);
                        let (t2, t3) = NeonButterfly::butterfly2_f64(u2, u3);
                        let (t4, t5) = NeonButterfly::butterfly2_f64(u4, u5);
                        let (t6, t7) = NeonButterfly::butterfly2_f64(u6, u7);
                        let (t8, t9) = NeonButterfly::butterfly2_f64(u8, u9);
                        let (t10, t11) = NeonButterfly::butterfly2_f64(u10, u11);
                        let (t12, t13) = NeonButterfly::butterfly2_f64(u12, u13);
                        let (t14, t15) = NeonButterfly::butterfly2_f64(u14, u15);
                        let (t16, t17) = NeonButterfly::butterfly2_f64(u16, u17);

                        let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                            self.bf9.exec(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                        let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                            self.bf9.exec(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                        vst1q_f64(chunk.as_mut_ptr().cast(), u0);
                        vst1q_f64(chunk.get_unchecked_mut(1..).as_mut_ptr().cast(), u1);

                        vst1q_f64(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), u2);
                        vst1q_f64(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), u3);

                        vst1q_f64(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), u4);
                        vst1q_f64(chunk.get_unchecked_mut(5..).as_mut_ptr().cast(), u5);

                        vst1q_f64(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), u6);
                        vst1q_f64(chunk.get_unchecked_mut(7..).as_mut_ptr().cast(), u7);

                        vst1q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), u8);
                        vst1q_f64(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), u9);

                        vst1q_f64(chunk.get_unchecked_mut(10..).as_mut_ptr().cast(), u10);
                        vst1q_f64(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), u11);

                        vst1q_f64(chunk.get_unchecked_mut(12..).as_mut_ptr().cast(), u12);
                        vst1q_f64(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), u13);

                        vst1q_f64(chunk.get_unchecked_mut(14..).as_mut_ptr().cast(), u14);
                        vst1q_f64(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), u15);

                        vst1q_f64(chunk.get_unchecked_mut(16..).as_mut_ptr().cast(), u16);
                        vst1q_f64(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), u17);
                    }
                }
                Ok(())
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                18
            }
        }
    };
}

butterfly18d!(NeonButterfly18d, NeonFastButterfly9d);
#[cfg(feature = "fcma")]
use crate::neon::butterflies::fast_bf9d::NeonFcmaFastButterfly9d;
#[cfg(feature = "fcma")]
butterfly18d!(NeonFcmaButterfly18d, NeonFcmaFastButterfly9d);

macro_rules! shift_load {
    ($chunk: expr, $offset0: expr) => {{
        let q0 = vld1q_f32($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = vld1q_f32($chunk.get_unchecked($offset0 + 18..).as_ptr().cast());
        (
            vcombine_f32(vget_low_f32(q0), vget_low_f32(q1)),
            vcombine_f32(vget_high_f32(q0), vget_high_f32(q1)),
        )
    }};
}

macro_rules! shift_store {
    ($chunk: expr, $offset0: expr, $r0: expr, $r1: expr) => {{
        vst1q_f32(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            vcombine_f32(vget_low_f32($r0), vget_low_f32($r1)),
        );
        vst1q_f32(
            $chunk
                .get_unchecked_mut($offset0 + 18..)
                .as_mut_ptr()
                .cast(),
            vcombine_f32(vget_high_f32($r0), vget_high_f32($r1)),
        );
    }};
}

macro_rules! butterfly18f {
    ($name: ident, $bf_name: ident) => {
        pub(crate) struct $name {
            direction: FftDirection,
            bf9: $bf_name,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf9: $bf_name::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f32> for $name {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if in_place.len() % 18 != 0 {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    for chunk in in_place.chunks_exact_mut(36) {
                        let (u0, u3) = shift_load!(chunk, 0);
                        let (u4, u7) = shift_load!(chunk, 2);
                        let (u8, u11) = shift_load!(chunk, 4);
                        let (u12, u15) = shift_load!(chunk, 6);
                        let (u16, u1) = shift_load!(chunk, 8);
                        let (u2, u5) = shift_load!(chunk, 10);
                        let (u6, u9) = shift_load!(chunk, 12);
                        let (u10, u13) = shift_load!(chunk, 14);
                        let (u14, u17) = shift_load!(chunk, 16);

                        let (t0, t1) = NeonButterfly::butterfly2_f32(u0, u1);
                        let (t2, t3) = NeonButterfly::butterfly2_f32(u2, u3);
                        let (t4, t5) = NeonButterfly::butterfly2_f32(u4, u5);
                        let (t6, t7) = NeonButterfly::butterfly2_f32(u6, u7);
                        let (t8, t9) = NeonButterfly::butterfly2_f32(u8, u9);
                        let (t10, t11) = NeonButterfly::butterfly2_f32(u10, u11);
                        let (t12, t13) = NeonButterfly::butterfly2_f32(u12, u13);
                        let (t14, t15) = NeonButterfly::butterfly2_f32(u14, u15);
                        let (t16, t17) = NeonButterfly::butterfly2_f32(u16, u17);

                        let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                            self.bf9.exec(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                        let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                            self.bf9.exec(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                        shift_store!(chunk, 0, u0, u1);
                        shift_store!(chunk, 2, u2, u3);
                        shift_store!(chunk, 4, u4, u5);
                        shift_store!(chunk, 6, u6, u7);
                        shift_store!(chunk, 8, u8, u9);
                        shift_store!(chunk, 10, u10, u11);
                        shift_store!(chunk, 12, u12, u13);
                        shift_store!(chunk, 14, u14, u15);
                        shift_store!(chunk, 16, u16, u17);
                    }

                    let rem = in_place.chunks_exact_mut(36).into_remainder();

                    for chunk in rem.chunks_exact_mut(18) {
                        let u0u3 = vld1q_f32(chunk.as_ptr().cast());
                        let u4u7 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());

                        let u8u11 = vld1q_f32(chunk.get_unchecked(4..).as_ptr().cast());
                        let u12u15 = vld1q_f32(chunk.get_unchecked(6..).as_ptr().cast());

                        let u16u1 = vld1q_f32(chunk.get_unchecked(8..).as_ptr().cast());
                        let u2u5 = vld1q_f32(chunk.get_unchecked(10..).as_ptr().cast());

                        let u6u9 = vld1q_f32(chunk.get_unchecked(12..).as_ptr().cast());
                        let u10u13 = vld1q_f32(chunk.get_unchecked(14..).as_ptr().cast());

                        let u14u17 = vld1q_f32(chunk.get_unchecked(16..).as_ptr().cast());

                        let (t0, t1) = NeonButterfly::butterfly2h_f32(
                            vget_low_f32(u0u3),
                            vget_high_f32(u16u1),
                        );
                        let (t2, t3) =
                            NeonButterfly::butterfly2h_f32(vget_low_f32(u2u5), vget_high_f32(u0u3));
                        let (t4, t5) =
                            NeonButterfly::butterfly2h_f32(vget_low_f32(u4u7), vget_high_f32(u2u5));
                        let (t6, t7) =
                            NeonButterfly::butterfly2h_f32(vget_low_f32(u6u9), vget_high_f32(u4u7));
                        let (t8, t9) = NeonButterfly::butterfly2h_f32(
                            vget_low_f32(u8u11),
                            vget_high_f32(u6u9),
                        );
                        let (t10, t11) = NeonButterfly::butterfly2h_f32(
                            vget_low_f32(u10u13),
                            vget_high_f32(u8u11),
                        );
                        let (t12, t13) = NeonButterfly::butterfly2h_f32(
                            vget_low_f32(u12u15),
                            vget_high_f32(u10u13),
                        );
                        let (t14, t15) = NeonButterfly::butterfly2h_f32(
                            vget_low_f32(u14u17),
                            vget_high_f32(u12u15),
                        );
                        let (t16, t17) = NeonButterfly::butterfly2h_f32(
                            vget_low_f32(u16u1),
                            vget_high_f32(u14u17),
                        );

                        let (u0, u2, u4, u6, u8, u10, u12, u14, u16) =
                            self.bf9.exech(t0, t2, t4, t6, t8, t10, t12, t14, t16);
                        let (u9, u11, u13, u15, u17, u1, u3, u5, u7) =
                            self.bf9.exech(t1, t3, t5, t7, t9, t11, t13, t15, t17);

                        vst1q_f32(chunk.as_mut_ptr().cast(), vcombine_f32(u0, u1));
                        vst1q_f32(
                            chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                            vcombine_f32(u2, u3),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                            vcombine_f32(u4, u5),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                            vcombine_f32(u6, u7),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                            vcombine_f32(u8, u9),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                            vcombine_f32(u10, u11),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                            vcombine_f32(u12, u13),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                            vcombine_f32(u14, u15),
                        );
                        vst1q_f32(
                            chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                            vcombine_f32(u16, u17),
                        );
                    }
                }
                Ok(())
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                18
            }
        }
    };
}

butterfly18f!(NeonButterfly18f, NeonFastButterfly9f);
#[cfg(feature = "fcma")]
use crate::neon::butterflies::fast_bf9f::NeonFcmaFastButterfly9f;
#[cfg(feature = "fcma")]
butterfly18f!(NeonFcmaButterfly18f, NeonFcmaFastButterfly9f);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::test_fcma_butterfly;

    test_butterfly!(test_neon_butterfly18, f32, NeonButterfly18f, 18, 1e-5);
    test_butterfly!(test_neon_butterfly18_f64, f64, NeonButterfly18d, 18, 1e-7);

    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(test_fcma_butterfly18, f32, NeonFcmaButterfly18f, 18, 1e-5);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly18_f64,
        f64,
        NeonFcmaButterfly18d,
        18,
        1e-7
    );
}
