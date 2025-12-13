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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::aarch64::*;
use std::sync::Arc;

macro_rules! gen_bf13d {
    ($name: ident, $features: literal, $internal_bf: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf13: $internal_bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf13: $internal_bf::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f64> for $name {
            fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                13
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(13) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows = [NeonStoreD::default(); 13];
                    for chunk in in_place.chunks_exact_mut(13) {
                        for i in 0..13 {
                            rows[i] = NeonStoreD::from_complex_ref(chunk.get_unchecked(i..));
                        }
                        rows = self.bf13.exec(rows);
                        for i in 0..13 {
                            rows[i].write(chunk.get_unchecked_mut(i..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl FftExecutorOutOfPlace<f64> for $name {
            fn execute_out_of_place(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_out_of_place_impl(src, dst) }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(13) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(13) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows = [NeonStoreD::default(); 13];

                    for (dst, src) in dst.chunks_exact_mut(13).zip(src.chunks_exact(13)) {
                        for i in 0..13 {
                            rows[i] = NeonStoreD::from_complex_ref(src.get_unchecked(i..));
                        }
                        rows = self.bf13.exec(rows);
                        for i in 0..13 {
                            rows[i].write(dst.get_unchecked_mut(i..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl CompositeFftExecutor<f64> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f64> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf13d!(NeonButterfly13d, "neon", ColumnButterfly13d);
#[cfg(feature = "fcma")]
gen_bf13d!(NeonFcmaButterfly13d, "fcma", ColumnFcmaButterfly13d);

macro_rules! gen_bf13f {
    ($name: ident, $features: literal, $internal_bf: ident) => {
        use crate::neon::mixed::$internal_bf;
        pub(crate) struct $name {
            direction: FftDirection,
            bf13: $internal_bf,
        }

        impl $name {
            pub(crate) fn new(fft_direction: FftDirection) -> Self {
                Self {
                    direction: fft_direction,
                    bf13: $internal_bf::new(fft_direction),
                }
            }
        }

        impl FftExecutor<f32> for $name {
            fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
            fn length(&self) -> usize {
                13
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of(13) {
                    return Err(ZaftError::InvalidSizeMultiplier(
                        in_place.len(),
                        self.length(),
                    ));
                }

                unsafe {
                    let mut rows = [NeonStoreF::default(); 13];
                    for chunk in in_place.chunks_exact_mut(26) {
                        for i in 0..6 {
                            let q0 = vld1q_f32(chunk.get_unchecked(i * 2..).as_ptr().cast());
                            let q1 = vld1q_f32(chunk.get_unchecked(i * 2 + 13..).as_ptr().cast());
                            rows[i * 2] =
                                NeonStoreF::raw(vcombine_f32(vget_low_f32(q0), vget_low_f32(q1)));
                            rows[i * 2 + 1] =
                                NeonStoreF::raw(vcombine_f32(vget_high_f32(q0), vget_high_f32(q1)));
                        }

                        let q0 = vld1_f32(chunk.get_unchecked(12..).as_ptr().cast());
                        let q1 = vld1_f32(chunk.get_unchecked(12 + 13..).as_ptr().cast());
                        rows[12] = NeonStoreF::raw(vcombine_f32(q0, q1));

                        rows = self.bf13.exec(rows);
                        for i in 0..6 {
                            let r0 = rows[i * 2];
                            let r1 = rows[i * 2 + 1];
                            let new_row0 = NeonStoreF::raw(vcombine_f32(
                                vget_low_f32(r0.v),
                                vget_low_f32(r1.v),
                            ));
                            let new_row1 = NeonStoreF::raw(vcombine_f32(
                                vget_high_f32(r0.v),
                                vget_high_f32(r1.v),
                            ));
                            new_row0.write(chunk.get_unchecked_mut(i * 2..));
                            new_row1.write(chunk.get_unchecked_mut(i * 2 + 13..));
                        }

                        let r0 = rows[12];
                        r0.write_lo(chunk.get_unchecked_mut(12..));
                        r0.write_hi(chunk.get_unchecked_mut(12 + 13..));
                    }
                    let rem = in_place.chunks_exact_mut(26).into_remainder();
                    for chunk in rem.chunks_exact_mut(13) {
                        for i in 0..13 {
                            rows[i] = NeonStoreF::from_complex(chunk.get_unchecked(i));
                        }
                        rows = self.bf13.exec(rows);
                        for i in 0..13 {
                            rows[i].write_lo(chunk.get_unchecked_mut(i..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl FftExecutorOutOfPlace<f32> for $name {
            fn execute_out_of_place(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_out_of_place_impl(src, dst) }
            }
        }

        impl $name {
            #[target_feature(enable = $features)]
            fn execute_out_of_place_impl(
                &self,
                src: &[Complex<f32>],
                dst: &mut [Complex<f32>],
            ) -> Result<(), ZaftError> {
                if !src.len().is_multiple_of(13) {
                    return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
                }
                if !dst.len().is_multiple_of(13) {
                    return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
                }

                unsafe {
                    let mut rows = [NeonStoreF::default(); 13];

                    for (dst, src) in dst.chunks_exact_mut(26).zip(src.chunks_exact(26)) {
                        for i in 0..6 {
                            let q0 = vld1q_f32(src.get_unchecked(i * 2..).as_ptr().cast());
                            let q1 = vld1q_f32(src.get_unchecked(i * 2 + 13..).as_ptr().cast());
                            rows[i * 2] =
                                NeonStoreF::raw(vcombine_f32(vget_low_f32(q0), vget_low_f32(q1)));
                            rows[i * 2 + 1] =
                                NeonStoreF::raw(vcombine_f32(vget_high_f32(q0), vget_high_f32(q1)));
                        }

                        let q0 = vld1_f32(src.get_unchecked(12..).as_ptr().cast());
                        let q1 = vld1_f32(src.get_unchecked(12 + 13..).as_ptr().cast());
                        rows[12] = NeonStoreF::raw(vcombine_f32(q0, q1));

                        rows = self.bf13.exec(rows);
                        for i in 0..6 {
                            let r0 = rows[i * 2];
                            let r1 = rows[i * 2 + 1];
                            let new_row0 = NeonStoreF::raw(vcombine_f32(
                                vget_low_f32(r0.v),
                                vget_low_f32(r1.v),
                            ));
                            let new_row1 = NeonStoreF::raw(vcombine_f32(
                                vget_high_f32(r0.v),
                                vget_high_f32(r1.v),
                            ));
                            new_row0.write(dst.get_unchecked_mut(i * 2..));
                            new_row1.write(dst.get_unchecked_mut(i * 2 + 13..));
                        }

                        let r0 = rows[12];
                        r0.write_lo(dst.get_unchecked_mut(12..));
                        r0.write_hi(dst.get_unchecked_mut(12 + 13..));
                    }

                    let rem_dst = dst.chunks_exact_mut(26).into_remainder();
                    let rem_src = src.chunks_exact(26).remainder();

                    for (dst, src) in rem_dst.chunks_exact_mut(13).zip(rem_src.chunks_exact(13)) {
                        for i in 0..13 {
                            rows[i] = NeonStoreF::from_complex(src.get_unchecked(i));
                        }
                        rows = self.bf13.exec(rows);
                        for i in 0..13 {
                            rows[i].write_lo(dst.get_unchecked_mut(i..));
                        }
                    }
                }
                Ok(())
            }
        }

        impl CompositeFftExecutor<f32> for $name {
            fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<f32> + Send + Sync> {
                self
            }
        }
    };
}

gen_bf13f!(NeonButterfly13f, "neon", ColumnButterfly13f);
#[cfg(feature = "fcma")]
gen_bf13f!(NeonFcmaButterfly13f, "fcma", ColumnFcmaButterfly13f);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    #[cfg(feature = "fcma")]
    use crate::neon::butterflies::{test_fcma_butterfly, test_oof_fcma_butterfly};

    test_butterfly!(test_neon_butterfly13, f32, NeonButterfly13f, 13, 1e-5);
    #[cfg(feature = "fcma")]
    test_butterfly!(test_fcma_butterfly13, f32, NeonFcmaButterfly13f, 13, 1e-5);
    test_butterfly!(test_neon_butterfly13_f64, f64, NeonButterfly13d, 13, 1e-7);
    #[cfg(feature = "fcma")]
    test_fcma_butterfly!(
        test_fcma_butterfly13_f64,
        f64,
        NeonFcmaButterfly13d,
        13,
        1e-7
    );
    test_oof_butterfly!(test_oof_butterfly13, f32, NeonButterfly13f, 13, 1e-5);
    test_oof_butterfly!(test_oof_butterfly13_f64, f64, NeonButterfly13d, 13, 1e-9);
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly13_f64,
        f64,
        NeonFcmaButterfly13d,
        13,
        1e-9
    );
    #[cfg(feature = "fcma")]
    test_oof_fcma_butterfly!(
        test_oof_fcma_butterfly13,
        f32,
        NeonFcmaButterfly13f,
        13,
        1e-4
    );
}
