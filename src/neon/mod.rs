/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
mod butterflies;
mod c2r;
#[cfg(feature = "fcma")]
mod c2r_fcma;
mod mixed;
mod r2c;
#[cfg(feature = "fcma")]
mod r2c_fcma;
mod raders;
mod radix10;
#[cfg(feature = "fcma")]
mod radix10_fcma;
mod radix11;
#[cfg(feature = "fcma")]
mod radix11_fcma;
mod radix13;
#[cfg(feature = "fcma")]
mod radix13_fcma;
mod radix3;
#[cfg(feature = "fcma")]
mod radix3_fcma;
mod radix4;
#[cfg(feature = "fcma")]
mod radix4_fcma;
mod radix5;
#[cfg(feature = "fcma")]
mod radix5_fcma;
mod radix6;
#[cfg(feature = "fcma")]
mod radix6_fcma;
mod radix7;
#[cfg(feature = "fcma")]
mod radix7_fcma;
mod spectrum_arithmetic;
#[cfg(feature = "fcma")]
mod spectrum_arithmetic_fcma;
mod transpose;
mod util;

pub(crate) use butterflies::{
    NeonButterfly2, NeonButterfly3, NeonButterfly4, NeonButterfly5, NeonButterfly6, NeonButterfly7,
    NeonButterfly8, NeonButterfly9, NeonButterfly10, NeonButterfly11, NeonButterfly12,
    NeonButterfly13, NeonButterfly14, NeonButterfly15, NeonButterfly16, NeonButterfly17,
    NeonButterfly18d, NeonButterfly18f, NeonButterfly19, NeonButterfly20, NeonButterfly23,
    NeonButterfly25d, NeonButterfly25f, NeonButterfly27d, NeonButterfly27f, NeonButterfly29,
    NeonButterfly31d, NeonButterfly31f, NeonButterfly32d, NeonButterfly32f, NeonButterfly35d,
    NeonButterfly35f, NeonButterfly36d, NeonButterfly36f, NeonButterfly42d, NeonButterfly42f,
    NeonButterfly48d, NeonButterfly48f, NeonButterfly49d, NeonButterfly49f, NeonButterfly64d,
    NeonButterfly64f, NeonButterfly81d, NeonButterfly81f, NeonButterfly100d, NeonButterfly100f,
    NeonButterfly121d, NeonButterfly121f, NeonButterfly128d, NeonButterfly128f, NeonButterfly256d,
    NeonButterfly256f,
};
#[cfg(feature = "fcma")]
pub(crate) use butterflies::{
    NeonFcmaButterfly4, NeonFcmaButterfly5, NeonFcmaButterfly6, NeonFcmaButterfly7,
    NeonFcmaButterfly8, NeonFcmaButterfly9, NeonFcmaButterfly10, NeonFcmaButterfly11,
    NeonFcmaButterfly12, NeonFcmaButterfly13, NeonFcmaButterfly14, NeonFcmaButterfly15,
    NeonFcmaButterfly16, NeonFcmaButterfly17, NeonFcmaButterfly18d, NeonFcmaButterfly18f,
    NeonFcmaButterfly19, NeonFcmaButterfly20, NeonFcmaButterfly23, NeonFcmaButterfly25d,
    NeonFcmaButterfly25f, NeonFcmaButterfly27d, NeonFcmaButterfly27f, NeonFcmaButterfly29,
    NeonFcmaButterfly31d, NeonFcmaButterfly31f, NeonFcmaButterfly32d, NeonFcmaButterfly32f,
    NeonFcmaButterfly35d, NeonFcmaButterfly35f, NeonFcmaButterfly36d, NeonFcmaButterfly36f,
    NeonFcmaButterfly42d, NeonFcmaButterfly42f, NeonFcmaButterfly48d, NeonFcmaButterfly48f,
    NeonFcmaButterfly49d, NeonFcmaButterfly49f, NeonFcmaButterfly64d, NeonFcmaButterfly64f,
    NeonFcmaButterfly81d, NeonFcmaButterfly81f, NeonFcmaButterfly100d, NeonFcmaButterfly100f,
    NeonFcmaButterfly121d, NeonFcmaButterfly121f, NeonFcmaButterfly128d, NeonFcmaButterfly128f,
    NeonFcmaButterfly256d, NeonFcmaButterfly256f,
};
pub(crate) use c2r::C2RNeonTwiddles;
#[cfg(feature = "fcma")]
pub(crate) use c2r_fcma::C2RNeonFcmaTwiddles;
#[cfg(feature = "fcma")]
pub(crate) use mixed::{
    NeonFcmaMixedRadix2, NeonFcmaMixedRadix2f, NeonFcmaMixedRadix3, NeonFcmaMixedRadix3f,
    NeonFcmaMixedRadix4, NeonFcmaMixedRadix4f, NeonFcmaMixedRadix5, NeonFcmaMixedRadix5f,
    NeonFcmaMixedRadix6, NeonFcmaMixedRadix6f, NeonFcmaMixedRadix7, NeonFcmaMixedRadix7f,
    NeonFcmaMixedRadix8, NeonFcmaMixedRadix8f, NeonFcmaMixedRadix9, NeonFcmaMixedRadix9f,
    NeonFcmaMixedRadix10, NeonFcmaMixedRadix10f, NeonFcmaMixedRadix11, NeonFcmaMixedRadix11f,
    NeonFcmaMixedRadix12, NeonFcmaMixedRadix12f, NeonFcmaMixedRadix13, NeonFcmaMixedRadix13f,
    NeonFcmaMixedRadix16, NeonFcmaMixedRadix16f,
};
pub(crate) use mixed::{
    NeonMixedRadix2, NeonMixedRadix2f, NeonMixedRadix3, NeonMixedRadix3f, NeonMixedRadix4,
    NeonMixedRadix4f, NeonMixedRadix5, NeonMixedRadix5f, NeonMixedRadix6, NeonMixedRadix6f,
    NeonMixedRadix7, NeonMixedRadix7f, NeonMixedRadix8, NeonMixedRadix8f, NeonMixedRadix9,
    NeonMixedRadix9f, NeonMixedRadix10, NeonMixedRadix10f, NeonMixedRadix11, NeonMixedRadix11f,
    NeonMixedRadix12, NeonMixedRadix12f, NeonMixedRadix13, NeonMixedRadix13f, NeonMixedRadix16,
    NeonMixedRadix16f,
};
pub(crate) use r2c::R2CNeonTwiddles;
#[cfg(feature = "fcma")]
pub(crate) use r2c_fcma::R2CNeonFcmaTwiddles;
pub(crate) use raders::NeonRadersFft;
pub(crate) use radix3::NeonRadix3;
#[cfg(feature = "fcma")]
pub(crate) use radix3_fcma::NeonFcmaRadix3;
pub(crate) use radix4::NeonRadix4;
#[cfg(feature = "fcma")]
pub(crate) use radix4_fcma::NeonFcmaRadix4;
pub(crate) use radix5::NeonRadix5;
#[cfg(feature = "fcma")]
pub(crate) use radix5_fcma::NeonFcmaRadix5;
pub(crate) use radix6::NeonRadix6;
#[cfg(feature = "fcma")]
pub(crate) use radix6_fcma::NeonFcmaRadix6;
pub(crate) use radix7::NeonRadix7;
#[cfg(feature = "fcma")]
pub(crate) use radix7_fcma::NeonFcmaRadix7;
pub(crate) use radix10::NeonRadix10;
#[cfg(feature = "fcma")]
pub(crate) use radix10_fcma::NeonFcmaRadix10;
pub(crate) use radix11::NeonRadix11;
#[cfg(feature = "fcma")]
pub(crate) use radix11_fcma::NeonFcmaRadix11;
pub(crate) use radix13::NeonRadix13;
#[cfg(feature = "fcma")]
pub(crate) use radix13_fcma::NeonFcmaRadix13;
#[allow(unused)]
pub(crate) use spectrum_arithmetic::NeonSpectrumArithmetic;
#[allow(unused)]
#[cfg(feature = "fcma")]
pub(crate) use spectrum_arithmetic_fcma::NeonFcmaSpectrumArithmetic;
pub(crate) use transpose::{
    NeonTranspose2x2F32, NeonTranspose2x2F64, NeonTranspose4x3F64, NeonTranspose4x4F32,
    NeonTranspose4x4F64, NeonTranspose4x7F32, NeonTranspose5x7F32, NeonTranspose6x4F32,
    NeonTranspose6x5F32, NeonTranspose7x2F32, NeonTranspose7x3F32, NeonTranspose7x5F32,
    NeonTranspose7x6F32, NeonTranspose7x7F32, NeonTranspose8x3F32, NeonTranspose9x2F32,
    NeonTranspose11x2F32, NeonTransposeNx2F32, NeonTransposeNx2F64, NeonTransposeNx3F32,
    NeonTransposeNx3F64, NeonTransposeNx4F32, NeonTransposeNx4F64, NeonTransposeNx5F32,
    NeonTransposeNx5F64, NeonTransposeNx6F32, NeonTransposeNx6F64, NeonTransposeNx7F32,
    NeonTransposeNx7F64, NeonTransposeNx8F32, NeonTransposeNx8F64, NeonTransposeNx9F32,
    NeonTransposeNx9F64, NeonTransposeNx10F32, NeonTransposeNx10F64, NeonTransposeNx11F32,
    NeonTransposeNx11F64, NeonTransposeNx12F32, NeonTransposeNx12F64, NeonTransposeNx13F32,
    NeonTransposeNx13F64, NeonTransposeNx14F32, NeonTransposeNx14F64, NeonTransposeNx15F32,
    NeonTransposeNx15F64, NeonTransposeNx16F32, NeonTransposeNx16F64, NeonTransposeNx17F32,
    NeonTransposeNx17F64, NeonTransposeNx18F32, NeonTransposeNx18F64, NeonTransposeNx19F32,
    NeonTransposeNx19F64, NeonTransposeNx20F32, block_transpose_f32x2_2x2,
    neon_transpose_f32x2_4x4, neon_transpose_f64x2_2x2,
};

#[cfg(test)]
#[cfg(feature = "fcma")]
macro_rules! test_fcma_radix {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $iters: expr, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            if !std::arch::is_aarch64_feature_detected!("fcma") {
                return;
            }
            use crate::FftDirection;
            use crate::FftExecutor;
            use num_complex::Complex;
            use rand::Rng;
            for i in 1..$iters {
                let val = $scale as usize;
                let size = val.pow(i);
                let mut input = vec![Complex::<$data_type>::default(); size];
                for z in input.iter_mut() {
                    *z = Complex {
                        re: rand::rng().random(),
                        im: rand::rng().random(),
                    };
                }
                let src = input.to_vec();
                use crate::dft::Dft;
                let reference_forward = Dft::new(size, FftDirection::Forward).unwrap();

                let mut ref_src = src.to_vec();
                reference_forward.execute(&mut ref_src).unwrap();

                let radix_forward = $butterfly::new(size, FftDirection::Forward).unwrap();
                let radix_inverse = $butterfly::new(size, FftDirection::Inverse).unwrap();
                radix_forward.execute(&mut input).unwrap();

                input
                    .iter()
                    .zip(ref_src.iter())
                    .enumerate()
                    .for_each(|(idx, (a, b))| {
                        assert!(
                            (a.re - b.re).abs() < $tol,
                            "a_re {} != b_re {} for size {} at {idx}",
                            a.re,
                            b.re,
                            size
                        );
                        assert!(
                            (a.im - b.im).abs() < $tol,
                            "a_im {} != b_im {} for size {} at {idx}",
                            a.im,
                            b.im,
                            size
                        );
                    });

                radix_inverse.execute(&mut input).unwrap();

                input = input
                    .iter()
                    .map(|&x| x * (1.0 / size as $data_type))
                    .collect();

                input.iter().zip(src.iter()).for_each(|(a, b)| {
                    assert!(
                        (a.re - b.re).abs() < $tol,
                        "a_re {} != b_re {} for size {}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < $tol,
                        "a_im {} != b_im {} for size {}",
                        a.im,
                        b.im,
                        size
                    );
                });
            }
        }
    };
}

#[cfg(test)]
#[cfg(feature = "fcma")]
pub(crate) use test_fcma_radix;
