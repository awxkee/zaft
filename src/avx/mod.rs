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
mod mixed;
mod r2c;
mod raders;
mod radix10;
mod radix11;
mod radix13;
mod radix3;
mod radix4;
mod radix5;
mod radix6;
mod radix7;
mod rotate;
mod spectrum_arithmetic;
mod transpose;
mod util;

pub(crate) use butterflies::{
    AvxButterfly2, AvxButterfly3, AvxButterfly4, AvxButterfly5, AvxButterfly6, AvxButterfly7,
    AvxButterfly8, AvxButterfly9d, AvxButterfly9f, AvxButterfly10d, AvxButterfly10f,
    AvxButterfly11, AvxButterfly11d, AvxButterfly12d, AvxButterfly12f, AvxButterfly13,
    AvxButterfly13d, AvxButterfly14d, AvxButterfly14f, AvxButterfly15d, AvxButterfly15f,
    AvxButterfly16d, AvxButterfly16f, AvxButterfly17, AvxButterfly18d, AvxButterfly18f,
    AvxButterfly19, AvxButterfly20d, AvxButterfly20f, AvxButterfly21d, AvxButterfly21f,
    AvxButterfly23, AvxButterfly24d, AvxButterfly24f, AvxButterfly25d, AvxButterfly25f,
    AvxButterfly27d, AvxButterfly27f, AvxButterfly29, AvxButterfly30d, AvxButterfly30f,
    AvxButterfly31, AvxButterfly32d, AvxButterfly32f, AvxButterfly35d, AvxButterfly35f,
    AvxButterfly36d, AvxButterfly36f, AvxButterfly40d, AvxButterfly40f, AvxButterfly42d,
    AvxButterfly42f, AvxButterfly48d, AvxButterfly48f, AvxButterfly49d, AvxButterfly49f,
    AvxButterfly54d, AvxButterfly54f, AvxButterfly63d, AvxButterfly63f, AvxButterfly64d,
    AvxButterfly64f, AvxButterfly66d, AvxButterfly66f, AvxButterfly70d, AvxButterfly70f,
    AvxButterfly72d, AvxButterfly72f, AvxButterfly81d, AvxButterfly81f, AvxButterfly96d,
    AvxButterfly96f, AvxButterfly100d, AvxButterfly100f, AvxButterfly121d, AvxButterfly121f,
    AvxButterfly125d, AvxButterfly125f, AvxButterfly128d, AvxButterfly128f, AvxButterfly144d,
    AvxButterfly144f, AvxButterfly169d, AvxButterfly169f, AvxButterfly243d, AvxButterfly243f,
    AvxButterfly256d, AvxButterfly256f, AvxButterfly512d, AvxButterfly512f,
};
pub(crate) use c2r::C2RAvxTwiddles;
pub(crate) use mixed::{
    AvxMixedRadix2d, AvxMixedRadix3d, AvxMixedRadix4d, AvxMixedRadix5d, AvxMixedRadix6d,
    AvxMixedRadix7d, AvxMixedRadix8d, AvxMixedRadix9d, AvxMixedRadix10d, AvxMixedRadix11d,
    AvxMixedRadix12d, AvxMixedRadix13d, AvxMixedRadix14d, AvxMixedRadix14f, AvxMixedRadix15d,
    AvxMixedRadix16d,
};
pub(crate) use mixed::{
    AvxMixedRadix2f, AvxMixedRadix3f, AvxMixedRadix4f, AvxMixedRadix5f, AvxMixedRadix6f,
    AvxMixedRadix7f, AvxMixedRadix8f, AvxMixedRadix9f, AvxMixedRadix10f, AvxMixedRadix11f,
    AvxMixedRadix12f, AvxMixedRadix13f, AvxMixedRadix15f, AvxMixedRadix16f,
};
pub(crate) use r2c::R2CAvxTwiddles;
pub(crate) use raders::AvxRadersFft;
pub(crate) use radix3::AvxFmaRadix3;
pub(crate) use radix4::AvxFmaRadix4;
pub(crate) use radix5::AvxFmaRadix5;
pub(crate) use radix6::AvxFmaRadix6;
pub(crate) use radix7::AvxFmaRadix7;
pub(crate) use radix10::{AvxFmaRadix10d, AvxFmaRadix10f};
pub(crate) use radix11::AvxFmaRadix11;
pub(crate) use radix13::AvxFmaRadix13;
#[allow(unused)]
pub(crate) use spectrum_arithmetic::AvxSpectrumArithmetic;
pub(crate) use transpose::{
    AvxTransposeF323x8, AvxTransposeF325x5, AvxTransposeF327x2, AvxTransposeF327x3,
    AvxTransposeF327x5, AvxTransposeF327x6, AvxTransposeF327x7, AvxTransposeF328x3,
    AvxTransposeF644x4, AvxTransposeNx2F32, AvxTransposeNx2F64, AvxTransposeNx3F32,
    AvxTransposeNx3F64, AvxTransposeNx4F32, AvxTransposeNx4F64, AvxTransposeNx5F32,
    AvxTransposeNx5F64, AvxTransposeNx6F32, AvxTransposeNx6F64, AvxTransposeNx7F32,
    AvxTransposeNx7F64, AvxTransposeNx8F32, AvxTransposeNx8F64, AvxTransposeNx9F32,
    AvxTransposeNx9F64, AvxTransposeNx10F32, AvxTransposeNx10F64, AvxTransposeNx11F32,
    AvxTransposeNx11F64, AvxTransposeNx12F32, AvxTransposeNx12F64, AvxTransposeNx13F32,
    AvxTransposeNx13F64, AvxTransposeNx14F32, AvxTransposeNx14F64, AvxTransposeNx15F32,
    AvxTransposeNx15F64, AvxTransposeNx16F32, AvxTransposeNx16F64, avx_transpose_f32x2_2x2,
    avx_transpose_f64x2_2x2, avx_transpose_f64x2_4x4, avx_transpose_f64x2_6x6_impl,
    avx2_transpose_f32x2_4x4, avx2_transpose_f32x2_8x4,
};

#[cfg(test)]
macro_rules! test_avx_radix {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $iters: expr, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use crate::util::has_valid_avx;
            if !has_valid_avx() {
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
                        re: rand::rng().random_range(-1.1..1.1),
                        im: rand::rng().random_range(-1.1..1.1),
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
macro_rules! test_avx_radix_fast {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $fast_bf: ident, $iters: expr, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use crate::util::has_valid_avx;
            if !has_valid_avx() {
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
                        re: rand::rng().random_range(-1.1..1.1),
                        im: rand::rng().random_range(-1.1..1.1),
                    };
                }
                let src = input.to_vec();
                use crate::$fast_bf;
                let reference_forward = $fast_bf::new(size, FftDirection::Forward).unwrap();

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
pub(crate) use test_avx_radix;

#[cfg(test)]
pub(crate) use test_avx_radix_fast;
