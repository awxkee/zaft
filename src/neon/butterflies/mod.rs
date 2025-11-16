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
mod bf10;
mod bf100;
#[cfg(feature = "fcma")]
mod bf100_fcma;
mod bf100d;
#[cfg(feature = "fcma")]
mod bf100d_fcma;
#[cfg(feature = "fcma")]
mod bf10_fcma;
mod bf11;
#[cfg(feature = "fcma")]
mod bf11_fcma;
mod bf12;
#[cfg(feature = "fcma")]
mod bf12_fcma;
mod bf13;
#[cfg(feature = "fcma")]
mod bf13_fcma;
mod bf14;
#[cfg(feature = "fcma")]
mod bf14_fcma;
mod bf15;
#[cfg(feature = "fcma")]
mod bf15_fcma;
mod bf16;
#[cfg(feature = "fcma")]
mod bf16_fcma;
mod bf17;
#[cfg(feature = "fcma")]
mod bf17_fcma;
mod bf18;
mod bf19;
#[cfg(feature = "fcma")]
mod bf19_fcma;
mod bf2;
mod bf20;
#[cfg(feature = "fcma")]
mod bf20_fcma;
mod bf23;
#[cfg(feature = "fcma")]
mod bf23_fcma;
mod bf25;
#[cfg(feature = "fcma")]
mod bf25_fcma;
mod bf27;
#[cfg(feature = "fcma")]
mod bf27_fcma;
mod bf29;
#[cfg(feature = "fcma")]
mod bf29_fcma;
mod bf3;
mod bf31d;
#[cfg(feature = "fcma")]
mod bf31d_fcma;
mod bf31f;
#[cfg(feature = "fcma")]
mod bf31f_fcma;
mod bf32;
#[cfg(feature = "fcma")]
mod bf32_fcma;
mod bf36d;
#[cfg(feature = "fcma")]
mod bf36d_fcma;
mod bf36f;
#[cfg(feature = "fcma")]
mod bf36f_fcma;
mod bf4;
mod bf49;
#[cfg(feature = "fcma")]
mod bf49_fcma;
mod bf49d;
#[cfg(feature = "fcma")]
mod bf49d_fcma;
#[cfg(feature = "fcma")]
mod bf4_fcma;
mod bf5;
#[cfg(feature = "fcma")]
mod bf5_fcma;
mod bf6;
mod bf64d;
#[cfg(feature = "fcma")]
mod bf64d_fcma;
mod bf64f;
#[cfg(feature = "fcma")]
mod bf64f_fcma;
#[cfg(feature = "fcma")]
mod bf6_fcma;
mod bf7;
#[cfg(feature = "fcma")]
mod bf7_fcma;
mod bf8;
mod bf81d;
#[cfg(feature = "fcma")]
mod bf81d_fcma;
mod bf81f;
#[cfg(feature = "fcma")]
mod bf81f_fcma;
#[cfg(feature = "fcma")]
mod bf8_fcma;
mod bf9;
#[cfg(feature = "fcma")]
mod bf9_fcma;
mod fast_bf16d;
mod fast_bf16f;
mod fast_bf5;
mod fast_bf7;
mod fast_bf8;
mod fast_bf9d;
mod fast_bf9f;
mod shared;

pub(crate) use bf2::NeonButterfly2;
pub(crate) use bf3::NeonButterfly3;
pub(crate) use bf4::NeonButterfly4;
#[cfg(feature = "fcma")]
pub(crate) use bf4_fcma::NeonFcmaButterfly4;
pub(crate) use bf5::NeonButterfly5;
#[cfg(feature = "fcma")]
pub(crate) use bf5_fcma::NeonFcmaButterfly5;
pub(crate) use bf6::NeonButterfly6;
#[cfg(feature = "fcma")]
pub(crate) use bf6_fcma::NeonFcmaButterfly6;
pub(crate) use bf7::NeonButterfly7;
#[cfg(feature = "fcma")]
pub(crate) use bf7_fcma::NeonFcmaButterfly7;
pub(crate) use bf8::NeonButterfly8;
#[cfg(feature = "fcma")]
pub(crate) use bf8_fcma::NeonFcmaButterfly8;
pub(crate) use bf9::NeonButterfly9;
#[cfg(feature = "fcma")]
pub(crate) use bf9_fcma::NeonFcmaButterfly9;
pub(crate) use bf10::NeonButterfly10;
#[cfg(feature = "fcma")]
pub(crate) use bf10_fcma::NeonFcmaButterfly10;
pub(crate) use bf11::NeonButterfly11;
#[cfg(feature = "fcma")]
pub(crate) use bf11_fcma::NeonFcmaButterfly11;
pub(crate) use bf12::NeonButterfly12;
#[cfg(feature = "fcma")]
pub(crate) use bf12_fcma::NeonFcmaButterfly12;
pub(crate) use bf13::NeonButterfly13;
#[cfg(feature = "fcma")]
pub(crate) use bf13_fcma::NeonFcmaButterfly13;
pub(crate) use bf14::NeonButterfly14;
#[cfg(feature = "fcma")]
pub(crate) use bf14_fcma::NeonFcmaButterfly14;
pub(crate) use bf15::NeonButterfly15;
#[cfg(feature = "fcma")]
pub(crate) use bf15_fcma::NeonFcmaButterfly15;
pub(crate) use bf16::NeonButterfly16;
#[cfg(feature = "fcma")]
pub(crate) use bf16_fcma::NeonFcmaButterfly16;
pub(crate) use bf17::NeonButterfly17;
#[cfg(feature = "fcma")]
pub(crate) use bf17_fcma::NeonFcmaButterfly17;
pub(crate) use bf18::{NeonButterfly18d, NeonButterfly18f};
#[cfg(feature = "fcma")]
pub(crate) use bf18::{NeonFcmaButterfly18d, NeonFcmaButterfly18f};
pub(crate) use bf19::NeonButterfly19;
#[cfg(feature = "fcma")]
pub(crate) use bf19_fcma::NeonFcmaButterfly19;
pub(crate) use bf20::NeonButterfly20;
#[cfg(feature = "fcma")]
pub(crate) use bf20_fcma::NeonFcmaButterfly20;
pub(crate) use bf23::NeonButterfly23;
#[cfg(feature = "fcma")]
pub(crate) use bf23_fcma::NeonFcmaButterfly23;
pub(crate) use bf25::{NeonButterfly25d, NeonButterfly25f};
#[cfg(feature = "fcma")]
pub(crate) use bf25_fcma::{NeonFcmaButterfly25d, NeonFcmaButterfly25f};
pub(crate) use bf27::{NeonButterfly27d, NeonButterfly27f};
#[cfg(feature = "fcma")]
pub(crate) use bf27_fcma::{NeonFcmaButterfly27d, NeonFcmaButterfly27f};
pub(crate) use bf29::NeonButterfly29;
#[cfg(feature = "fcma")]
pub(crate) use bf29_fcma::NeonFcmaButterfly29;
pub(crate) use bf31d::NeonButterfly31d;
#[cfg(feature = "fcma")]
pub(crate) use bf31d_fcma::NeonFcmaButterfly31d;
pub(crate) use bf31f::NeonButterfly31f;
#[cfg(feature = "fcma")]
pub(crate) use bf31f_fcma::NeonFcmaButterfly31f;
pub(crate) use bf32::{NeonButterfly32d, NeonButterfly32f};
#[cfg(feature = "fcma")]
pub(crate) use bf32_fcma::{NeonFcmaButterfly32d, NeonFcmaButterfly32f};
pub(crate) use bf36d::NeonButterfly36d;
#[cfg(feature = "fcma")]
pub(crate) use bf36d_fcma::NeonFcmaButterfly36d;
pub(crate) use bf36f::NeonButterfly36f;
#[cfg(feature = "fcma")]
pub(crate) use bf36f_fcma::NeonFcmaButterfly36f;
pub(crate) use bf49::NeonButterfly49f;
#[cfg(feature = "fcma")]
pub(crate) use bf49_fcma::NeonFcmaButterfly49f;
pub(crate) use bf49d::NeonButterfly49d;
#[cfg(feature = "fcma")]
pub(crate) use bf49d_fcma::NeonFcmaButterfly49d;
pub(crate) use bf64d::NeonButterfly64d;
#[cfg(feature = "fcma")]
pub(crate) use bf64d_fcma::NeonFcmaButterfly64d;
pub(crate) use bf64f::NeonButterfly64f;
#[cfg(feature = "fcma")]
pub(crate) use bf64f_fcma::NeonFcmaButterfly64f;
pub(crate) use bf81d::NeonButterfly81d;
#[cfg(feature = "fcma")]
pub(crate) use bf81d_fcma::NeonFcmaButterfly81d;
pub(crate) use bf81f::NeonButterfly81f;
#[cfg(feature = "fcma")]
pub(crate) use bf81f_fcma::NeonFcmaButterfly81f;
pub(crate) use bf100::NeonButterfly100f;
#[cfg(feature = "fcma")]
pub(crate) use bf100_fcma::NeonFcmaButterfly100f;
pub(crate) use bf100d::NeonButterfly100d;
#[cfg(feature = "fcma")]
pub(crate) use bf100d_fcma::NeonFcmaButterfly100d;
pub(crate) use fast_bf5::NeonFastButterfly5;
pub(crate) use fast_bf8::NeonFastButterfly8;
use num_complex::Complex;
pub(crate) use shared::NeonButterfly;

#[inline]
pub(crate) fn make_mixedradix_twiddle_chunk_f32(
    x: usize,
    y: usize,
    len: usize,
    direction: FftDirection,
) -> NeonStoreF {
    let mut twiddle_chunk = [Complex::<f32>::default(); 2];
    use crate::util::compute_twiddle;
    #[allow(clippy::needless_range_loop)]
    for i in 0..2 {
        twiddle_chunk[i] = compute_twiddle(y * (x + i), len, direction);
    }

    NeonStoreF::from_complex_ref(twiddle_chunk.as_slice())
}

macro_rules! gen_butterfly_twiddles_separated_columns_f32 {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $direction: expr) => {{
        const FFT_LEN: usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS: usize = $num_rows - 1;
        const TWIDDLE_COLS: usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS: usize = TWIDDLE_COLS / 2;
        const TWIDDLE_VECTOR_COUNT: usize = TWIDDLE_VECTOR_COLS * TWIDDLE_ROWS;
        let mut twiddles = [NeonStoreF::default(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index % TWIDDLE_ROWS) + 1;
            let x = (index / TWIDDLE_ROWS) * 2 + $skip_cols;

            use crate::neon::butterflies::make_mixedradix_twiddle_chunk_f32;

            twiddles[index] = make_mixedradix_twiddle_chunk_f32(x, y, FFT_LEN, $direction);
        }
        twiddles
    }};
}

pub(crate) use gen_butterfly_twiddles_separated_columns_f32;

#[cfg(test)]
#[cfg(feature = "fcma")]
macro_rules! test_fcma_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            if !std::arch::is_aarch64_feature_detected!("fcma") {
                return;
            }
            use rand::Rng;
            for i in 1..4 {
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
                let reference_forward = Dft::new($scale, FftDirection::Forward).unwrap();

                let mut ref_src = src.to_vec();
                reference_forward.execute(&mut ref_src).unwrap();

                let radix_forward = $butterfly::new(FftDirection::Forward);
                let radix_inverse = $butterfly::new(FftDirection::Inverse);
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

                let val = $scale as $data_type;
                input = input.iter().map(|&x| x * (1.0 / val)).collect();

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
pub(crate) use test_fcma_butterfly;

#[cfg(test)]
#[cfg(feature = "fcma")]
macro_rules! test_fcma_butterfly_small {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            if !std::arch::is_aarch64_feature_detected!("fcma") {
                return;
            }
            use rand::Rng;
            for i in 1..2 {
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
                let reference_forward = Dft::new($scale, FftDirection::Forward).unwrap();

                let mut ref_src = src.to_vec();
                reference_forward.execute(&mut ref_src).unwrap();

                let radix_forward = $butterfly::new(FftDirection::Forward);
                let radix_inverse = $butterfly::new(FftDirection::Inverse);
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

                let val = $scale as $data_type;
                input = input.iter().map(|&x| x * (1.0 / val)).collect();

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
pub(crate) use test_fcma_butterfly_small;

#[cfg(test)]
#[cfg(feature = "fcma")]
macro_rules! test_oof_fcma_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            if !std::arch::is_aarch64_feature_detected!("fcma") {
                return;
            }
            use rand::Rng;
            for i in 1..4 {
                let kern = $scale;
                let size = (kern as usize).pow(i);
                let mut input = vec![Complex::<$data_type>::default(); size];
                for z in input.iter_mut() {
                    *z = Complex {
                        re: rand::rng().random(),
                        im: rand::rng().random(),
                    };
                }
                let src = input.to_vec();
                let mut out_of_place = vec![Complex::<$data_type>::default(); size];
                let mut ref_input = input.to_vec();
                let radix_forward = $butterfly::new(FftDirection::Forward);
                let radix_inverse = $butterfly::new(FftDirection::Inverse);

                use crate::dft::Dft;
                let reference_dft = Dft::new($scale, FftDirection::Forward).unwrap();
                reference_dft.execute(&mut ref_input).unwrap();

                radix_forward
                    .execute_out_of_place(&input, &mut out_of_place)
                    .unwrap();

                out_of_place
                    .iter()
                    .zip(ref_input.iter())
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

                radix_inverse
                    .execute_out_of_place(&out_of_place, &mut input)
                    .unwrap();

                input = input
                    .iter()
                    .map(|&x| x * (1.0 / (kern as $data_type)))
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

use crate::FftDirection;
use crate::neon::mixed::NeonStoreF;
#[cfg(test)]
#[cfg(feature = "fcma")]
pub(crate) use test_oof_fcma_butterfly;

#[cfg(test)]
#[cfg(feature = "fcma")]
macro_rules! test_oof_fcma_butterfly_small {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            if !std::arch::is_aarch64_feature_detected!("fcma") {
                return;
            }
            use rand::Rng;
            for i in 1..2 {
                let kern = $scale;
                let size = (kern as usize).pow(i);
                let mut input = vec![Complex::<$data_type>::default(); size];
                for z in input.iter_mut() {
                    *z = Complex {
                        re: rand::rng().random(),
                        im: rand::rng().random(),
                    };
                }
                let src = input.to_vec();
                let mut out_of_place = vec![Complex::<$data_type>::default(); size];
                let mut ref_input = input.to_vec();
                let radix_forward = $butterfly::new(FftDirection::Forward);
                let radix_inverse = $butterfly::new(FftDirection::Inverse);

                use crate::dft::Dft;
                let reference_dft = Dft::new($scale, FftDirection::Forward).unwrap();
                reference_dft.execute(&mut ref_input).unwrap();

                radix_forward
                    .execute_out_of_place(&input, &mut out_of_place)
                    .unwrap();

                out_of_place
                    .iter()
                    .zip(ref_input.iter())
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

                radix_inverse
                    .execute_out_of_place(&out_of_place, &mut input)
                    .unwrap();

                input = input
                    .iter()
                    .map(|&x| x * (1.0 / (kern as $data_type)))
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
pub(crate) use test_oof_fcma_butterfly_small;
