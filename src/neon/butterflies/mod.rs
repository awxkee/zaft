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
#![allow(clippy::needless_range_loop)]
mod bf10;
mod bf100d;
mod bf100f;
mod bf1024;
mod bf108d;
mod bf108f;
mod bf11;
mod bf12;
mod bf121d;
mod bf121f;
mod bf125d;
mod bf125f;
mod bf128d;
mod bf128f;
mod bf13;
mod bf14;
mod bf144d;
mod bf144f;
mod bf15;
mod bf16;
mod bf169d;
mod bf169f;
mod bf17;
mod bf18;
mod bf19;
mod bf192d;
mod bf192f;
mod bf2;
mod bf20;
mod bf21;
mod bf216d;
mod bf216f;
mod bf23;
mod bf24;
mod bf243d;
mod bf243f;
mod bf25;
mod bf256d;
mod bf256f;
mod bf27;
mod bf28;
mod bf3;
mod bf30d;
mod bf30f;
mod bf32;
mod bf35d;
mod bf35f;
mod bf36d;
mod bf36f;
mod bf4;
mod bf40d;
mod bf40f;
mod bf42d;
mod bf42f;
mod bf48d;
mod bf48f;
mod bf49;
mod bf49d;
#[cfg(feature = "fcma")]
mod bf4_fcma;
mod bf5;
mod bf512f;
mod bf54d;
mod bf54f;
#[cfg(feature = "fcma")]
mod bf5_fcma;
mod bf6;
mod bf63d;
mod bf63f;
mod bf64d;
mod bf64f;
mod bf66d;
mod bf66f;
mod bf7;
mod bf70d;
mod bf70f;
mod bf72d;
mod bf72f;
mod bf78d;
mod bf78f;
mod bf8;
mod bf81d;
mod bf81f;
mod bf88d;
mod bf88f;
mod bf9;
mod bf96d;
mod bf96f;
mod fast_bf5;
mod shared;

pub(crate) use bf2::NeonButterfly2;
pub(crate) use bf3::NeonButterfly3;
pub(crate) use bf4::NeonButterfly4;
#[cfg(feature = "fcma")]
pub(crate) use bf4_fcma::NeonFcmaButterfly4;
pub(crate) use bf5::NeonButterfly5;
#[cfg(feature = "fcma")]
pub(crate) use bf5_fcma::NeonFcmaButterfly5;
pub(crate) use bf6::{NeonButterfly6d, NeonButterfly6f};
#[cfg(feature = "fcma")]
pub(crate) use bf6::{NeonFcmaButterfly6d, NeonFcmaButterfly6f};
pub(crate) use bf7::{NeonButterfly7d, NeonButterfly7f};
#[cfg(feature = "fcma")]
pub(crate) use bf7::{NeonFcmaButterfly7d, NeonFcmaButterfly7f};
pub(crate) use bf8::{NeonButterfly8d, NeonButterfly8f};
#[cfg(feature = "fcma")]
pub(crate) use bf8::{NeonFcmaButterfly8d, NeonFcmaButterfly8f};
pub(crate) use bf9::{NeonButterfly9d, NeonButterfly9f};
#[cfg(feature = "fcma")]
pub(crate) use bf9::{NeonFcmaButterfly9d, NeonFcmaButterfly9f};
pub(crate) use bf10::{NeonButterfly10d, NeonButterfly10f};
#[cfg(feature = "fcma")]
pub(crate) use bf10::{NeonFcmaButterfly10d, NeonFcmaButterfly10f};
pub(crate) use bf11::{NeonButterfly11d, NeonButterfly11f};
#[cfg(feature = "fcma")]
pub(crate) use bf11::{NeonFcmaButterfly11d, NeonFcmaButterfly11f};
pub(crate) use bf12::{NeonButterfly12d, NeonButterfly12f};
#[cfg(feature = "fcma")]
pub(crate) use bf12::{NeonFcmaButterfly12d, NeonFcmaButterfly12f};
pub(crate) use bf13::{NeonButterfly13d, NeonButterfly13f};
#[cfg(feature = "fcma")]
pub(crate) use bf13::{NeonFcmaButterfly13d, NeonFcmaButterfly13f};
pub(crate) use bf14::{NeonButterfly14d, NeonButterfly14f};
#[cfg(feature = "fcma")]
pub(crate) use bf14::{NeonFcmaButterfly14d, NeonFcmaButterfly14f};
pub(crate) use bf15::{NeonButterfly15d, NeonButterfly15f};
#[cfg(feature = "fcma")]
pub(crate) use bf15::{NeonFcmaButterfly15d, NeonFcmaButterfly15f};
pub(crate) use bf16::{NeonButterfly16d, NeonButterfly16f};
#[cfg(feature = "fcma")]
pub(crate) use bf16::{NeonFcmaButterfly16d, NeonFcmaButterfly16f};
pub(crate) use bf17::{NeonButterfly17d, NeonButterfly17f};
#[cfg(feature = "fcma")]
pub(crate) use bf17::{NeonFcmaButterfly17d, NeonFcmaButterfly17f};
pub(crate) use bf18::{NeonButterfly18d, NeonButterfly18f};
#[cfg(feature = "fcma")]
pub(crate) use bf18::{NeonFcmaButterfly18d, NeonFcmaButterfly18f};
pub(crate) use bf19::{NeonButterfly19d, NeonButterfly19f};
#[cfg(feature = "fcma")]
pub(crate) use bf19::{NeonFcmaButterfly19d, NeonFcmaButterfly19f};
pub(crate) use bf20::{NeonButterfly20d, NeonButterfly20f};
#[cfg(feature = "fcma")]
pub(crate) use bf20::{NeonFcmaButterfly20d, NeonFcmaButterfly20f};
pub(crate) use bf21::{NeonButterfly21d, NeonButterfly21f};
#[cfg(feature = "fcma")]
pub(crate) use bf21::{NeonFcmaButterfly21d, NeonFcmaButterfly21f};
pub(crate) use bf23::{NeonButterfly23d, NeonButterfly23f};
#[cfg(feature = "fcma")]
pub(crate) use bf23::{NeonFcmaButterfly23d, NeonFcmaButterfly23f};
pub(crate) use bf24::{NeonButterfly24d, NeonButterfly24f};
#[cfg(feature = "fcma")]
pub(crate) use bf24::{NeonFcmaButterfly24d, NeonFcmaButterfly24f};
pub(crate) use bf25::{NeonButterfly25d, NeonButterfly25f};
#[cfg(feature = "fcma")]
pub(crate) use bf25::{NeonFcmaButterfly25d, NeonFcmaButterfly25f};
pub(crate) use bf27::{NeonButterfly27d, NeonButterfly27f};
#[cfg(feature = "fcma")]
pub(crate) use bf27::{NeonFcmaButterfly27d, NeonFcmaButterfly27f};
pub(crate) use bf28::{NeonButterfly28d, NeonButterfly28f};
#[cfg(feature = "fcma")]
pub(crate) use bf28::{NeonFcmaButterfly28d, NeonFcmaButterfly28f};
pub(crate) use bf30d::NeonButterfly30d;
#[cfg(feature = "fcma")]
pub(crate) use bf30d::NeonFcmaButterfly30d;
pub(crate) use bf30f::NeonButterfly30f;
#[cfg(feature = "fcma")]
pub(crate) use bf30f::NeonFcmaButterfly30f;
pub(crate) use bf32::{NeonButterfly32d, NeonButterfly32f};
#[cfg(feature = "fcma")]
pub(crate) use bf32::{NeonFcmaButterfly32d, NeonFcmaButterfly32f};
pub(crate) use bf35d::NeonButterfly35d;
#[cfg(feature = "fcma")]
pub(crate) use bf35d::NeonFcmaButterfly35d;
pub(crate) use bf35f::NeonButterfly35f;
#[cfg(feature = "fcma")]
pub(crate) use bf35f::NeonFcmaButterfly35f;
pub(crate) use bf36d::NeonButterfly36d;
#[cfg(feature = "fcma")]
pub(crate) use bf36d::NeonFcmaButterfly36d;
pub(crate) use bf36f::NeonButterfly36f;
#[cfg(feature = "fcma")]
pub(crate) use bf36f::NeonFcmaButterfly36f;
pub(crate) use bf40d::NeonButterfly40d;
#[cfg(feature = "fcma")]
pub(crate) use bf40d::NeonFcmaButterfly40d;
pub(crate) use bf40f::NeonButterfly40f;
#[cfg(feature = "fcma")]
pub(crate) use bf40f::NeonFcmaButterfly40f;
pub(crate) use bf42d::NeonButterfly42d;
#[cfg(feature = "fcma")]
pub(crate) use bf42d::NeonFcmaButterfly42d;
pub(crate) use bf42f::NeonButterfly42f;
#[cfg(feature = "fcma")]
pub(crate) use bf42f::NeonFcmaButterfly42f;
pub(crate) use bf48d::NeonButterfly48d;
#[cfg(feature = "fcma")]
pub(crate) use bf48d::NeonFcmaButterfly48d;
pub(crate) use bf48f::NeonButterfly48f;
#[cfg(feature = "fcma")]
pub(crate) use bf48f::NeonFcmaButterfly48f;
pub(crate) use bf49::NeonButterfly49f;
#[cfg(feature = "fcma")]
pub(crate) use bf49::NeonFcmaButterfly49f;
pub(crate) use bf49d::NeonButterfly49d;
#[cfg(feature = "fcma")]
pub(crate) use bf49d::NeonFcmaButterfly49d;
pub(crate) use bf54d::NeonButterfly54d;
#[cfg(feature = "fcma")]
pub(crate) use bf54d::NeonFcmaButterfly54d;
pub(crate) use bf54f::NeonButterfly54f;
#[cfg(feature = "fcma")]
pub(crate) use bf54f::NeonFcmaButterfly54f;
pub(crate) use bf63d::NeonButterfly63d;
#[cfg(feature = "fcma")]
pub(crate) use bf63d::NeonFcmaButterfly63d;
pub(crate) use bf63f::NeonButterfly63f;
#[cfg(feature = "fcma")]
pub(crate) use bf63f::NeonFcmaButterfly63f;
pub(crate) use bf64d::NeonButterfly64d;
#[cfg(feature = "fcma")]
pub(crate) use bf64d::NeonFcmaButterfly64d;
pub(crate) use bf64f::NeonButterfly64f;
#[cfg(feature = "fcma")]
pub(crate) use bf64f::NeonFcmaButterfly64f;
pub(crate) use bf66d::NeonButterfly66d;
#[cfg(feature = "fcma")]
pub(crate) use bf66d::NeonFcmaButterfly66d;
pub(crate) use bf66f::NeonButterfly66f;
#[cfg(feature = "fcma")]
pub(crate) use bf66f::NeonFcmaButterfly66f;
pub(crate) use bf70d::NeonButterfly70d;
#[cfg(feature = "fcma")]
pub(crate) use bf70d::NeonFcmaButterfly70d;
pub(crate) use bf70f::NeonButterfly70f;
#[cfg(feature = "fcma")]
pub(crate) use bf70f::NeonFcmaButterfly70f;
pub(crate) use bf72d::NeonButterfly72d;
#[cfg(feature = "fcma")]
pub(crate) use bf72d::NeonFcmaButterfly72d;
pub(crate) use bf72f::NeonButterfly72f;
#[cfg(feature = "fcma")]
pub(crate) use bf72f::NeonFcmaButterfly72f;
pub(crate) use bf78d::NeonButterfly78d;
#[cfg(feature = "fcma")]
pub(crate) use bf78d::NeonFcmaButterfly78d;
pub(crate) use bf78f::NeonButterfly78f;
#[cfg(feature = "fcma")]
pub(crate) use bf78f::NeonFcmaButterfly78f;
pub(crate) use bf81d::NeonButterfly81d;
#[cfg(feature = "fcma")]
pub(crate) use bf81d::NeonFcmaButterfly81d;
pub(crate) use bf81f::NeonButterfly81f;
#[cfg(feature = "fcma")]
pub(crate) use bf81f::NeonFcmaButterfly81f;
pub(crate) use bf88d::NeonButterfly88d;
#[cfg(feature = "fcma")]
pub(crate) use bf88d::NeonFcmaButterfly88d;
pub(crate) use bf88f::NeonButterfly88f;
#[cfg(feature = "fcma")]
pub(crate) use bf88f::NeonFcmaButterfly88f;
pub(crate) use bf96d::NeonButterfly96d;
#[cfg(feature = "fcma")]
pub(crate) use bf96d::NeonFcmaButterfly96d;
pub(crate) use bf96f::NeonButterfly96f;
#[cfg(feature = "fcma")]
pub(crate) use bf96f::NeonFcmaButterfly96f;
pub(crate) use bf100d::NeonButterfly100d;
#[cfg(feature = "fcma")]
pub(crate) use bf100d::NeonFcmaButterfly100d;
pub(crate) use bf100f::NeonButterfly100f;
#[cfg(feature = "fcma")]
pub(crate) use bf100f::NeonFcmaButterfly100f;
pub(crate) use bf108d::NeonButterfly108d;
#[cfg(feature = "fcma")]
pub(crate) use bf108d::NeonFcmaButterfly108d;
pub(crate) use bf108f::NeonButterfly108f;
#[cfg(feature = "fcma")]
pub(crate) use bf108f::NeonFcmaButterfly108f;
pub(crate) use bf121d::NeonButterfly121d;
#[cfg(feature = "fcma")]
pub(crate) use bf121d::NeonFcmaButterfly121d;
pub(crate) use bf121f::NeonButterfly121f;
#[cfg(feature = "fcma")]
pub(crate) use bf121f::NeonFcmaButterfly121f;
pub(crate) use bf125d::NeonButterfly125d;
#[cfg(feature = "fcma")]
pub(crate) use bf125d::NeonFcmaButterfly125d;
pub(crate) use bf125f::NeonButterfly125f;
#[cfg(feature = "fcma")]
pub(crate) use bf125f::NeonFcmaButterfly125f;
pub(crate) use bf128d::NeonButterfly128d;
#[cfg(feature = "fcma")]
pub(crate) use bf128d::NeonFcmaButterfly128d;
pub(crate) use bf128f::NeonButterfly128f;
#[cfg(feature = "fcma")]
pub(crate) use bf128f::NeonFcmaButterfly128f;
pub(crate) use bf144d::NeonButterfly144d;
#[cfg(feature = "fcma")]
pub(crate) use bf144d::NeonFcmaButterfly144d;
pub(crate) use bf144f::NeonButterfly144f;
#[cfg(feature = "fcma")]
pub(crate) use bf144f::NeonFcmaButterfly144f;
pub(crate) use bf169d::NeonButterfly169d;
#[cfg(feature = "fcma")]
pub(crate) use bf169d::NeonFcmaButterfly169d;
pub(crate) use bf169f::NeonButterfly169f;
#[cfg(feature = "fcma")]
pub(crate) use bf169f::NeonFcmaButterfly169f;
pub(crate) use bf192d::NeonButterfly192d;
#[cfg(feature = "fcma")]
pub(crate) use bf192d::NeonFcmaButterfly192d;
pub(crate) use bf192f::NeonButterfly192f;
#[cfg(feature = "fcma")]
pub(crate) use bf192f::NeonFcmaButterfly192f;
pub(crate) use bf216d::NeonButterfly216d;
#[cfg(feature = "fcma")]
pub(crate) use bf216d::NeonFcmaButterfly216d;
pub(crate) use bf216f::NeonButterfly216f;
#[cfg(feature = "fcma")]
pub(crate) use bf216f::NeonFcmaButterfly216f;
pub(crate) use bf243d::NeonButterfly243d;
#[cfg(feature = "fcma")]
pub(crate) use bf243d::NeonFcmaButterfly243d;
pub(crate) use bf243f::NeonButterfly243f;
#[cfg(feature = "fcma")]
pub(crate) use bf243f::NeonFcmaButterfly243f;
pub(crate) use bf256d::NeonButterfly256d;
#[cfg(feature = "fcma")]
pub(crate) use bf256d::NeonFcmaButterfly256d;
pub(crate) use bf256f::NeonButterfly256f;
#[cfg(feature = "fcma")]
pub(crate) use bf256f::NeonFcmaButterfly256f;
pub(crate) use bf512f::NeonButterfly512f;
#[cfg(feature = "fcma")]
pub(crate) use bf512f::NeonFcmaButterfly512f;
pub(crate) use bf1024::NeonButterfly1024f;
#[cfg(feature = "fcma")]
pub(crate) use bf1024::{NeonFcmaForwardButterfly1024f, NeonFcmaInverseButterfly1024f};
pub(crate) use fast_bf5::NeonFastButterfly5;
use num_complex::Complex;
#[cfg(feature = "fcma")]
pub(crate) use shared::FastFcmaBf4f;
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
            let radix_forward = $butterfly::new(FftDirection::Forward);
            let radix_inverse = $butterfly::new(FftDirection::Inverse);
            assert_eq!(radix_forward.length(), $scale);
            for i in 1..20 {
                let val = $scale as usize;
                let size = val * i;
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

                FftExecutor::execute(&radix_forward, &mut input).unwrap();

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

                FftExecutor::execute(&radix_inverse, &mut input).unwrap();

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
macro_rules! test_oof_fcma_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            if !std::arch::is_aarch64_feature_detected!("fcma") {
                return;
            }
            use rand::Rng;
            for i in 1..20 {
                let kern = $scale;
                let size = (kern as usize) * i;
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
