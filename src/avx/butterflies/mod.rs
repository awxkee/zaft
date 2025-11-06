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
mod bf10;
mod bf11;
mod bf12;
mod bf13;
mod bf14;
mod bf15;
mod bf16;
mod bf17;
mod bf19;
mod bf2;
mod bf23;
mod bf27d;
mod bf27f;
mod bf29;
mod bf3;
mod bf32d;
mod bf32f;
mod bf4;
mod bf5;
mod bf6;
mod bf7;
mod bf8;
mod bf9;
mod fast_bf3;
mod fast_bf4;
mod fast_bf5;
mod fast_bf7;
mod fast_bf8;
mod fast_bf9;
mod fast_bf9d;
mod shared;

pub(crate) use bf2::AvxButterfly2;
pub(crate) use bf3::AvxButterfly3;
pub(crate) use bf4::AvxButterfly4;
pub(crate) use bf5::AvxButterfly5;
pub(crate) use bf6::AvxButterfly6;
pub(crate) use bf7::AvxButterfly7;
pub(crate) use bf8::AvxButterfly8;
pub(crate) use bf9::AvxButterfly9;
pub(crate) use bf10::{AvxButterfly10d, AvxButterfly10f};
pub(crate) use bf11::AvxButterfly11;
pub(crate) use bf12::AvxButterfly12;
pub(crate) use bf13::AvxButterfly13;
pub(crate) use bf14::AvxButterfly14;
pub(crate) use bf15::{AvxButterfly15d, AvxButterfly15f};
pub(crate) use bf16::AvxButterfly16;
pub(crate) use bf17::AvxButterfly17;
pub(crate) use bf19::AvxButterfly19;
pub(crate) use bf23::AvxButterfly23;
pub(crate) use bf27d::AvxButterfly27d;
pub(crate) use bf27f::AvxButterfly27f;
pub(crate) use bf29::AvxButterfly29;
pub(crate) use bf32d::AvxButterfly32d;
pub(crate) use bf32f::AvxButterfly32f;
pub(crate) use fast_bf3::AvxFastButterfly3;
pub(crate) use fast_bf4::AvxFastButterfly4;
pub(crate) use fast_bf5::{AvxFastButterfly5d, AvxFastButterfly5f};
pub(crate) use fast_bf8::AvxFastButterfly8;
pub(crate) use shared::AvxButterfly;

#[cfg(test)]
macro_rules! test_avx_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use crate::util::has_valid_avx;
            if !has_valid_avx() {
                return;
            }
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
pub(crate) use test_avx_butterfly;

#[cfg(test)]
macro_rules! test_oof_avx_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use crate::util::has_valid_avx;
            if !has_valid_avx() {
                return;
            }
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

#[cfg(test)]
pub(crate) use test_oof_avx_butterfly;
