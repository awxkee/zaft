/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
mod butterflies;
mod c2r;
mod column;
mod mixed_radix;
mod r2c;
mod rotate;
mod store;
mod transpose;

pub(crate) use butterflies::{
    WasmButterfly4, WasmButterfly8d, WasmButterfly8f, WasmButterfly16d, WasmButterfly16f,
    WasmButterfly32d, WasmButterfly32f, WasmButterfly64d, WasmButterfly64f, WasmButterfly128d,
    WasmButterfly128f, WasmButterfly256d, WasmButterfly256f, WasmButterfly512f,
};
pub(crate) use c2r::C2RWasmTwiddles;
pub(crate) use mixed_radix::{
    WasmMixedRadix2, WasmMixedRadix2f, WasmMixedRadix3, WasmMixedRadix3f, WasmMixedRadix4,
    WasmMixedRadix4f, WasmMixedRadix5, WasmMixedRadix5f, WasmMixedRadix7, WasmMixedRadix7f,
    WasmMixedRadix8, WasmMixedRadix8f, WasmMixedRadix9, WasmMixedRadix9f,
};
pub(crate) use r2c::R2CWasmTwiddles;
pub(crate) use transpose::{
    WasmTransposeNx2F32, WasmTransposeNx3F32, WasmTransposeNx4F32, WasmTransposeNx5F32,
    WasmTransposeNx7F32, WasmTransposeNx8F32, WasmTransposeNx9F32,
};

macro_rules! boring_wasm_butterfly {
    ($bf_name: ident, $f_type: ident, $size: expr) => {
        impl $bf_name {
            fn execute_impl(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.as_chunks_mut::<$size>().0.iter_mut() {
                    use crate::store::InPlaceStore;
                    self.run(&mut InPlaceStore::new(chunk));
                }

                Ok(())
            }

            fn execute_oof_impl(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, $size);

                for (dst, src) in dst
                    .as_chunks_mut::<$size>()
                    .0
                    .iter_mut()
                    .zip(src.as_chunks::<$size>().0.iter())
                {
                    use crate::store::BiStore;
                    self.run(&mut BiStore::new(src, dst));
                }
                Ok(())
            }
        }

        impl FftExecutor<$f_type> for $bf_name {
            fn execute(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                FftExecutor::execute_with_scratch(self, in_place, &mut [])
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_impl(in_place)
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_oof_impl(src, dst)
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_oof_impl(src, dst)
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            fn length(&self) -> usize {
                $size
            }

            fn scratch_length(&self) -> usize {
                0
            }

            fn out_of_place_scratch_length(&self) -> usize {
                0
            }

            fn destructive_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

pub(crate) use boring_wasm_butterfly;

#[cfg(test)]
macro_rules! test_wasm_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[wasm_bindgen_test::wasm_bindgen_test]
        fn $method_name() {
            use rand::RngExt;
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
                            "forward a_re {} != b_re {} for size {} at {idx}",
                            a.re,
                            b.re,
                            size
                        );
                        assert!(
                            (a.im - b.im).abs() < $tol,
                            "forward a_im {} != b_im {} for size {} at {idx}",
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
                        "inverse a_re {} != b_re {} for size {}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < $tol,
                        "inverse a_im {} != b_im {} for size {}",
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
pub(crate) use test_wasm_butterfly;

#[cfg(test)]
macro_rules! test_wasm_oof_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[wasm_bindgen_test::wasm_bindgen_test]
        fn $method_name() {
            use rand::RngExt;
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
use crate::util::compute_twiddle;
use crate::wasm::store::{WasmStoreD, WasmStoreF};
#[cfg(test)]
pub(crate) use test_wasm_oof_butterfly;

pub(crate) fn gen_butterfly_twiddles_f32<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [WasmStoreF; N] {
    let mut twiddles = [WasmStoreF::default(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 2;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = WasmStoreF::from_complex2(
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR), size, direction),
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), size, direction),
            );
            q += 1;
        }
    }
    twiddles
}

pub(crate) fn gen_butterfly_twiddles_f64<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [WasmStoreD; N] {
    let mut twiddles = [WasmStoreD::default(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 1;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = WasmStoreD::from_complex(&compute_twiddle(
                y * (x * COMPLEX_PER_VECTOR),
                size,
                direction,
            ));
            q += 1;
        }
    }
    twiddles
}
