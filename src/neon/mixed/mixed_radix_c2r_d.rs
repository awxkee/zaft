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

/*
   Layout 7x7
   00 01 02 03 04 05 06
   07 08 09 10 11 12 13
   14 15 16 17 18 19 20
   21 22 23 24 24*23*22*
   21*20*19*18*17*16*15*
   14*13*12*11*10*09*08*
   07*06*05*04*03*02*01*
*/

/*
    Layout 3x5
    00 01 02 03 04
    05 06 07 07*06*
    05*04*03*02*01*
*/

/*
    Layout 5x3
    00 01 02
    03 04 05
    06 07 07*
    06*05*04*
    03*02*01*
*/

use crate::err::try_vec;
use crate::neon::mixed::NeonStoreD;
use crate::r2c::C2rChild;
use crate::transpose::{TransposeExecutorRealInv, TransposeFactory};
use crate::util::compute_twiddle;
use crate::{C2RFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::Zero;

macro_rules! define_mixed_radix_neon_d {
    ($radix_name: ident, $features: literal, $bf_name: ident, $row_count: expr, $mul: ident, $transpose: ident) => {

        use crate::neon::mixed::$bf_name;
        pub(crate) struct $radix_name {
            execution_length: usize,
            twiddles: Vec<NeonStoreD>,
            width_executor: C2rChild<f64>,
            width: usize,
            spectrum_width: usize,
            stage_len: usize,
            transpose_executor: Option<Box<dyn TransposeExecutorRealInv<f64> + Send + Sync>>,
            inner_bf: $bf_name,
            width_scratch_length: usize,
        }

        impl $radix_name {
            pub(crate) fn new(width_executor: C2rChild<f64>) -> Result<Self, ZaftError> {
                let direction = crate::FftDirection::Inverse;
                let (width, child_scratch_length, complex_leaf) = match &width_executor {
                    C2rChild::Complex(child) => (child.length(), child.scratch_length(), true),
                    C2rChild::Real(child) => (child.real_length(), child.complex_scratch_length(), false),
                };
                let spectrum_width = width / 2 + 1;
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 1;
                let len = width.checked_mul(ROW_COUNT).ok_or(ZaftError::Overflow)?;
                let stage_len = if complex_leaf {
                    len
                } else {
                    spectrum_width
                        .checked_mul(ROW_COUNT)
                        .ok_or(ZaftError::Overflow)?
                };
                let twiddle_width = if complex_leaf { width } else { spectrum_width };
                let num_twiddle_columns = twiddle_width.div_ceil(COMPLEX_PER_VECTOR);
                let twiddle_count = num_twiddle_columns
                    .checked_mul(TWIDDLES_PER_COLUMN)
                    .ok_or(ZaftError::Overflow)?;
                let mut twiddles = Vec::new();
                twiddles
                    .try_reserve_exact(twiddle_count)
                    .map_err(|_| ZaftError::OutOfMemory(twiddle_count))?;
                for x in 0..num_twiddle_columns {
                    for y in 1..ROW_COUNT {
                        let mut data = [Complex::<f64>::zero(); COMPLEX_PER_VECTOR];
                        for (i, value) in data.iter_mut().enumerate() {
                            *value = compute_twiddle(y * (x * COMPLEX_PER_VECTOR + i), len, direction);
                        }
                        twiddles.push(NeonStoreD::from_complex_ref(&data));
                    }
                }
                // Real children keep one output row as a temporary. Complex
                // leaves can use the whole output for scratch until transpose.
                let available_scratch = if complex_leaf {
                    len / 2
                } else {
                    (len - width) / 2
                };
                let width_scratch_length = if child_scratch_length <= available_scratch {
                    0
                } else {
                    child_scratch_length
                };
                stage_len
                    .checked_add(width_scratch_length)
                    .ok_or(ZaftError::Overflow)?;
                Ok(Self {
                    execution_length: len,
                    width_executor,
                    width,
                    spectrum_width,
                    stage_len,
                    transpose_executor: if complex_leaf {
                        Some(f64::transpose_strategy_real_inv(width, ROW_COUNT))
                    } else {
                        None
                    },
                    twiddles,
                    inner_bf: $bf_name::new(direction),
                    width_scratch_length,
                })
            }
        }

        impl C2RFftExecutor<f64> for $radix_name {
            fn execute(&self, input: &[Complex<f64>], output: &mut [f64]) -> Result<(), ZaftError> {
                crate::util::validate_oof_block_sizes(
                    input.len(),
                    self.complex_length(),
                    output.len(),
                    self.real_length(),
                )?;
                let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
                self.execute_with_scratch(input, output, &mut scratch)
            }

            fn execute_with_scratch(
                &self,
                input: &[Complex<f64>],
                output: &mut [f64],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                crate::util::validate_oof_block_sizes(
                    input.len(),
                    self.complex_length(),
                    output.len(),
                    self.real_length(),
                )?;
                unsafe { self.execute_oof_impl(input, output, scratch) }
            }

            fn complex_length(&self) -> usize {
                self.execution_length / 2 + 1
            }
            fn complex_scratch_length(&self) -> usize {
                self.stage_len + self.width_scratch_length
            }
            fn real_length(&self) -> usize {
                self.execution_length
            }
        }

        impl $radix_name {
            #[target_feature(enable = $features)]
            fn process_full_columns(&self, src: &[Complex<f64>], complex: &mut [Complex<f64>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = $row_count - 1;
                const COMPLEX_PER_VECTOR: usize = 1;

                let len_per_row = self.real_length() / ROW_COUNT;
                let chunk_count = len_per_row / COMPLEX_PER_VECTOR;

                let conj_flag = NeonStoreD::conj_flag();

                let complex_chunks_count = chunk_count / 2 + 1;

                let src_len = src.len();

                // Pass 1: chunks fully within the stored half-spectrum (c < complex_chunks_count).
                // Both forward and mirror rows load from src without boundary special-casing.
                for (c, twiddle_chunk) in self
                    .twiddles
                    .as_chunks::<TWIDDLES_PER_COLUMN>().0.iter()
                    .take(complex_chunks_count)
                    .enumerate()
                {
                    let index_base = c * COMPLEX_PER_VECTOR;
                    // Reversed kx index: for kx, the mirror is at position (chunk_count - c)
                    let reversed_index_base = (chunk_count - c) * COMPLEX_PER_VECTOR;

                    // Load columns from the input into registers
                    let mut columns = [NeonStoreD::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT / 2 + 1 {
                        unsafe {
                            let q = NeonStoreD::from_complex_ref(
                                src.get_unchecked(index_base + len_per_row * i..),
                            );
                            columns[i] = q;
                        }
                    }
                    // Load mirror rows from C2R conjugate-reversed indices
                    // Row ROW_COUNT-i mirrors row i, but kx is negated:
                    // physical index = reversed_index_base + len_per_row * (i - 1)
                    // but kx=0 col stays at offset 0, so reversed chunk index is:
                    // chunk_count - 1 - c   for the non-DC columns
                    for i in 1..=ROW_COUNT / 2 {
                        unsafe {
                            let mirror_row = ROW_COUNT - i;
                            // +COMPLEX_PER_VECTOR because kx=0 is at offset 0 and not mirrored
                            let r = reversed_index_base + len_per_row * (i - 1);
                            let q = NeonStoreD::from_complex_ref(src.get_unchecked(r..));
                            columns[mirror_row] = q.xor(conj_flag);
                        }
                    }

                    #[allow(unused_unsafe)]
                    let output = unsafe { self.inner_bf.exec(columns) };

                    unsafe {
                        output[0].write(complex.get_unchecked_mut(index_base..));
                    }

                    // here LLVM doesn't "see" NeonStoreD as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [NeonStoreD::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = NeonStoreD::$mul(output[i], twiddle);
                        unsafe {
                            output.write(complex.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }

                let twiddles = &self.twiddles[complex_chunks_count * TWIDDLES_PER_COLUMN..];

                // Pass 2: chunks that cross or lie beyond the Middle boundary.
                // The middle row must be assembled from the end of src based on middle_lane.
                for (c, twiddle_chunk) in twiddles
                    .as_chunks::<TWIDDLES_PER_COLUMN>().0.iter()
                    .take(chunk_count - complex_chunks_count)
                    .enumerate()
                {
                    let c0 = c;
                    let c = complex_chunks_count + c;
                    let index_base = c * COMPLEX_PER_VECTOR;
                    // Reversed kx index: for kx, the mirror is at position (chunk_count - c)
                    let reversed_index_base = (chunk_count - c) * COMPLEX_PER_VECTOR;

                    // Load columns from the input into registers
                    let mut columns = [NeonStoreD::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT / 2 {
                        unsafe {
                            let input = index_base + len_per_row * i;
                            let q = NeonStoreD::from_complex_ref(src.get_unchecked(input..));
                            columns[i] = q;
                        }
                    }
                    unsafe {
                        // middle lane after half have odd conjugation we need to handle this
                        let input = src_len - c0 - 1;
                        let q = NeonStoreD::from_complex_ref(src.get_unchecked(input..));
                        columns[ROW_COUNT / 2] = q.xor(conj_flag);
                    }
                    // Load mirror rows from C2R conjugate-reversed indices
                    // Row ROW_COUNT-i mirrors row i, but kx is negated:
                    // physical index = reversed_index_base + len_per_row * (i - 1)
                    // but kx=0 col stays at offset 0, so reversed chunk index is:
                    // chunk_count - 1 - c   for the non-DC columns
                    for i in 1..=ROW_COUNT / 2 {
                        unsafe {
                            let mirror_row = ROW_COUNT - i;
                            // +COMPLEX_PER_VECTOR because kx=0 is at offset 0 and not mirrored
                            let r = reversed_index_base + len_per_row * (i - 1);
                            let q = NeonStoreD::from_complex_ref(src.get_unchecked(r..));
                            columns[mirror_row] = q.xor(conj_flag);
                        }
                    }

                    #[allow(unused_unsafe)]
                    let output = unsafe { self.inner_bf.exec(columns) };

                    unsafe {
                        output[0].write(complex.get_unchecked_mut(index_base..));
                    }

                    // here LLVM doesn't "see" NeonStoreD as the same type returned by output
                    // so we need to force cast it onwards to the same type
                    let mut twiddles = [NeonStoreD::default(); ROW_COUNT - 1];
                    for i in 0..ROW_COUNT - 1 {
                        twiddles[i] = twiddle_chunk[i];
                    }

                    for i in 1..ROW_COUNT {
                        let twiddle = twiddles[i - 1];
                        let output = NeonStoreD::$mul(output[i], twiddle);
                        unsafe {
                            output.write(complex.get_unchecked_mut(index_base + len_per_row * i..))
                        }
                    }
                }
            }

            #[target_feature(enable = $features)]
            fn process_columns(&self, src: &[Complex<f64>], complex: &mut [Complex<f64>]) {
                const ROW_COUNT: usize = $row_count;
                const TWIDDLES_PER_COLUMN: usize = ROW_COUNT - 1;
                const COMPLEX_PER_VECTOR: usize = 1;
                let width = self.width;
                let k = self.spectrum_width;
                let conj_flag = NeonStoreD::conj_flag();
                // Only k=0..floor(width/2) is needed by each real child.
                // For a mirror row i, X[N-(i*width-x)] = conj(X[i*width-x]).
                for (c, twiddle_chunk) in self
                    .twiddles
                    .as_chunks::<TWIDDLES_PER_COLUMN>()
                    .0
                    .iter()
                    .take(k / COMPLEX_PER_VECTOR)
                    .enumerate()
                {
                    let x = c * COMPLEX_PER_VECTOR;
                    let mut columns = [NeonStoreD::default(); ROW_COUNT];
                    for i in 0..=ROW_COUNT / 2 {
                        columns[i] = NeonStoreD::from_complex_ref(&src[x + width * i..]);
                    }
                    for i in 1..=ROW_COUNT / 2 {
                        let mirror = i * width - x - (COMPLEX_PER_VECTOR - 1);
                        columns[ROW_COUNT - i] =
                            NeonStoreD::from_complex_ref(&src[mirror..]).xor(conj_flag);
                    }
                    let output = self.inner_bf.exec(columns);
                    output[0].write(&mut complex[x..]);
                    for i in 1..ROW_COUNT {
                        NeonStoreD::$mul(output[i], twiddle_chunk[i - 1])
                            .write(&mut complex[i * k + x..]);
                    }
                }
            }

            #[target_feature(enable = $features)]
            fn transpose_real_rows(&self, input: &[Complex<f64>], output: &mut [f64]) {
                use crate::neon::transpose::$transpose;
                const ROW_COUNT: usize = $row_count;
                const LANES: usize = 2;
                let input = crate::r2c::mixed_radix_c2r::as_real(input);
                let stride = self.spectrum_width * 2;
                let mut x = 0;
                while x + LANES <= self.width {
                    let mut rows = [NeonStoreD::default(); ROW_COUNT];
                    for i in 0..ROW_COUNT {
                        let physical = (i + ROW_COUNT - 1) % ROW_COUNT;
                        rows[i] = NeonStoreD::load(&input[physical * stride + x..]);
                    }
                    let values = $transpose(rows);
                    for column in 0..LANES {
                        for (group, block) in values
                            .chunks_exact(LANES)
                            .take(ROW_COUNT / LANES)
                            .enumerate()
                        {
                            block[column]
                                .write_real(&mut output[(x + column) * ROW_COUNT + group * LANES..]);
                        }
                        values[(ROW_COUNT / 2) * 2 + column]
                            .write_real_lo(&mut output[(x + column) * ROW_COUNT + (ROW_COUNT / 2) * 2..]);
                    }
                    x += LANES;
                }
                while x < self.width {
                    for i in 0..ROW_COUNT {
                        output[x * ROW_COUNT + i] = input[((i + ROW_COUNT - 1) % ROW_COUNT) * stride + x];
                    }
                    x += 1;
                }
            }

            #[target_feature(enable = $features)]
            fn execute_oof_impl(
                &self,
                src: &[Complex<f64>],
                dst: &mut [f64],
                scratch: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_scratch;
                let scratch = validate_scratch!(scratch, self.complex_scratch_length());
                let (spectra, width_scratch) = scratch.split_at_mut(self.stage_len);
                for (output, input) in dst
                    .chunks_exact_mut(self.execution_length)
                    .zip(src.chunks_exact(self.complex_length()))
                {
                    match &self.width_executor {
                        C2rChild::Real(child) => {
                            self.process_columns(input, spectra);
                            crate::r2c::mixed_radix_c2r::execute_rows(
                                child.as_ref(),
                                spectra,
                                output,
                                width_scratch,
                                $row_count,
                            )?;
                            self.transpose_real_rows(spectra, output);
                        }
                        C2rChild::Complex(child) => {
                            self.process_full_columns(input, spectra);
                            crate::r2c::mixed_radix_c2r::execute_complex_rows(
                                child.as_ref(),
                                spectra,
                                output,
                                width_scratch,
                            )?;
                            self.transpose_executor.as_ref().unwrap().transpose(
                                spectra,
                                output,
                                self.width,
                                $row_count,
                            );
                        }
                    }
                }
                Ok(())
            }
        }
    };
}

define_mixed_radix_neon_d!(
    NeonC2RMixedRadix3d,
    "neon",
    ColumnButterfly3d,
    3,
    mul_by_complex,
    transpose_f64x2_2x3
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaC2RMixedRadix3d,
    "fcma",
    ColumnFcmaButterfly3d,
    3,
    fcmul_fcma,
    transpose_f64x2_2x3
);
define_mixed_radix_neon_d!(
    NeonC2RMixedRadix5d,
    "neon",
    ColumnButterfly5d,
    5,
    mul_by_complex,
    transpose_f64x2_2x5
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaC2RMixedRadix5d,
    "fcma",
    ColumnFcmaButterfly5d,
    5,
    fcmul_fcma,
    transpose_f64x2_2x5
);
define_mixed_radix_neon_d!(
    NeonC2RMixedRadix7d,
    "neon",
    ColumnButterfly7d,
    7,
    mul_by_complex,
    transpose_f64x2_2x7
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaC2RMixedRadix7d,
    "fcma",
    ColumnFcmaButterfly7d,
    7,
    fcmul_fcma,
    transpose_f64x2_2x7
);
define_mixed_radix_neon_d!(
    NeonC2RMixedRadix9d,
    "neon",
    ColumnButterfly9d,
    9,
    mul_by_complex,
    transpose_f64x2_2x9
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaC2RMixedRadix9d,
    "fcma",
    ColumnFcmaButterfly9d,
    9,
    fcmul_fcma,
    transpose_f64x2_2x9
);
define_mixed_radix_neon_d!(
    NeonC2RMixedRadix11d,
    "neon",
    ColumnButterfly11d,
    11,
    mul_by_complex,
    transpose_f64x2_2x11
);
#[cfg(feature = "fcma")]
define_mixed_radix_neon_d!(
    NeonFcmaC2RMixedRadix11d,
    "fcma",
    ColumnFcmaButterfly11d,
    11,
    fcmul_fcma,
    transpose_f64x2_2x11
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use crate::{FftDirection, FftExecutor};

    #[test]
    fn test_mixed_radixd() {
        let src: [f64; 15] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.1,
        ];

        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(15, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let mut input_c = reference_value[..15 / 2 + 1].to_vec();
        for w in input_c.iter_mut() {
            *w = *w / 15.0;
        }

        let local_r2c =
            NeonC2RMixedRadix3d::new(C2rChild::Real(crate::r2c::strategy_c2r(5).unwrap())).unwrap();
        let mut complex_output = vec![f64::zero(); 15];
        local_r2c.execute(&input_c, &mut complex_output).unwrap();
        println!("complex_output: {:?}", complex_output);

        src.iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_re {} != b_re {} for at {idx}",
                    a,
                    b,
                );
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_im {} != b_im {} for at {idx}",
                    a,
                    b,
                );
            });
    }

    #[test]
    fn test_mixed_radixd_5() {
        let src: [f64; 15] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.1,
        ];

        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(15, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let mut input_c = reference_value[..15 / 2 + 1].to_vec();
        for w in input_c.iter_mut() {
            *w = *w / 15.0;
        }

        let local_r2c =
            NeonC2RMixedRadix5d::new(C2rChild::Real(crate::r2c::strategy_c2r(3).unwrap())).unwrap();
        let mut complex_output = vec![f64::zero(); 15];
        local_r2c.execute(&input_c, &mut complex_output).unwrap();
        println!("complex_output: {:?}", complex_output);

        src.iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_re {} != b_re {} for at {idx}",
                    a,
                    b,
                );
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_im {} != b_im {} for at {idx}",
                    a,
                    b,
                );
            });
    }

    #[test]
    fn test_mixed_radixd_21() {
        let src: [f64; 21] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.1, 4.1, 5.4,
            1.4, 12.5, 6.2, 7.2,
        ];

        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(21, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let mut input_c = reference_value[..21 / 2 + 1].to_vec();
        for w in input_c.iter_mut() {
            *w = *w / 21.0;
        }

        let local_r2c =
            NeonC2RMixedRadix3d::new(C2rChild::Real(crate::r2c::strategy_c2r(7).unwrap())).unwrap();
        let mut complex_output = vec![f64::zero(); 21];
        local_r2c.execute(&input_c, &mut complex_output).unwrap();
        println!("complex_output: {:?}", complex_output);

        src.iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_re {} != b_re {} for at {idx}",
                    a,
                    b,
                );
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_im {} != b_im {} for at {idx}",
                    a,
                    b,
                );
            });
    }

    #[test]
    fn test_mixed_radixd_21_7() {
        let src: [f64; 21] = [
            7.2, 6.2, 6.4, 7.9, 1.3, 5.6, 2.6, 6.4, 7.4, 3.4, 2.1, 3.2, 3.3, 9.8, 5.1, 4.1, 5.4,
            1.4, 12.5, 6.2, 7.2,
        ];

        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(21, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let mut input_c = reference_value[..21 / 2 + 1].to_vec();
        for w in input_c.iter_mut() {
            *w = *w / 21.0;
        }

        let local_r2c =
            NeonC2RMixedRadix7d::new(C2rChild::Real(crate::r2c::strategy_c2r(3).unwrap())).unwrap();
        let mut complex_output = vec![f64::zero(); 21];
        local_r2c.execute(&input_c, &mut complex_output).unwrap();
        println!("complex_output: {:?}", complex_output);

        src.iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_re {} != b_re {} for at {idx}",
                    a,
                    b,
                );
                assert!(
                    (a - b).abs() < 1e-3,
                    "a_im {} != b_im {} for at {idx}",
                    a,
                    b,
                );
            });
    }
}
