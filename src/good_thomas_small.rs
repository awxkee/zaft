/*
 * // Copyright (c) Radzivon Bartoshyk 05/2026. All rights reserved.
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
use crate::err::try_vec;
use crate::transpose::TransposeExecutor;
use crate::util::{validate_oof_sizes, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Zero};
use std::sync::Arc;

pub(crate) trait LutGather<T> {
    fn gather(&self, source: &[Complex<T>], destination: &mut [Complex<T>], lut: &[u32]);
}

pub(crate) trait LutGatherFactory<T> {
    fn make_gatherer() -> Arc<dyn LutGather<T> + Send + Sync>;
}

#[allow(unused)]
struct DefaultLutGather<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[allow(unused)]
impl<T: Copy> LutGather<T> for DefaultLutGather<T> {
    fn gather(&self, source: &[Complex<T>], destination: &mut [Complex<T>], lut: &[u32]) {
        for (dst, &src_idx) in destination.iter_mut().zip(lut.iter()) {
            // SAFETY: input_perm is built from indices 0..n and source.len() == n.
            *dst = unsafe { *source.get_unchecked(src_idx as usize) };
        }
    }
}

impl LutGatherFactory<f32> for f32 {
    fn make_gatherer() -> Arc<dyn LutGather<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "sve"))]
        {
            if std::arch::is_aarch64_feature_detected!("sve2") {
                use crate::sve::SveLutGather;
                return Arc::new(SveLutGather);
            }
        }
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonRadersIndicer;
            Arc::new(NeonRadersIndicer)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(DefaultLutGather {
                _phantom: Default::default(),
            })
        }
    }
}

impl LutGatherFactory<f64> for f64 {
    fn make_gatherer() -> Arc<dyn LutGather<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::NeonRadersIndicer;
            Arc::new(NeonRadersIndicer)
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(DefaultLutGather {
                _phantom: Default::default(),
            })
        }
    }
}

pub(crate) struct GoodThomasSmallFft<T> {
    width: usize,
    width_size_fft: Arc<dyn FftExecutor<T> + Send + Sync>,

    height: usize,
    height_size_fft: Arc<dyn FftExecutor<T> + Send + Sync>,

    execution_length: usize,
    direction: FftDirection,
    transpose_ops: Box<dyn TransposeExecutor<T> + Send + Sync>,
    width_scratch_length: usize,
    height_scratch_length: usize,
    height_destructive_scratch: usize,
    input_permutation: Vec<u32>,
    output_permutation: Vec<u32>,
    gather: Arc<dyn LutGather<T> + Send + Sync>,
}

impl<T: FftSample> GoodThomasSmallFft<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(
        mut width_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        mut height_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    ) -> Result<GoodThomasSmallFft<T>, ZaftError> {
        assert_eq!(
            width_fft.direction(),
            height_fft.direction(),
            "width_fft and height_fft must have the same direction. got width direction={}, height direction={}",
            width_fft.direction(),
            height_fft.direction()
        );

        let mut width = width_fft.length();
        let mut height = height_fft.length();
        let direction = width_fft.direction();

        // This algorithm doesn't work if width and height aren't coprime
        let gcd = num_integer::gcd(width as i64, height as i64);
        assert_eq!(
            gcd, 1,
            "Invalid width and height for Good-Thomas Algorithm (width={width}, height={height}): Inputs must be coprime"
        );

        // The trick we're using for our index remapping will only work if width < height, so just swap them if it isn't
        if width > height {
            std::mem::swap(&mut width, &mut height);
            std::mem::swap(&mut width_fft, &mut height_fft);
        }

        let len = width.checked_mul(height).ok_or(ZaftError::Overflow)?;

        let width_scratch_length = width_fft.destructive_scratch_length();
        let height_scratch_length = height_fft.scratch_length();
        let height_destructive_scratch = height_fft.destructive_scratch_length();

        Ok(Self {
            width,
            width_size_fft: width_fft,

            height,
            height_size_fft: height_fft,

            execution_length: len,
            direction,
            transpose_ops: T::transpose_strategy(width, height),
            width_scratch_length,
            height_scratch_length,
            height_destructive_scratch,
            input_permutation: build_input_permutation(width, height)?,
            output_permutation: build_output_permutation(width, height)?,
            gather: T::make_gatherer(),
        })
    }

    // fn generate_code(&self) -> String {
    //     let n = self.execution_length;
    //     let n1 = self.width;
    //     let n2 = self.height;
    //     let mut code = String::new();
    //
    //     // Precompute output_perm inverse: for each (row,col) in result grid,
    //     // what sequential output index does it map to?
    //     // output_perm[i] = src means chunk[i] = r{src/n1}_{src%n1}
    //     // We need the reverse: given r{row}_{col}, which chunk[i] does it write to?
    //     let mut grid_to_dst = vec![0usize; n];
    //     for i in 0..n {
    //         let src = self.output_permutation[i] as usize;
    //         grid_to_dst[src] = i;
    //     }
    //
    //     // bf_n1 — load directly from chunk via input_perm, no staging
    //     for row in 0..n2 {
    //         let base = row * n1;
    //         let inputs: Vec<String> = (0..n1)
    //             .map(|col| format!("chunk[{}]", self.input_permutation[base + col]))
    //             .collect();
    //         let outputs: Vec<String> = (0..n1).map(|col| format!("t{}_{}", row, col)).collect();
    //         code.push_str(&format!(
    //             "\t\tlet ({}) = self.bf{n1}.bf{n1}({});\n",
    //             outputs.join(", "),
    //             inputs.join(", ")
    //         ));
    //     }
    //     code.push('\n');
    //
    //     // bf_n2 — compute each column and store outputs immediately
    //     for col in 0..n1 {
    //         let inputs: Vec<String> = (0..n2).map(|row| format!("t{}_{}", row, col)).collect();
    //         let outputs: Vec<String> = (0..n2).map(|row| format!("r{}_{}", row, col)).collect();
    //         code.push_str(&format!(
    //             "\t\tlet ({}) = self.bf{n2}.bf{n2}({});\n",
    //             outputs.join(", "),
    //             inputs.join(", ")
    //         ));
    //         // Store immediately
    //         for row in 0..n2 {
    //             let flat = col * n2 + row;
    //             let dst = grid_to_dst[flat];
    //             code.push_str(&format!("\t\tchunk[{dst}] = r{row}_{col};\n"));
    //         }
    //         code.push('\n');
    //     }
    //
    //     code
    // }
}

fn build_input_permutation(width: usize, height: usize) -> Result<Vec<u32>, ZaftError> {
    let n = width * height;
    let mut perm = try_vec![0u32; n];
    let mut destination_index = 0usize;
    for row in 0..height {
        let row_start = row * width;
        let increments_until_cycle = 1 + (n - destination_index) / (width + 1);
        let mut src_col = 0usize;
        if increments_until_cycle < width {
            for c in 0..increments_until_cycle {
                perm[destination_index] = (row_start + c) as u32;
                destination_index += width + 1;
            }
            src_col = increments_until_cycle;
            destination_index -= n;
        }
        for c in src_col..width {
            perm[destination_index] = (row_start + c) as u32;
            destination_index += width + 1;
        }
        destination_index -= width;
    }
    Ok(perm)
}

fn build_output_permutation(width: usize, height: usize) -> Result<Vec<u32>, ZaftError> {
    let n = width * height;
    let mut perm = try_vec![0u32; n];
    for y in 0..width {
        let src_base = y * height;
        let yh = y * height;
        let quotient = yh / width;
        let remainder = yh % width;
        let mut destination_index = remainder;
        let start_x = height - quotient;
        for x in start_x..height {
            perm[destination_index] = (src_base + x) as u32;
            destination_index += width;
        }
        for x in 0..start_x {
            perm[destination_index] = (src_base + x) as u32;
            destination_index += width;
        }
    }
    Ok(perm)
}

impl<T: FftSample> FftExecutor<T> for GoodThomasSmallFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.scratch_length()];
        self.execute_with_scratch(in_place, scratch.as_mut_slice())
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.execution_length) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.execution_length,
            ));
        }

        let scratch = validate_scratch!(scratch, self.scratch_length());
        let (scratch_left, sr) = scratch.split_at_mut(self.execution_length);

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            self.gather
                .gather(chunk, scratch_left, self.input_permutation.as_slice());

            let (width_scratch, _) = sr.split_at_mut(self.width_scratch_length);

            self.width_size_fft.execute_destructive_with_scratch(
                scratch_left,
                chunk,
                width_scratch,
            )?;

            self.transpose_ops
                .transpose(chunk, scratch_left, self.width, self.height);

            let (height_scratch, _) = sr.split_at_mut(self.height_scratch_length);
            self.height_size_fft
                .execute_with_scratch(scratch_left, height_scratch)?;

            self.gather
                .gather(scratch_left, chunk, self.output_permutation.as_slice());
        }
        Ok(())
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        let mut scratch = try_vec![Complex::zero(); self.out_of_place_scratch_length()];
        self.execute_out_of_place_with_scratch(src, dst, scratch.as_mut_slice())
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        let scratch = validate_scratch!(scratch, self.out_of_place_scratch_length());
        let (scratch_left, sr) = scratch.split_at_mut(self.execution_length);

        for (chunk, output_chunk) in src
            .chunks_exact(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            self.gather
                .gather(chunk, scratch_left, self.input_permutation.as_slice());

            let (width_scratch, _) = sr.split_at_mut(self.width_scratch_length);

            self.width_size_fft.execute_destructive_with_scratch(
                scratch_left,
                output_chunk,
                width_scratch,
            )?;

            self.transpose_ops
                .transpose(output_chunk, scratch_left, self.width, self.height);

            let (height_scratch, _) = sr.split_at_mut(self.height_scratch_length);
            self.height_size_fft
                .execute_with_scratch(scratch_left, height_scratch)?;

            self.gather.gather(
                scratch_left,
                output_chunk,
                self.output_permutation.as_slice(),
            );
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        validate_oof_sizes!(src, dst, self.execution_length);

        let scratch = validate_scratch!(scratch, self.destructive_scratch_length());

        for (src_chunk, output_chunk) in src
            .chunks_exact_mut(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            self.gather
                .gather(src_chunk, output_chunk, self.input_permutation.as_slice());

            let (width_scratch, _) = scratch.split_at_mut(self.width_scratch_length);

            self.width_size_fft.execute_destructive_with_scratch(
                output_chunk,
                src_chunk,
                width_scratch,
            )?;

            self.transpose_ops
                .transpose(src_chunk, output_chunk, self.width, self.height);

            let (height_scratch, _) = scratch.split_at_mut(self.height_destructive_scratch);
            self.height_size_fft.execute_destructive_with_scratch(
                output_chunk,
                src_chunk,
                height_scratch,
            )?;

            self.gather
                .gather(src_chunk, output_chunk, self.output_permutation.as_slice());
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_length(&self) -> usize {
        self.execution_length + self.width_scratch_length.max(self.height_scratch_length)
    }

    #[inline]
    fn out_of_place_scratch_length(&self) -> usize {
        self.scratch_length()
    }

    #[inline]
    fn destructive_scratch_length(&self) -> usize {
        self.width_scratch_length
            .max(self.height_destructive_scratch)
    }
}

#[cfg(test)]
mod tests {
    use crate::dft::Dft;
    use crate::good_thomas_small::GoodThomasSmallFft;
    use crate::{FftDirection, FftExecutor, Zaft};
    use num_complex::Complex;

    #[test]
    fn test_mixed_radixd() {
        let src: [Complex<f64>; 45] = [
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(1.3, 1.6),
            Complex::new(1.7, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
            Complex::new(-0.45, -0.4),
            Complex::new(0.45, -0.4),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.45, -0.4),
            Complex::new(8.2, -0.1),
            Complex::new(0.9, 0.13),
            Complex::new(3.25, 2.7),
            Complex::new(0.654, 0.324),
        ];
        let good_thomas20 = GoodThomasSmallFft::new(
            Zaft::strategy(5, FftDirection::Forward).unwrap(),
            Zaft::strategy(9, FftDirection::Forward).unwrap(),
        )
        .unwrap();
        // let code = good_thomas20.generate_code();
        // println!("{}", code);
        let mx = Dft::new(45, FftDirection::Forward).unwrap();
        let mut reference_value = src.to_vec();
        good_thomas20.execute(&mut reference_value).unwrap();
        let mut test_value = src.to_vec();
        mx.execute(&mut test_value).unwrap();
        reference_value
            .iter()
            .zip(test_value.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for at {idx}",
                    a.im,
                    b.im,
                );
            });
    }
}
