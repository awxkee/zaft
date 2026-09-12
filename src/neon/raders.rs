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
use crate::err::try_vec;
use crate::fast_divider::DividerU64;
use crate::good_thomas_small::LutGather;
use crate::neon::util::{conj_f64, conjq_f32};
use crate::prime_factors::{PrimeFactors, primitive_root};
use crate::spectrum_arithmetic::ComplexArith;
use crate::util::{compute_twiddle, validate_oof_sizes, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::arch::aarch64::*;
use std::sync::Arc;

pub(crate) struct NeonRadersFft<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    input_indices: Vec<u32>,
    output_indices: Vec<u32>,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
    // Extra scratch is only needed when the inner FFT cannot use the caller's buffer.
    extra_scratch_length: usize,
    indicer: Arc<dyn RadersIndicer<T> + Send + Sync>,
}

pub(crate) trait RadersIndicer<T> {
    fn index_inputs(&self, buffer: &[Complex<T>], output: &mut [Complex<T>], indices: &[u32]);
    fn output_indices(&self, buffer: &mut [Complex<T>], scratch: &[Complex<T>], indices: &[u32]);
}

pub(crate) struct NeonRadersIndicer;

impl LutGather<f32> for NeonRadersIndicer {
    fn gather(&self, source: &[Complex<f32>], destination: &mut [Complex<f32>], lut: &[u32]) {
        self.index_inputs(source, destination, lut)
    }
}

impl LutGather<f64> for NeonRadersIndicer {
    fn gather(&self, source: &[Complex<f64>], destination: &mut [Complex<f64>], lut: &[u32]) {
        self.index_inputs(source, destination, lut)
    }
}

impl RadersIndicer<f32> for NeonRadersIndicer {
    fn index_inputs(&self, buffer: &[Complex<f32>], output: &mut [Complex<f32>], indices: &[u32]) {
        unsafe {
            for (scratch_element, buffer_idx) in output
                .as_chunks_mut::<6>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<6>().0.iter())
            {
                let idx0 = buffer_idx[0] as usize;
                let idx1 = buffer_idx[1] as usize;

                let v0 = vld1_f32(buffer.get_unchecked(idx0..).as_ptr().cast());
                let v1 = vld1_f32(buffer.get_unchecked(idx1..).as_ptr().cast());

                let idx2 = buffer_idx[2] as usize;
                let idx3 = buffer_idx[3] as usize;

                let v2 = vld1_f32(buffer.get_unchecked(idx2..).as_ptr().cast());
                let v3 = vld1_f32(buffer.get_unchecked(idx3..).as_ptr().cast());

                let idx4 = buffer_idx[4] as usize;
                let idx5 = buffer_idx[5] as usize;

                let v4 = vld1_f32(buffer.get_unchecked(idx4..).as_ptr().cast());
                let v5 = vld1_f32(buffer.get_unchecked(idx5..).as_ptr().cast());

                vst1_f32(scratch_element.as_mut_ptr().cast(), v0);
                vst1_f32(
                    scratch_element.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    v1,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    v2,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    v3,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    v4,
                );
                vst1_f32(
                    scratch_element.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    v5,
                );
            }

            let rem = output.as_chunks_mut::<6>().1;
            let rem_indices = indices.as_chunks::<6>().1;

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                let v0 = vld1_f32(buffer.get_unchecked(buffer_idx as usize..).as_ptr().cast());
                vst1_f32(scratch_element as *mut Complex<f32> as *mut f32, v0);
            }
        }
    }

    fn output_indices(
        &self,
        buffer: &mut [Complex<f32>],
        scratch: &[Complex<f32>],
        indices: &[u32],
    ) {
        unsafe {
            static CONJ: [f32; 4] = [0.0, -0.0, 0.0, -0.0];
            let conj = vld1q_f32(CONJ.as_ptr());
            for (dst, buffer_idx) in buffer
                .as_chunks_mut::<6>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<6>().0.iter())
            {
                let idx0 = buffer_idx[0] as usize;
                let idx1 = buffer_idx[1] as usize;
                let idx2 = buffer_idx[2] as usize;
                let idx3 = buffer_idx[3] as usize;
                let idx4 = buffer_idx[4] as usize;
                let idx5 = buffer_idx[5] as usize;

                let mut v0 = vcombine_f32(
                    vld1_f32(scratch.get_unchecked(idx0..).as_ptr().cast()),
                    vld1_f32(scratch.get_unchecked(idx1..).as_ptr().cast()),
                );
                let mut v1 = vcombine_f32(
                    vld1_f32(scratch.get_unchecked(idx2..).as_ptr().cast()),
                    vld1_f32(scratch.get_unchecked(idx3..).as_ptr().cast()),
                );
                let mut v2 = vcombine_f32(
                    vld1_f32(scratch.get_unchecked(idx4..).as_ptr().cast()),
                    vld1_f32(scratch.get_unchecked(idx5..).as_ptr().cast()),
                );

                v0 = conjq_f32(v0, conj);
                v1 = conjq_f32(v1, conj);
                v2 = conjq_f32(v2, conj);

                vst1q_f32(dst.as_mut_ptr().cast(), v0);
                vst1q_f32(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), v1);
                vst1q_f32(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), v2);
            }

            let rem = buffer.as_chunks_mut::<6>().1;
            let rem_indices = indices.as_chunks::<6>().1;

            for (dst, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *dst = scratch.get_unchecked(buffer_idx as usize).conj();
            }
        }
    }
}

impl RadersIndicer<f64> for NeonRadersIndicer {
    fn index_inputs(&self, buffer: &[Complex<f64>], output: &mut [Complex<f64>], indices: &[u32]) {
        unsafe {
            for (scratch_element, buffer_idx) in output
                .as_chunks_mut::<6>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<6>().0.iter())
            {
                let idx0 = buffer_idx[0] as usize;
                let idx1 = buffer_idx[1] as usize;

                let v0 = vld1q_f64(buffer.get_unchecked(idx0..).as_ptr().cast());
                let v1 = vld1q_f64(buffer.get_unchecked(idx1..).as_ptr().cast());

                let idx2 = buffer_idx[2] as usize;
                let idx3 = buffer_idx[3] as usize;

                let v2 = vld1q_f64(buffer.get_unchecked(idx2..).as_ptr().cast());
                let v3 = vld1q_f64(buffer.get_unchecked(idx3..).as_ptr().cast());

                let idx4 = buffer_idx[4] as usize;
                let idx5 = buffer_idx[5] as usize;

                let v4 = vld1q_f64(buffer.get_unchecked(idx4..).as_ptr().cast());
                let v5 = vld1q_f64(buffer.get_unchecked(idx5..).as_ptr().cast());

                vst1q_f64(scratch_element.as_mut_ptr().cast(), v0);
                vst1q_f64(
                    scratch_element.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    v1,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    v2,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    v3,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    v4,
                );
                vst1q_f64(
                    scratch_element.get_unchecked_mut(5..).as_mut_ptr().cast(),
                    v5,
                );
            }

            let rem = output.as_chunks_mut::<6>().1;
            let rem_indices = indices.as_chunks::<6>().1;

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                let v0 = vld1q_f64(buffer.get_unchecked(buffer_idx as usize..).as_ptr().cast());
                vst1q_f64(scratch_element as *mut Complex<f64> as *mut f64, v0);
            }
        }
    }

    fn output_indices(
        &self,
        buffer: &mut [Complex<f64>],
        scratch: &[Complex<f64>],
        indices: &[u32],
    ) {
        unsafe {
            static CONJ: [f64; 2] = [0.0, -0.0];
            let conj = vld1q_f64(CONJ.as_ptr());
            for (dst, buffer_idx) in buffer
                .as_chunks_mut::<6>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<6>().0.iter())
            {
                let idx0 = buffer_idx[0] as usize;
                let idx1 = buffer_idx[1] as usize;

                let mut v0 = vld1q_f64(scratch.get_unchecked(idx0..).as_ptr().cast());
                let mut v1 = vld1q_f64(scratch.get_unchecked(idx1..).as_ptr().cast());

                v0 = conj_f64(v0, conj);
                v1 = conj_f64(v1, conj);

                let idx2 = buffer_idx[2] as usize;
                let idx3 = buffer_idx[3] as usize;

                let mut v2 = vld1q_f64(scratch.get_unchecked(idx2..).as_ptr().cast());
                let mut v3 = vld1q_f64(scratch.get_unchecked(idx3..).as_ptr().cast());

                let idx4 = buffer_idx[4] as usize;
                let idx5 = buffer_idx[5] as usize;

                let mut v4 = vld1q_f64(scratch.get_unchecked(idx4..).as_ptr().cast());
                let mut v5 = vld1q_f64(scratch.get_unchecked(idx5..).as_ptr().cast());

                v2 = conj_f64(v2, conj);
                v3 = conj_f64(v3, conj);

                v4 = conj_f64(v4, conj);
                v5 = conj_f64(v5, conj);

                vst1q_f64(dst.as_mut_ptr().cast(), v0);
                vst1q_f64(dst.get_unchecked_mut(1..).as_mut_ptr().cast(), v1);
                vst1q_f64(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), v2);
                vst1q_f64(dst.get_unchecked_mut(3..).as_mut_ptr().cast(), v3);
                vst1q_f64(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), v4);
                vst1q_f64(dst.get_unchecked_mut(5..).as_mut_ptr().cast(), v5);
            }

            let rem = buffer.as_chunks_mut::<6>().1;
            let rem_indices = indices.as_chunks::<6>().1;

            for (dst, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *dst = scratch.get_unchecked(buffer_idx as usize).conj();
            }
        }
    }
}

impl<T: FftSample> NeonRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
        indicer: Arc<dyn RadersIndicer<T> + Send + Sync>,
    ) -> Result<NeonRadersFft<T>, ZaftError> {
        assert!(
            PrimeFactors::from_number(size as u64).is_prime(),
            "Input length for Rader's must be a prime number"
        );

        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let dividing_len = DividerU64::new(size as u64);

        // compute the primitive root and its inverse for this size
        let primitive_root =
            primitive_root(size as u64).ok_or(ZaftError::CantFindPrimitiveRootFor(size as u64))?;

        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(size as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + size as i64
        } as u64;

        // precompute the coefficients to use inside the process method
        let inner_fft_scale: T = (1f64 / convolve_fft_len as f64).as_();
        let mut inner_fft_input = try_vec![Complex::zero(); convolve_fft_len];
        let (first_half, second_half) = inner_fft_input.split_at_mut(convolve_fft_len / 2);
        let mut twiddle_input = 1;
        // For H = (size - 1) / 2, g^H = -1 mod size, so kernel[q + H] = conj(kernel[q]).
        for (dst, conjugate_dst) in first_half.iter_mut().zip(second_half) {
            let twiddle = compute_twiddle(twiddle_input, size, direction) * inner_fft_scale;
            *dst = twiddle;
            *conjugate_dst = twiddle.conj();

            twiddle_input =
                ((twiddle_input as u64 * primitive_root_inverse) % dividing_len) as usize;
        }

        convolve_fft.execute(&mut inner_fft_input)?;

        let mut input_index = 1;
        let input_indices = (0..size - 1)
            .map(|_| {
                input_index = ((input_index as u64 * primitive_root) % dividing_len) as usize;
                (input_index - 1) as u32
            })
            .collect::<Vec<_>>();

        let mut output_index = 1;
        let mut output_indices = try_vec![0u32; size - 1];
        // Invert the output permutation so execution gathers into contiguous output slots.
        for scratch_index in 0..size - 1 {
            output_index = ((output_index as u64 * primitive_root_inverse) % dividing_len) as usize;
            output_indices[output_index - 1] = scratch_index as u32;
        }

        let inner_scratch_length = convolve_fft.scratch_length();
        let extra_scratch_length = if inner_scratch_length <= size {
            0
        } else {
            inner_scratch_length
        };

        Ok(NeonRadersFft {
            execution_length: size,
            convolve_fft,
            convolve_fft_twiddles: inner_fft_input,
            direction: fft_direction,
            input_indices,
            output_indices,
            spectrum_ops: T::make_complex_arith(),
            extra_scratch_length,
            indicer,
        })
    }
}

impl<T: FftSample> FftExecutor<T> for NeonRadersFft<T>
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
        let (scratch, convolve_scratch) = scratch.split_at_mut(self.execution_length);

        for chunk in in_place.chunks_exact_mut(self.execution_length) {
            let (buffer_first, buffer) = chunk.split_first_mut().unwrap();
            let buffer_first_val = *buffer_first;

            let (scratch, _) = scratch.split_at_mut(self.length() - 1);

            self.indicer
                .index_inputs(buffer, scratch, &self.input_indices);

            let convolve_scratch = if self.extra_scratch_length == 0 {
                &mut *chunk
            } else {
                &mut *convolve_scratch
            };
            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            // Both inner FFTs may overwrite the caller's entire buffer, including DC.
            let dc = buffer_first_val + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            let (buffer_first, buffer) = chunk.split_first_mut().unwrap();
            *buffer_first = dc;
            self.indicer
                .output_indices(buffer, scratch, &self.output_indices);
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
        let (scratch, convolve_scratch) = scratch.split_at_mut(self.execution_length);

        for (chunk, output_chunk) in src
            .chunks_exact(self.execution_length)
            .zip(dst.chunks_exact_mut(self.execution_length))
        {
            let (buffer_first, buffer) = chunk.split_first().unwrap();
            let buffer_first_val = *buffer_first;

            let (scratch, _) = scratch.split_at_mut(self.length() - 1);

            self.indicer
                .index_inputs(buffer, scratch, &self.input_indices);

            let convolve_scratch = if self.extra_scratch_length == 0 {
                &mut *output_chunk
            } else {
                &mut *convolve_scratch
            };
            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            let dc = buffer_first_val + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            let (buffer_first, buffer) = output_chunk.split_first_mut().unwrap();
            *buffer_first = dc;

            self.indicer
                .output_indices(buffer, scratch, &self.output_indices);
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place_with_scratch(src, dst, scratch)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn scratch_length(&self) -> usize {
        self.execution_length + self.extra_scratch_length
    }

    #[inline]
    fn out_of_place_scratch_length(&self) -> usize {
        self.scratch_length()
    }

    #[inline]
    fn destructive_scratch_length(&self) -> usize {
        self.out_of_place_scratch_length()
    }
}
