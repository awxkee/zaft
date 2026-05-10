/*
 * // Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //2028
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
use crate::prime_factors::{PrimeFactors, primitive_root};
use crate::spectrum_arithmetic::ComplexArith;
use crate::util::{compute_twiddle, validate_oof_sizes, validate_scratch};
use crate::{FftDirection, FftExecutor, FftSample, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::arch::x86_64::*;
use std::sync::Arc;

pub(crate) struct AvxRadersFft<T> {
    convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
    convolve_fft_twiddles: Vec<Complex<T>>,
    execution_length: usize,
    direction: FftDirection,
    input_indices: Vec<u32>,
    output_indices: Vec<u32>,
    spectrum_ops: Arc<dyn ComplexArith<T> + Send + Sync>,
    convolve_fft_scratch_length: usize,
    indicer: Arc<dyn RadersIndicer<T> + Send + Sync>,
}

pub(crate) trait RadersIndicer<T> {
    unsafe fn index_inputs(
        &self,
        buffer: &[Complex<T>],
        output: &mut [Complex<T>],
        indices: &[u32],
    );
    unsafe fn output_indices(
        &self,
        buffer: &mut [Complex<T>],
        scratch: &[Complex<T>],
        indices: &[u32],
    );
}

pub(crate) struct AvxRadersIndicer;

pub(crate) trait AvxRadersFactory<T> {
    fn make_raders_indicer() -> Arc<dyn RadersIndicer<T> + Send + Sync>;
}

impl AvxRadersFactory<f32> for f32 {
    fn make_raders_indicer() -> Arc<dyn RadersIndicer<f32> + Send + Sync> {
        Arc::new(AvxRadersIndicer)
    }
}

impl AvxRadersFactory<f64> for f64 {
    fn make_raders_indicer() -> Arc<dyn RadersIndicer<f64> + Send + Sync> {
        Arc::new(AvxRadersIndicer)
    }
}

impl LutGather<f32> for AvxRadersIndicer {
    fn gather(&self, source: &[Complex<f32>], destination: &mut [Complex<f32>], lut: &[u32]) {
        unsafe {
            self.index_inputs(source, destination, lut);
        }
    }
}

impl LutGather<f64> for AvxRadersIndicer {
    fn gather(&self, source: &[Complex<f64>], destination: &mut [Complex<f64>], lut: &[u32]) {
        unsafe {
            self.index_inputs(source, destination, lut);
        }
    }
}

impl RadersIndicer<f32> for AvxRadersIndicer {
    #[target_feature(enable = "avx2")]
    unsafe fn index_inputs(
        &self,
        buffer: &[Complex<f32>],
        output: &mut [Complex<f32>],
        indices: &[u32],
    ) {
        unsafe {
            let one = _mm256_set1_epi32(1); // [1, 1, 1, 1]

            for (scratch_element, buffer_idx) in output
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<8>().0.iter())
            {
                let all_indices =
                    _mm256_slli_epi32::<1>(_mm256_loadu_si256(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm256_add_epi32(all_indices, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                let r0 = _mm256_unpacklo_epi32(all_indices, idx_plus_one);
                let r1 = _mm256_unpackhi_epi32(all_indices, idx_plus_one);
                let xy0 = _mm256_permute2f128_si256::<32>(r0, r1);
                let xy1 = _mm256_permute2f128_si256::<49>(r0, r1);

                let v0 = _mm256_i32gather_ps::<4>(buffer.as_ptr().cast(), xy0);
                let v1 = _mm256_i32gather_ps::<4>(buffer.as_ptr().cast(), xy1);

                _mm256_storeu_ps(scratch_element.as_mut_ptr().cast(), v0);
                _mm256_storeu_ps(scratch_element[4..].as_mut_ptr().cast(), v1);
            }

            let rem = output.as_chunks_mut::<8>().1;
            let rem_indices = indices.as_chunks::<8>().1;

            for (scratch_element, buffer_idx) in rem
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(rem_indices.as_chunks::<4>().0.iter())
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, _mm256_castsi256_si128(one)); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_i32gather_ps::<4>(
                    buffer.as_ptr().cast(),
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(idx0), idx1),
                );

                _mm256_storeu_ps(scratch_element.as_mut_ptr().cast(), v0);
            }

            let rem = rem.as_chunks_mut::<4>().1;
            let rem_indices = rem_indices.as_chunks::<4>().1;

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *scratch_element = *buffer.get_unchecked(buffer_idx as usize);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn output_indices(
        &self,
        buffer: &mut [Complex<f32>],
        scratch: &[Complex<f32>],
        indices: &[u32],
    ) {
        unsafe {
            let one = _mm256_set1_epi32(1); // [1, 1, 1, 1]
            let conj_factors =
                _mm256_loadu_ps([0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0].as_ptr());

            for (scratch_element, buffer_idx) in buffer
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<8>().0.iter())
            {
                let all_indices =
                    _mm256_slli_epi32::<1>(_mm256_loadu_si256(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm256_add_epi32(all_indices, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                let r0 = _mm256_unpacklo_epi32(all_indices, idx_plus_one);
                let r1 = _mm256_unpackhi_epi32(all_indices, idx_plus_one);
                let xy0 = _mm256_permute2f128_si256::<32>(r0, r1);
                let xy1 = _mm256_permute2f128_si256::<49>(r0, r1);

                let u0 = _mm256_i32gather_ps::<4>(scratch.as_ptr().cast(), xy0);
                let u1 = _mm256_i32gather_ps::<4>(scratch.as_ptr().cast(), xy1);

                let v0 = _mm256_xor_ps(u0, conj_factors);
                let v1 = _mm256_xor_ps(u1, conj_factors);

                _mm256_storeu_ps(scratch_element.as_mut_ptr().cast(), v0);
                _mm256_storeu_ps(scratch_element[4..].as_mut_ptr().cast(), v1);
            }

            let rem = buffer.as_chunks_mut::<8>().1;
            let rem_indices = indices.as_chunks::<8>().1;

            for (scratch_element, buffer_idx) in rem
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(rem_indices.as_chunks::<4>().0.iter())
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, _mm256_castsi256_si128(one)); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_xor_ps(
                    _mm256_i32gather_ps::<4>(
                        scratch.as_ptr().cast(),
                        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(idx0), idx1),
                    ),
                    conj_factors,
                );

                _mm256_storeu_ps(scratch_element.as_mut_ptr().cast(), v0);
            }

            let rem = rem.as_chunks_mut::<4>().1;
            let rem_indices = rem_indices.as_chunks::<4>().1;

            for (dst, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *dst = scratch.get_unchecked(buffer_idx as usize).conj();
            }
        }
    }
}

impl RadersIndicer<f64> for AvxRadersIndicer {
    #[target_feature(enable = "avx2")]
    unsafe fn index_inputs(
        &self,
        buffer: &[Complex<f64>],
        output: &mut [Complex<f64>],
        indices: &[u32],
    ) {
        unsafe {
            let one = _mm_set1_epi32(1); // [1, 1, 1, 1]

            for (scratch_element, buffer_idx) in output
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<4>().0.iter())
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_i32gather_pd::<8>(buffer.as_ptr().cast(), idx0);
                let v1 = _mm256_i32gather_pd::<8>(buffer.as_ptr().cast(), idx1);

                _mm256_storeu_pd(scratch_element.as_mut_ptr().cast(), v0);
                _mm256_storeu_pd(scratch_element[2..].as_mut_ptr().cast(), v1);
            }

            let rem = output.as_chunks_mut::<4>().1;
            let rem_indices = indices.as_chunks::<4>().1;

            for (scratch_element, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *scratch_element = *buffer.get_unchecked(buffer_idx as usize);
            }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn output_indices(
        &self,
        buffer: &mut [Complex<f64>],
        scratch: &[Complex<f64>],
        indices: &[u32],
    ) {
        unsafe {
            let one = _mm_set1_epi32(1); // [1, 1, 1, 1]
            let conj_factors = _mm256_loadu_pd([0.0, -0.0, 0.0, -0.0].as_ptr());

            for (scratch_element, buffer_idx) in buffer
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(indices.as_chunks::<4>().0.iter())
            {
                let idx = _mm_slli_epi32::<1>(_mm_loadu_si128(buffer_idx.as_ptr().cast()));

                let idx_plus_one = _mm_add_epi32(idx, one); // [idx0+1, idx1+1, idx2+1, idx3+1]

                // Interleave: [idx0, idx0+1, idx1, idx1+1]
                let idx0 = _mm_unpacklo_epi32(idx, idx_plus_one); // low 2 elements

                // Interleave: [idx2, idx2+1, idx3, idx3+1]
                let idx1 = _mm_unpackhi_epi32(idx, idx_plus_one); // high 2 elements

                let v0 = _mm256_xor_pd(
                    _mm256_i32gather_pd::<8>(scratch.as_ptr().cast(), idx0),
                    conj_factors,
                );
                let v1 = _mm256_xor_pd(
                    _mm256_i32gather_pd::<8>(scratch.as_ptr().cast(), idx1),
                    conj_factors,
                );

                _mm256_storeu_pd(scratch_element.as_mut_ptr().cast(), v0);
                _mm256_storeu_pd(
                    scratch_element.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    v1,
                );
            }

            let rem = buffer.as_chunks_mut::<4>().1;
            let rem_indices = indices.as_chunks::<4>().1;

            for (dst, &buffer_idx) in rem.iter_mut().zip(rem_indices.iter()) {
                *dst = scratch.get_unchecked(buffer_idx as usize).conj();
            }
        }
    }
}

impl<T: FftSample + AvxRadersFactory<T>> AvxRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    pub(crate) fn new(
        size: usize,
        convolve_fft: Arc<dyn FftExecutor<T> + Send + Sync>,
        fft_direction: FftDirection,
    ) -> Result<AvxRadersFft<T>, ZaftError> {
        assert!(
            PrimeFactors::from_number(size as u64).is_prime(),
            "Input length for Rader's must be a prime number"
        );

        let direction = convolve_fft.direction();
        let convolve_fft_len = convolve_fft.length();
        assert_eq!(fft_direction, direction);
        let dividing_len = DividerU64::new(size as u64);

        let primitive_root =
            primitive_root(size as u64).ok_or(ZaftError::CantFindPrimitiveRootFor(size as u64))?;

        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(size as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + size as i64
        } as u64;

        let inner_fft_scale: T = (1f64 / convolve_fft_len as f64).as_();
        let mut inner_fft_input = try_vec![Complex::zero(); convolve_fft_len];
        let mut twiddle_input = 1;
        for dst in &mut inner_fft_input {
            let twiddle = compute_twiddle(twiddle_input, size, direction);
            *dst = twiddle * inner_fft_scale;

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
        let output_indices = (0..size - 1)
            .map(|_| {
                output_index =
                    ((output_index as u64 * primitive_root_inverse) % dividing_len) as usize;
                (output_index - 1) as u32
            })
            .collect::<Vec<_>>();

        let mut z_output = try_vec![0u32; size - 1];
        for (input_idx, &output_idx) in output_indices.iter().enumerate() {
            z_output[output_idx as usize] = input_idx as u32;
        }

        let convolve_fft_scratch = convolve_fft.scratch_length();

        Ok(AvxRadersFft {
            execution_length: size,
            convolve_fft,
            input_indices,
            output_indices: z_output,
            convolve_fft_twiddles: inner_fft_input,
            direction: fft_direction,
            spectrum_ops: T::make_complex_arith(),
            convolve_fft_scratch_length: convolve_fft_scratch,
            indicer: T::make_raders_indicer(),
        })
    }
}

impl<T: FftSample> AvxRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_impl(
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

            unsafe {
                self.indicer
                    .index_inputs(buffer, scratch, &self.input_indices);
            }

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            *buffer_first = *buffer_first + scratch[0];

            self.spectrum_ops
                .mul_conjugate_in_place(scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            unsafe {
                self.indicer
                    .output_indices(buffer, scratch, &self.output_indices);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn execute_oof_impl(
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

            unsafe {
                self.indicer
                    .index_inputs(buffer, scratch, &self.input_indices);
            }

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            unsafe {
                *output_chunk.get_unchecked_mut(0) = *buffer_first + scratch[0];
            }

            self.spectrum_ops
                .mul_conjugate_in_place(scratch, &self.convolve_fft_twiddles);

            scratch[0] = scratch[0] + buffer_first_val.conj();

            self.convolve_fft
                .execute_with_scratch(scratch, convolve_scratch)?;

            let (_, buffer) = output_chunk.split_first_mut().unwrap();

            unsafe {
                self.indicer
                    .output_indices(buffer, scratch, &self.output_indices);
            }
        }
        Ok(())
    }
}

impl<T: FftSample> FftExecutor<T> for AvxRadersFft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        let mut scratch = vec![Complex::zero(); self.scratch_length()];
        unsafe { self.execute_impl(in_place, scratch.as_mut_slice()) }
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_impl(in_place, scratch) }
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        let mut scratch = vec![Complex::zero(); self.out_of_place_scratch_length()];
        self.execute_out_of_place_with_scratch(src, dst, scratch.as_mut_slice())
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_oof_impl(src, dst, scratch) }
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
        self.execution_length + self.convolve_fft_scratch_length
    }

    #[inline]
    fn out_of_place_scratch_length(&self) -> usize {
        self.scratch_length()
    }

    fn destructive_scratch_length(&self) -> usize {
        self.scratch_length()
    }
}
