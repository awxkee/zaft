/*
 * // Copyright (c) Radzivon Bartoshyk 4/2026. All rights reserved.
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
use crate::neon::RadersIndicer;
use num_complex::Complex;
use std::arch::aarch64::*;

pub(crate) struct SveRadersIndicer;

impl SveRadersIndicer {
    #[target_feature(enable = "sve")]
    fn index_inputs_impl_f32(
        &self,
        buffer: &[Complex<f32>],
        output: &mut [Complex<f32>],
        indices: &[u32],
    ) {
        assert_eq!(output.len(), indices.len());
        let vl = svcntd() as usize;

        for (out, idx) in output
            .chunks_exact_mut(vl * 4)
            .zip(indices.chunks_exact(vl * 4))
        {
            let pg_all = svptrue_b64();
            let raw_idx0 = unsafe { svld1uw_u64(pg_all, idx.as_ptr()) };
            let raw_idx1 = unsafe { svld1uw_u64(pg_all, idx.get_unchecked(vl..).as_ptr()) };
            let raw_idx2 = unsafe { svld1uw_u64(pg_all, idx.get_unchecked(vl * 2..).as_ptr()) };
            let raw_idx3 = unsafe { svld1uw_u64(pg_all, idx.get_unchecked(vl * 3..).as_ptr()) };

            let gathered0 =
                unsafe { svld1_gather_u64index_u64(pg_all, buffer.as_ptr().cast(), raw_idx0) };
            let gathered1 =
                unsafe { svld1_gather_u64index_u64(pg_all, buffer.as_ptr().cast(), raw_idx1) };
            let gathered2 =
                unsafe { svld1_gather_u64index_u64(pg_all, buffer.as_ptr().cast(), raw_idx2) };
            let gathered3 =
                unsafe { svld1_gather_u64index_u64(pg_all, buffer.as_ptr().cast(), raw_idx3) };

            unsafe {
                svst1_u64(pg_all, out.as_mut_ptr().cast(), gathered0);
                svst1_u64(
                    pg_all,
                    out.get_unchecked_mut(vl..).as_mut_ptr().cast(),
                    gathered1,
                );
                svst1_u64(
                    pg_all,
                    out.get_unchecked_mut(vl * 2..).as_mut_ptr().cast(),
                    gathered2,
                );
                svst1_u64(
                    pg_all,
                    out.get_unchecked_mut(vl * 3..).as_mut_ptr().cast(),
                    gathered3,
                );
            }
        }

        let out_rem = output.chunks_exact_mut(vl).into_remainder();
        let idx_rem = indices.chunks_exact(vl).remainder();

        for (out, idx) in out_rem.chunks_exact_mut(vl).zip(idx_rem.chunks_exact(vl)) {
            let pg_all = svptrue_b64();
            let raw_idx: svuint64_t = unsafe { svld1uw_u64(pg_all, idx.as_ptr()) };

            let gathered: svuint64_t =
                unsafe { svld1_gather_u64index_u64(pg_all, buffer.as_ptr().cast(), raw_idx) };

            unsafe {
                svst1_u64(pg_all, out.as_mut_ptr().cast(), gathered);
            }
        }

        let out_rem = out_rem.chunks_exact_mut(vl).into_remainder();
        let idx_rem = idx_rem.chunks_exact(vl).remainder();

        if !out_rem.is_empty() {
            let pg_tail = svwhilelt_b64_u64(0u64, out_rem.len() as u64);
            let raw_idx: svuint64_t = unsafe { svld1uw_u64(pg_tail, idx_rem.as_ptr()) };
            let gathered: svuint64_t =
                unsafe { svld1_gather_u64index_u64(pg_tail, buffer.as_ptr().cast(), raw_idx) };
            unsafe {
                svst1_u64(pg_tail, out_rem.as_mut_ptr().cast(), gathered);
            }
        }
    }
}

impl RadersIndicer<f32> for SveRadersIndicer {
    fn index_inputs(&self, buffer: &[Complex<f32>], output: &mut [Complex<f32>], indices: &[u32]) {
        unsafe {
            self.index_inputs_impl_f32(buffer, output, indices);
        }
    }

    fn output_indices(
        &self,
        buffer: &mut [Complex<f32>],
        scratch: &[Complex<f32>],
        indices: &[u32],
    ) {
        unsafe {
            for (scratch_element, &buffer_idx) in scratch.iter().zip(indices.iter()) {
                *buffer.get_unchecked_mut(buffer_idx as usize) = scratch_element.conj();
            }
        }
    }
}

impl RadersIndicer<f64> for SveRadersIndicer {
    fn index_inputs(&self, buffer: &[Complex<f64>], output: &mut [Complex<f64>], indices: &[u32]) {
        unsafe {
            for (scratch_element, buffer_idx) in
                output.chunks_exact_mut(6).zip(indices.chunks_exact(6))
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

            let rem = output.chunks_exact_mut(6).into_remainder();
            let rem_indices = indices.chunks_exact(6).remainder();

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
            for (scratch_element, &buffer_idx) in scratch.iter().zip(indices.iter()) {
                *buffer.get_unchecked_mut(buffer_idx as usize) = scratch_element.conj();
            }
        }
    }
}
