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
    #[target_feature(enable = "sve,sve2")]
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

        let out_rem = output.chunks_exact_mut(vl * 4).into_remainder();
        let idx_rem = indices.chunks_exact(vl * 4).remainder();

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

    #[target_feature(enable = "sve,sve2")]
    fn output_indices_f32(
        &self,
        buffer: &mut [Complex<f32>],
        scratch: &[Complex<f32>],
        indices: &[u32],
    ) {
        let vl = svcntd() as usize;

        let conj_mask_val: u64 = 0x8000_0000_0000_0000;

        let conj_mask = svdup_n_u64(conj_mask_val);

        for (src, idx) in scratch
            .chunks_exact(vl * 4)
            .zip(indices.chunks_exact(vl * 4))
        {
            let pg = svptrue_b64();
            let mut v0 = unsafe { svld1_u64(pg, src.as_ptr().cast()) };
            let mut v1 = unsafe { svld1_u64(pg, src.get_unchecked(vl..).as_ptr().cast()) };
            let mut v2 = unsafe { svld1_u64(pg, src.get_unchecked(vl * 2..).as_ptr().cast()) };
            let mut v3 = unsafe { svld1_u64(pg, src.get_unchecked(vl * 3..).as_ptr().cast()) };

            v0 = sveor_u64_m(pg, v0, conj_mask);
            v1 = sveor_u64_m(pg, v1, conj_mask);
            v2 = sveor_u64_m(pg, v2, conj_mask);
            v3 = sveor_u64_m(pg, v3, conj_mask);

            let i0 = unsafe { svld1uw_u64(pg, idx.as_ptr()) };
            let i1 = unsafe { svld1uw_u64(pg, idx.get_unchecked(vl..).as_ptr()) };
            let i2 = unsafe { svld1uw_u64(pg, idx.get_unchecked(vl * 2..).as_ptr()) };
            let i3 = unsafe { svld1uw_u64(pg, idx.get_unchecked(vl * 3..).as_ptr()) };

            unsafe {
                svst1_scatter_u64index_u64(pg, buffer.as_mut_ptr().cast(), i0, v0);
                svst1_scatter_u64index_u64(pg, buffer.as_mut_ptr().cast(), i1, v1);
                svst1_scatter_u64index_u64(pg, buffer.as_mut_ptr().cast(), i2, v2);
                svst1_scatter_u64index_u64(pg, buffer.as_mut_ptr().cast(), i3, v3);
            }
        }

        let src_rem = scratch.chunks_exact(vl * 4).remainder();
        let idx_rem = indices.chunks_exact(vl * 4).remainder();

        for (src, idx) in src_rem.chunks_exact(vl).zip(idx_rem.chunks_exact(vl)) {
            let pg = svptrue_b64();
            let mut v0 = unsafe { svld1_u64(pg, src.as_ptr().cast()) };

            v0 = sveor_u64_m(pg, v0, conj_mask);

            let i0 = unsafe { svld1uw_u64(pg, idx.as_ptr()) };

            unsafe {
                svst1_scatter_u64index_u64(pg, buffer.as_mut_ptr().cast(), i0, v0);
            }
        }

        let src_tail = src_rem.chunks_exact(vl).remainder();
        let idx_tail = idx_rem.chunks_exact(vl).remainder();

        if !src_tail.is_empty() {
            let pg = svwhilelt_b64_u64(0u64, src_tail.len() as u64);
            let mut v0 = unsafe { svld1_u64(pg, src_tail.as_ptr().cast()) };
            v0 = sveor_u64_m(pg, v0, conj_mask);
            let i = unsafe { svld1uw_u64(pg, idx_tail.as_ptr()) };
            unsafe {
                svst1_scatter_u64index_u64(pg, buffer.as_mut_ptr().cast(), i, v0);
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
            self.output_indices_f32(buffer, scratch, indices);
        }
    }
}
