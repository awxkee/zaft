/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
use crate::r2c::C2ROddExpander;
use num_complex::Complex;
use num_traits::AsPrimitive;

#[derive(Default)]
pub(crate) struct AvxC2RExpander {}

impl AvxC2RExpander {
    #[target_feature(enable = "avx2", enable = "fma")]
    fn expand_impl_f64(
        &self,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
        complex_length: usize,
        _: usize,
    ) {
        let mut start = &input[1..];
        let (mut out_left, mut out_right) = output.split_at_mut(complex_length);

        out_left = &mut out_left[1..];

        let conj = AvxStoreD::conj_flag();

        for ((buf_left, buf_right), val) in out_left
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(out_right.as_rchunks_mut::<2>().1.iter_mut().rev())
            .zip(start.as_chunks::<2>().0.iter())
        {
            let val = AvxStoreD::from_complex_ref(val);
            val.write(buf_left);
            val.xor(conj).reverse_complex().write(buf_right);
        }

        out_left = out_left.as_chunks_mut::<2>().1;
        out_right = out_right.as_rchunks_mut::<2>().0;
        start = start.as_chunks::<2>().1;

        for ((buf_left, buf_right), val) in out_left
            .iter_mut()
            .zip(out_right.iter_mut().rev())
            .zip(start)
        {
            *buf_left = *val;
            *buf_right = val.conj();
        }

        output[0].re = input[0].re;
        output[0].im = 0.0.as_();
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    fn expand_impl_f32(
        &self,
        input: &[Complex<f32>],
        output: &mut [Complex<f32>],
        complex_length: usize,
        _: usize,
    ) {
        let mut start = &input[1..];
        let (mut out_left, mut out_right) = output.split_at_mut(complex_length);

        out_left = &mut out_left[1..];

        let conj = AvxStoreF::conj_flag();

        for ((buf_left, buf_right), val) in out_left
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(out_right.as_rchunks_mut::<4>().1.iter_mut().rev())
            .zip(start.as_chunks::<4>().0.iter())
        {
            let val = AvxStoreF::from_complex_ref(val);
            val.write(buf_left);
            val.xor(conj).reverse_complex().write(buf_right);
        }

        out_left = out_left.as_chunks_mut::<4>().1;
        out_right = out_right.as_rchunks_mut::<4>().0;
        start = start.as_chunks::<4>().1;

        for ((buf_left, buf_right), val) in out_left
            .iter_mut()
            .zip(out_right.iter_mut().rev())
            .zip(start)
        {
            *buf_left = *val;
            *buf_right = val.conj();
        }

        output[0].re = input[0].re;
        output[0].im = 0.0.as_();
    }
}

impl C2ROddExpander<f64> for AvxC2RExpander {
    fn expand(
        &self,
        input: &[Complex<f64>],
        output: &mut [Complex<f64>],
        complex_length: usize,
        length: usize,
    ) {
        unsafe {
            self.expand_impl_f64(input, output, complex_length, length);
        }
    }
}

impl C2ROddExpander<f32> for AvxC2RExpander {
    fn expand(
        &self,
        input: &[Complex<f32>],
        output: &mut [Complex<f32>],
        complex_length: usize,
        length: usize,
    ) {
        unsafe {
            self.expand_impl_f32(input, output, complex_length, length);
        }
    }
}
