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

#![allow(unused)]

use crate::butterflies::fast_bf16::FastButterfly16;
use crate::butterflies::util::boring_scalar_butterfly;
use crate::butterflies::{Butterfly6, rotate_90};
use crate::complex_fma::{c_mul_fast, c_mul_fast_conj};
use crate::store::{BidirectionalStore, InPlaceStore};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::ops::Neg;

pub(crate) struct Butterfly36<T> {
    direction: FftDirection,
    twiddles36: [Complex<T>; 13],
    bf6: Butterfly6<T>,
}

impl<T: FftTrigonometry + Float + 'static + Default + FftSample> Butterfly36<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(direction: FftDirection) -> Self {
        Butterfly36 {
            direction,
            twiddles36: [
                compute_twiddle(1, 36, direction),
                compute_twiddle(2, 36, direction),
                compute_twiddle(3, 36, direction),
                compute_twiddle(4, 36, direction),
                compute_twiddle(5, 36, direction),
                compute_twiddle(6, 36, direction),
                compute_twiddle(8, 36, direction),
                compute_twiddle(9, 36, direction),
                compute_twiddle(10, 36, direction),
                compute_twiddle(12, 36, direction),
                compute_twiddle(15, 36, direction),
                compute_twiddle(16, 36, direction),
                compute_twiddle(25, 36, direction),
            ],
            bf6: Butterfly6::new(direction),
        }
    }
}

impl<T: FftSample> Butterfly36<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let mut mid0 = [
            chunk[0], chunk[6], chunk[12], chunk[18], chunk[24], chunk[30],
        ];
        self.bf6.run(&mut InPlaceStore::new(&mut mid0));

        let mut mid1 = [
            chunk[1], chunk[7], chunk[13], chunk[19], chunk[25], chunk[31],
        ];
        self.bf6.run(&mut InPlaceStore::new(&mut mid1));
        mid1[1] = c_mul_fast(mid1[1], self.twiddles36[0]);
        mid1[2] = c_mul_fast(mid1[2], self.twiddles36[1]);
        mid1[3] = c_mul_fast(mid1[3], self.twiddles36[2]);
        mid1[4] = c_mul_fast(mid1[4], self.twiddles36[3]);
        mid1[5] = c_mul_fast(mid1[5], self.twiddles36[4]);

        let mut mid2 = [
            chunk[2], chunk[8], chunk[14], chunk[20], chunk[26], chunk[32],
        ];
        self.bf6.run(&mut InPlaceStore::new(&mut mid2));
        mid2[1] = c_mul_fast(mid2[1], self.twiddles36[1]);
        mid2[2] = c_mul_fast(mid2[2], self.twiddles36[3]);
        mid2[3] = c_mul_fast(mid2[3], self.twiddles36[5]);
        mid2[4] = c_mul_fast(mid2[4], self.twiddles36[6]);
        mid2[5] = c_mul_fast(mid2[5], self.twiddles36[8]);

        let mut mid3 = [
            chunk[3], chunk[9], chunk[15], chunk[21], chunk[27], chunk[33],
        ];
        self.bf6.run(&mut InPlaceStore::new(&mut mid3));
        mid3[1] = c_mul_fast(mid3[1], self.twiddles36[2]);
        mid3[2] = c_mul_fast(mid3[2], self.twiddles36[5]);
        mid3[3] = c_mul_fast(mid3[3], self.twiddles36[7]);
        mid3[4] = c_mul_fast(mid3[4], self.twiddles36[9]);
        mid3[5] = c_mul_fast(mid3[5], self.twiddles36[10]);

        let mut mid4 = [
            chunk[4], chunk[10], chunk[16], chunk[22], chunk[28], chunk[34],
        ];
        self.bf6.run(&mut InPlaceStore::new(&mut mid4));
        mid4[1] = c_mul_fast(mid4[1], self.twiddles36[3]);
        mid4[2] = c_mul_fast(mid4[2], self.twiddles36[6]);
        mid4[3] = c_mul_fast(mid4[3], self.twiddles36[9]);
        mid4[4] = c_mul_fast(mid4[4], self.twiddles36[11]);
        mid4[5] = c_mul_fast(mid4[5], self.twiddles36[1].neg());

        let mut mid5 = [
            chunk[5], chunk[11], chunk[17], chunk[23], chunk[29], chunk[35],
        ];
        self.bf6.run(&mut InPlaceStore::new(&mut mid5));
        mid5[1] = c_mul_fast(mid5[1], self.twiddles36[4]);
        mid5[2] = c_mul_fast(mid5[2], self.twiddles36[8]);
        mid5[3] = c_mul_fast(mid5[3], self.twiddles36[10]);
        mid5[4] = c_mul_fast(mid5[4], self.twiddles36[1].neg());
        mid5[5] = c_mul_fast(mid5[5], self.twiddles36[12]);

        for i in 0..6 {
            let mut output = [mid0[i], mid1[i], mid2[i], mid3[i], mid4[i], mid5[i]];
            self.bf6.run(&mut InPlaceStore::new(&mut output));
            chunk[i] = output[0];
            chunk[i + 6] = output[1];
            chunk[i + 12] = output[2];
            chunk[i + 18] = output[3];
            chunk[i + 24] = output[4];
            chunk[i + 30] = output[5];
        }
    }
}

boring_scalar_butterfly!(Butterfly36, 36);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_butterfly!(test_butterfly36, f32, Butterfly36, 36, 1e-5);
    test_oof_butterfly!(test_oof_butterfly36, f32, Butterfly36, 36, 1e-5);
}
