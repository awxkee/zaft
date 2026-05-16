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

use crate::FftExecutor;
use crate::butterflies::fast_bf8::FastButterfly8;
use crate::butterflies::short_butterflies::FastButterfly5;
use crate::butterflies::util::boring_scalar_butterfly;
use crate::store::BidirectionalStore;
use crate::{FftDirection, FftSample, ZaftError};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;

pub(crate) struct Butterfly40<T> {
    direction: FftDirection,
    phantom_data: PhantomData<T>,
    bf5: FastButterfly5<T>,
    bf8: FastButterfly8<T>,
}

impl<T: FftSample> Butterfly40<T>
where
    f64: AsPrimitive<T>,
{
    #[allow(unused)]
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Butterfly40 {
            direction: fft_direction,
            phantom_data: PhantomData,
            bf5: FastButterfly5::new(fft_direction),
            bf8: FastButterfly8::new(fft_direction),
        }
    }
}

impl<T: FftSample> Butterfly40<T>
where
    f64: AsPrimitive<T>,
{
    #[inline(always)]
    pub(crate) fn run<S: BidirectionalStore<Complex<T>>>(&self, chunk: &mut S) {
        let (t0_0, t0_1, t0_2, t0_3, t0_4) = self
            .bf5
            .bf5(chunk[0], chunk[16], chunk[32], chunk[8], chunk[24]);
        let (t1_0, t1_1, t1_2, t1_3, t1_4) = self
            .bf5
            .bf5(chunk[25], chunk[1], chunk[17], chunk[33], chunk[9]);
        let (t2_0, t2_1, t2_2, t2_3, t2_4) = self
            .bf5
            .bf5(chunk[10], chunk[26], chunk[2], chunk[18], chunk[34]);
        let (t3_0, t3_1, t3_2, t3_3, t3_4) = self
            .bf5
            .bf5(chunk[35], chunk[11], chunk[27], chunk[3], chunk[19]);
        let (t4_0, t4_1, t4_2, t4_3, t4_4) = self
            .bf5
            .bf5(chunk[20], chunk[36], chunk[12], chunk[28], chunk[4]);
        let (t5_0, t5_1, t5_2, t5_3, t5_4) = self
            .bf5
            .bf5(chunk[5], chunk[21], chunk[37], chunk[13], chunk[29]);
        let (t6_0, t6_1, t6_2, t6_3, t6_4) = self
            .bf5
            .bf5(chunk[30], chunk[6], chunk[22], chunk[38], chunk[14]);
        let (t7_0, t7_1, t7_2, t7_3, t7_4) = self
            .bf5
            .bf5(chunk[15], chunk[31], chunk[7], chunk[23], chunk[39]);

        let (r0_0, r1_0, r2_0, r3_0, r4_0, r5_0, r6_0, r7_0) = self
            .bf8
            .exec(t0_0, t1_0, t2_0, t3_0, t4_0, t5_0, t6_0, t7_0);
        chunk[0] = r0_0;
        chunk[5] = r1_0;
        chunk[10] = r2_0;
        chunk[15] = r3_0;
        chunk[20] = r4_0;
        chunk[25] = r5_0;
        chunk[30] = r6_0;
        chunk[35] = r7_0;

        let (r0_1, r1_1, r2_1, r3_1, r4_1, r5_1, r6_1, r7_1) = self
            .bf8
            .exec(t0_1, t1_1, t2_1, t3_1, t4_1, t5_1, t6_1, t7_1);
        chunk[8] = r0_1;
        chunk[13] = r1_1;
        chunk[18] = r2_1;
        chunk[23] = r3_1;
        chunk[28] = r4_1;
        chunk[33] = r5_1;
        chunk[38] = r6_1;
        chunk[3] = r7_1;

        let (r0_2, r1_2, r2_2, r3_2, r4_2, r5_2, r6_2, r7_2) = self
            .bf8
            .exec(t0_2, t1_2, t2_2, t3_2, t4_2, t5_2, t6_2, t7_2);
        chunk[16] = r0_2;
        chunk[21] = r1_2;
        chunk[26] = r2_2;
        chunk[31] = r3_2;
        chunk[36] = r4_2;
        chunk[1] = r5_2;
        chunk[6] = r6_2;
        chunk[11] = r7_2;

        let (r0_3, r1_3, r2_3, r3_3, r4_3, r5_3, r6_3, r7_3) = self
            .bf8
            .exec(t0_3, t1_3, t2_3, t3_3, t4_3, t5_3, t6_3, t7_3);
        chunk[24] = r0_3;
        chunk[29] = r1_3;
        chunk[34] = r2_3;
        chunk[39] = r3_3;
        chunk[4] = r4_3;
        chunk[9] = r5_3;
        chunk[14] = r6_3;
        chunk[19] = r7_3;

        let (r0_4, r1_4, r2_4, r3_4, r4_4, r5_4, r6_4, r7_4) = self
            .bf8
            .exec(t0_4, t1_4, t2_4, t3_4, t4_4, t5_4, t6_4, t7_4);
        chunk[32] = r0_4;
        chunk[37] = r1_4;
        chunk[2] = r2_4;
        chunk[7] = r3_4;
        chunk[12] = r4_4;
        chunk[17] = r5_4;
        chunk[22] = r6_4;
        chunk[27] = r7_4;
    }
}

boring_scalar_butterfly!(Butterfly40, 40);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::test_butterfly;

    test_butterfly!(test_butterfly40, f32, Butterfly40, 40, 1e-5);
}
