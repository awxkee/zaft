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
use crate::FftDirection;
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::neon::mixed::{ColumnButterfly3d, ColumnButterfly3f};
#[cfg(feature = "fcma")]
use crate::neon::mixed::{ColumnFcmaButterfly3d, ColumnFcmaButterfly3f};
use crate::util::compute_twiddle;

pub(crate) struct ColumnButterfly9d {
    tw1: NeonStoreD,
    tw2: NeonStoreD,
    tw4: NeonStoreD,
    pub(crate) bf3: ColumnButterfly3d,
}

impl ColumnButterfly9d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f64>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f64>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f64>(4, 9, fft_direction);
        Self {
            tw1: NeonStoreD::from_complex(&tw1),
            tw2: NeonStoreD::from_complex(&tw2),
            tw4: NeonStoreD::from_complex(&tw4),
            bf3: ColumnButterfly3d::new(fft_direction),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 9]) -> [NeonStoreD; 9] {
        let [u0, u3, u6] = self.bf3.exec([store[0], store[3], store[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([store[1], store[4], store[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([store[2], store[5], store[8]]);

        u4 = NeonStoreD::mul_by_complex(u4, self.tw1);
        u7 = NeonStoreD::mul_by_complex(u7, self.tw2);
        u5 = NeonStoreD::mul_by_complex(u5, self.tw2);
        u8 = NeonStoreD::mul_by_complex(u8, self.tw4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly9d {
    tw1: NeonStoreD,
    tw2: NeonStoreD,
    tw4: NeonStoreD,
    pub(crate) bf3: ColumnFcmaButterfly3d,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly9d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f64>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f64>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f64>(4, 9, fft_direction);
        Self {
            tw1: NeonStoreD::from_complex(&tw1),
            tw2: NeonStoreD::from_complex(&tw2),
            tw4: NeonStoreD::from_complex(&tw4),
            bf3: ColumnFcmaButterfly3d::new(fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 9]) -> [NeonStoreD; 9] {
        let [u0, u3, u6] = self.bf3.exec([store[0], store[3], store[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([store[1], store[4], store[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([store[2], store[5], store[8]]);

        u4 = NeonStoreD::fcmul_fcma(u4, self.tw1);
        u7 = NeonStoreD::fcmul_fcma(u7, self.tw2);
        u5 = NeonStoreD::fcmul_fcma(u5, self.tw2);
        u8 = NeonStoreD::fcmul_fcma(u8, self.tw4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}

pub(crate) struct ColumnButterfly9f {
    tw1: NeonStoreF,
    tw2: NeonStoreF,
    tw4: NeonStoreF,
    pub(crate) bf3: ColumnButterfly3f,
}

impl ColumnButterfly9f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f32>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f32>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f32>(4, 9, fft_direction);
        Self {
            tw1: NeonStoreF::from_complex(&tw1),
            tw2: NeonStoreF::from_complex(&tw2),
            tw4: NeonStoreF::from_complex(&tw4),
            bf3: ColumnButterfly3f::new(fft_direction),
        }
    }

    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 9]) -> [NeonStoreF; 9] {
        let [u0, u3, u6] = self.bf3.exec([store[0], store[3], store[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([store[1], store[4], store[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([store[2], store[5], store[8]]);

        u4 = NeonStoreF::mul_by_complex(u4, self.tw1);
        u7 = NeonStoreF::mul_by_complex(u7, self.tw2);
        u5 = NeonStoreF::mul_by_complex(u5, self.tw2);
        u8 = NeonStoreF::mul_by_complex(u8, self.tw4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 9]) -> [NeonStoreFh; 9] {
        let [u0, u3, u6] = self.bf3.exech([store[0], store[3], store[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exech([store[1], store[4], store[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exech([store[2], store[5], store[8]]);

        u4 = NeonStoreFh::mul_by_complex(u4, self.tw1.to_lo());
        u7 = NeonStoreFh::mul_by_complex(u7, self.tw2.to_lo());
        u5 = NeonStoreFh::mul_by_complex(u5, self.tw2.to_lo());
        u8 = NeonStoreFh::mul_by_complex(u8, self.tw4.to_lo());

        let [y0, y3, y6] = self.bf3.exech([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exech([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exech([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct ColumnFcmaButterfly9f {
    tw1: NeonStoreF,
    tw2: NeonStoreF,
    tw4: NeonStoreF,
    pub(crate) bf3: ColumnFcmaButterfly3f,
}

#[cfg(feature = "fcma")]
impl ColumnFcmaButterfly9f {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f32>(1, 9, fft_direction);
        let tw2 = compute_twiddle::<f32>(2, 9, fft_direction);
        let tw4 = compute_twiddle::<f32>(4, 9, fft_direction);
        Self {
            tw1: NeonStoreF::from_complex(&tw1),
            tw2: NeonStoreF::from_complex(&tw2),
            tw4: NeonStoreF::from_complex(&tw4),
            bf3: ColumnFcmaButterfly3f::new(fft_direction),
        }
    }

    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 9]) -> [NeonStoreF; 9] {
        let [u0, u3, u6] = self.bf3.exec([store[0], store[3], store[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exec([store[1], store[4], store[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exec([store[2], store[5], store[8]]);

        u4 = NeonStoreF::fcmul_fcma(u4, self.tw1);
        u7 = NeonStoreF::fcmul_fcma(u7, self.tw2);
        u5 = NeonStoreF::fcmul_fcma(u5, self.tw2);
        u8 = NeonStoreF::fcmul_fcma(u8, self.tw4);

        let [y0, y3, y6] = self.bf3.exec([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exec([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exec([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 9]) -> [NeonStoreFh; 9] {
        let [u0, u3, u6] = self.bf3.exech([store[0], store[3], store[6]]);
        let [u1, mut u4, mut u7] = self.bf3.exech([store[1], store[4], store[7]]);
        let [u2, mut u5, mut u8] = self.bf3.exech([store[2], store[5], store[8]]);

        u4 = NeonStoreFh::fcmul_fcma(u4, self.tw1.to_lo());
        u7 = NeonStoreFh::fcmul_fcma(u7, self.tw2.to_lo());
        u5 = NeonStoreFh::fcmul_fcma(u5, self.tw2.to_lo());
        u8 = NeonStoreFh::fcmul_fcma(u8, self.tw4.to_lo());

        let [y0, y3, y6] = self.bf3.exech([u0, u1, u2]);
        let [y1, y4, y7] = self.bf3.exech([u3, u4, u5]);
        let [y2, y5, y8] = self.bf3.exech([u6, u7, u8]);
        [y0, y1, y2, y3, y4, y5, y6, y7, y8]
    }
}
