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
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use crate::neon::mixed::{ColumnButterfly3d, ColumnButterfly3f};
#[cfg(feature = "fcma")]
use crate::neon::mixed::{ColumnFcmaButterfly3d, ColumnFcmaButterfly3f};
use crate::util::compute_twiddle;
use crate::{FftDirection, FftSample};

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

    #[inline]
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

    #[inline]
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

pub(crate) struct ColumnRdftButterfly9f {
    twiddle1: f32,
    twiddle2: f32,
    twiddle3: f32,
    twiddle4: f32,
    d6: f32,
    d7: f32,
    d8: f32,
}

impl ColumnRdftButterfly9f {
    pub(crate) fn new() -> Self {
        let one_over_3 = (1f64 / 3f64) as f32;
        let twiddle1 = compute_twiddle::<f32>(2, 9, FftDirection::Forward);
        let twiddle2 = compute_twiddle::<f32>(4, 9, FftDirection::Forward);
        let twiddle3 = compute_twiddle::<f32>(6, 9, FftDirection::Forward);
        let twiddle4 = compute_twiddle::<f32>(8, 9, FftDirection::Forward);
        let h0 = twiddle4.re + twiddle1.re; // cos(φ) + cos(2φ)
        let d6 = (2.0 * twiddle4.re - twiddle1.re - twiddle2.re) * one_over_3;
        let d7 = (-twiddle4.re + 2.0 * twiddle1.re - twiddle2.re) * one_over_3;
        let d8 = (-h0 + 2.0 * twiddle2.re) * one_over_3;

        Self {
            twiddle1: twiddle1.im,
            twiddle2: twiddle2.im,
            twiddle3: twiddle3.im,
            twiddle4: twiddle4.im,
            d6,
            d7,
            d8,
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn exec(&self, store: [NeonStoreF; 9]) -> [[NeonStoreF; 5]; 2] {
        let t1 = store[1] + store[8];
        let t2 = store[2] + store[7];
        let t3 = store[3] + store[6];
        let t4 = store[4] + store[5];
        let t5 = store[4] - store[5];
        let t6 = store[3] - store[6];
        let t7 = store[2] - store[7];
        let t8 = store[1] - store[8];

        // DC
        let r0 = t1 + t2 + t4;
        let z0 = t8 - t7 + t5;
        let y0 = store[0] + r0 + t3;

        let y3 = r0.mul_f32_add(-f32::HALF, store[0] + t3);

        let r1 = t1 - t4;
        let r2 = t2 - t4;
        let r3 = -t1 + t2;

        let m2 = self.d6 * r1;
        let m3 = self.d7 * r2;
        let m4 = self.d8 * r3;

        let re_base = t3.mul_f32_add(-f32::HALF, store[0]);
        let y1 = re_base + m2 + m3;
        let y2 = re_base - m2 + m4;
        let y4 = re_base - m3 - m4;

        let y5 = -t8.mul_f32_add(
            self.twiddle4,
            t7.mul_f32_add(
                -self.twiddle1,
                t6.mul_f32_add(self.twiddle3, t5 * -self.twiddle2),
            ),
        ); // X[1].im

        let y6 = -t8.mul_f32_add(
            -self.twiddle1,
            t7.mul_f32_add(
                -self.twiddle2,
                t6.mul_f32_add(-self.twiddle3, t5 * -self.twiddle4),
            ),
        ); // X[2].im

        let y7 = z0 * -self.twiddle3; // X[3].im: -sin(3φ)*(t8-t7+t5)

        let y8 = t8.mul_f32_add(
            self.twiddle2,
            t7.mul_f32_add(
                self.twiddle4,
                t6.mul_f32_add(-self.twiddle3, t5 * -self.twiddle1),
            ),
        );

        let v0 = y0.zip_complex(NeonStoreF::zero());
        let v1 = y1.zip_complex(y5);
        let v2 = y2.zip_complex(y6);
        let v3 = y3.zip_complex(y7);
        let v4 = y4.zip_complex(y8);
        [
            [v0[0], v1[0], v2[0], v3[0], v4[0]],
            [v0[1], v1[1], v2[1], v3[1], v4[1]],
        ]
    }
}

pub(crate) struct ColumnRdftButterfly9d {
    twiddle1: f64,
    twiddle2: f64,
    twiddle3: f64,
    twiddle4: f64,
    d6: f64,
    d7: f64,
    d8: f64,
}

impl ColumnRdftButterfly9d {
    pub(crate) fn new() -> Self {
        let one_over_3 = 1f64 / 3f64;
        let twiddle1 = compute_twiddle::<f64>(2, 9, FftDirection::Forward);
        let twiddle2 = compute_twiddle::<f64>(4, 9, FftDirection::Forward);
        let twiddle3 = compute_twiddle::<f64>(6, 9, FftDirection::Forward);
        let twiddle4 = compute_twiddle::<f64>(8, 9, FftDirection::Forward);
        let h0 = twiddle4.re + twiddle1.re; // cos(φ) + cos(2φ)
        let d6 = (2.0 * twiddle4.re - twiddle1.re - twiddle2.re) * one_over_3;
        let d7 = (-twiddle4.re + 2.0 * twiddle1.re - twiddle2.re) * one_over_3;
        let d8 = (-h0 + 2.0 * twiddle2.re) * one_over_3;

        Self {
            twiddle1: twiddle1.im,
            twiddle2: twiddle2.im,
            twiddle3: twiddle3.im,
            twiddle4: twiddle4.im,
            d6,
            d7,
            d8,
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn exec(&self, store: [NeonStoreD; 9]) -> [[NeonStoreD; 5]; 2] {
        let t1 = store[1] + store[8];
        let t2 = store[2] + store[7];
        let t3 = store[3] + store[6];
        let t4 = store[4] + store[5];
        let t5 = store[4] - store[5];
        let t6 = store[3] - store[6];
        let t7 = store[2] - store[7];
        let t8 = store[1] - store[8];

        // DC
        let r0 = t1 + t2 + t4;
        let z0 = t8 - t7 + t5;
        let y0 = store[0] + r0 + t3;

        let y3 = r0.mul_f64_add(-f64::HALF, store[0] + t3);

        let r1 = t1 - t4;
        let r2 = t2 - t4;
        let r3 = -t1 + t2;

        let m2 = self.d6 * r1;
        let m3 = self.d7 * r2;
        let m4 = self.d8 * r3;

        let re_base = t3.mul_f64_add(-f64::HALF, store[0]);
        let y1 = re_base + m2 + m3;
        let y2 = re_base - m2 + m4;
        let y4 = re_base - m3 - m4;

        let y5 = -t8.mul_f64_add(
            self.twiddle4,
            t7.mul_f64_add(
                -self.twiddle1,
                t6.mul_f64_add(self.twiddle3, t5 * -self.twiddle2),
            ),
        ); // X[1].im

        let y6 = -t8.mul_f64_add(
            -self.twiddle1,
            t7.mul_f64_add(
                -self.twiddle2,
                t6.mul_f64_add(-self.twiddle3, t5 * -self.twiddle4),
            ),
        ); // X[2].im

        let y7 = z0 * -self.twiddle3; // X[3].im: -sin(3φ)*(t8-t7+t5)

        let y8 = t8.mul_f64_add(
            self.twiddle2,
            t7.mul_f64_add(
                self.twiddle4,
                t6.mul_f64_add(-self.twiddle3, t5 * -self.twiddle1),
            ),
        );

        let v0 = y0.zip_complex(NeonStoreD::zero());
        let v1 = y1.zip_complex(y5);
        let v2 = y2.zip_complex(y6);
        let v3 = y3.zip_complex(y7);
        let v4 = y4.zip_complex(y8);
        [
            [v0[0], v1[0], v2[0], v3[0], v4[0]],
            [v0[1], v1[1], v2[1], v3[1], v4[1]],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;
    use std::arch::aarch64::vgetq_lane_f32;
    use std::f32::consts::PI;

    fn naive_dft9(x: &[f32; 9]) -> Vec<Complex<f32>> {
        (0..5)
            .map(|k| {
                (0..9usize).fold(Complex::new(0.0f32, 0.0f32), |acc, n| {
                    let angle = -2.0 * PI * k as f32 * n as f32 / 9.0;
                    acc + x[n] * Complex::new(angle.cos(), angle.sin())
                })
            })
            .collect()
    }

    #[test]
    fn test_column_rdft9_two_lanes_independent() {
        unsafe {
            let bf = ColumnRdftButterfly9f::new();

            let input_a = [-1.3f32, 1.32, 2.62, -2.3, -6.0, 3.12, 5.2, 8.0, 9.0];

            let store: [NeonStoreF; 9] = std::array::from_fn(|i| NeonStoreF::dup(input_a[i]));

            let result = bf.exec(store);

            let expected_a = naive_dft9(&input_a);

            for k in 0..5 {
                let re_a = vgetq_lane_f32::<0>(result[0][k].v);
                let im_a = vgetq_lane_f32::<1>(result[0][k].v);

                assert!(
                    (re_a - expected_a[k].re).abs() < 1e-4,
                    "lane0 X[{k}].re: got {re_a} expected {}",
                    expected_a[k].re
                );
                assert!(
                    (im_a - expected_a[k].im).abs() < 1e-4,
                    "lane0 X[{k}].im: got {im_a} expected {}",
                    expected_a[k].im
                );
            }
        }
    }
}
