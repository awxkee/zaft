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
use crate::avx::butterflies::AvxFastButterfly3;
use crate::avx::util::{_mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_ps, _mm256_fcmul_ps};
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct AvxFastButterfly9f {
    tw1: __m256,
    pub(crate) bf3: AvxFastButterfly3<f32>,
}

impl AvxFastButterfly9f {
    #[target_feature(enable = "avx")]
    pub(crate) unsafe fn new(direction: FftDirection) -> Self {
        let tw1 = compute_twiddle::<f32>(1, 9, direction);
        let tw2 = compute_twiddle::<f32>(2, 9, direction);
        let tw4 = compute_twiddle::<f32>(4, 9, direction);
        unsafe {
            Self {
                tw1: _mm256_setr_ps(
                    tw1.re, tw1.im, tw2.re, tw2.im, tw2.re, tw2.im, tw4.re, tw4.im,
                ),
                bf3: AvxFastButterfly3::<f32>::new(direction),
            }
        }
    }

    #[inline]
    #[target_feature(enable = "avx", enable = "fma")]
    pub(crate) fn exec(
        &self,
        u0: __m128,
        u1: __m128,
        u2: __m128,
        u3: __m128,
        u4: __m128,
        u5: __m128,
        u6: __m128,
        u7: __m128,
        u8: __m128,
    ) -> (
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
        __m128,
    ) {
        unsafe {
            let (u0, u3, u6) = self.bf3.exec_m128(u0, u3, u6);
            let (u1, mut u4, mut u7) = self.bf3.exec_m128(u1, u4, u7);
            let (u2, mut u5, mut u8) = self.bf3.exec_m128(u2, u5, u8);

            let mut u4u7u5u8 =
                _mm256_create_ps(_mm_unpacklo_ps64(u4, u7), _mm_unpacklo_ps64(u5, u8));

            u4u7u5u8 = _mm256_fcmul_ps(u4u7u5u8, self.tw1);

            u4 = _mm256_castps256_ps128(u4u7u5u8);
            u7 = _mm_unpackhi_ps64(
                _mm256_castps256_ps128(u4u7u5u8),
                _mm256_castps256_ps128(u4u7u5u8),
            );
            let u5u8 = _mm256_extractf128_ps::<1>(u4u7u5u8);
            u5 = u5u8;
            u8 = _mm_unpackhi_ps64(u5u8, u5u8);

            let (y0y1, y3y4, y6y7) = self.bf3.exec_m128(
                _mm_unpacklo_ps64(u0, u3),
                _mm_unpacklo_ps64(u1, u4),
                _mm_unpacklo_ps64(u2, u5),
            );
            let (y2, y5, y8) = self.bf3.exec_m128(u6, u7, u8);
            (
                y0y1,
                _mm_unpackhi_ps64(y0y1, y0y1),
                y2,
                y3y4,
                _mm_unpackhi_ps64(y3y4, y3y4),
                y5,
                y6y7,
                _mm_unpackhi_ps64(y6y7, y6y7),
                y8,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::avx::butterflies::fast_bf9::AvxFastButterfly9f;
    use crate::avx::util::{_m128s_load_f32x2, _m128s_store_f32x2};
    use crate::dft::Dft;
    use crate::{FftDirection, FftExecutor};
    use num_complex::Complex;
    use std::arch::x86_64::_mm_setzero_ps;

    #[test]
    fn test_fast_bf9() {
        let mut input = vec![
            Complex::new(0., -1.),
            Complex::new(0.2f32, -0.9),
            Complex::new(0.4, -0.7),
            Complex::new(-0.2, -0.9),
            Complex::new(-0.4, -0.7),
            Complex::new(0.35, 0.65),
            Complex::new(-0.5, 0.65),
            Complex::new(0.5, 0.65),
            Complex::new(-0.321, 0.854),
        ];
        let dft = Dft::new(9, FftDirection::Forward).unwrap();
        let mut dft_ref = input.to_vec();
        dft.execute(&mut dft_ref).unwrap();
        unsafe {
            let fast_bf9 = AvxFastButterfly9f::new(FftDirection::Forward);
            let mut u = [_mm_setzero_ps(); 9];
            for i in 0..9 {
                u[i] = _m128s_load_f32x2(input.get_unchecked(i..).as_ptr().cast());
            }
            let w = fast_bf9.exec(u[0], u[1], u[2], u[3], u[4], u[5], u[6], u[7], u[8]);
            _m128s_store_f32x2(input.get_unchecked_mut(0..).as_mut_ptr().cast(), w.0);
            _m128s_store_f32x2(input.get_unchecked_mut(1..).as_mut_ptr().cast(), w.1);
            _m128s_store_f32x2(input.get_unchecked_mut(2..).as_mut_ptr().cast(), w.2);
            _m128s_store_f32x2(input.get_unchecked_mut(3..).as_mut_ptr().cast(), w.3);
            _m128s_store_f32x2(input.get_unchecked_mut(4..).as_mut_ptr().cast(), w.4);
            _m128s_store_f32x2(input.get_unchecked_mut(5..).as_mut_ptr().cast(), w.5);
            _m128s_store_f32x2(input.get_unchecked_mut(6..).as_mut_ptr().cast(), w.6);
            _m128s_store_f32x2(input.get_unchecked_mut(7..).as_mut_ptr().cast(), w.7);
            _m128s_store_f32x2(input.get_unchecked_mut(8..).as_mut_ptr().cast(), w.8);
        }
        input
            .iter()
            .zip(dft_ref.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < 1e-5f32,
                    "forward at {idx} a_re {} != b_re {} for  at {idx}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "forward at {idx} a_im {} != b_im {} for  at {idx}",
                    a.im,
                    b.im,
                );
            });
    }
}
