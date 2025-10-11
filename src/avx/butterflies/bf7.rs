// Copyright (c) Radzivon Bartoshyk 10/2025. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// 1.  Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2.  Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3.  Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

use crate::avx::butterflies::AvxButterfly;
use crate::avx::rotate::AvxRotate;
use crate::avx::util::{_mm_unpacklo_ps64, _mm256_create_pd, _mm256_create_ps, shuffle};
use crate::traits::FftTrigonometry;
use crate::util::compute_twiddle;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly7<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    tw1tw2r: [T; 8],
    tw2tw3r: [T; 8],
    tw3tw1r: [T; 8],
    tw1tw2i: [T; 8],
    tw2ntw3i: [T; 8],
    tw3ntw1i: [T; 8],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> AvxButterfly7<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        let tw1 = compute_twiddle(1, 7, fft_direction);
        let tw2 = compute_twiddle(2, 7, fft_direction);
        let tw3 = compute_twiddle(3, 7, fft_direction);
        Self {
            direction: fft_direction,
            twiddle1: tw1,
            twiddle2: tw2,
            twiddle3: tw3,
            tw1tw2r: [
                tw1.re, tw1.re, tw2.re, tw2.re, tw1.re, tw1.re, tw2.re, tw2.re,
            ],
            tw2tw3r: [
                tw2.re, tw2.re, tw3.re, tw3.re, tw2.re, tw2.re, tw3.re, tw3.re,
            ],
            tw3tw1r: [
                tw3.re, tw3.re, tw1.re, tw1.re, tw3.re, tw3.re, tw1.re, tw1.re,
            ],
            tw1tw2i: [
                tw1.im, tw1.im, tw2.im, tw2.im, tw1.im, tw1.im, tw2.im, tw2.im,
            ],
            tw2ntw3i: [
                tw2.im, tw2.im, -tw3.im, -tw3.im, tw2.im, tw2.im, -tw3.im, -tw3.im,
            ],
            tw3ntw1i: [
                tw3.im, tw3.im, -tw1.im, -tw1.im, tw3.im, tw3.im, -tw1.im, -tw1.im,
            ],
        }
    }
}

impl AvxButterfly7<f64> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let rotate = AvxRotate::<f64>::new(FftDirection::Inverse);

            let tw1tw2r = _mm256_loadu_pd(self.tw1tw2r.as_ptr().cast());
            let tw2tw3r = _mm256_loadu_pd(self.tw2tw3r.as_ptr().cast());
            let tw3tw1r = _mm256_loadu_pd(self.tw3tw1r.as_ptr().cast());
            let tw1tw2i = _mm256_loadu_pd(self.tw1tw2i.as_ptr().cast());
            let tw2ntw3i = _mm256_loadu_pd(self.tw2ntw3i.as_ptr().cast());
            let tw3ntw1i = _mm256_loadu_pd(self.tw3ntw1i.as_ptr().cast());

            const LO_HI: i32 = 0b0011_0000;
            const HI_LO: i32 = 0b0010_0001;
            const HI_HI: i32 = 0b0011_0001;
            const LO_LO: i32 = 0b0010_0000;

            for chunk in in_place.chunks_exact_mut(7) {
                let u0u1 = _mm256_loadu_pd(chunk.get_unchecked(0..).as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6 = _mm_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());

                let u0u0 = _mm256_permute2f128_pd::<LO_LO>(u0u1, u0u1);
                let u3 = _mm256_extractf128_pd::<1>(u2u3);
                let u4 = _mm256_castpd256_pd128(u4u5);

                let u1u2 = _mm256_permute2f128_pd::<HI_LO>(u0u1, u2u3);
                let u6u5 = _mm256_permute2f128_pd::<LO_HI>(_mm256_castpd128_pd256(u6), u4u5);

                let (x1p6x2p5, x1m6x2m5) = AvxButterfly::butterfly2_f64(u1u2, u6u5);
                let x1m6x2m5 = rotate.rotate_m256d(x1m6x2m5);

                let x2m5 = _mm256_extractf128_pd::<1>(x1m6x2m5);
                let x2p5 = _mm256_extractf128_pd::<1>(x1p6x2p5);

                let y00 = _mm_add_pd(
                    _mm_add_pd(_mm256_castpd256_pd128(u0u0), x2p5),
                    _mm256_castpd256_pd128(x1p6x2p5),
                );
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f64_m128(u3, u4);
                let x3m4 = rotate.rotate_m128d(x3m4);
                let y00 = _mm_add_pd(y00, x3p4);

                let m0106a_m0205a0 = _mm256_fmadd_pd(
                    _mm256_permute2f128_pd::<LO_LO>(x1p6x2p5, x1p6x2p5),
                    tw1tw2r,
                    u0u0,
                );
                let m0106a_m0205a1 = _mm256_fmadd_pd(
                    _mm256_permute2f128_pd::<HI_HI>(x1p6x2p5, x1p6x2p5),
                    tw2tw3r,
                    m0106a_m0205a0,
                );
                let m0106a_m0205a =
                    _mm256_fmadd_pd(_mm256_create_pd(x3p4, x3p4), tw3tw1r, m0106a_m0205a1);
                let m0106b_m0205b0 =
                    _mm256_mul_pd(_mm256_permute2f128_pd::<LO_LO>(x1m6x2m5, x1m6x2m5), tw1tw2i);
                let m0106b_m0205b1 = _mm256_fmadd_pd(
                    _mm256_permute2f128_pd::<HI_HI>(x1m6x2m5, x1m6x2m5),
                    tw2ntw3i,
                    m0106b_m0205b0,
                );
                let m0106b_m0205b =
                    _mm256_fmadd_pd(_mm256_create_pd(x3m4, x3m4), tw3ntw1i, m0106b_m0205b1);
                let (y01y02, y06y05) = AvxButterfly::butterfly2_f64(m0106a_m0205a, m0106b_m0205b);

                let m0304a = _mm_fmadd_pd(
                    _mm256_castpd256_pd128(x1p6x2p5),
                    _mm256_castpd256_pd128(tw3tw1r),
                    _mm256_castpd256_pd128(u0u0),
                );
                let m0304a = _mm_fmadd_pd(x2p5, _mm256_castpd256_pd128(tw1tw2r), m0304a);
                let m0304a = _mm_fmadd_pd(x3p4, _mm256_castpd256_pd128(tw2tw3r), m0304a);
                let m0304b = _mm_mul_pd(
                    _mm256_castpd256_pd128(x1m6x2m5),
                    _mm256_castpd256_pd128(tw3ntw1i),
                );
                let m0304b = _mm_fnmadd_pd(x2m5, _mm256_castpd256_pd128(tw1tw2i), m0304b);
                let m0304b = _mm_fmadd_pd(x3m4, _mm256_castpd256_pd128(tw2ntw3i), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f64_m128(m0304a, m0304b);

                let y0y1 = _mm256_permute2f128_pd::<LO_LO>(_mm256_castpd128_pd256(y00), y01y02);
                let y2y3 = _mm256_permute2f128_pd::<HI_LO>(y01y02, _mm256_castpd128_pd256(y03));
                let y4y5 = _mm256_permute2f128_pd::<LO_HI>(_mm256_castpd128_pd256(y04), y06y05);

                _mm256_storeu_pd(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), y2y3);
                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), y4y5);
                _mm_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_castpd256_pd128(y06y05),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f64> for AvxButterfly7<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        7
    }
}

impl AvxButterfly7<f32> {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe {
            if in_place.len() % 7 != 0 {
                return Err(ZaftError::InvalidSizeMultiplier(
                    in_place.len(),
                    self.length(),
                ));
            }

            let rotate = AvxRotate::<f32>::new(FftDirection::Inverse);

            for chunk in in_place.chunks_exact_mut(7) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.get_unchecked(0..).as_ptr().cast());
                let u3u4u5u6 = _mm256_loadu_ps(chunk.get_unchecked(3..).as_ptr().cast());

                let u2u3 = _mm256_extractf128_ps::<1>(u0u1u2u3);

                let u0 = _mm256_castps256_ps128(u0u1u2u3);
                let u1 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u0, u0);
                let u2 = u2u3;
                let u3 = _mm256_castps256_ps128(u3u4u5u6);
                let u4 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u3, u3);
                let u5u6 = _mm256_extractf128_ps::<1>(u3u4u5u6);
                let u5 = u5u6;
                let u6 = _mm_shuffle_ps::<{ shuffle(3, 2, 3, 2) }>(u5u6, u5u6);

                let (x1p6, x1m6) = AvxButterfly::butterfly2_f32_m128(u1, u6);
                let x1m6 = rotate.rotate_m128(x1m6);
                let y00 = _mm_add_ps(u0, x1p6);
                let (x2p5, x2m5) = AvxButterfly::butterfly2_f32_m128(u2, u5);
                let x2m5 = rotate.rotate_m128(x2m5);
                let y00 = _mm_add_ps(y00, x2p5);
                let (x3p4, x3m4) = AvxButterfly::butterfly2_f32_m128(u3, u4);
                let x3m4 = rotate.rotate_m128(x3m4);
                let y00 = _mm_add_ps(y00, x3p4);

                let m0106a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle1.re), u0);
                let m0106a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle2.re), m0106a);
                let m0106a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle3.re), m0106a);
                let m0106b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle1.im));
                let m0106b = _mm_fmadd_ps(x2m5, _mm_set1_ps(self.twiddle2.im), m0106b);
                let m0106b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle3.im), m0106b);
                let (y01, y06) = AvxButterfly::butterfly2_f32_m128(m0106a, m0106b);

                let m0205a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle2.re), u0);
                let m0205a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle3.re), m0205a);
                let m0205a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle1.re), m0205a);
                let m0205b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle2.im));
                let m0205b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle3.im), m0205b);
                let m0205b = _mm_fnmadd_ps(x3m4, _mm_set1_ps(self.twiddle1.im), m0205b);
                let (y02, y05) = AvxButterfly::butterfly2_f32_m128(m0205a, m0205b);

                let m0304a = _mm_fmadd_ps(x1p6, _mm_set1_ps(self.twiddle3.re), u0);
                let m0304a = _mm_fmadd_ps(x2p5, _mm_set1_ps(self.twiddle1.re), m0304a);
                let m0304a = _mm_fmadd_ps(x3p4, _mm_set1_ps(self.twiddle2.re), m0304a);
                let m0304b = _mm_mul_ps(x1m6, _mm_set1_ps(self.twiddle3.im));
                let m0304b = _mm_fnmadd_ps(x2m5, _mm_set1_ps(self.twiddle1.im), m0304b);
                let m0304b = _mm_fmadd_ps(x3m4, _mm_set1_ps(self.twiddle2.im), m0304b);
                let (y03, y04) = AvxButterfly::butterfly2_f32_m128(m0304a, m0304b);

                let y0y1y2y3 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y00, y01), _mm_unpacklo_ps64(y02, y03));
                let y3y4y5y6 =
                    _mm256_create_ps(_mm_unpacklo_ps64(y03, y04), _mm_unpacklo_ps64(y05, y06));

                _mm256_storeu_ps(chunk.get_unchecked_mut(0..).as_mut_ptr().cast(), y0y1y2y3);
                _mm256_storeu_ps(chunk.get_unchecked_mut(3..).as_mut_ptr().cast(), y3y4y5y6);
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly7<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        7
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;

    #[test]
    fn test_butterfly7_f32() {
        for i in 1..6 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly7::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly7::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 7f32)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly7_f64() {
        for i in 1..6 {
            let size = 7usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = AvxButterfly7::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly7::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 7f64)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-9,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-9,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }
}
