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
use crate::avx::butterflies::fast_bf9d::AvxFastButterfly9d;
use crate::avx::util::{_mm256_create_pd, _mm256_fcmul_pd, _mm256_set2_complexd};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly27d {
    direction: FftDirection,
    bf9: AvxFastButterfly9d,
    tw1tw2: __m256d,
    tw2tw4: __m256d,
    tw3tw6: __m256d,
    tw4tw8: __m256d,
    tw5tw9: __m256d,
    tw6tw10: __m256d,
    tw7tw11: __m256d,
    tw8tw12: __m256d,
}

impl AvxButterfly27d {
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        unsafe {
            let t1 = compute_twiddle(1, 27, fft_direction);
            let t2 = compute_twiddle(2, 27, fft_direction);
            let t3 = compute_twiddle(3, 27, fft_direction);
            let t4 = compute_twiddle(4, 27, fft_direction);
            let t5 = compute_twiddle(5, 27, fft_direction);
            let t6 = compute_twiddle(6, 27, fft_direction);
            let t7 = compute_twiddle(7, 27, fft_direction);
            let t8 = compute_twiddle(8, 27, fft_direction);
            let t9 = compute_twiddle(10, 27, fft_direction);
            let t10 = compute_twiddle(12, 27, fft_direction);
            let t11 = compute_twiddle(14, 27, fft_direction);
            let t12 = compute_twiddle(16, 27, fft_direction);
            Self {
                direction: fft_direction,
                tw1tw2: _mm256_set2_complexd(t1, t2),
                tw2tw4: _mm256_set2_complexd(t2, t4),
                tw3tw6: _mm256_set2_complexd(t3, t6),
                tw4tw8: _mm256_set2_complexd(t4, t8),
                tw5tw9: _mm256_set2_complexd(t5, t9),
                tw6tw10: _mm256_set2_complexd(t6, t10),
                tw7tw11: _mm256_set2_complexd(t7, t11),
                tw8tw12: _mm256_set2_complexd(t8, t12),
                bf9: AvxFastButterfly9d::new(fft_direction),
            }
        }
    }
}

impl AvxButterfly27d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(27) {
                let u0u1 = _mm256_loadu_pd(chunk.as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(chunk.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(chunk.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(chunk.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(chunk.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(chunk.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(chunk.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(chunk.get_unchecked(14..).as_ptr().cast());
                let u16u17 = _mm256_loadu_pd(chunk.get_unchecked(16..).as_ptr().cast());
                let u18u19 = _mm256_loadu_pd(chunk.get_unchecked(18..).as_ptr().cast());
                let u20u21 = _mm256_loadu_pd(chunk.get_unchecked(20..).as_ptr().cast());
                let u22u23 = _mm256_loadu_pd(chunk.get_unchecked(22..).as_ptr().cast());
                let u24u25 = _mm256_loadu_pd(chunk.get_unchecked(24..).as_ptr().cast());
                let u26 = _mm_loadu_pd(chunk.get_unchecked(26..).as_ptr().cast());

                let s0 = self.bf9.exec_m128d(
                    _mm256_castpd256_pd128(u0u1),
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u8u9),
                    _mm256_castpd256_pd128(u12u13),
                    _mm256_extractf128_pd::<1>(u14u15),
                    _mm256_castpd256_pd128(u18u19),
                    _mm256_extractf128_pd::<1>(u20u21),
                    _mm256_castpd256_pd128(u24u25),
                );

                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let mut s1s2 = self.bf9.exec(
                    _mm256_permute2f128_pd::<HI_LO>(u0u1, u2u3),
                    u4u5,
                    _mm256_permute2f128_pd::<HI_LO>(u6u7, u8u9),
                    u10u11,
                    _mm256_permute2f128_pd::<HI_LO>(u12u13, u14u15),
                    u16u17,
                    _mm256_permute2f128_pd::<HI_LO>(u18u19, u20u21),
                    u22u23,
                    _mm256_permute2f128_pd::<HI_LO>(u24u25, _mm256_castpd128_pd256(u26)),
                );

                s1s2.1 = _mm256_fcmul_pd(s1s2.1, self.tw1tw2);
                s1s2.2 = _mm256_fcmul_pd(s1s2.2, self.tw2tw4);
                s1s2.3 = _mm256_fcmul_pd(s1s2.3, self.tw3tw6);
                s1s2.4 = _mm256_fcmul_pd(s1s2.4, self.tw4tw8);
                s1s2.5 = _mm256_fcmul_pd(s1s2.5, self.tw5tw9);
                s1s2.6 = _mm256_fcmul_pd(s1s2.6, self.tw6tw10);
                s1s2.7 = _mm256_fcmul_pd(s1s2.7, self.tw7tw11);
                s1s2.8 = _mm256_fcmul_pd(s1s2.8, self.tw8tw12);

                let z0z1 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.0, s0.1),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.0, s1s2.1),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.0, s1s2.1),
                );

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), z0z1.0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(9..).as_mut_ptr().cast(), z0z1.1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(18..).as_mut_ptr().cast(), z0z1.2);

                let z2z3 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.2, s0.3),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.2, s1s2.3),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.2, s1s2.3),
                );

                _mm256_storeu_pd(chunk.get_unchecked_mut(2..).as_mut_ptr().cast(), z2z3.0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(11..).as_mut_ptr().cast(), z2z3.1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(20..).as_mut_ptr().cast(), z2z3.2);

                let z4z5 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.4, s0.5),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.4, s1s2.5),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.4, s1s2.5),
                );

                _mm256_storeu_pd(chunk.get_unchecked_mut(4..).as_mut_ptr().cast(), z4z5.0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(13..).as_mut_ptr().cast(), z4z5.1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(22..).as_mut_ptr().cast(), z4z5.2);

                let z6z7 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.6, s0.7),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.6, s1s2.7),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.6, s1s2.7),
                );

                _mm256_storeu_pd(chunk.get_unchecked_mut(6..).as_mut_ptr().cast(), z6z7.0);
                _mm256_storeu_pd(chunk.get_unchecked_mut(15..).as_mut_ptr().cast(), z6z7.1);
                _mm256_storeu_pd(chunk.get_unchecked_mut(24..).as_mut_ptr().cast(), z6z7.2);

                let z8 = self.bf9.bf3.exec_m128(
                    s0.8,
                    _mm256_castpd256_pd128(s1s2.8),
                    _mm256_extractf128_pd::<1>(s1s2.8),
                );

                _mm_storeu_pd(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), z8.0);
                _mm_storeu_pd(chunk.get_unchecked_mut(17..).as_mut_ptr().cast(), z8.1);
                _mm_storeu_pd(chunk.get_unchecked_mut(26..).as_mut_ptr().cast(), z8.2);
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }

        if dst.len() % 27 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(27).zip(src.chunks_exact(27)) {
                let u0u1 = _mm256_loadu_pd(src.as_ptr().cast());
                let u2u3 = _mm256_loadu_pd(src.get_unchecked(2..).as_ptr().cast());
                let u4u5 = _mm256_loadu_pd(src.get_unchecked(4..).as_ptr().cast());
                let u6u7 = _mm256_loadu_pd(src.get_unchecked(6..).as_ptr().cast());
                let u8u9 = _mm256_loadu_pd(src.get_unchecked(8..).as_ptr().cast());
                let u10u11 = _mm256_loadu_pd(src.get_unchecked(10..).as_ptr().cast());
                let u12u13 = _mm256_loadu_pd(src.get_unchecked(12..).as_ptr().cast());
                let u14u15 = _mm256_loadu_pd(src.get_unchecked(14..).as_ptr().cast());
                let u16u17 = _mm256_loadu_pd(src.get_unchecked(16..).as_ptr().cast());
                let u18u19 = _mm256_loadu_pd(src.get_unchecked(18..).as_ptr().cast());
                let u20u21 = _mm256_loadu_pd(src.get_unchecked(20..).as_ptr().cast());
                let u22u23 = _mm256_loadu_pd(src.get_unchecked(22..).as_ptr().cast());
                let u24u25 = _mm256_loadu_pd(src.get_unchecked(24..).as_ptr().cast());
                let u26 = _mm_loadu_pd(src.get_unchecked(26..).as_ptr().cast());

                let s0 = self.bf9.exec_m128d(
                    _mm256_castpd256_pd128(u0u1),
                    _mm256_extractf128_pd::<1>(u2u3),
                    _mm256_castpd256_pd128(u6u7),
                    _mm256_extractf128_pd::<1>(u8u9),
                    _mm256_castpd256_pd128(u12u13),
                    _mm256_extractf128_pd::<1>(u14u15),
                    _mm256_castpd256_pd128(u18u19),
                    _mm256_extractf128_pd::<1>(u20u21),
                    _mm256_castpd256_pd128(u24u25),
                );

                const HI_LO: i32 = 0b0010_0001;
                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let mut s1s2 = self.bf9.exec(
                    _mm256_permute2f128_pd::<HI_LO>(u0u1, u2u3),
                    u4u5,
                    _mm256_permute2f128_pd::<HI_LO>(u6u7, u8u9),
                    u10u11,
                    _mm256_permute2f128_pd::<HI_LO>(u12u13, u14u15),
                    u16u17,
                    _mm256_permute2f128_pd::<HI_LO>(u18u19, u20u21),
                    u22u23,
                    _mm256_permute2f128_pd::<HI_LO>(u24u25, _mm256_castpd128_pd256(u26)),
                );

                s1s2.1 = _mm256_fcmul_pd(s1s2.1, self.tw1tw2);
                s1s2.2 = _mm256_fcmul_pd(s1s2.2, self.tw2tw4);
                s1s2.3 = _mm256_fcmul_pd(s1s2.3, self.tw3tw6);
                s1s2.4 = _mm256_fcmul_pd(s1s2.4, self.tw4tw8);
                s1s2.5 = _mm256_fcmul_pd(s1s2.5, self.tw5tw9);
                s1s2.6 = _mm256_fcmul_pd(s1s2.6, self.tw6tw10);
                s1s2.7 = _mm256_fcmul_pd(s1s2.7, self.tw7tw11);
                s1s2.8 = _mm256_fcmul_pd(s1s2.8, self.tw8tw12);

                let z0z1 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.0, s0.1),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.0, s1s2.1),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.0, s1s2.1),
                );

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), z0z1.0);
                _mm256_storeu_pd(dst.get_unchecked_mut(9..).as_mut_ptr().cast(), z0z1.1);
                _mm256_storeu_pd(dst.get_unchecked_mut(18..).as_mut_ptr().cast(), z0z1.2);

                let z2z3 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.2, s0.3),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.2, s1s2.3),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.2, s1s2.3),
                );

                _mm256_storeu_pd(dst.get_unchecked_mut(2..).as_mut_ptr().cast(), z2z3.0);
                _mm256_storeu_pd(dst.get_unchecked_mut(11..).as_mut_ptr().cast(), z2z3.1);
                _mm256_storeu_pd(dst.get_unchecked_mut(20..).as_mut_ptr().cast(), z2z3.2);

                let z4z5 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.4, s0.5),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.4, s1s2.5),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.4, s1s2.5),
                );

                _mm256_storeu_pd(dst.get_unchecked_mut(4..).as_mut_ptr().cast(), z4z5.0);
                _mm256_storeu_pd(dst.get_unchecked_mut(13..).as_mut_ptr().cast(), z4z5.1);
                _mm256_storeu_pd(dst.get_unchecked_mut(22..).as_mut_ptr().cast(), z4z5.2);

                let z6z7 = self.bf9.bf3.exec(
                    _mm256_create_pd(s0.6, s0.7),
                    _mm256_permute2f128_pd::<LO_LO>(s1s2.6, s1s2.7),
                    _mm256_permute2f128_pd::<HI_HI>(s1s2.6, s1s2.7),
                );

                _mm256_storeu_pd(dst.get_unchecked_mut(6..).as_mut_ptr().cast(), z6z7.0);
                _mm256_storeu_pd(dst.get_unchecked_mut(15..).as_mut_ptr().cast(), z6z7.1);
                _mm256_storeu_pd(dst.get_unchecked_mut(24..).as_mut_ptr().cast(), z6z7.2);

                let z8 = self.bf9.bf3.exec_m128(
                    s0.8,
                    _mm256_castpd256_pd128(s1s2.8),
                    _mm256_extractf128_pd::<1>(s1s2.8),
                );

                _mm_storeu_pd(dst.get_unchecked_mut(8..).as_mut_ptr().cast(), z8.0);
                _mm_storeu_pd(dst.get_unchecked_mut(17..).as_mut_ptr().cast(), z8.1);
                _mm_storeu_pd(dst.get_unchecked_mut(26..).as_mut_ptr().cast(), z8.2);
            }
        }
        Ok(())
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly27d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly27d {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly27d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        27
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_butterfly27_f64() {
        for i in 1..4 {
            let size = 27usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut ref0 = input.to_vec();
            let radix_forward = AvxButterfly27d::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly27d::new(FftDirection::Inverse);

            let radix_forward_ref = Dft::new(27, FftDirection::Forward).unwrap();

            radix_forward.execute(&mut input).unwrap();
            radix_forward_ref.execute(&mut ref0).unwrap();

            input
                .iter()
                .zip(ref0.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "forward at {idx} a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "forward at {idx} a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 27f64)).collect();

            input
                .iter()
                .zip(src.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });
        }
    }

    #[test]
    fn test_butterfly27_out_of_place_f64() {
        for i in 1..4 {
            let size = 27usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut out_of_place = vec![Complex::<f64>::default(); size];
            let mut ref_input = input.to_vec();
            let radix_forward = AvxButterfly27d::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly27d::new(FftDirection::Inverse);

            let reference_dft = Dft::new(27, FftDirection::Forward).unwrap();
            reference_dft.execute(&mut ref_input).unwrap();

            radix_forward
                .execute_out_of_place(&input, &mut out_of_place)
                .unwrap();

            out_of_place
                .iter()
                .zip(ref_input.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {} at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {} at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse
                .execute_out_of_place(&out_of_place, &mut input)
                .unwrap();

            input = input.iter().map(|&x| x * (1.0 / 27f64)).collect();

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
