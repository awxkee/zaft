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
use crate::traits::FftTrigonometry;
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};
use std::arch::aarch64::*;

pub(crate) struct NeonButterfly4<T> {
    direction: FftDirection,
    multiplier: [T; 4],
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> NeonButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            multiplier: match fft_direction {
                FftDirection::Inverse => [-0.0f64.as_(), 0.0.as_(), -0.0.as_(), 0.0.as_()],
                FftDirection::Forward => [0.0f64.as_(), -0.0.as_(), 0.0.as_(), -0.0.as_()],
            },
        }
    }
}

impl FftExecutor<f32> for NeonButterfly4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let z_mul = unsafe { vld1q_f32(self.multiplier.as_ptr()) };
        let v_i_multiplier = unsafe { vreinterpretq_u32_f32(z_mul) };

        for chunk in in_place.chunks_exact_mut(16) {
            unsafe {
                let uzp0 = vld4q_f64(chunk.as_ptr().cast());
                let uzp1 = vld4q_f64(chunk.get_unchecked(8..).as_ptr().cast());

                let u0 = vreinterpretq_f32_f64(uzp0.0);
                let u1 = vreinterpretq_f32_f64(uzp0.1);
                let u2 = vreinterpretq_f32_f64(uzp0.2);
                let u3 = vreinterpretq_f32_f64(uzp0.3);

                let u4 = vreinterpretq_f32_f64(uzp1.0);
                let u5 = vreinterpretq_f32_f64(uzp1.1);
                let u6 = vreinterpretq_f32_f64(uzp1.2);
                let u7 = vreinterpretq_f32_f64(uzp1.3);

                let t0 = vaddq_f32(u0, u2);
                let t4 = vaddq_f32(u4, u6);
                let t1 = vsubq_f32(u0, u2);
                let t5 = vsubq_f32(u4, u6);
                let t2 = vaddq_f32(u1, u3);
                let t6 = vaddq_f32(u5, u7);
                let mut t3 = vsubq_f32(u1, u3);
                let mut t7 = vsubq_f32(u5, u7);
                t3 = vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(vrev64q_f32(t3)),
                    v_i_multiplier,
                ));
                t7 = vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(vrev64q_f32(t7)),
                    v_i_multiplier,
                ));

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vaddq_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vsubq_f32(t1, t3)),
                );
                let rw1 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vaddq_f32(t5, t7)),
                    vreinterpretq_f64_f32(vsubq_f32(t4, t6)),
                    vreinterpretq_f64_f32(vsubq_f32(t5, t7)),
                );

                vst4q_f64(chunk.as_mut_ptr().cast(), rw0);
                vst4q_f64(chunk.get_unchecked_mut(8..).as_mut_ptr().cast(), rw1);
            }
        }

        let rem = in_place.chunks_exact_mut(16).into_remainder();

        for chunk in rem.chunks_exact_mut(8) {
            unsafe {
                let uzp = vld4q_f64(chunk.as_ptr().cast());

                let a = vreinterpretq_f32_f64(uzp.0);
                let b = vreinterpretq_f32_f64(uzp.1);
                let c = vreinterpretq_f32_f64(uzp.2);
                let d = vreinterpretq_f32_f64(uzp.3);

                let t0 = vaddq_f32(a, c);
                let t1 = vsubq_f32(a, c);
                let t2 = vaddq_f32(b, d);
                let mut t3 = vsubq_f32(b, d);
                t3 = vreinterpretq_f32_u32(veorq_u32(
                    vreinterpretq_u32_f32(vrev64q_f32(t3)),
                    v_i_multiplier,
                ));

                let rw0 = float64x2x4_t(
                    vreinterpretq_f64_f32(vaddq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vaddq_f32(t1, t3)),
                    vreinterpretq_f64_f32(vsubq_f32(t0, t2)),
                    vreinterpretq_f64_f32(vsubq_f32(t1, t3)),
                );

                vst4q_f64(chunk.as_mut_ptr().cast(), rw0);
            }
        }

        let rem = rem.chunks_exact_mut(8).into_remainder();

        for chunk in rem.chunks_exact_mut(4) {
            unsafe {
                let uz0 = vld1q_f32(chunk.get_unchecked(0..).as_ptr().cast());
                let uz1 = vld1q_f32(chunk.get_unchecked(2..).as_ptr().cast());

                let a = vget_low_f32(uz0);
                let b = vget_high_f32(uz0);
                let c = vget_low_f32(uz1);
                let d = vget_high_f32(uz1);

                let t0 = vadd_f32(a, c);
                let t1 = vsub_f32(a, c);
                let t2 = vadd_f32(b, d);
                let mut t3 = vsub_f32(b, d);
                t3 = vreinterpret_f32_u32(veor_u32(
                    vreinterpret_u32_f32(vext_f32::<1>(t3, t3)),
                    vget_low_u32(v_i_multiplier),
                ));

                vst1q_f32(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vcombine_f32(vadd_f32(t0, t2), vadd_f32(t1, t3)),
                );
                vst1q_f32(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vcombine_f32(vsub_f32(t0, t2), vsub_f32(t1, t3)),
                );
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
    }
}

impl FftExecutor<f64> for NeonButterfly4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 4 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }
        let v_i_multiplier = unsafe { vreinterpretq_u64_f64(vld1q_f64(self.multiplier.as_ptr())) };

        for chunk in in_place.chunks_exact_mut(4) {
            unsafe {
                let a = vld1q_f64(chunk.get_unchecked(0..).as_ptr().cast());
                let b = vld1q_f64(chunk.get_unchecked(1..).as_ptr().cast());
                let c = vld1q_f64(chunk.get_unchecked(2..).as_ptr().cast());
                let d = vld1q_f64(chunk.get_unchecked(3..).as_ptr().cast());

                let t0 = vaddq_f64(a, c);
                let t1 = vsubq_f64(a, c);
                let t2 = vaddq_f64(b, d);
                let mut t3 = vsubq_f64(b, d);
                t3 = vreinterpretq_f64_u64(veorq_u64(
                    vreinterpretq_u64_f64(vextq_f64::<1>(t3, t3)),
                    v_i_multiplier,
                ));

                vst1q_f64(
                    chunk.get_unchecked_mut(0..).as_mut_ptr().cast(),
                    vaddq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(1..).as_mut_ptr().cast(),
                    vaddq_f64(t1, t3),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    vsubq_f64(t0, t2),
                );
                vst1q_f64(
                    chunk.get_unchecked_mut(3..).as_mut_ptr().cast(),
                    vsubq_f64(t1, t3),
                );
            }
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::Butterfly4;
    use rand::Rng;

    #[test]
    fn test_butterfly4_f32() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f32>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let mut z_ref = input.to_vec();

            let radix_forward = NeonButterfly4::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly4::new(FftDirection::Inverse);

            let radix4_reference = Butterfly4::new(FftDirection::Forward);
            let radix4_inv_reference = Butterfly4::new(FftDirection::Inverse);

            radix_forward.execute(&mut input).unwrap();
            radix4_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference failed",
                    a.im,
                    b.im,
                    size
                );
            });

            radix_inverse.execute(&mut input).unwrap();
            radix4_inv_reference.execute(&mut z_ref).unwrap();

            input.iter().zip(z_ref.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-5,
                    "a_re {} != b_re {} for size {}, reference inv failed",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-5,
                    "a_im {} != b_im {} for size {}, reference inv failed",
                    a.im,
                    b.im,
                    size
                );
            });

            input = input.iter().map(|&x| x * (1.0 / 4f32)).collect();

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
    fn test_butterfly4_f64() {
        for i in 1..6 {
            let size = 4usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();
            let radix_forward = NeonButterfly4::new(FftDirection::Forward);
            let radix_inverse = NeonButterfly4::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 4f64)).collect();

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
