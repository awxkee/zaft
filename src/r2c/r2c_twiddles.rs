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
use num_complex::Complex;
use num_traits::{AsPrimitive, MulAdd, Num};
use std::marker::PhantomData;
use std::ops::{Add, Neg, Sub};

pub(crate) trait R2CTwiddlesFactory<T> {
    fn make_r2c_twiddles_handler() -> Box<dyn R2CTwiddlesHandler<T> + Send + Sync>;
}

pub(crate) trait R2CTwiddlesHandler<T> {
    fn handle(&self, twiddles: &[Complex<T>], left: &mut [Complex<T>], right: &mut [Complex<T>]);
}

#[allow(unused)]
struct R2CHandler<T> {
    phantom_data: PhantomData<T>,
}

impl R2CTwiddlesFactory<f32> for f32 {
    fn make_r2c_twiddles_handler() -> Box<dyn R2CTwiddlesHandler<f32> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::R2CNeonFcmaTwiddles;
                    return Box::new(R2CNeonFcmaTwiddles {});
                }
            }
            use crate::neon::R2CNeonTwiddles;
            Box::new(R2CNeonTwiddles {})
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Box::new(R2CHandler {
                phantom_data: PhantomData::<f32>,
            })
        }
    }
}

impl R2CTwiddlesFactory<f64> for f64 {
    fn make_r2c_twiddles_handler() -> Box<dyn R2CTwiddlesHandler<f64> + Send + Sync> {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            {
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    use crate::neon::R2CNeonFcmaTwiddles;
                    return Box::new(R2CNeonFcmaTwiddles {});
                }
            }
            use crate::neon::R2CNeonTwiddles;
            Box::new(R2CNeonTwiddles {})
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Box::new(R2CHandler {
                phantom_data: PhantomData::<f64>,
            })
        }
    }
}

impl<
    T: Copy
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + MulAdd<T, Output = T>
        + Neg<Output = T>
        + 'static
        + Num,
> R2CTwiddlesHandler<T> for R2CHandler<T>
where
    f64: AsPrimitive<T>,
{
    fn handle(&self, twiddles: &[Complex<T>], left: &mut [Complex<T>], right: &mut [Complex<T>]) {
        for ((twiddle, out), out_rev) in twiddles
            .iter()
            .zip(left.iter_mut())
            .zip(right.iter_mut().rev())
        {
            let sum = *out + *out_rev;
            let diff = *out - *out_rev;
            let half: T = 0.5f64.as_();
            //
            let twiddled_re_sum = sum.im * twiddle.re;
            let twiddled_im_sum = sum.im * twiddle.im;
            let twiddled_re_diff = diff.re * twiddle.re;
            let twiddled_im_diff = diff.re * twiddle.im;
            let half_sum_re = half * sum.re;
            let half_diff_im = half * diff.im;

            let output_twiddled_real = twiddled_re_sum + twiddled_im_diff;
            let output_twiddled_im = twiddled_im_sum - twiddled_re_diff;

            // We finally have all the data we need to write the transformed data back out where we found it.
            *out = Complex {
                re: half_sum_re + output_twiddled_real,
                im: half_diff_im + output_twiddled_im,
            };

            *out_rev = Complex {
                re: half_sum_re - output_twiddled_real,
                im: output_twiddled_im - half_diff_im,
            };
        }
    }
}
