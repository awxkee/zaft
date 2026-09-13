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
use crate::FftSample;
use crate::complex_fma::c_conj_mul_fast;
use crate::complex_fma::c_mul_fast;
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

pub(crate) trait ComplexArith<T> {
    // a * b
    fn mul(&self, a: &[Complex<T>], b: &[Complex<T>], dst: &mut [Complex<T>]);
    // for each chunk(a, cut_width) * chunk(b, cut_width)
    fn mul_and_cut(
        &self,
        a: &[Complex<T>],
        original_width: usize,
        b: &[Complex<T>],
        cut_width: usize,
        dst: &mut [Complex<T>],
    );
    // Real a times complex b: duplicate a into both lanes for a componentwise multiply.
    #[cfg(not(target_arch = "wasm32"))]
    fn mul_expand_to_complex(&self, a: &[T], b: &[Complex<T>], dst: &mut [Complex<T>]);
    // (a*b).conj()
    fn mul_conjugate_in_place(&self, dst: &mut [Complex<T>], b: &[Complex<T>]);
    // a.conj() * b
    fn conjugate_mul_by_b(&self, a: &[Complex<T>], b: &[Complex<T>], dst: &mut [Complex<T>]);
    // 2 * Re(a.conj() * b). All three slices must have the same length.
    fn conjugate_mul_real_doubled(&self, a: &[Complex<T>], b: &[Complex<T>], dst: &mut [T]);
}

pub(crate) trait ComplexArithFactory<T> {
    fn make_complex_arith() -> Arc<dyn ComplexArith<T> + Send + Sync>;
}

macro_rules! default_arith_module {
    () => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            #[cfg(feature = "fcma")]
            if std::arch::is_aarch64_feature_detected!("fcma") {
                use crate::neon::NeonFcmaSpectrumArithmetic;
                return Arc::new(NeonFcmaSpectrumArithmetic {
                    phantom_data: Default::default(),
                });
            }
            use crate::neon::NeonSpectrumArithmetic;
            Arc::new(NeonSpectrumArithmetic {
                phantom_data: Default::default(),
            })
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                use crate::avx::AvxSpectrumArithmetic;
                return Arc::new(AvxSpectrumArithmetic {
                    phantom_data: Default::default(),
                });
            }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(ScalarSpectrumArithmetic {
                phantom_data: Default::default(),
            })
        }
    }};
}

impl ComplexArithFactory<f32> for f32 {
    fn make_complex_arith() -> Arc<dyn ComplexArith<f32> + Send + Sync> {
        static ARITHMETIC_MODULE_SINGLE: OnceLock<Arc<dyn ComplexArith<f32> + Send + Sync>> =
            OnceLock::new();
        ARITHMETIC_MODULE_SINGLE
            .get_or_init(|| default_arith_module!())
            .clone()
    }
}

impl ComplexArithFactory<f64> for f64 {
    fn make_complex_arith() -> Arc<dyn ComplexArith<f64> + Send + Sync> {
        static ARITHMETIC_MODULE_DOUBLE: OnceLock<Arc<dyn ComplexArith<f64> + Send + Sync>> =
            OnceLock::new();
        ARITHMETIC_MODULE_DOUBLE
            .get_or_init(|| default_arith_module!())
            .clone()
    }
}

#[allow(unused)]
#[derive(Clone)]
pub(crate) struct ScalarSpectrumArithmetic<T: Clone> {
    phantom_data: PhantomData<T>,
}

impl<T: FftSample> ComplexArith<T> for ScalarSpectrumArithmetic<T>
where
    f64: AsPrimitive<T>,
{
    fn mul(&self, a: &[Complex<T>], b: &[Complex<T>], dst: &mut [Complex<T>]) {
        for ((dst, src), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = c_mul_fast(*src, *twiddle);
        }
    }

    fn mul_and_cut(
        &self,
        a: &[Complex<T>],
        original_width: usize,
        b: &[Complex<T>],
        cut_width: usize,
        dst: &mut [Complex<T>],
    ) {
        for ((source, twiddle), dst) in b
            .chunks_exact(cut_width)
            .zip(a.chunks_exact(original_width))
            .zip(dst.chunks_exact_mut(cut_width))
        {
            for ((&source, &twiddle), dst) in source.iter().zip(twiddle.iter()).zip(dst.iter_mut())
            {
                *dst = c_mul_fast(source, twiddle);
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn mul_expand_to_complex(&self, a: &[T], b: &[Complex<T>], dst: &mut [Complex<T>]) {
        for ((dst, &src), &twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *dst = Complex::new(src * twiddle.re, src * twiddle.im);
        }
    }

    fn mul_conjugate_in_place(&self, dst: &mut [Complex<T>], b: &[Complex<T>]) {
        for (scratch_cell, &twiddle) in dst.iter_mut().zip(b.iter()) {
            *scratch_cell = c_mul_fast(*scratch_cell, twiddle).conj();
        }
    }

    fn conjugate_mul_by_b(&self, a: &[Complex<T>], b: &[Complex<T>], dst: &mut [Complex<T>]) {
        for ((buffer_entry, inner_entry), twiddle) in dst.iter_mut().zip(a.iter()).zip(b.iter()) {
            *buffer_entry = c_conj_mul_fast(*inner_entry, *twiddle);
        }
    }

    fn conjugate_mul_real_doubled(&self, a: &[Complex<T>], b: &[Complex<T>], dst: &mut [T]) {
        assert_eq!(a.len(), dst.len());
        assert_eq!(b.len(), dst.len());
        for ((dst, a), b) in dst.iter_mut().zip(a).zip(b) {
            let re = a.re * b.re + a.im * b.im;
            *dst = re + re;
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    macro_rules! test_real_chirp {
        ($name:ident, $ty:ty) => {
            #[test]
            fn $name() {
                #[allow(unused_mut)]
                let mut backends: Vec<Box<dyn ComplexArith<$ty>>> =
                    vec![Box::new(ScalarSpectrumArithmetic {
                        phantom_data: PhantomData,
                    })];
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                backends.push(Box::new(crate::neon::NeonSpectrumArithmetic {
                    phantom_data: PhantomData,
                }));
                #[cfg(all(target_arch = "aarch64", feature = "fcma"))]
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    backends.push(Box::new(crate::neon::NeonFcmaSpectrumArithmetic {
                        phantom_data: PhantomData,
                    }));
                }
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    backends.push(Box::new(crate::avx::AvxSpectrumArithmetic {
                        phantom_data: PhantomData,
                    }));
                }
                for (backend, ops) in backends.iter().enumerate() {
                    // Exercise every SIMD remainder and unaligned subslices, with guards
                    // to catch stores beyond the requested output.
                    for n in (0..=65).chain([127, 257]) {
                        for offset in [0, 1, 3] {
                            let input: Vec<$ty> = (0..n + offset)
                                .map(|i| ((i * 17 % 31) as $ty - 15.0) * 0.25)
                                .collect();
                            let twiddles: Vec<Complex<$ty>> = (0..n + offset)
                                .map(|i| {
                                    let angle = i as f64 * 0.37;
                                    Complex::new(angle.cos() as $ty, angle.sin() as $ty)
                                })
                                .collect();
                            let guard = Complex::new(123.0, -456.0);
                            let mut output = vec![guard; n + offset + 3];
                            ops.mul_expand_to_complex(
                                &input[offset..],
                                &twiddles[offset..],
                                &mut output[offset..offset + n],
                            );
                            for i in offset..offset + n {
                                let expected = twiddles[i] * input[i];
                                assert_eq!(
                                    output[i], expected,
                                    "backend {backend}, n {n}, offset {offset}, index {i}"
                                );
                            }
                            assert!(output[..offset].iter().all(|&v| v == guard));
                            assert!(output[offset + n..].iter().all(|&v| v == guard));
                        }
                    }
                }
            }
        };
    }

    test_real_chirp!(real_chirp_f32, f32);
    test_real_chirp!(real_chirp_f64, f64);

    macro_rules! test_real_projection {
        ($name:ident, $ty:ty) => {
            #[test]
            fn $name() {
                #[allow(unused_mut)]
                let mut backends: Vec<(Box<dyn ComplexArith<$ty>>, bool)> = vec![(
                    Box::new(ScalarSpectrumArithmetic {
                        phantom_data: PhantomData,
                    }),
                    false,
                )];
                #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                backends.push((
                    Box::new(crate::neon::NeonSpectrumArithmetic {
                        phantom_data: PhantomData,
                    }),
                    false,
                ));
                #[cfg(all(target_arch = "aarch64", feature = "fcma"))]
                if std::arch::is_aarch64_feature_detected!("fcma") {
                    backends.push((
                        Box::new(crate::neon::NeonFcmaSpectrumArithmetic {
                            phantom_data: PhantomData,
                        }),
                        false,
                    ));
                }
                #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                if std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("fma")
                {
                    backends.push((
                        Box::new(crate::avx::AvxSpectrumArithmetic {
                            phantom_data: PhantomData,
                        }),
                        true,
                    ));
                }
                for (backend, (ops, fused)) in backends.iter().enumerate() {
                    // Every unrolled/vector remainder, independently unaligned
                    // inputs/output, and guards beyond the requested stores.
                    for n in (0..=97).chain([127, 257]) {
                        for offset in [0, 1, 3] {
                            let a_offset = (offset + 1) % 4;
                            let b_offset = (offset + 2) % 4;
                            let a: Vec<_> = (0..n + a_offset)
                                .map(|i| {
                                    Complex::new(
                                        ((i * 17 % 97) as $ty - 48.0) / 63.0,
                                        ((i * 31 % 101) as $ty - 50.0) / 63.0,
                                    )
                                })
                                .collect();
                            let b: Vec<_> = (0..n + b_offset)
                                .map(|i| {
                                    Complex::new(
                                        (i as f64 * 0.37).cos() as $ty,
                                        (i as f64 * 0.37).sin() as $ty,
                                    )
                                })
                                .collect();
                            let guard = 123.0;
                            let mut output = vec![guard; n + offset + 3];
                            ops.conjugate_mul_real_doubled(
                                &a[a_offset..],
                                &b[b_offset..],
                                &mut output[offset..offset + n],
                            );
                            for i in 0..n {
                                let a = a[a_offset + i];
                                let b = b[b_offset + i];
                                let re = if *fused {
                                    a.re.mul_add(b.re, a.im * b.im)
                                } else {
                                    a.re * b.re + a.im * b.im
                                };
                                assert_eq!(
                                    output[offset + i].to_bits(),
                                    (re + re).to_bits(),
                                    "backend {backend}, n {n}, offset {offset}, index {i}"
                                );
                            }
                            assert!(output[..offset].iter().all(|&v| v == guard));
                            assert!(output[offset + n..].iter().all(|&v| v == guard));
                        }
                    }
                    // Cancellation distinguishes FMA from two separately rounded
                    // products. Include zeros, subnormals and non-finite values.
                    let cases = [
                        (
                            Complex::new(1.0 + <$ty>::EPSILON, -1.0),
                            Complex::new(1.0 - <$ty>::EPSILON, 1.0),
                        ),
                        (Complex::new(-0.0, -0.0), Complex::new(1.0, 1.0)),
                        (
                            Complex::new(<$ty>::MIN_POSITIVE, 0.0),
                            Complex::new(0.5, 0.0),
                        ),
                        (Complex::new(<$ty>::INFINITY, 1.0), Complex::new(1.0, 0.0)),
                        (Complex::new(<$ty>::NAN, 1.0), Complex::new(1.0, 1.0)),
                    ];
                    // Rotate cases across all vector lanes and the scalar tail.
                    for shift in 0..cases.len() {
                        let a: Vec<_> = (0..65)
                            .map(|i| cases[(i + shift) % cases.len()].0)
                            .collect();
                        let b: Vec<_> = (0..65)
                            .map(|i| cases[(i + shift) % cases.len()].1)
                            .collect();
                        let mut output = vec![0.0; a.len()];
                        ops.conjugate_mul_real_doubled(&a, &b, &mut output);
                        for ((actual, a), b) in output.into_iter().zip(a).zip(b) {
                            let re = if *fused {
                                a.re.mul_add(b.re, a.im * b.im)
                            } else {
                                a.re * b.re + a.im * b.im
                            };
                            let expected = re + re;
                            if expected.is_nan() {
                                assert!(actual.is_nan());
                            } else {
                                assert_eq!(
                                    actual.to_bits(),
                                    expected.to_bits(),
                                    "backend {backend}"
                                );
                            }
                        }
                    }
                }
            }
        };
    }

    test_real_projection!(real_projection_f32, f32);
    test_real_projection!(real_projection_f64, f64);
}
