/*
 * // Copyright (c) Radzivon Bartoshyk 01/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //2028
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
use crate::err::try_vec;
use crate::fast_divider::DividerU64;
use crate::good_thomas_small::{LutGather, LutGatherFactory};
use crate::prime_factors::{PrimeFactors, primitive_root};
use crate::util::{compute_twiddle, validate_scratch};
use crate::{C2RFftExecutor, FftDirection, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

/// Reconstruct complex bins from the two halves of the scaled real convolution.
/// `y[p] = (first + lo[p] + hi[p], (lo[p] - hi[p]) * signs[p])`.
pub(crate) trait RadersRfftCombiner<T> {
    fn combine(&self, y: &mut [Complex<T>], lo: &[T], hi: &[T], signs: &[T], first: T);
}

pub(crate) trait RadersRfftCombinerFactory<T> {
    fn make_raders_rfft_combiner() -> Arc<dyn RadersRfftCombiner<T> + Send + Sync>;
}

#[allow(unused)]
pub(crate) struct ScalarRadersRfftCombiner<T> {
    phantom_data: PhantomData<T>,
}

impl<T: FftSample> RadersRfftCombiner<T> for ScalarRadersRfftCombiner<T> {
    fn combine(&self, y: &mut [Complex<T>], lo: &[T], hi: &[T], signs: &[T], first: T) {
        combine_scalar(y, lo, hi, signs, first);
    }
}

#[inline]
pub(crate) fn combine_scalar<T: FftSample>(
    y: &mut [Complex<T>],
    lo: &[T],
    hi: &[T],
    signs: &[T],
    first: T,
) {
    for (((y, &lo), &hi), &sign) in y.iter_mut().zip(lo).zip(hi).zip(signs) {
        *y = Complex::new(first + (lo + hi), (lo - hi) * sign);
    }
}

macro_rules! raders_rfft_combiner_factory {
    ($ty: ty) => {
        impl RadersRfftCombinerFactory<$ty> for $ty {
            fn make_raders_rfft_combiner() -> Arc<dyn RadersRfftCombiner<$ty> + Send + Sync> {
                static Q: OnceLock<Arc<dyn RadersRfftCombiner<$ty> + Send + Sync>> =
                    OnceLock::new();
                Q.get_or_init(|| {
                    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
                    {
                        use crate::neon::NeonRadersRfftCombiner;
                        Arc::new(NeonRadersRfftCombiner {})
                    }
                    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
                    {
                        use crate::util::has_valid_avx;
                        if has_valid_avx() {
                            use crate::avx::AvxRadersRfftCombiner;
                            return Arc::new(AvxRadersRfftCombiner {});
                        }
                    }
                    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
                    {
                        Arc::new(ScalarRadersRfftCombiner {
                            phantom_data: PhantomData::<$ty>,
                        })
                    }
                })
                .clone()
            }
        }
    };
}

raders_rfft_combiner_factory!(f32);
raders_rfft_combiner_factory!(f64);

/// Rader's algorithm for a prime-length real-to-complex FFT using one real convolution.
///
/// For prime `N` with primitive root `g`, `X[g^-p] - x[0]` is a cyclic convolution
/// of length `M = N - 1` between `a[q] = x[g^q]` and `b[q] = w^(g^-q)`.
///
/// Let `H = M / 2`. The real part of `b` is H-periodic and its imaginary part is
/// H-antiperiodic. Their spectra are nonzero only on even and odd bins, respectively.
/// Interleaving those spectra gives the transform of the real kernel `Re(b) + Im(b)`.
///
/// ```text
/// z = c2r_M(r2c_M(a) * r2c_M((Re(b) + Im(b)) / (2M)))
/// re[p] = z[p] + z[p + H]
/// im[p] = z[p] - z[p + H]
/// ```
///
/// The kernel includes both inverse-FFT normalization and the reconstruction's 1/2.
/// Execution uses one R2C and one C2R of length M, without odd-spectrum expansion
/// or post-transform phase rotations. The shifted input permutation used below
/// maps convolution position p to output bin `g^-(p + 1)`.
pub(crate) struct RadersRfft<T> {
    convolve_r2c: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    convolve_c2r: Arc<dyn C2RFftExecutor<T> + Send + Sync>,
    /// Interleaved C[2k] and S[2k+1], scaled by 1/(2M); H + 1 entries.
    convolve_twiddles: Vec<Complex<T>>,
    /// +1 for a stored output bin, -1 if the output permutation needs its conjugate.
    output_signs: Vec<T>,
    input_indices: Vec<usize>,
    /// Output bin `k + 1` is read from convolution index `output_indices[k]`.
    output_indices: Vec<u32>,
    gatherer: Arc<dyn LutGather<T> + Send + Sync>,
    combiner: Arc<dyn RadersRfftCombiner<T> + Send + Sync>,
    execution_length: usize,
    convolve_scratch_length: usize,
}

impl<T: FftSample + LutGatherFactory<T> + RadersRfftCombinerFactory<T>> RadersRfft<T>
where
    f64: AsPrimitive<T>,
{
    /// Both convolution transforms must have real length N - 1.
    pub(crate) fn new(
        size: usize,
        convolve_r2c: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
        convolve_c2r: Arc<dyn C2RFftExecutor<T> + Send + Sync>,
    ) -> Result<RadersRfft<T>, ZaftError> {
        assert!(
            PrimeFactors::from_number(size as u64).is_prime(),
            "Input length for Rader's must be a prime number"
        );
        assert!(size >= 3, "Rader's rfft requires an odd prime");
        assert!(
            size <= u32::MAX as usize,
            "Rader's rfft output gather uses 32-bit indices"
        );

        let m = size - 1;
        let h = m / 2;

        assert_eq!(convolve_r2c.real_length(), m);
        assert_eq!(convolve_r2c.complex_length(), h + 1);
        assert_eq!(convolve_c2r.real_length(), m);
        assert_eq!(convolve_c2r.complex_length(), h + 1);

        let dividing_len = DividerU64::new(size as u64);

        // compute the primitive root and its inverse for this size
        let primitive_root =
            primitive_root(size as u64).ok_or(ZaftError::CantFindPrimitiveRootFor(size as u64))?;

        // compute the multiplicative inverse of primative_root mod len and vice versa.
        // i64::extended_gcd will compute both the inverse of left mod right, and the inverse of right mod left,
        // but we're only going to use one of them
        // the primitive root inverse might be negative, if o make it positive by wrapping
        let gcd_data = i64::extended_gcd(&(primitive_root as i64), &(size as i64));
        let primitive_root_inverse = if gcd_data.x >= 0 {
            gcd_data.x
        } else {
            gcd_data.x + size as i64
        } as u64;

        // Keep the even and odd kernel spectra separate during planning so the
        // known zero bins do not acquire rounding noise from adding the kernels.
        // 1/M normalizes the inverse FFT, and 1/2 is folded in for reconstruction.
        let inner_fft_scale: T = (0.5f64 / m as f64).as_();
        let mut b_re = try_vec![T::zero(); m];
        let mut b_im = try_vec![T::zero(); m];
        let mut twiddle_input = 1u64;
        for (re, im) in b_re.iter_mut().zip(b_im.iter_mut()) {
            let twiddle = compute_twiddle::<T>(twiddle_input as usize, size, FftDirection::Forward)
                * inner_fft_scale;
            *re = twiddle.re;
            *im = twiddle.im;
            twiddle_input = (twiddle_input * primitive_root_inverse) % dividing_len;
        }

        let mut c_spectrum = try_vec![Complex::<T>::zero(); h + 1];
        let mut s_spectrum = try_vec![Complex::<T>::zero(); h + 1];
        convolve_r2c.execute(&b_re, &mut c_spectrum)?;
        convolve_r2c.execute(&b_im, &mut s_spectrum)?;

        // Re(b) is H-periodic so C is non-zero on even bins only,
        // Im(b) is H-antiperiodic so S is non-zero on odd bins only.
        let mut convolve_twiddles = c_spectrum;
        for (dst, &src) in convolve_twiddles
            .iter_mut()
            .skip(1)
            .step_by(2)
            .zip(s_spectrum.iter().skip(1).step_by(2))
        {
            *dst = src;
        }

        // a[q] = x[g^(q + 1)]
        let mut input_index = 1u64;
        let input_indices = (0..m)
            .map(|_| {
                input_index = (input_index * primitive_root) % dividing_len;
                (input_index - 1) as usize
            })
            .collect::<Vec<_>>();

        // y[p] = X[g^-(p + 1)] - x[0] for p in 0..H, the second half is conj(y[p]).
        // Every output bin k in 1..=H is either g^-(p + 1) itself or its mirror N - g^-(p + 1).
        let mut output_indices = try_vec![0u32; h];
        let mut output_signs = try_vec![1f64.as_(); h];
        let mut output_index = 1u64;
        for (p, sign) in output_signs.iter_mut().enumerate() {
            output_index = (output_index * primitive_root_inverse) % dividing_len;
            let k = output_index as usize;
            if k <= h {
                output_indices[k - 1] = p as u32;
            } else {
                output_indices[size - k - 1] = p as u32;
                *sign = (-1f64).as_();
            }
        }

        let convolve_scratch_length = convolve_r2c
            .complex_scratch_length()
            .max(convolve_c2r.complex_scratch_length());

        Ok(RadersRfft {
            convolve_r2c,
            convolve_c2r,
            convolve_twiddles,
            output_signs,
            input_indices,
            output_indices,
            gatherer: T::make_gatherer(),
            combiner: T::make_raders_rfft_combiner(),
            execution_length: size,
            convolve_scratch_length,
        })
    }

    #[inline(always)]
    pub(crate) fn execute_impl(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;

        assert_eq!(size_of::<T>() * 2, size_of::<Complex<T>>());
        assert_eq!(align_of::<T>(), align_of::<Complex<T>>());

        let h = self.execution_length / 2;
        let m = h * 2;

        let scratch = validate_scratch!(scratch, self.complex_scratch_length());
        let (a_spectrum, rem) = scratch.split_at_mut(h + 1);
        // M reals == H complex: reuse the permuted input buffer for the convolution output.
        let (real_buffer_c, convolve_scratch) = rem.split_at_mut(h);
        let real_buffer: &mut [T] =
            unsafe { std::slice::from_raw_parts_mut(real_buffer_c.as_mut_ptr().cast(), m) };

        for (input, complex) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.complex_length()))
        {
            let (buffer_first, buffer) = input.split_first().unwrap();
            let buffer_first_val = *buffer_first;

            for (scratch_element, &buffer_idx) in
                real_buffer.iter_mut().zip(self.input_indices.iter())
            {
                *scratch_element = unsafe { *buffer.get_unchecked(buffer_idx) }
            }
            self.convolve_r2c
                .execute_with_scratch(real_buffer, a_spectrum, convolve_scratch)?;

            // a_spectrum[0] is the sum of elements 1..len, add the first input to get X[0]
            complex[0] = Complex::new(buffer_first_val + a_spectrum[0].re, T::zero());

            for (value, &twiddle) in a_spectrum.iter_mut().zip(&self.convolve_twiddles) {
                *value = *value * twiddle;
            }

            self.convolve_c2r
                .execute_with_scratch(a_spectrum, real_buffer, convolve_scratch)?;

            let (lo, hi) = real_buffer.split_at(h);
            // The spectrum is dead after C2R, so reuse it for the reconstructed bins.
            self.combiner.combine(
                &mut a_spectrum[..h],
                lo,
                hi,
                &self.output_signs,
                buffer_first_val,
            );

            self.gatherer
                .gather(&a_spectrum[..h], &mut complex[1..], &self.output_indices);
        }
        Ok(())
    }
}

impl<T: FftSample + LutGatherFactory<T> + RadersRfftCombinerFactory<T>> R2CFftExecutor<T>
    for RadersRfft<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        crate::util::validate_oof_block_sizes(
            input.len(),
            self.real_length(),
            output.len(),
            self.complex_length(),
        )?;
        let mut scratch = try_vec![Complex::zero(); self.complex_scratch_length()];
        self.execute_impl(input, output, scratch.as_mut_slice())
    }

    fn execute_with_scratch(
        &self,
        input: &[T],
        output: &mut [Complex<T>],
        scratch: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        self.execute_impl(input, output, scratch)
    }

    #[inline]
    fn real_length(&self) -> usize {
        self.execution_length
    }

    #[inline]
    fn complex_length(&self) -> usize {
        self.execution_length / 2 + 1
    }

    #[inline]
    fn complex_scratch_length(&self) -> usize {
        let h = self.execution_length / 2;
        (h + 1) + h + self.convolve_scratch_length
    }
}

#[cfg(test)]
mod tests {
    use crate::dft::Dft;
    use crate::r2c::rfft_raders::RadersRfft;
    use crate::{FftDirection, FftExecutor, R2CFftExecutor, Zaft};
    use num_complex::Complex;
    use num_traits::Zero;

    fn check_f64(n: usize) {
        let src = (0..n)
            .map(|i| ((i * 7919 + 13) % 1000) as f64 / 250.0 - 2.0)
            .collect::<Vec<f64>>();
        let mx = RadersRfft::new(
            n,
            Zaft::make_r2c_fft_f64(n - 1).unwrap(),
            Zaft::make_c2r_fft_f64(n - 1).unwrap(),
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(n, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let mut complex_output = vec![Complex::<f64>::zero(); n / 2 + 1];
        mx.execute(&src, &mut complex_output).unwrap();
        let tol = 1e-9 * n as f64;
        reference_value
            .iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re).abs() < tol,
                    "a_re {} != b_re {} at {idx} for n={n}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im).abs() < tol,
                    "a_im {} != b_im {} at {idx} for n={n}",
                    a.im,
                    b.im,
                );
            });
    }

    fn check_f32(n: usize) {
        let src = (0..n)
            .map(|i| ((i * 7919 + 13) % 1000) as f32 / 250.0 - 2.0)
            .collect::<Vec<f32>>();
        let mx = RadersRfft::new(
            n,
            Zaft::make_r2c_fft_f32(n - 1).unwrap(),
            Zaft::make_c2r_fft_f32(n - 1).unwrap(),
        )
        .unwrap();
        let mut reference_value = src
            .iter()
            .map(|x| Complex::new(*x as f64, 0.0))
            .collect::<Vec<_>>();
        let dft = Dft::new(n, FftDirection::Forward).unwrap();
        dft.execute(&mut reference_value).unwrap();

        let mut complex_output = vec![Complex::<f32>::zero(); n / 2 + 1];
        mx.execute(&src, &mut complex_output).unwrap();
        let tol = 1e-4 * n as f64;
        reference_value
            .iter()
            .zip(complex_output.iter())
            .enumerate()
            .for_each(|(idx, (a, b))| {
                assert!(
                    (a.re - b.re as f64).abs() < tol,
                    "a_re {} != b_re {} at {idx} for n={n}",
                    a.re,
                    b.re,
                );
                assert!(
                    (a.im - b.im as f64).abs() < tol,
                    "a_im {} != b_im {} at {idx} for n={n}",
                    a.im,
                    b.im,
                );
            });
    }

    #[test]
    fn test_raders_rfft() {
        // covers odd and even H = (n - 1) / 2
        for n in [3, 5, 7, 11, 13, 17, 19, 23, 101, 103, 1009, 1013, 4099] {
            check_f64(n);
        }
    }

    #[test]
    fn test_raders_rfft_f32() {
        for n in [3, 5, 7, 11, 13, 17, 19, 23, 101, 103, 1009, 1013] {
            check_f32(n);
        }
    }

    #[test]
    fn test_raders_rfft_structured_inputs() {
        // DC, impulses, and high/low frequency tones expose normalization and
        // conjugation errors that can hide in arbitrary input. Both parities of H
        // and SIMD tails are exercised in one batched call with poisoned scratch.
        for n in [3usize, 7, 11, 13, 17, 31, 101, 257] {
            let mut input = vec![0.0f64; n * 8];
            input[n..2 * n].fill(1.0);
            input[2 * n] = 1.0;
            input[4 * n - 1] = 1.0;
            for (row, k, sine) in [
                (4, 1, false),
                (5, 1, true),
                (6, n / 2, false),
                (7, n / 2, true),
            ] {
                for i in 0..n {
                    let phase = std::f64::consts::TAU * (i * k) as f64 / n as f64;
                    input[row * n + i] = if sine { phase.sin() } else { phase.cos() };
                }
            }
            let mut reference = input
                .iter()
                .map(|&v| Complex::new(v, 0.0))
                .collect::<Vec<_>>();
            Dft::new(n, FftDirection::Forward)
                .unwrap()
                .execute(&mut reference)
                .unwrap();
            let plan = RadersRfft::new(
                n,
                Zaft::make_r2c_fft_f64(n - 1).unwrap(),
                Zaft::make_c2r_fft_f64(n - 1).unwrap(),
            )
            .unwrap();
            let mut output = vec![Complex::new(f64::NAN, f64::NAN); (n / 2 + 1) * 8];
            let mut scratch = vec![Complex::new(f64::NAN, f64::NAN); plan.complex_scratch_length()];
            plan.execute_with_scratch(&input, &mut output, &mut scratch)
                .unwrap();
            let input_f32 = input.iter().map(|&v| v as f32).collect::<Vec<_>>();
            let plan_f32 = RadersRfft::new(
                n,
                Zaft::make_r2c_fft_f32(n - 1).unwrap(),
                Zaft::make_c2r_fft_f32(n - 1).unwrap(),
            )
            .unwrap();
            let mut output_f32 = vec![Complex::new(f32::NAN, f32::NAN); output.len()];
            let mut scratch_f32 =
                vec![Complex::new(f32::NAN, f32::NAN); plan_f32.complex_scratch_length()];
            plan_f32
                .execute_with_scratch(&input_f32, &mut output_f32, &mut scratch_f32)
                .unwrap();
            for row in 0..8 {
                for k in 0..n / 2 + 1 {
                    let expected = reference[row * n + k];
                    let index = row * (n / 2 + 1) + k;
                    assert!(
                        (output[index] - expected).norm() < 1e-10 * n as f64,
                        "f64 n {n} row {row} bin {k}"
                    );
                    let value = output_f32[index];
                    assert!(
                        (Complex::new(value.re as f64, value.im as f64) - expected).norm()
                            < 2e-6 * n as f64,
                        "f32 n {n} row {row} bin {k}"
                    );
                }
            }
            assert!(matches!(
                plan.execute_with_scratch(&input, &mut output, &mut scratch[1..]),
                Err(crate::ZaftError::ScratchBufferIsTooSmall(_, _))
            ));
        }
    }

    #[test]
    fn test_raders_rfft_large_against_complex_fft() {
        for n in [2003usize, 4093, 65537] {
            let input = (0..n)
                .map(|i| ((i * 7919 + 13) % 97) as f32 * 0.125 - 6.0)
                .collect::<Vec<_>>();
            let input_f64 = input.iter().map(|&v| v as f64).collect::<Vec<_>>();
            let mut reference = input_f64
                .iter()
                .map(|&v| Complex::new(v, 0.0))
                .collect::<Vec<_>>();
            Zaft::make_forward_fft_f64(n)
                .unwrap()
                .execute(&mut reference)
                .unwrap();
            let mut output = vec![Complex::<f64>::zero(); n / 2 + 1];
            Zaft::make_r2c_fft_f64(n)
                .unwrap()
                .execute(&input_f64, &mut output)
                .unwrap();
            let mut output_f32 = vec![Complex::<f32>::zero(); n / 2 + 1];
            Zaft::make_r2c_fft_f32(n)
                .unwrap()
                .execute(&input, &mut output_f32)
                .unwrap();
            let mut error_f64 = 0.0;
            let mut error_f32 = 0.0;
            let mut norm = 0.0;
            for ((&a, &b), &c) in reference.iter().zip(&output).zip(&output_f32) {
                error_f64 += (a - b).norm_sqr();
                error_f32 += (a - Complex::new(c.re as f64, c.im as f64)).norm_sqr();
                norm += a.norm_sqr();
            }
            assert!((error_f64 / norm).sqrt() < 1e-12, "f64 n {n}");
            assert!((error_f32 / norm).sqrt() < 2e-6, "f32 n {n}");
        }
    }

    #[test]
    fn test_raders_rfft_combiner() {
        use crate::r2c::rfft_raders::{RadersRfftCombinerFactory, combine_scalar};
        for len in (0..=65).chain([127, 257]) {
            // Offset slices exercise unaligned loads; guards catch overlong stores.
            for offset in [0, 1, 3] {
                let lo: Vec<f64> = (0..len + offset).map(|i| (i % 7) as f64 - 3.0).collect();
                let hi: Vec<f64> = (0..len + offset).map(|i| (i % 5) as f64 * 0.5).collect();
                let signs: Vec<f64> = (0..len + offset)
                    .map(|i| if i % 3 == 0 { -1.0 } else { 1.0 })
                    .collect();
                let guard = Complex::new(123.0, -456.0);
                let mut expected = vec![guard; len + offset + 3];
                combine_scalar(
                    &mut expected[offset..offset + len],
                    &lo[offset..],
                    &hi[offset..],
                    &signs[offset..],
                    0.75,
                );
                let mut actual = vec![guard; len + offset + 3];
                f64::make_raders_rfft_combiner().combine(
                    &mut actual[offset..offset + len],
                    &lo[offset..],
                    &hi[offset..],
                    &signs[offset..],
                    0.75,
                );
                assert_eq!(expected, actual, "f64 len {len} offset {offset}");
                let lo: Vec<f32> = lo.iter().map(|&x| x as f32).collect();
                let hi: Vec<f32> = hi.iter().map(|&x| x as f32).collect();
                let signs: Vec<f32> = signs.iter().map(|&x| x as f32).collect();
                let guard = Complex::new(123.0f32, -456.0);
                let mut actual = vec![guard; len + offset + 3];
                f32::make_raders_rfft_combiner().combine(
                    &mut actual[offset..offset + len],
                    &lo[offset..],
                    &hi[offset..],
                    &signs[offset..],
                    0.75,
                );
                for (e, a) in expected.iter().zip(&actual) {
                    assert_eq!(
                        *e,
                        Complex::new(a.re as f64, a.im as f64),
                        "f32 len {len} offset {offset}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_raders_rfft_multiple_blocks() {
        let n = 13usize;
        let blocks = 3usize;
        let src = (0..n * blocks)
            .map(|i| ((i * 31 + 7) % 97) as f64 / 10.0)
            .collect::<Vec<f64>>();
        let mx = RadersRfft::new(
            n,
            Zaft::make_r2c_fft_f64(n - 1).unwrap(),
            Zaft::make_c2r_fft_f64(n - 1).unwrap(),
        )
        .unwrap();
        let mut output = vec![Complex::<f64>::zero(); (n / 2 + 1) * blocks];
        let mut scratch = vec![Complex::<f64>::zero(); mx.complex_scratch_length()];
        mx.execute_with_scratch(&src, &mut output, &mut scratch)
            .unwrap();
        let dft = Dft::new(n, FftDirection::Forward).unwrap();
        for (block, out) in src.chunks_exact(n).zip(output.chunks_exact(n / 2 + 1)) {
            let mut reference = block
                .iter()
                .map(|x| Complex::new(*x, 0.0))
                .collect::<Vec<_>>();
            dft.execute(&mut reference).unwrap();
            for (a, b) in reference.iter().zip(out.iter()) {
                assert!((a.re - b.re).abs() < 1e-9, "{a} != {b}");
                assert!((a.im - b.im).abs() < 1e-9, "{a} != {b}");
            }
        }
    }
}
