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
use crate::{C2RFftExecutor, FftDirection, FftExecutor, FftSample, R2CFftExecutor, ZaftError};
use num_complex::Complex;
use num_integer::Integer;
use num_traits::{AsPrimitive, Zero};
use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

/// Final pass of Raders rfft, `y[p] = (first + re[p], Re(y[p] * twiddles[p]))`.
pub(crate) trait RadersRfftCombiner<T> {
    fn combine(&self, y: &mut [Complex<T>], twiddles: &[Complex<T>], re: &[T], first: T);
}

pub(crate) trait RadersRfftCombinerFactory<T> {
    fn make_raders_rfft_combiner() -> Arc<dyn RadersRfftCombiner<T> + Send + Sync>;
}

#[allow(unused)]
pub(crate) struct ScalarRadersRfftCombiner<T> {
    phantom_data: PhantomData<T>,
}

impl<T: FftSample> RadersRfftCombiner<T> for ScalarRadersRfftCombiner<T> {
    fn combine(&self, y: &mut [Complex<T>], twiddles: &[Complex<T>], re: &[T], first: T) {
        combine_scalar(y, twiddles, re, first);
    }
}

#[inline]
pub(crate) fn combine_scalar<T: FftSample>(
    y: &mut [Complex<T>],
    twiddles: &[Complex<T>],
    re: &[T],
    first: T,
) {
    for ((y, &twiddle), &re) in y.iter_mut().zip(twiddles.iter()).zip(re.iter()) {
        *y = Complex::new(first + re, y.re * twiddle.re - y.im * twiddle.im);
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

/// Rader's algorithm for a prime-length real-to-complex FFT where every inner
/// transform touches only half of a spectrum.
///
/// For prime `N` with primitive root `g`, `X[g^-p] - x[0]` is a cyclic convolution
/// of length `M = N - 1` between `a[q] = x[g^q]` and `b[q] = w^(g^-q)`.
///
/// The input is real, so `a (*) b = a (*) Re(b) + i * (a (*) Im(b))`: two real
/// convolutions `re` and `im`. Additionally, `X[N - k] = conj(X[k])` together with
/// `g^(p + M/2) = -g^p (mod N)` gives `re[p + M/2] = re[p]` and `im[p + M/2] = -im[p]`,
/// i.e. the spectrum of `re` is non-zero on even bins only and the spectrum of `im`
/// on odd bins only.
///
/// ```text
/// A      = r2c_M(a)
/// re[p]  = c2r_H(A[2k] * C[2k])[p]                        p in 0..H
/// im[p]  = Re(ifft_H(A[2k+1] * S[2k+1])[p] * e^(2 pi i p / M))
/// ```
///
/// where `C = r2c_M(Re(b))` and `S = r2c_M(Im(b))` are precomputed once per `N`.
/// The forward stage is one r2c of length `M`, the inverse stage is one c2r of
/// length `H` plus one complex inverse FFT of length `H`.
pub(crate) struct RadersRfft<T> {
    convolve_r2c: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
    convolve_c2r: Arc<dyn C2RFftExecutor<T> + Send + Sync>,
    convolve_ifft: Arc<dyn FftExecutor<T> + Send + Sync>,
    /// `C[2k] / M`, `H / 2 + 1` entries.
    even_twiddles: Vec<Complex<T>>,
    /// `S[2k + 1] / M`, `(H + 1) / 2` entries.
    odd_twiddles: Vec<Complex<T>>,
    /// `±e^(2 pi i p / M)` for `p in 0..H`, the sign folds the conjugation of the
    /// mirrored half of the output.
    post_twiddles: Vec<Complex<T>>,
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
    /// `convolve_r2c` must have real length `N - 1`, `convolve_c2r` and `convolve_ifft`
    /// must have length `(N - 1) / 2` and `convolve_ifft` must be an inverse transform.
    pub(crate) fn new(
        size: usize,
        convolve_r2c: Arc<dyn R2CFftExecutor<T> + Send + Sync>,
        convolve_c2r: Arc<dyn C2RFftExecutor<T> + Send + Sync>,
        convolve_ifft: Arc<dyn FftExecutor<T> + Send + Sync>,
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
        assert_eq!(convolve_c2r.real_length(), h);
        assert_eq!(convolve_ifft.length(), h);
        assert_eq!(convolve_ifft.direction(), FftDirection::Inverse);

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

        // b[q] = w^(g^-q) / M, split into its real and imaginary sequences.
        // The 1/M compensates the unscaled inverse transforms.
        let inner_fft_scale: T = (1f64 / m as f64).as_();
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
        let even_count = h / 2 + 1;
        let odd_count = h.div_ceil(2);
        assert_eq!(convolve_c2r.complex_length(), even_count);
        let mut even_twiddles = try_vec![Complex::<T>::zero(); even_count];
        let mut odd_twiddles = try_vec![Complex::<T>::zero(); odd_count];
        for (dst, &src) in even_twiddles.iter_mut().zip(c_spectrum.iter().step_by(2)) {
            *dst = src;
        }
        for (dst, &src) in odd_twiddles
            .iter_mut()
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
        let mut post_twiddles = try_vec![Complex::<T>::zero(); h];
        let mut output_index = 1u64;
        for (p, post_twiddle) in post_twiddles.iter_mut().enumerate() {
            output_index = (output_index * primitive_root_inverse) % dividing_len;
            let k = output_index as usize;
            let twiddle = compute_twiddle::<T>(p, m, FftDirection::Inverse);
            if k <= h {
                output_indices[k - 1] = p as u32;
                *post_twiddle = twiddle;
            } else {
                output_indices[size - k - 1] = p as u32;
                *post_twiddle = -twiddle;
            }
        }

        let convolve_scratch_length = convolve_r2c
            .complex_scratch_length()
            .max(convolve_c2r.complex_scratch_length())
            .max(convolve_ifft.scratch_length());

        Ok(RadersRfft {
            convolve_r2c,
            convolve_c2r,
            convolve_ifft,
            even_twiddles,
            odd_twiddles,
            post_twiddles,
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
        // odd_buffer doubles as the real input of the forward r2c: M reals == H complex
        let (odd_buffer, rem) = rem.split_at_mut(h);
        let (even_buffer, rem) = rem.split_at_mut(h / 2 + 1);
        let (re_buffer_c, convolve_scratch) = rem.split_at_mut(h.div_ceil(2));
        let re_buffer: &mut [T] =
            unsafe { std::slice::from_raw_parts_mut(re_buffer_c.as_mut_ptr().cast(), h) };

        for (input, complex) in input
            .chunks_exact(self.execution_length)
            .zip(output.chunks_exact_mut(self.complex_length()))
        {
            let (buffer_first, buffer) = input.split_first().unwrap();
            let buffer_first_val = *buffer_first;

            {
                let real_input: &mut [T] =
                    unsafe { std::slice::from_raw_parts_mut(odd_buffer.as_mut_ptr().cast(), m) };
                for (scratch_element, &buffer_idx) in
                    real_input.iter_mut().zip(self.input_indices.iter())
                {
                    *scratch_element = unsafe { *buffer.get_unchecked(buffer_idx) }
                }
                self.convolve_r2c
                    .execute_with_scratch(real_input, a_spectrum, convolve_scratch)?;
            }

            // a_spectrum[0] is the sum of elements 1..len, add the first input to get X[0]
            complex[0] = Complex::new(buffer_first_val + a_spectrum[0].re, T::zero());

            // even bins feed the real part of the convolution, odd bins the imaginary part
            let (pairs, tail) = a_spectrum.as_chunks::<2>();
            for (((pair, &even_twiddle), &odd_twiddle), (even, odd)) in pairs
                .iter()
                .zip(self.even_twiddles.iter())
                .zip(self.odd_twiddles.iter())
                .zip(even_buffer.iter_mut().zip(odd_buffer.iter_mut()))
            {
                *even = pair[0] * even_twiddle;
                *odd = pair[1] * odd_twiddle;
            }
            if let (Some(&last), Some(&twiddle)) = (tail.first(), self.even_twiddles.last()) {
                even_buffer[h / 2] = last * twiddle;
            }

            // odd bins above H follow from Hermitian symmetry of the real spectrum
            let (odd_lower, odd_upper) = odd_buffer.split_at_mut(h.div_ceil(2));
            for (dst, &src) in odd_upper.iter_mut().rev().zip(odd_lower.iter()) {
                *dst = src.conj();
            }

            self.convolve_c2r
                .execute_with_scratch(even_buffer, re_buffer, convolve_scratch)?;
            self.convolve_ifft
                .execute_with_scratch(odd_buffer, convolve_scratch)?;

            // y[p] = x[0] + re[p] + i * im[p], im[p] = Re(v[p] * e^(2 pi i p / M))
            self.combiner
                .combine(odd_buffer, &self.post_twiddles, re_buffer, buffer_first_val);

            self.gatherer
                .gather(odd_buffer, &mut complex[1..], &self.output_indices);
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
        (h + 1) + h + (h / 2 + 1) + h.div_ceil(2) + self.convolve_scratch_length
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
            Zaft::make_c2r_fft_f64((n - 1) / 2).unwrap(),
            Zaft::strategy((n - 1) / 2, FftDirection::Inverse).unwrap(),
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
            Zaft::make_c2r_fft_f32((n - 1) / 2).unwrap(),
            Zaft::strategy((n - 1) / 2, FftDirection::Inverse).unwrap(),
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
    fn test_raders_rfft_combiner() {
        use crate::r2c::rfft_raders::{RadersRfftCombinerFactory, combine_scalar};
        for len in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 64, 101] {
            let y: Vec<Complex<f64>> = (0..len)
                .map(|i| Complex::new((i % 7) as f64 - 3.0, (i % 5) as f64 * 0.5))
                .collect();
            let twiddles: Vec<Complex<f64>> = (0..len)
                .map(|i| Complex::new(((i * 3) % 11) as f64 / 11.0, -((i % 4) as f64) / 4.0))
                .collect();
            let re: Vec<f64> = (0..len).map(|i| (i % 9) as f64 * 1.25).collect();

            let mut expected = y.clone();
            combine_scalar(&mut expected, &twiddles, &re, 0.75);
            let mut actual = y.clone();
            f64::make_raders_rfft_combiner().combine(&mut actual, &twiddles, &re, 0.75);
            assert_eq!(expected, actual, "f64 len {len}");

            let y_f32: Vec<Complex<f32>> = y
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();
            let twiddles_f32: Vec<Complex<f32>> = twiddles
                .iter()
                .map(|c| Complex::new(c.re as f32, c.im as f32))
                .collect();
            let re_f32: Vec<f32> = re.iter().map(|&r| r as f32).collect();
            let mut expected = y_f32.clone();
            combine_scalar(&mut expected, &twiddles_f32, &re_f32, 0.75);
            let mut actual = y_f32.clone();
            f32::make_raders_rfft_combiner().combine(&mut actual, &twiddles_f32, &re_f32, 0.75);
            for (e, a) in expected.iter().zip(actual.iter()) {
                assert!(
                    (e.re - a.re).abs() < 1e-5 && (e.im - a.im).abs() < 1e-5,
                    "f32 len {len}: {e} != {a}"
                );
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
            Zaft::make_c2r_fft_f64((n - 1) / 2).unwrap(),
            Zaft::strategy((n - 1) / 2, FftDirection::Inverse).unwrap(),
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
