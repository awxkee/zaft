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
use crate::neon::mixed::{NeonStoreD, NeonStoreF};
use crate::util::compute_twiddle;
use std::arch::aarch64::*;

pub(crate) fn gen_butterfly_twiddles_f64<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [NeonStoreD; N] {
    let mut twiddles = [NeonStoreD::default(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 1;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = NeonStoreD::from_complex(&compute_twiddle(
                y * (x * COMPLEX_PER_VECTOR),
                size,
                direction,
            ));
            q += 1;
        }
    }
    twiddles
}

pub(crate) fn gen_butterfly_twiddles_f32<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [NeonStoreF; N] {
    let mut twiddles = [NeonStoreF::default(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 2;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = NeonStoreF::from_complex2(
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR), size, direction),
                compute_twiddle(y * (x * COMPLEX_PER_VECTOR + 1), size, direction),
            );
            q += 1;
        }
    }
    twiddles
}

pub(crate) fn gen_butterfly_separate_cols_twiddles_f32<const N: usize>(
    rows: usize,
    cols: usize,
    direction: FftDirection,
    size: usize,
) -> [NeonStoreF; N] {
    let mut twiddles = [NeonStoreF::default(); N];
    let mut q = 0usize;
    let len_per_row = rows;
    const COMPLEX_PER_VECTOR: usize = 1;
    let quotient = len_per_row / COMPLEX_PER_VECTOR;
    let remainder = len_per_row % COMPLEX_PER_VECTOR;

    let num_twiddle_columns = quotient + remainder.div_ceil(COMPLEX_PER_VECTOR);
    for x in 0..num_twiddle_columns {
        for y in 1..cols {
            twiddles[q] = NeonStoreF::from_complex(&compute_twiddle(
                y * (x * COMPLEX_PER_VECTOR),
                size,
                direction,
            ));
            q += 1;
        }
    }
    twiddles
}

pub(crate) struct NeonButterfly {}

impl NeonButterfly {
    #[inline]
    pub(crate) fn butterfly3h_f32(
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        tw_re: float32x2_t,
        tw_w_2: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t) {
        unsafe {
            let xp = vadd_f32(u1, u2);
            let xn = vsub_f32(u1, u2);
            let sum = vadd_f32(u0, xp);

            let w_1 = vfma_f32(u0, tw_re, xp);
            let xn_rot = vext_f32::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfma_f32(w_1, tw_w_2, xn_rot);
            let y2 = vfms_f32(w_1, tw_w_2, xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline]
    pub(crate) fn butterfly3_f32(
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        tw_re: float32x4_t,
        tw_w_2: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        unsafe {
            let xp = vaddq_f32(u1, u2);
            let xn = vsubq_f32(u1, u2);
            let sum = vaddq_f32(u0, xp);

            let w_1 = vfmaq_f32(u0, tw_re, xp);
            let xn_rot = vrev64q_f32(xn);

            let y0 = sum;
            let y1 = vfmaq_f32(w_1, tw_w_2, xn_rot);
            let y2 = vfmsq_f32(w_1, tw_w_2, xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn butterfly3h_f32_fcma(
        u0: float32x2_t,
        u1: float32x2_t,
        u2: float32x2_t,
        tw_re: float32x2_t,
        tw_w_2: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t) {
        let xp = vadd_f32(u1, u2);
        let xn = vsub_f32(u1, u2);
        let sum = vadd_f32(u0, xp);

        let w_1 = vfma_f32(u0, tw_re, xp);

        let y0 = sum;
        let y1 = vcmla_rot90_f32(w_1, tw_w_2, xn);
        let y2 = vcmla_rot270_f32(w_1, tw_w_2, xn);
        (y0, y1, y2)
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn butterfly3_f32_fcma(
        u0: float32x4_t,
        u1: float32x4_t,
        u2: float32x4_t,
        tw_re: float32x4_t,
        tw_w_2: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t) {
        let xp = vaddq_f32(u1, u2);
        let xn = vsubq_f32(u1, u2);
        let sum = vaddq_f32(u0, xp);

        let w_1 = vfmaq_f32(u0, tw_re, xp);

        let y0 = sum;
        let y1 = vcmlaq_rot90_f32(w_1, tw_w_2, xn);
        let y2 = vcmlaq_rot270_f32(w_1, tw_w_2, xn);
        (y0, y1, y2)
    }

    #[inline]
    pub(crate) fn butterfly2h_f32(u0: float32x2_t, u1: float32x2_t) -> (float32x2_t, float32x2_t) {
        unsafe {
            let t = vadd_f32(u0, u1);
            let y1 = vsub_f32(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline]
    pub(crate) fn butterfly2_f32(u0: float32x4_t, u1: float32x4_t) -> (float32x4_t, float32x4_t) {
        unsafe {
            let t = vaddq_f32(u0, u1);
            let y1 = vsubq_f32(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }

    #[inline]
    pub(crate) fn butterfly3_f64(
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        tw_re: float64x2_t,
        tw_w_2: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t) {
        unsafe {
            let xp = vaddq_f64(u1, u2);
            let xn = vsubq_f64(u1, u2);
            let sum = vaddq_f64(u0, xp);

            let w_1 = vfmaq_f64(u0, tw_re, xp);
            let xn_rot = vextq_f64::<1>(xn, xn);

            let y0 = sum;
            let y1 = vfmaq_f64(w_1, tw_w_2, xn_rot);
            let y2 = vfmsq_f64(w_1, tw_w_2, xn_rot);
            (y0, y1, y2)
        }
    }

    #[inline]
    #[cfg(feature = "fcma")]
    #[target_feature(enable = "fcma")]
    pub(crate) fn butterfly3_f64_fcma(
        u0: float64x2_t,
        u1: float64x2_t,
        u2: float64x2_t,
        tw_re: float64x2_t,
        tw_w_2: float64x2_t,
    ) -> (float64x2_t, float64x2_t, float64x2_t) {
        let xp = vaddq_f64(u1, u2);
        let xn = vsubq_f64(u1, u2);
        let sum = vaddq_f64(u0, xp);

        let w_1 = vfmaq_f64(u0, tw_re, xp);

        let y0 = sum;
        let y1 = vcmlaq_rot90_f64(w_1, tw_w_2, xn);
        let y2 = vcmlaq_rot270_f64(w_1, tw_w_2, xn);
        (y0, y1, y2)
    }

    #[inline]
    pub(crate) fn butterfly4_f32(
        a: float32x4_t,
        b: float32x4_t,
        c: float32x4_t,
        d: float32x4_t,
        rotate: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        unsafe {
            let t0 = vaddq_f32(a, c);
            let t1 = vsubq_f32(a, c);
            let t2 = vaddq_f32(b, d);
            let mut t3 = vsubq_f32(b, d);
            t3 = vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(vrev64q_f32(t3)),
                vreinterpretq_u32_f32(rotate),
            ));
            (
                vaddq_f32(t0, t2),
                vaddq_f32(t1, t3),
                vsubq_f32(t0, t2),
                vsubq_f32(t1, t3),
            )
        }
    }

    #[inline]
    pub(crate) fn butterfly2_f64(u0: float64x2_t, u1: float64x2_t) -> (float64x2_t, float64x2_t) {
        unsafe {
            let t = vaddq_f64(u0, u1);

            let y1 = vsubq_f64(u0, u1);
            let y0 = t;
            (y0, y1)
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct FastFcmaBf4f<const FORWARD: bool> {}

#[cfg(feature = "fcma")]
impl<const FORWARD: bool> FastFcmaBf4f<FORWARD> {
    #[inline]
    pub(crate) fn new() -> Self {
        Self {}
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exec(
        &self,
        a: float32x4_t,
        b: float32x4_t,
        c: float32x4_t,
        d: float32x4_t,
    ) -> (float32x4_t, float32x4_t, float32x4_t, float32x4_t) {
        let t0 = vaddq_f32(a, c);
        let t1 = vsubq_f32(a, c);
        let t2 = vaddq_f32(b, d);
        let t3 = vsubq_f32(b, d);
        if FORWARD {
            (
                vaddq_f32(t0, t2),
                vcaddq_rot270_f32(t1, t3),
                vsubq_f32(t0, t2),
                vcaddq_rot90_f32(t1, t3),
            )
        } else {
            (
                vaddq_f32(t0, t2),
                vcaddq_rot90_f32(t1, t3),
                vsubq_f32(t0, t2),
                vcaddq_rot270_f32(t1, t3),
            )
        }
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn exech(
        &self,
        a: float32x2_t,
        b: float32x2_t,
        c: float32x2_t,
        d: float32x2_t,
    ) -> (float32x2_t, float32x2_t, float32x2_t, float32x2_t) {
        let t0 = vadd_f32(a, c);
        let t1 = vsub_f32(a, c);
        let t2 = vadd_f32(b, d);
        let t3 = vsub_f32(b, d);
        if FORWARD {
            (
                vadd_f32(t0, t2),
                vcadd_rot270_f32(t1, t3),
                vsub_f32(t0, t2),
                vcadd_rot90_f32(t1, t3),
            )
        } else {
            (
                vadd_f32(t0, t2),
                vcadd_rot90_f32(t1, t3),
                vsub_f32(t0, t2),
                vcadd_rot270_f32(t1, t3),
            )
        }
    }
}

pub(crate) struct NeonRotate90F {
    sign: float32x4_t,
}

impl NeonRotate90F {
    pub(crate) fn new() -> Self {
        Self {
            sign: unsafe { vld1q_f32([-0.0, 0.0, -0.0, 0.0].as_ptr()) },
        }
    }

    #[inline(always)]
    pub(crate) fn roth(&self, values: float32x2_t) -> float32x2_t {
        unsafe {
            let temp = vext_f32::<1>(values, values);
            vreinterpret_f32_u32(veor_u32(
                vreinterpret_u32_f32(temp),
                vreinterpret_u32_f32(vget_low_f32(self.sign)),
            ))
        }
    }

    #[inline(always)]
    pub(crate) fn rot(&self, values: float32x4_t) -> float32x4_t {
        unsafe {
            let temp = vrev64q_f32(values);
            vreinterpretq_f32_u32(veorq_u32(
                vreinterpretq_u32_f32(temp),
                vreinterpretq_u32_f32(self.sign),
            ))
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct NeonFcmaRotate90F {}

#[cfg(feature = "fcma")]
impl NeonFcmaRotate90F {
    pub(crate) fn new() -> Self {
        Self {}
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn roth(&self, values: float32x2_t) -> float32x2_t {
        vcadd_rot90_f32(vdup_n_f32(0.), values)
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rot(&self, values: float32x4_t) -> float32x4_t {
        vcaddq_rot90_f32(vdupq_n_f32(0.), values)
    }
}

pub(crate) struct NeonRotate90D {
    sign: float64x2_t,
}

impl NeonRotate90D {
    pub(crate) fn new() -> Self {
        Self {
            sign: unsafe { vld1q_f64([-0.0, 0.0].as_ptr()) },
        }
    }

    #[inline(always)]
    pub(crate) fn rot(&self, values: float64x2_t) -> float64x2_t {
        unsafe {
            let temp = vextq_f64::<1>(values, values);
            vreinterpretq_f64_u64(veorq_u64(
                vreinterpretq_u64_f64(temp),
                vreinterpretq_u64_f64(self.sign),
            ))
        }
    }
}

#[cfg(feature = "fcma")]
pub(crate) struct NeonFcmaRotate90D {}

#[cfg(feature = "fcma")]
impl NeonFcmaRotate90D {
    pub(crate) fn new() -> Self {
        Self {}
    }

    #[inline]
    #[target_feature(enable = "fcma")]
    pub(crate) fn rot(&self, values: float64x2_t) -> float64x2_t {
        vcaddq_rot90_f64(vdupq_n_f64(0.), values)
    }
}

macro_rules! boring_simple_neon_butterfly {
    ($bf_name: ident, $f_type: ident, $size: expr) => {
        impl FftExecutor<$f_type> for $bf_name<$f_type> {
            fn execute(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.chunks_exact_mut($size) {
                    self.run(&mut InPlaceStore::new(chunk));
                }

                Ok(())
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<f64>],
                _: &mut [Complex<f64>],
            ) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.chunks_exact_mut($size) {
                    use crate::store::InPlaceStore;
                    self.run(&mut InPlaceStore::new(chunk));
                }

                Ok(())
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, $size);

                for (dst, src) in dst.chunks_exact_mut($size).zip(src.chunks_exact($size)) {
                    use crate::store::BiStore;
                    self.run(&mut BiStore::new(src, dst));
                }
                Ok(())
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            fn length(&self) -> usize {
                $size
            }

            fn scratch_length(&self) -> usize {
                0
            }

            fn out_of_place_scratch_length(&self) -> usize {
                0
            }

            fn destructive_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

pub(crate) use boring_simple_neon_butterfly;

macro_rules! boring_neon_butterfly {
    ($bf_name: ident, $features: literal, $f_type: ident, $size: expr) => {
        impl $bf_name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.chunks_exact_mut($size) {
                    use crate::store::InPlaceStore;
                    self.run(&mut InPlaceStore::new(chunk));
                }

                Ok(())
            }

            #[target_feature(enable = $features)]
            fn execute_oof_impl(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, $size);

                for (dst, src) in dst.chunks_exact_mut($size).zip(src.chunks_exact($size)) {
                    use crate::store::BiStore;
                    self.run(&mut BiStore::new(src, dst));
                }
                Ok(())
            }
        }

        impl FftExecutor<$f_type> for $bf_name {
            fn execute(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                FftExecutor::execute_with_scratch(self, in_place, &mut [])
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            fn length(&self) -> usize {
                $size
            }

            fn scratch_length(&self) -> usize {
                0
            }

            fn out_of_place_scratch_length(&self) -> usize {
                0
            }

            fn destructive_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

pub(crate) use boring_neon_butterfly;

macro_rules! boring_neon_butterfly2 {
    ($bf_name: ident, $features: literal, $f_type: ident, $size: expr) => {
        impl $bf_name {
            #[target_feature(enable = $features)]
            fn execute_impl(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.chunks_exact_mut($size * 2) {
                    use crate::store::InPlaceStore;
                    self.run2(&mut InPlaceStore::new(chunk));
                }

                let rem = in_place.chunks_exact_mut($size * 2).into_remainder();

                for chunk in rem.chunks_exact_mut($size) {
                    use crate::store::InPlaceStore;
                    self.run(&mut InPlaceStore::new(chunk));
                }

                Ok(())
            }

            #[target_feature(enable = $features)]
            fn execute_oof_impl(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                use crate::util::validate_oof_sizes;
                validate_oof_sizes!(src, dst, $size);

                for (dst, src) in dst
                    .chunks_exact_mut($size * 2)
                    .zip(src.chunks_exact($size * 2))
                {
                    use crate::store::BiStore;
                    self.run2(&mut BiStore::new(src, dst));
                }

                let rem_dst = dst.chunks_exact_mut($size * 2).into_remainder();
                let rem_src = src.chunks_exact($size * 2).remainder();

                for (dst, src) in rem_dst
                    .chunks_exact_mut($size)
                    .zip(rem_src.chunks_exact($size))
                {
                    use crate::store::BiStore;
                    self.run(&mut BiStore::new(src, dst));
                }
                Ok(())
            }
        }

        impl FftExecutor<$f_type> for $bf_name {
            fn execute(&self, in_place: &mut [Complex<$f_type>]) -> Result<(), ZaftError> {
                FftExecutor::execute_with_scratch(self, in_place, &mut [])
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_impl(in_place) }
            }

            fn execute_out_of_place(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                unsafe { self.execute_oof_impl(src, dst) }
            }

            fn execute_destructive_with_scratch(
                &self,
                src: &mut [Complex<$f_type>],
                dst: &mut [Complex<$f_type>],
                _: &mut [Complex<$f_type>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            fn length(&self) -> usize {
                $size
            }

            fn scratch_length(&self) -> usize {
                0
            }

            fn out_of_place_scratch_length(&self) -> usize {
                0
            }

            fn destructive_scratch_length(&self) -> usize {
                0
            }
        }
    };
}

pub(crate) use boring_neon_butterfly2;
