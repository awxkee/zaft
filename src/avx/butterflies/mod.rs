/*
 * // Copyright (c) Radzivon Bartoshyk 9/2025. All rights reserved.
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
#![allow(clippy::needless_range_loop)]
mod bf10;
mod bf100d;
mod bf100f;
mod bf11;
mod bf12;
mod bf121d;
mod bf121f;
mod bf125d;
mod bf125f;
mod bf128d;
mod bf128f;
mod bf13;
mod bf14;
mod bf144d;
mod bf144f;
mod bf15;
mod bf16;
mod bf169d;
mod bf169f;
mod bf17;
mod bf18;
mod bf19;
mod bf2;
mod bf20;
mod bf21d;
mod bf21f;
mod bf23;
mod bf243d;
mod bf243f;
mod bf24d;
mod bf24f;
mod bf25;
mod bf256d;
mod bf256f;
mod bf27d;
mod bf27f;
mod bf29;
mod bf3;
mod bf30d;
mod bf30f;
mod bf31;
mod bf32d;
mod bf32f;
mod bf35d;
mod bf35f;
mod bf36d;
mod bf36f;
mod bf4;
mod bf40d;
mod bf40f;
mod bf42d;
mod bf42f;
mod bf48d;
mod bf48f;
mod bf49;
mod bf5;
mod bf512d;
mod bf512f;
mod bf54d;
mod bf54f;
mod bf6;
mod bf63d;
mod bf63f;
mod bf64d;
mod bf64f;
mod bf66d;
mod bf66f;
mod bf7;
mod bf70d;
mod bf70f;
mod bf72d;
mod bf72f;
mod bf78d;
mod bf78f;
mod bf8;
mod bf81d;
mod bf81f;
mod bf9;
mod bf96d;
mod bf96f;
mod fast_bf3;
mod fast_bf4;
mod fast_bf5;
mod fast_bf8;
mod shared;

pub(crate) use bf2::AvxButterfly2;
pub(crate) use bf3::AvxButterfly3;
pub(crate) use bf4::AvxButterfly4;
pub(crate) use bf5::AvxButterfly5;
pub(crate) use bf6::{AvxButterfly6d, AvxButterfly6f};
pub(crate) use bf7::AvxButterfly7;
pub(crate) use bf8::{AvxButterfly8d, AvxButterfly8f};
pub(crate) use bf9::{AvxButterfly9d, AvxButterfly9f};
pub(crate) use bf10::{AvxButterfly10d, AvxButterfly10f};
pub(crate) use bf11::{AvxButterfly11, AvxButterfly11d};
pub(crate) use bf12::{AvxButterfly12d, AvxButterfly12f};
pub(crate) use bf13::{AvxButterfly13, AvxButterfly13d};
pub(crate) use bf14::{AvxButterfly14d, AvxButterfly14f};
pub(crate) use bf15::{AvxButterfly15d, AvxButterfly15f};
pub(crate) use bf16::{AvxButterfly16d, AvxButterfly16f};
pub(crate) use bf17::AvxButterfly17;
pub(crate) use bf18::{AvxButterfly18d, AvxButterfly18f};
pub(crate) use bf19::AvxButterfly19;
pub(crate) use bf20::{AvxButterfly20d, AvxButterfly20f};
pub(crate) use bf21d::AvxButterfly21d;
pub(crate) use bf21f::AvxButterfly21f;
pub(crate) use bf23::AvxButterfly23;
pub(crate) use bf24d::AvxButterfly24d;
pub(crate) use bf24f::AvxButterfly24f;
pub(crate) use bf25::{AvxButterfly25d, AvxButterfly25f};
pub(crate) use bf27d::AvxButterfly27d;
pub(crate) use bf27f::AvxButterfly27f;
pub(crate) use bf29::AvxButterfly29;
pub(crate) use bf30d::AvxButterfly30d;
pub(crate) use bf30f::AvxButterfly30f;
pub(crate) use bf31::AvxButterfly31;
pub(crate) use bf32d::AvxButterfly32d;
pub(crate) use bf32f::AvxButterfly32f;
pub(crate) use bf35d::AvxButterfly35d;
pub(crate) use bf35f::AvxButterfly35f;
pub(crate) use bf36d::AvxButterfly36d;
pub(crate) use bf36f::AvxButterfly36f;
pub(crate) use bf40d::AvxButterfly40d;
pub(crate) use bf40f::AvxButterfly40f;
pub(crate) use bf42d::AvxButterfly42d;
pub(crate) use bf42f::AvxButterfly42f;
pub(crate) use bf48d::AvxButterfly48d;
pub(crate) use bf48f::AvxButterfly48f;
pub(crate) use bf49::{AvxButterfly49d, AvxButterfly49f};
pub(crate) use bf54d::AvxButterfly54d;
pub(crate) use bf54f::AvxButterfly54f;
pub(crate) use bf63d::AvxButterfly63d;
pub(crate) use bf63f::AvxButterfly63f;
pub(crate) use bf64d::AvxButterfly64d;
pub(crate) use bf64f::AvxButterfly64f;
pub(crate) use bf66d::AvxButterfly66d;
pub(crate) use bf66f::AvxButterfly66f;
pub(crate) use bf70d::AvxButterfly70d;
pub(crate) use bf70f::AvxButterfly70f;
pub(crate) use bf72d::AvxButterfly72d;
pub(crate) use bf72f::AvxButterfly72f;
pub(crate) use bf78d::AvxButterfly78d;
pub(crate) use bf78f::AvxButterfly78f;
pub(crate) use bf81d::AvxButterfly81d;
pub(crate) use bf81f::AvxButterfly81f;
pub(crate) use bf96d::AvxButterfly96d;
pub(crate) use bf96f::AvxButterfly96f;
pub(crate) use bf100d::AvxButterfly100d;
pub(crate) use bf100f::AvxButterfly100f;
pub(crate) use bf121d::AvxButterfly121d;
pub(crate) use bf121f::AvxButterfly121f;
pub(crate) use bf125d::AvxButterfly125d;
pub(crate) use bf125f::AvxButterfly125f;
pub(crate) use bf128d::AvxButterfly128d;
pub(crate) use bf128f::AvxButterfly128f;
pub(crate) use bf144d::AvxButterfly144d;
pub(crate) use bf144f::AvxButterfly144f;
pub(crate) use bf169d::AvxButterfly169d;
pub(crate) use bf169f::AvxButterfly169f;
pub(crate) use bf243d::AvxButterfly243d;
pub(crate) use bf243f::AvxButterfly243f;
pub(crate) use bf256d::AvxButterfly256d;
pub(crate) use bf256f::AvxButterfly256f;
pub(crate) use bf512d::AvxButterfly512d;
pub(crate) use bf512f::AvxButterfly512f;
pub(crate) use fast_bf3::AvxFastButterfly3;
pub(crate) use fast_bf4::AvxFastButterfly4;
pub(crate) use fast_bf5::{AvxFastButterfly5d, AvxFastButterfly5f};
pub(crate) use fast_bf8::AvxFastButterfly8;
use num_complex::Complex;
pub(crate) use shared::AvxButterfly;

macro_rules! shift_loadl {
    ($chunk: expr, $size: expr, $offset0: expr) => {{
        let q0 = _m128s_load_f32x2($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = _m128s_load_f32x2($chunk.get_unchecked($offset0 + $size..).as_ptr().cast());
        _mm_unpacklo_ps64(q0, q1)
    }};
}

pub(crate) use shift_loadl;

macro_rules! shift_storel {
    ($chunk: expr, $size: expr, $offset0: expr, $r0: expr) => {{
        use crate::avx::util::_m128s_store_f32x2;
        _m128s_store_f32x2(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            $r0,
        );
        use crate::avx::util::_m128s_storeh_f32x2;
        _m128s_storeh_f32x2(
            $chunk
                .get_unchecked_mut($offset0 + $size..)
                .as_mut_ptr()
                .cast(),
            $r0,
        );
    }};
}

pub(crate) use shift_storel;

macro_rules! shift_load8 {
    ($chunk: expr, $size: expr, $offset0: expr) => {{
        let q0 = _mm256_loadu_ps($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = _mm256_loadu_ps($chunk.get_unchecked($offset0 + $size..).as_ptr().cast());
        let q2 = _mm256_loadu_ps($chunk.get_unchecked($offset0 + $size * 2..).as_ptr().cast());
        let q3 = _mm256_loadu_ps($chunk.get_unchecked($offset0 + $size * 3..).as_ptr().cast());
        let u0u1 = _mm256_castps256_ps128(q0);
        let u2u3 = _mm256_extractf128_ps::<1>(q0);
        let u0u1_1 = _mm256_castps256_ps128(q1);
        let u2u3_1 = _mm256_extractf128_ps::<1>(q1);
        let u0u1_2 = _mm256_castps256_ps128(q2);
        let u2u3_2 = _mm256_extractf128_ps::<1>(q2);
        let u0u1_3 = _mm256_castps256_ps128(q3);
        let u2u3_3 = _mm256_extractf128_ps::<1>(q3);
        (
            _mm256_setr_m128(
                _mm_unpacklo_ps64(u0u1, u0u1_1),
                _mm_unpacklo_ps64(u0u1_2, u0u1_3),
            ),
            _mm256_setr_m128(
                _mm_unpackhi_ps64(u0u1, u0u1_1),
                _mm_unpackhi_ps64(u0u1_2, u0u1_3),
            ),
            _mm256_setr_m128(
                _mm_unpacklo_ps64(u2u3, u2u3_1),
                _mm_unpacklo_ps64(u2u3_2, u2u3_3),
            ),
            _mm256_setr_m128(
                _mm_unpackhi_ps64(u2u3, u2u3_1),
                _mm_unpackhi_ps64(u2u3_2, u2u3_3),
            ),
        )
    }};
}

pub(crate) use shift_load8;

macro_rules! shift_load4 {
    ($chunk: expr, $size: expr, $offset0: expr) => {{
        let q0 = _mm256_loadu_ps($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = _mm256_loadu_ps($chunk.get_unchecked($offset0 + $size..).as_ptr().cast());
        let u0u1 = _mm256_castps256_ps128(q0);
        let u2u3 = _mm256_extractf128_ps::<1>(q0);
        let u0u1_1 = _mm256_castps256_ps128(q1);
        let u2u3_1 = _mm256_extractf128_ps::<1>(q1);
        (
            _mm_unpacklo_ps64(u0u1, u0u1_1),
            _mm_unpackhi_ps64(u0u1, u0u1_1),
            _mm_unpacklo_ps64(u2u3, u2u3_1),
            _mm_unpackhi_ps64(u2u3, u2u3_1),
        )
    }};
}

pub(crate) use shift_load4;

macro_rules! shift_load2dd {
    ($chunk: expr, $size: expr, $offset0: expr) => {{
        let q0 = _mm256_loadu_pd($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = _mm256_loadu_pd($chunk.get_unchecked($offset0 + $size..).as_ptr().cast());
        const HI_HI: i32 = 0b0011_0001;
        const LO_LO: i32 = 0b0010_0000;
        (
            _mm256_permute2f128_pd::<LO_LO>(q0, q1),
            _mm256_permute2f128_pd::<HI_HI>(q0, q1),
        )
    }};
}

pub(crate) use shift_load2dd;

macro_rules! shift_load2d {
    ($chunk: expr, $size: expr, $offset0: expr) => {{
        let q0 = _mm_loadu_pd($chunk.get_unchecked($offset0..).as_ptr().cast());
        let q1 = _mm_loadu_pd($chunk.get_unchecked($offset0 + $size..).as_ptr().cast());
        _mm256_setr_m128d(q0, q1)
    }};
}

pub(crate) use shift_load2d;

macro_rules! shift_store2d {
    ($chunk: expr, $size: expr, $offset0: expr, $r0: expr) => {{
        _mm_storeu_pd(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            _mm256_castpd256_pd128($r0),
        );
        _mm_storeu_pd(
            $chunk
                .get_unchecked_mut($offset0 + $size..)
                .as_mut_ptr()
                .cast(),
            _mm256_extractf128_pd::<1>($r0),
        );
    }};
}

pub(crate) use shift_store2d;

macro_rules! shift_store2dd {
    ($chunk: expr, $size: expr, $offset0: expr, $r0: expr, $r1: expr) => {{
        const HI_HI: i32 = 0b0011_0001;
        const LO_LO: i32 = 0b0010_0000;
        _mm256_storeu_pd(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            _mm256_permute2f128_pd::<LO_LO>($r0, $r1),
        );
        _mm256_storeu_pd(
            $chunk
                .get_unchecked_mut($offset0 + $size..)
                .as_mut_ptr()
                .cast(),
            _mm256_permute2f128_pd::<HI_HI>($r0, $r1),
        );
    }};
}

pub(crate) use shift_store2dd;

macro_rules! shift_store4 {
    ($chunk: expr, $size: expr, $offset0: expr, $r0: expr, $r1: expr, $r2: expr, $r3: expr) => {{
        let l0 = _mm_unpacklo_ps64($r0, $r1);
        let l1 = _mm_unpacklo_ps64($r2, $r3);
        _mm256_storeu_ps(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            _mm256_setr_m128(l0, l1),
        );
        let q0 = _mm_unpackhi_ps64($r0, $r1);
        let q1 = _mm_unpackhi_ps64($r2, $r3);
        _mm256_storeu_ps(
            $chunk
                .get_unchecked_mut($offset0 + $size..)
                .as_mut_ptr()
                .cast(),
            _mm256_setr_m128(q0, q1),
        );
    }};
}

pub(crate) use shift_store4;

macro_rules! shift_store8 {
    ($chunk: expr, $size: expr, $offset0: expr, $q0: expr, $q1: expr, $q2: expr, $q3: expr) => {{
        let r0 = _mm256_castps256_ps128($q0);
        let r1 = _mm256_castps256_ps128($q1);
        let r2 = _mm256_castps256_ps128($q2);
        let r3 = _mm256_castps256_ps128($q3);
        let l0 = _mm_unpacklo_ps64(r0, r1);
        let l1 = _mm_unpacklo_ps64(r2, r3);
        _mm256_storeu_ps(
            $chunk.get_unchecked_mut($offset0..).as_mut_ptr().cast(),
            _mm256_setr_m128(l0, l1),
        );
        let q0 = _mm_unpackhi_ps64(r0, r1);
        let q1 = _mm_unpackhi_ps64(r2, r3);
        _mm256_storeu_ps(
            $chunk
                .get_unchecked_mut($offset0 + $size..)
                .as_mut_ptr()
                .cast(),
            _mm256_setr_m128(q0, q1),
        );

        let r0 = _mm256_extractf128_ps::<1>($q0);
        let r1 = _mm256_extractf128_ps::<1>($q1);
        let r2 = _mm256_extractf128_ps::<1>($q2);
        let r3 = _mm256_extractf128_ps::<1>($q3);
        let l0 = _mm_unpacklo_ps64(r0, r1);
        let l1 = _mm_unpacklo_ps64(r2, r3);
        _mm256_storeu_ps(
            $chunk
                .get_unchecked_mut($offset0 + $size * 2..)
                .as_mut_ptr()
                .cast(),
            _mm256_setr_m128(l0, l1),
        );
        let q0 = _mm_unpackhi_ps64(r0, r1);
        let q1 = _mm_unpackhi_ps64(r2, r3);
        _mm256_storeu_ps(
            $chunk
                .get_unchecked_mut($offset0 + $size * 3..)
                .as_mut_ptr()
                .cast(),
            _mm256_setr_m128(q0, q1),
        );
    }};
}

pub(crate) use shift_store8;

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn make_mixedradix_twiddle_chunk_f64(
    x: usize,
    y: usize,
    len: usize,
    direction: FftDirection,
) -> AvxStoreD {
    let mut twiddle_chunk = [Complex::<f64>::default(); 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        twiddle_chunk[i] = compute_twiddle(y * (x + i), len, direction);
    }

    AvxStoreD::from_complex_ref(twiddle_chunk.as_slice())
}

#[inline]
#[target_feature(enable = "avx")]
pub(crate) unsafe fn make_mixedradix_twiddle_chunk_f32(
    x: usize,
    y: usize,
    len: usize,
    direction: FftDirection,
) -> AvxStoreF {
    let mut twiddle_chunk = [Complex::<f32>::default(); 4];
    #[allow(clippy::needless_range_loop)]
    for i in 0..4 {
        twiddle_chunk[i] = compute_twiddle(y * (x + i), len, direction);
    }

    AvxStoreF::from_complex_ref(twiddle_chunk.as_slice())
}

macro_rules! gen_butterfly_twiddles_interleaved_columns_f64 {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $direction: expr) => {{
        const FFT_LEN: usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS: usize = $num_rows - 1;
        const TWIDDLE_COLS: usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS: usize = TWIDDLE_COLS / 2;
        const TWIDDLE_VECTOR_COUNT: usize = TWIDDLE_VECTOR_COLS * TWIDDLE_ROWS;
        let mut twiddles = [AvxStoreD::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index / TWIDDLE_VECTOR_COLS) + 1;
            let x = (index % TWIDDLE_VECTOR_COLS) * 2 + $skip_cols;

            use crate::avx::butterflies::make_mixedradix_twiddle_chunk_f64;

            twiddles[index] = make_mixedradix_twiddle_chunk_f64(x, y, FFT_LEN, $direction);
        }
        twiddles
    }};
}

macro_rules! gen_butterfly_twiddles_interleaved_columns_f32 {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $direction: expr) => {{
        const FFT_LEN: usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS: usize = $num_rows - 1;
        const TWIDDLE_COLS: usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS: usize = TWIDDLE_COLS / 4;
        const TWIDDLE_VECTOR_COUNT: usize = TWIDDLE_VECTOR_COLS * TWIDDLE_ROWS;
        let mut twiddles = [AvxStoreF::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index / TWIDDLE_VECTOR_COLS) + 1;
            let x = (index % TWIDDLE_VECTOR_COLS) * 4 + $skip_cols;

            use crate::avx::butterflies::make_mixedradix_twiddle_chunk_f32;

            twiddles[index] = make_mixedradix_twiddle_chunk_f32(x, y, FFT_LEN, $direction);
        }
        twiddles
    }};
}

macro_rules! gen_butterfly_twiddles_separated_columns_f32 {
    ($num_rows:expr, $num_cols:expr, $skip_cols:expr, $direction: expr) => {{
        const FFT_LEN: usize = $num_rows * $num_cols;
        const TWIDDLE_ROWS: usize = $num_rows - 1;
        const TWIDDLE_COLS: usize = $num_cols - $skip_cols;
        const TWIDDLE_VECTOR_COLS: usize = TWIDDLE_COLS / 4;
        const TWIDDLE_VECTOR_COUNT: usize = TWIDDLE_VECTOR_COLS * TWIDDLE_ROWS;
        let mut twiddles = [AvxStoreF::zero(); TWIDDLE_VECTOR_COUNT];
        for index in 0..TWIDDLE_VECTOR_COUNT {
            let y = (index % TWIDDLE_ROWS) + 1;
            let x = (index / TWIDDLE_ROWS) * 4 + $skip_cols;
            use crate::avx::butterflies::make_mixedradix_twiddle_chunk_f32;
            twiddles[index] = make_mixedradix_twiddle_chunk_f32(x, y, FFT_LEN, $direction);
        }
        twiddles
    }};
}

pub(crate) use gen_butterfly_twiddles_separated_columns_f32;

pub(crate) use gen_butterfly_twiddles_interleaved_columns_f32;
pub(crate) use gen_butterfly_twiddles_interleaved_columns_f64;

#[cfg(test)]
macro_rules! test_avx_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use crate::util::has_valid_avx;
            if !has_valid_avx() {
                return;
            }
            let radix_forward = $butterfly::new(FftDirection::Forward);
            let radix_inverse = $butterfly::new(FftDirection::Inverse);
            assert_eq!(radix_forward.length(), $scale);
            use rand::Rng;
            for i in 1..20 {
                let val = $scale as usize;
                let size = val * i;
                let mut input = vec![Complex::<$data_type>::default(); size];
                for z in input.iter_mut() {
                    *z = Complex {
                        re: rand::rng().random(),
                        im: rand::rng().random(),
                    };
                }
                let src = input.to_vec();
                use crate::dft::Dft;
                let reference_forward = Dft::new($scale, FftDirection::Forward).unwrap();

                let mut ref_src = src.to_vec();
                reference_forward.execute(&mut ref_src).unwrap();

                radix_forward.execute(&mut input).unwrap();

                input
                    .iter()
                    .zip(ref_src.iter())
                    .enumerate()
                    .for_each(|(idx, (a, b))| {
                        assert!(
                            (a.re - b.re).abs() < $tol,
                            "a_re {} != b_re {} for size {} at {idx}",
                            a.re,
                            b.re,
                            size
                        );
                        assert!(
                            (a.im - b.im).abs() < $tol,
                            "a_im {} != b_im {} for size {} at {idx}",
                            a.im,
                            b.im,
                            size
                        );
                    });

                radix_inverse.execute(&mut input).unwrap();

                let val = $scale as $data_type;
                input = input.iter().map(|&x| x * (1.0 / val)).collect();

                input.iter().zip(src.iter()).for_each(|(a, b)| {
                    assert!(
                        (a.re - b.re).abs() < $tol,
                        "a_re {} != b_re {} for size {}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < $tol,
                        "a_im {} != b_im {} for size {}",
                        a.im,
                        b.im,
                        size
                    );
                });
            }
        }
    };
}

#[cfg(test)]
pub(crate) use test_avx_butterfly;

#[cfg(test)]
macro_rules! test_oof_avx_butterfly {
    ($method_name: ident, $data_type: ident, $butterfly: ident, $scale: expr, $tol: expr) => {
        #[test]
        fn $method_name() {
            use crate::util::has_valid_avx;
            if !has_valid_avx() {
                return;
            }
            use rand::Rng;
            for i in 1..20 {
                let kern = $scale;
                let size = (kern as usize) * i;
                let mut input = vec![Complex::<$data_type>::default(); size];
                for z in input.iter_mut() {
                    *z = Complex {
                        re: rand::rng().random(),
                        im: rand::rng().random(),
                    };
                }
                let src = input.to_vec();
                let mut out_of_place = vec![Complex::<$data_type>::default(); size];
                let mut ref_input = input.to_vec();
                let radix_forward = $butterfly::new(FftDirection::Forward);
                let radix_inverse = $butterfly::new(FftDirection::Inverse);

                use crate::dft::Dft;
                let reference_dft = Dft::new($scale, FftDirection::Forward).unwrap();
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
                            (a.re - b.re).abs() < $tol,
                            "a_re {} != b_re {} for size {} at {idx}",
                            a.re,
                            b.re,
                            size
                        );
                        assert!(
                            (a.im - b.im).abs() < $tol,
                            "a_im {} != b_im {} for size {} at {idx}",
                            a.im,
                            b.im,
                            size
                        );
                    });

                radix_inverse
                    .execute_out_of_place(&out_of_place, &mut input)
                    .unwrap();

                input = input
                    .iter()
                    .map(|&x| x * (1.0 / (kern as $data_type)))
                    .collect();

                input.iter().zip(src.iter()).for_each(|(a, b)| {
                    assert!(
                        (a.re - b.re).abs() < $tol,
                        "a_re {} != b_re {} for size {}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < $tol,
                        "a_im {} != b_im {} for size {}",
                        a.im,
                        b.im,
                        size
                    );
                });
            }
        }
    };
}

use crate::FftDirection;
use crate::avx::mixed::{AvxStoreD, AvxStoreF};
use crate::util::compute_twiddle;
#[cfg(test)]
pub(crate) use test_oof_avx_butterfly;
