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
use crate::avx::butterflies::AvxButterfly;
use crate::avx::mixed::{AvxStoreD, ColumnButterfly16d};
use crate::avx::util::{_mm256_create_pd, _mm256_fcmul_pd, _mm256_set2_complexd};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly32d {
    direction: FftDirection,
    bf16: ColumnButterfly16d,
    twiddle1: __m256d,
    twiddle2: __m256d,
    twiddle3: __m256d,
    twiddle4: __m256d,
    twiddle5: __m256d,
    twiddle6: __m256d,
    twiddle7: __m256d,
}

impl AvxButterfly32d {
    pub(crate) fn new(direction: FftDirection) -> AvxButterfly32d {
        unsafe {
            let t1 = compute_twiddle(1, 32, direction);
            let t2 = compute_twiddle(2, 32, direction);
            let t3 = compute_twiddle(3, 32, direction);
            let t4 = compute_twiddle(4, 32, direction);
            let t5 = compute_twiddle(5, 32, direction);
            let t6 = compute_twiddle(6, 32, direction);
            let t7 = compute_twiddle(7, 32, direction);
            Self {
                direction,
                bf16: ColumnButterfly16d::new(direction),
                twiddle1: _mm256_set2_complexd(t1, t1.conj()),
                twiddle2: _mm256_set2_complexd(t2, t2.conj()),
                twiddle3: _mm256_set2_complexd(t3, t3.conj()),
                twiddle4: _mm256_set2_complexd(t4, t4.conj()),
                twiddle5: _mm256_set2_complexd(t5, t5.conj()),
                twiddle6: _mm256_set2_complexd(t6, t6.conj()),
                twiddle7: _mm256_set2_complexd(t7, t7.conj()),
            }
        }
    }
}

impl AvxButterfly32d {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f64(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if in_place.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(32) {
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
                let u26u27 = _mm256_loadu_pd(chunk.get_unchecked(26..).as_ptr().cast());
                let u28u29 = _mm256_loadu_pd(chunk.get_unchecked(28..).as_ptr().cast());
                let u30u31 = _mm256_loadu_pd(chunk.get_unchecked(30..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let s_evens = self.bf16.exec([
                    AvxStoreD::raw(u0u1),
                    AvxStoreD::raw(u2u3),
                    AvxStoreD::raw(u4u5),
                    AvxStoreD::raw(u6u7),
                    AvxStoreD::raw(u8u9),
                    AvxStoreD::raw(u10u11),
                    AvxStoreD::raw(u12u13),
                    AvxStoreD::raw(u14u15),
                    AvxStoreD::raw(u16u17),
                    AvxStoreD::raw(u18u19),
                    AvxStoreD::raw(u20u21),
                    AvxStoreD::raw(u22u23),
                    AvxStoreD::raw(u24u25),
                    AvxStoreD::raw(u26u27),
                    AvxStoreD::raw(u28u29),
                    AvxStoreD::raw(u30u31),
                ]); //  u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30,
                let mut odds1 = self.bf16.bf8.exec(
                    _mm256_permute2f128_pd::<HI_HI>(u0u1, u30u31),
                    _mm256_permute2f128_pd::<HI_HI>(u4u5, u2u3),
                    _mm256_permute2f128_pd::<HI_HI>(u8u9, u6u7),
                    _mm256_permute2f128_pd::<HI_HI>(u12u13, u10u11),
                    _mm256_permute2f128_pd::<HI_HI>(u16u17, u14u15),
                    _mm256_permute2f128_pd::<HI_HI>(u20u21, u18u19),
                    _mm256_permute2f128_pd::<HI_HI>(u24u25, u22u23),
                    _mm256_permute2f128_pd::<HI_HI>(u28u29, u26u27),
                ); // u1, u5, u9, u13, u17, u21, u25, u29
                // u31, u3, u7, u11, u15, u19, u23, u27

                let evens01 = _mm256_permute2f128_pd::<LO_LO>(s_evens[0].v, s_evens[1].v);
                let evens23 = _mm256_permute2f128_pd::<LO_LO>(s_evens[2].v, s_evens[3].v);
                let evens45 = _mm256_permute2f128_pd::<LO_LO>(s_evens[4].v, s_evens[5].v);
                let evens67 = _mm256_permute2f128_pd::<LO_LO>(s_evens[6].v, s_evens[7].v);
                let evens89 = _mm256_permute2f128_pd::<LO_LO>(s_evens[8].v, s_evens[9].v);
                let evens1011 = _mm256_permute2f128_pd::<LO_LO>(s_evens[10].v, s_evens[11].v);
                let evens1213 = _mm256_permute2f128_pd::<LO_LO>(s_evens[12].v, s_evens[13].v);
                let evens1415 = _mm256_permute2f128_pd::<LO_LO>(s_evens[14].v, s_evens[15].v);

                odds1.1 = _mm256_fcmul_pd(odds1.1, self.twiddle1);
                odds1.2 = _mm256_fcmul_pd(odds1.2, self.twiddle2);
                odds1.3 = _mm256_fcmul_pd(odds1.3, self.twiddle3);
                odds1.4 = _mm256_fcmul_pd(odds1.4, self.twiddle4);
                odds1.5 = _mm256_fcmul_pd(odds1.5, self.twiddle5);
                odds1.6 = _mm256_fcmul_pd(odds1.6, self.twiddle6);
                odds1.7 = _mm256_fcmul_pd(odds1.7, self.twiddle7);

                let mut q0 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.0),
                    _mm256_extractf128_pd::<1>(odds1.0),
                );
                let mut q1 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.1),
                    _mm256_extractf128_pd::<1>(odds1.1),
                );
                let mut q2 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.2),
                    _mm256_extractf128_pd::<1>(odds1.2),
                );
                let mut q3 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.3),
                    _mm256_extractf128_pd::<1>(odds1.3),
                );
                let mut q4 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.4),
                    _mm256_extractf128_pd::<1>(odds1.4),
                );
                let mut q5 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.5),
                    _mm256_extractf128_pd::<1>(odds1.5),
                );
                let mut q6 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.6),
                    _mm256_extractf128_pd::<1>(odds1.6),
                );
                let mut q7 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.7),
                    _mm256_extractf128_pd::<1>(odds1.7),
                );

                q0.1 = self.bf16.rotate.rotate_m128d(q0.1);
                q1.1 = self.bf16.rotate.rotate_m128d(q1.1);
                q2.1 = self.bf16.rotate.rotate_m128d(q2.1);
                q3.1 = self.bf16.rotate.rotate_m128d(q3.1);
                q4.1 = self.bf16.rotate.rotate_m128d(q4.1);
                q5.1 = self.bf16.rotate.rotate_m128d(q5.1);
                q6.1 = self.bf16.rotate.rotate_m128d(q6.1);
                q7.1 = self.bf16.rotate.rotate_m128d(q7.1);

                let q00 = _mm256_create_pd(q0.0, q1.0);
                let q01 = _mm256_create_pd(q2.0, q3.0);
                let q02 = _mm256_create_pd(q4.0, q5.0);
                let q03 = _mm256_create_pd(q6.0, q7.0);
                let q04 = _mm256_create_pd(q0.1, q1.1);
                let q05 = _mm256_create_pd(q2.1, q3.1);
                let q06 = _mm256_create_pd(q4.1, q5.1);
                let q07 = _mm256_create_pd(q6.1, q7.1);

                _mm256_storeu_pd(chunk.as_mut_ptr().cast(), _mm256_add_pd(evens01, q00));
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens23, q01),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens45, q02),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens67, q03),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens89, q04),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens1011, q05),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens1213, q06),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens1415, q07),
                );

                // upper part

                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens01, q00),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens23, q01),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens45, q02),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens67, q03),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens89, q04),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens1011, q05),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens1213, q06),
                );
                _mm256_storeu_pd(
                    chunk.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens1415, q07),
                );
            }

            Ok(())
        }
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f64(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if src.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
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
                let u26u27 = _mm256_loadu_pd(src.get_unchecked(26..).as_ptr().cast());
                let u28u29 = _mm256_loadu_pd(src.get_unchecked(28..).as_ptr().cast());
                let u30u31 = _mm256_loadu_pd(src.get_unchecked(30..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let s_evens = self.bf16.exec([
                    AvxStoreD::raw(u0u1),
                    AvxStoreD::raw(u2u3),
                    AvxStoreD::raw(u4u5),
                    AvxStoreD::raw(u6u7),
                    AvxStoreD::raw(u8u9),
                    AvxStoreD::raw(u10u11),
                    AvxStoreD::raw(u12u13),
                    AvxStoreD::raw(u14u15),
                    AvxStoreD::raw(u16u17),
                    AvxStoreD::raw(u18u19),
                    AvxStoreD::raw(u20u21),
                    AvxStoreD::raw(u22u23),
                    AvxStoreD::raw(u24u25),
                    AvxStoreD::raw(u26u27),
                    AvxStoreD::raw(u28u29),
                    AvxStoreD::raw(u30u31),
                ]); //  u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30,
                let mut odds1 = self.bf16.bf8.exec(
                    _mm256_permute2f128_pd::<HI_HI>(u0u1, u30u31),
                    _mm256_permute2f128_pd::<HI_HI>(u4u5, u2u3),
                    _mm256_permute2f128_pd::<HI_HI>(u8u9, u6u7),
                    _mm256_permute2f128_pd::<HI_HI>(u12u13, u10u11),
                    _mm256_permute2f128_pd::<HI_HI>(u16u17, u14u15),
                    _mm256_permute2f128_pd::<HI_HI>(u20u21, u18u19),
                    _mm256_permute2f128_pd::<HI_HI>(u24u25, u22u23),
                    _mm256_permute2f128_pd::<HI_HI>(u28u29, u26u27),
                ); // u1, u5, u9, u13, u17, u21, u25, u29
                // u31, u3, u7, u11, u15, u19, u23, u27

                let evens01 = _mm256_permute2f128_pd::<LO_LO>(s_evens[0].v, s_evens[1].v);
                let evens23 = _mm256_permute2f128_pd::<LO_LO>(s_evens[2].v, s_evens[3].v);
                let evens45 = _mm256_permute2f128_pd::<LO_LO>(s_evens[4].v, s_evens[5].v);
                let evens67 = _mm256_permute2f128_pd::<LO_LO>(s_evens[6].v, s_evens[7].v);
                let evens89 = _mm256_permute2f128_pd::<LO_LO>(s_evens[8].v, s_evens[9].v);
                let evens1011 = _mm256_permute2f128_pd::<LO_LO>(s_evens[10].v, s_evens[11].v);
                let evens1213 = _mm256_permute2f128_pd::<LO_LO>(s_evens[12].v, s_evens[13].v);
                let evens1415 = _mm256_permute2f128_pd::<LO_LO>(s_evens[14].v, s_evens[15].v);

                odds1.1 = _mm256_fcmul_pd(odds1.1, self.twiddle1);
                odds1.2 = _mm256_fcmul_pd(odds1.2, self.twiddle2);
                odds1.3 = _mm256_fcmul_pd(odds1.3, self.twiddle3);
                odds1.4 = _mm256_fcmul_pd(odds1.4, self.twiddle4);
                odds1.5 = _mm256_fcmul_pd(odds1.5, self.twiddle5);
                odds1.6 = _mm256_fcmul_pd(odds1.6, self.twiddle6);
                odds1.7 = _mm256_fcmul_pd(odds1.7, self.twiddle7);

                let mut q0 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.0),
                    _mm256_extractf128_pd::<1>(odds1.0),
                );
                let mut q1 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.1),
                    _mm256_extractf128_pd::<1>(odds1.1),
                );
                let mut q2 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.2),
                    _mm256_extractf128_pd::<1>(odds1.2),
                );
                let mut q3 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.3),
                    _mm256_extractf128_pd::<1>(odds1.3),
                );
                let mut q4 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.4),
                    _mm256_extractf128_pd::<1>(odds1.4),
                );
                let mut q5 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.5),
                    _mm256_extractf128_pd::<1>(odds1.5),
                );
                let mut q6 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.6),
                    _mm256_extractf128_pd::<1>(odds1.6),
                );
                let mut q7 = AvxButterfly::butterfly2_f64_m128(
                    _mm256_castpd256_pd128(odds1.7),
                    _mm256_extractf128_pd::<1>(odds1.7),
                );

                q0.1 = self.bf16.rotate.rotate_m128d(q0.1);
                q1.1 = self.bf16.rotate.rotate_m128d(q1.1);
                q2.1 = self.bf16.rotate.rotate_m128d(q2.1);
                q3.1 = self.bf16.rotate.rotate_m128d(q3.1);
                q4.1 = self.bf16.rotate.rotate_m128d(q4.1);
                q5.1 = self.bf16.rotate.rotate_m128d(q5.1);
                q6.1 = self.bf16.rotate.rotate_m128d(q6.1);
                q7.1 = self.bf16.rotate.rotate_m128d(q7.1);

                let q00 = _mm256_create_pd(q0.0, q1.0);
                let q01 = _mm256_create_pd(q2.0, q3.0);
                let q02 = _mm256_create_pd(q4.0, q5.0);
                let q03 = _mm256_create_pd(q6.0, q7.0);
                let q04 = _mm256_create_pd(q0.1, q1.1);
                let q05 = _mm256_create_pd(q2.1, q3.1);
                let q06 = _mm256_create_pd(q4.1, q5.1);
                let q07 = _mm256_create_pd(q6.1, q7.1);

                _mm256_storeu_pd(dst.as_mut_ptr().cast(), _mm256_add_pd(evens01, q00));
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(2..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens23, q01),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens45, q02),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(6..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens67, q03),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens89, q04),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(10..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens1011, q05),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens1213, q06),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(14..).as_mut_ptr().cast(),
                    _mm256_add_pd(evens1415, q07),
                );

                // upper part

                _mm256_storeu_pd(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens01, q00),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(18..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens23, q01),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens45, q02),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(22..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens67, q03),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens89, q04),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(26..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens1011, q05),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens1213, q06),
                );
                _mm256_storeu_pd(
                    dst.get_unchecked_mut(30..).as_mut_ptr().cast(),
                    _mm256_sub_pd(evens1415, q07),
                );
            }

            Ok(())
        }
    }
}

impl FftExecutorOutOfPlace<f64> for AvxButterfly32d {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f64(src, dst) }
    }
}

impl CompositeFftExecutor<f64> for AvxButterfly32d {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f64> + Send + Sync> {
        self
    }
}

impl FftExecutor<f64> for AvxButterfly32d {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f64(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dft::Dft;
    use rand::Rng;

    #[test]
    fn test_avx_butterfly32_f64() {
        for i in 1..4 {
            let size = 32usize.pow(i);
            let mut input = vec![Complex::<f64>::default(); size];
            for z in input.iter_mut() {
                *z = Complex {
                    re: rand::rng().random(),
                    im: rand::rng().random(),
                };
            }
            let src = input.to_vec();

            let mut z_ref = input.to_vec();

            let bf23_reference = Dft::new(32, FftDirection::Forward).unwrap();

            let radix_forward = AvxButterfly32d::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly32d::new(FftDirection::Inverse);
            radix_forward.execute(&mut input).unwrap();
            bf23_reference.execute(&mut z_ref).unwrap();

            input
                .iter()
                .zip(z_ref.iter())
                .enumerate()
                .for_each(|(idx, (a, b))| {
                    assert!(
                        (a.re - b.re).abs() < 1e-9,
                        "a_re {} != b_re {} for size {}, reference failed at {idx}",
                        a.re,
                        b.re,
                        size
                    );
                    assert!(
                        (a.im - b.im).abs() < 1e-9,
                        "a_im {} != b_im {} for size {}, reference failed at {idx}",
                        a.im,
                        b.im,
                        size
                    );
                });

            radix_inverse.execute(&mut input).unwrap();

            input = input.iter().map(|&x| x * (1.0 / 32f64)).collect();

            input.iter().zip(src.iter()).for_each(|(a, b)| {
                assert!(
                    (a.re - b.re).abs() < 1e-8,
                    "a_re {} != b_re {} for size {}",
                    a.re,
                    b.re,
                    size
                );
                assert!(
                    (a.im - b.im).abs() < 1e-8,
                    "a_im {} != b_im {} for size {}",
                    a.im,
                    b.im,
                    size
                );
            });
        }
    }

    #[test]
    fn test_butterfly32_out_of_place_f64() {
        for i in 1..4 {
            let size = 32usize.pow(i);
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
            let radix_forward = AvxButterfly32d::new(FftDirection::Forward);
            let radix_inverse = AvxButterfly32d::new(FftDirection::Inverse);

            let reference_dft = Dft::new(32, FftDirection::Forward).unwrap();
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

            input = input.iter().map(|&x| x * (1.0 / 32f64)).collect();

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
