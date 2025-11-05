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
use crate::avx::mixed::{AvxStoreF, ColumnButterfly16f};
use crate::avx::util::{
    _mm_fcmul_ps, _mm_unpackhi_ps64, _mm_unpacklo_ps64, _mm256_create_ps, _mm256_set4_complex,
};
use crate::util::compute_twiddle;
use crate::{CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, ZaftError};
use num_complex::Complex;
use std::arch::x86_64::*;

pub(crate) struct AvxButterfly32f {
    direction: FftDirection,
    bf16: ColumnButterfly16f,
    twiddle1: __m256,
    twiddle2: __m256,
    twiddle3: __m256,
    twiddle4: __m256,
    twiddle5: __m256,
    twiddle6: __m256,
    twiddle7: __m256,
}

impl AvxButterfly32f {
    pub(crate) fn new(direction: FftDirection) -> AvxButterfly32f {
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
                bf16: ColumnButterfly16f::new(direction),
                twiddle1: _mm256_set4_complex(t1, t1.conj(), t1, t1.conj()),
                twiddle2: _mm256_set4_complex(t2, t2.conj(), t2, t2.conj()),
                twiddle3: _mm256_set4_complex(t3, t3.conj(), t3, t3.conj()),
                twiddle4: _mm256_set4_complex(t4, t4.conj(), t4, t4.conj()),
                twiddle5: _mm256_set4_complex(t5, t5.conj(), t5, t5.conj()),
                twiddle6: _mm256_set4_complex(t6, t6.conj(), t6, t6.conj()),
                twiddle7: _mm256_set4_complex(t7, t7.conj(), t7, t7.conj()),
            }
        }
    }
}

impl AvxButterfly32f {
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_f32(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if in_place.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        unsafe {
            for chunk in in_place.chunks_exact_mut(32) {
                let u0u1u2u3 = _mm256_loadu_ps(chunk.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(chunk.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(chunk.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(chunk.get_unchecked(12..).as_ptr().cast());
                let u16u17u18u19 = _mm256_loadu_ps(chunk.get_unchecked(16..).as_ptr().cast());
                let u20u21u22u23 = _mm256_loadu_ps(chunk.get_unchecked(20..).as_ptr().cast());
                let u24u25u26u27 = _mm256_loadu_ps(chunk.get_unchecked(24..).as_ptr().cast());
                let u28u29u30u31 = _mm256_loadu_ps(chunk.get_unchecked(28..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let s_evens = self.bf16.exec([
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u0u1u2u3, u0u1u2u3)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u0u1u2u3, u0u1u2u3)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u4u5u6u7, u4u5u6u7)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u4u5u6u7, u4u5u6u7)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u8u9u10u11, u8u9u10u11)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u8u9u10u11, u8u9u10u11)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u12u13u14u15, u12u13u14u15)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u12u13u14u15, u12u13u14u15)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u16u17u18u19, u16u17u18u19)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u16u17u18u19, u16u17u18u19)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u20u21u22u23, u20u21u22u23)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u20u21u22u23, u20u21u22u23)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u24u25u26u27, u24u25u26u27)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u24u25u26u27, u24u25u26u27)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u28u29u30u31, u28u29u30u31)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u28u29u30u31, u28u29u30u31)),
                ]); //  u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30

                let mut odds1 = self.bf16.bf8.exec_short(
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_extractf128_ps::<1>(u28u29u30u31),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u4u5u6u7),
                        _mm256_extractf128_ps::<1>(u0u1u2u3),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u8u9u10u11),
                        _mm256_extractf128_ps::<1>(u4u5u6u7),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u12u13u14u15),
                        _mm256_extractf128_ps::<1>(u8u9u10u11),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u16u17u18u19),
                        _mm256_extractf128_ps::<1>(u12u13u14u15),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u20u21u22u23),
                        _mm256_extractf128_ps::<1>(u16u17u18u19),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u24u25u26u27),
                        _mm256_extractf128_ps::<1>(u20u21u22u23),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u28u29u30u31),
                        _mm256_extractf128_ps::<1>(u24u25u26u27),
                    ),
                ); // u1, u5, u9, u13, u17, u21, u25, u29
                // u31, u3, u7, u11, u15, u19, u23, u27

                let evens01 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[0].v),
                    _mm256_castps256_ps128(s_evens[1].v),
                );
                let evens23 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[2].v),
                    _mm256_castps256_ps128(s_evens[3].v),
                );
                let evens45 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[4].v),
                    _mm256_castps256_ps128(s_evens[5].v),
                );
                let evens67 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[6].v),
                    _mm256_castps256_ps128(s_evens[7].v),
                );
                let evens89 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[8].v),
                    _mm256_castps256_ps128(s_evens[9].v),
                );
                let evens1011 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[10].v),
                    _mm256_castps256_ps128(s_evens[11].v),
                );
                let evens1213 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[12].v),
                    _mm256_castps256_ps128(s_evens[13].v),
                );
                let evens1415 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[14].v),
                    _mm256_castps256_ps128(s_evens[15].v),
                );

                odds1.1 = _mm_fcmul_ps(odds1.1, _mm256_castps256_ps128(self.twiddle1));
                odds1.2 = _mm_fcmul_ps(odds1.2, _mm256_castps256_ps128(self.twiddle2));
                odds1.3 = _mm_fcmul_ps(odds1.3, _mm256_castps256_ps128(self.twiddle3));
                odds1.4 = _mm_fcmul_ps(odds1.4, _mm256_castps256_ps128(self.twiddle4));
                odds1.5 = _mm_fcmul_ps(odds1.5, _mm256_castps256_ps128(self.twiddle5));
                odds1.6 = _mm_fcmul_ps(odds1.6, _mm256_castps256_ps128(self.twiddle6));
                odds1.7 = _mm_fcmul_ps(odds1.7, _mm256_castps256_ps128(self.twiddle7));

                let mut q0 =
                    AvxButterfly::butterfly2_f32_m128(odds1.0, _mm_unpackhi_ps64(odds1.0, odds1.0));
                let mut q1 =
                    AvxButterfly::butterfly2_f32_m128(odds1.1, _mm_unpackhi_ps64(odds1.1, odds1.1));
                let mut q2 =
                    AvxButterfly::butterfly2_f32_m128(odds1.2, _mm_unpackhi_ps64(odds1.2, odds1.2));
                let mut q3 =
                    AvxButterfly::butterfly2_f32_m128(odds1.3, _mm_unpackhi_ps64(odds1.3, odds1.3));
                let mut q4 =
                    AvxButterfly::butterfly2_f32_m128(odds1.4, _mm_unpackhi_ps64(odds1.4, odds1.4));
                let mut q5 =
                    AvxButterfly::butterfly2_f32_m128(odds1.5, _mm_unpackhi_ps64(odds1.5, odds1.5));
                let mut q6 =
                    AvxButterfly::butterfly2_f32_m128(odds1.6, _mm_unpackhi_ps64(odds1.6, odds1.6));
                let mut q7 =
                    AvxButterfly::butterfly2_f32_m128(odds1.7, _mm_unpackhi_ps64(odds1.7, odds1.7));

                q0.1 = self.bf16.rotate.rotate_m128(q0.1);
                q1.1 = self.bf16.rotate.rotate_m128(q1.1);
                q2.1 = self.bf16.rotate.rotate_m128(q2.1);
                q3.1 = self.bf16.rotate.rotate_m128(q3.1);
                q4.1 = self.bf16.rotate.rotate_m128(q4.1);
                q5.1 = self.bf16.rotate.rotate_m128(q5.1);
                q6.1 = self.bf16.rotate.rotate_m128(q6.1);
                q7.1 = self.bf16.rotate.rotate_m128(q7.1);

                let q00 = _mm_unpacklo_ps64(q0.0, q1.0);
                let q01 = _mm_unpacklo_ps64(q2.0, q3.0);
                let q02 = _mm_unpacklo_ps64(q4.0, q5.0);
                let q03 = _mm_unpacklo_ps64(q6.0, q7.0);
                let q04 = _mm_unpacklo_ps64(q0.1, q1.1);
                let q05 = _mm_unpacklo_ps64(q2.1, q3.1);
                let q06 = _mm_unpacklo_ps64(q4.1, q5.1);
                let q07 = _mm_unpacklo_ps64(q6.1, q7.1);

                _mm256_storeu_ps(
                    chunk.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens01, q00), _mm_add_ps(evens23, q01)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens45, q02), _mm_add_ps(evens67, q03)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens89, q04), _mm_add_ps(evens1011, q05)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens1213, q06), _mm_add_ps(evens1415, q07)),
                );

                // upper part

                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens01, q00), _mm_sub_ps(evens23, q01)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens45, q02), _mm_sub_ps(evens67, q03)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens89, q04), _mm_sub_ps(evens1011, q05)),
                );
                _mm256_storeu_ps(
                    chunk.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens1213, q06), _mm_sub_ps(evens1415, q07)),
                );
            }
        }
        Ok(())
    }

    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn execute_out_of_place_f32(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if src.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if dst.len() % 32 != 0 {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        unsafe {
            for (dst, src) in dst.chunks_exact_mut(32).zip(src.chunks_exact(32)) {
                let u0u1u2u3 = _mm256_loadu_ps(src.as_ptr().cast());
                let u4u5u6u7 = _mm256_loadu_ps(src.get_unchecked(4..).as_ptr().cast());
                let u8u9u10u11 = _mm256_loadu_ps(src.get_unchecked(8..).as_ptr().cast());
                let u12u13u14u15 = _mm256_loadu_ps(src.get_unchecked(12..).as_ptr().cast());
                let u16u17u18u19 = _mm256_loadu_ps(src.get_unchecked(16..).as_ptr().cast());
                let u20u21u22u23 = _mm256_loadu_ps(src.get_unchecked(20..).as_ptr().cast());
                let u24u25u26u27 = _mm256_loadu_ps(src.get_unchecked(24..).as_ptr().cast());
                let u28u29u30u31 = _mm256_loadu_ps(src.get_unchecked(28..).as_ptr().cast());

                const HI_HI: i32 = 0b0011_0001;
                const LO_LO: i32 = 0b0010_0000;

                let s_evens = self.bf16.exec([
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u0u1u2u3, u0u1u2u3)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u0u1u2u3, u0u1u2u3)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u4u5u6u7, u4u5u6u7)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u4u5u6u7, u4u5u6u7)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u8u9u10u11, u8u9u10u11)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u8u9u10u11, u8u9u10u11)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u12u13u14u15, u12u13u14u15)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u12u13u14u15, u12u13u14u15)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u16u17u18u19, u16u17u18u19)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u16u17u18u19, u16u17u18u19)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u20u21u22u23, u20u21u22u23)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u20u21u22u23, u20u21u22u23)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u24u25u26u27, u24u25u26u27)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u24u25u26u27, u24u25u26u27)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<LO_LO>(u28u29u30u31, u28u29u30u31)),
                    AvxStoreF::raw(_mm256_permute2f128_ps::<HI_HI>(u28u29u30u31, u28u29u30u31)),
                ]); //  u0, u2, u4, u6, u8, u10, u12, u14, u16, u18, u20, u22, u24, u26, u28, u30

                let mut odds1 = self.bf16.bf8.exec_short(
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u0u1u2u3),
                        _mm256_extractf128_ps::<1>(u28u29u30u31),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u4u5u6u7),
                        _mm256_extractf128_ps::<1>(u0u1u2u3),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u8u9u10u11),
                        _mm256_extractf128_ps::<1>(u4u5u6u7),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u12u13u14u15),
                        _mm256_extractf128_ps::<1>(u8u9u10u11),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u16u17u18u19),
                        _mm256_extractf128_ps::<1>(u12u13u14u15),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u20u21u22u23),
                        _mm256_extractf128_ps::<1>(u16u17u18u19),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u24u25u26u27),
                        _mm256_extractf128_ps::<1>(u20u21u22u23),
                    ),
                    _mm_unpackhi_ps64(
                        _mm256_castps256_ps128(u28u29u30u31),
                        _mm256_extractf128_ps::<1>(u24u25u26u27),
                    ),
                ); // u1, u5, u9, u13, u17, u21, u25, u29
                // u31, u3, u7, u11, u15, u19, u23, u27

                let evens01 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[0].v),
                    _mm256_castps256_ps128(s_evens[1].v),
                );
                let evens23 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[2].v),
                    _mm256_castps256_ps128(s_evens[3].v),
                );
                let evens45 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[4].v),
                    _mm256_castps256_ps128(s_evens[5].v),
                );
                let evens67 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[6].v),
                    _mm256_castps256_ps128(s_evens[7].v),
                );
                let evens89 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[8].v),
                    _mm256_castps256_ps128(s_evens[9].v),
                );
                let evens1011 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[10].v),
                    _mm256_castps256_ps128(s_evens[11].v),
                );
                let evens1213 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[12].v),
                    _mm256_castps256_ps128(s_evens[13].v),
                );
                let evens1415 = _mm_unpacklo_ps64(
                    _mm256_castps256_ps128(s_evens[14].v),
                    _mm256_castps256_ps128(s_evens[15].v),
                );

                odds1.1 = _mm_fcmul_ps(odds1.1, _mm256_castps256_ps128(self.twiddle1));
                odds1.2 = _mm_fcmul_ps(odds1.2, _mm256_castps256_ps128(self.twiddle2));
                odds1.3 = _mm_fcmul_ps(odds1.3, _mm256_castps256_ps128(self.twiddle3));
                odds1.4 = _mm_fcmul_ps(odds1.4, _mm256_castps256_ps128(self.twiddle4));
                odds1.5 = _mm_fcmul_ps(odds1.5, _mm256_castps256_ps128(self.twiddle5));
                odds1.6 = _mm_fcmul_ps(odds1.6, _mm256_castps256_ps128(self.twiddle6));
                odds1.7 = _mm_fcmul_ps(odds1.7, _mm256_castps256_ps128(self.twiddle7));

                let mut q0 =
                    AvxButterfly::butterfly2_f32_m128(odds1.0, _mm_unpackhi_ps64(odds1.0, odds1.0));
                let mut q1 =
                    AvxButterfly::butterfly2_f32_m128(odds1.1, _mm_unpackhi_ps64(odds1.1, odds1.1));
                let mut q2 =
                    AvxButterfly::butterfly2_f32_m128(odds1.2, _mm_unpackhi_ps64(odds1.2, odds1.2));
                let mut q3 =
                    AvxButterfly::butterfly2_f32_m128(odds1.3, _mm_unpackhi_ps64(odds1.3, odds1.3));
                let mut q4 =
                    AvxButterfly::butterfly2_f32_m128(odds1.4, _mm_unpackhi_ps64(odds1.4, odds1.4));
                let mut q5 =
                    AvxButterfly::butterfly2_f32_m128(odds1.5, _mm_unpackhi_ps64(odds1.5, odds1.5));
                let mut q6 =
                    AvxButterfly::butterfly2_f32_m128(odds1.6, _mm_unpackhi_ps64(odds1.6, odds1.6));
                let mut q7 =
                    AvxButterfly::butterfly2_f32_m128(odds1.7, _mm_unpackhi_ps64(odds1.7, odds1.7));

                q0.1 = self.bf16.rotate.rotate_m128(q0.1);
                q1.1 = self.bf16.rotate.rotate_m128(q1.1);
                q2.1 = self.bf16.rotate.rotate_m128(q2.1);
                q3.1 = self.bf16.rotate.rotate_m128(q3.1);
                q4.1 = self.bf16.rotate.rotate_m128(q4.1);
                q5.1 = self.bf16.rotate.rotate_m128(q5.1);
                q6.1 = self.bf16.rotate.rotate_m128(q6.1);
                q7.1 = self.bf16.rotate.rotate_m128(q7.1);

                let q00 = _mm_unpacklo_ps64(q0.0, q1.0);
                let q01 = _mm_unpacklo_ps64(q2.0, q3.0);
                let q02 = _mm_unpacklo_ps64(q4.0, q5.0);
                let q03 = _mm_unpacklo_ps64(q6.0, q7.0);
                let q04 = _mm_unpacklo_ps64(q0.1, q1.1);
                let q05 = _mm_unpacklo_ps64(q2.1, q3.1);
                let q06 = _mm_unpacklo_ps64(q4.1, q5.1);
                let q07 = _mm_unpacklo_ps64(q6.1, q7.1);

                _mm256_storeu_ps(
                    dst.as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens01, q00), _mm_add_ps(evens23, q01)),
                );
                _mm256_storeu_ps(
                    dst.get_unchecked_mut(4..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens45, q02), _mm_add_ps(evens67, q03)),
                );
                _mm256_storeu_ps(
                    dst.get_unchecked_mut(8..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens89, q04), _mm_add_ps(evens1011, q05)),
                );
                _mm256_storeu_ps(
                    dst.get_unchecked_mut(12..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_add_ps(evens1213, q06), _mm_add_ps(evens1415, q07)),
                );

                // upper part

                _mm256_storeu_ps(
                    dst.get_unchecked_mut(16..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens01, q00), _mm_sub_ps(evens23, q01)),
                );
                _mm256_storeu_ps(
                    dst.get_unchecked_mut(20..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens45, q02), _mm_sub_ps(evens67, q03)),
                );
                _mm256_storeu_ps(
                    dst.get_unchecked_mut(24..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens89, q04), _mm_sub_ps(evens1011, q05)),
                );
                _mm256_storeu_ps(
                    dst.get_unchecked_mut(28..).as_mut_ptr().cast(),
                    _mm256_create_ps(_mm_sub_ps(evens1213, q06), _mm_sub_ps(evens1415, q07)),
                );
            }
        }
        Ok(())
    }
}

impl FftExecutor<f32> for AvxButterfly32f {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        unsafe { self.execute_f32(in_place) }
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    fn length(&self) -> usize {
        32
    }
}

impl FftExecutorOutOfPlace<f32> for AvxButterfly32f {
    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        unsafe { self.execute_out_of_place_f32(src, dst) }
    }
}

impl CompositeFftExecutor<f32> for AvxButterfly32f {
    fn into_fft_executor(self: Box<Self>) -> Box<dyn FftExecutor<f32> + Send + Sync> {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::avx::butterflies::{test_avx_butterfly, test_oof_avx_butterfly};
    use rand::Rng;

    test_avx_butterfly!(test_avx_butterfly32, f32, AvxButterfly32f, 32, 1e-5);

    test_oof_avx_butterfly!(test_oof_avx_butterfly32, f32, AvxButterfly32f, 32, 1e-5);
}
