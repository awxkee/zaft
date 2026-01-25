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
use crate::avx::mixed::avx_stored::AvxStoreD;
use crate::avx::mixed::avx_storef::AvxStoreF;
use crate::avx::util::shuffle;
use crate::util::compute_twiddle;
use std::arch::x86_64::*;

pub(crate) struct ColumnButterfly3d {
    twiddle_re: __m256d,
    twiddle_im: __m256d,
}

impl ColumnButterfly3d {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly3d {
        let twiddle = compute_twiddle::<f64>(1, 3, direction);
        unsafe {
            Self {
                twiddle_re: _mm256_set1_pd(twiddle.re),
                twiddle_im: _mm256_loadu_pd(
                    [-twiddle.im, twiddle.im, -twiddle.im, twiddle.im].as_ptr(),
                ),
            }
        }
    }
}

impl ColumnButterfly3d {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreD; 3]) -> [AvxStoreD; 3] {
        let xp = _mm256_add_pd(v[1].v, v[2].v);
        let xn = _mm256_sub_pd(v[1].v, v[2].v);
        let sum = _mm256_add_pd(v[0].v, xp);

        let w_1 = _mm256_fmadd_pd(self.twiddle_re, xp, v[0].v);
        let xn_rot = _mm256_shuffle_pd::<0b0101>(xn, xn);

        let y0 = sum;
        let y1 = _mm256_fmadd_pd(self.twiddle_im, xn_rot, w_1);
        let y2 = _mm256_fnmadd_pd(self.twiddle_im, xn_rot, w_1);
        [AvxStoreD::raw(y0), AvxStoreD::raw(y1), AvxStoreD::raw(y2)]
    }
}

pub(crate) struct ColumnButterfly3f {
    twiddle_re: __m256,
    twiddle_im: __m256,
}

impl ColumnButterfly3f {
    #[target_feature(enable = "avx2")]
    pub(crate) fn new(direction: FftDirection) -> ColumnButterfly3f {
        let twiddle = compute_twiddle::<f32>(1, 3, direction);
        unsafe {
            Self {
                twiddle_re: _mm256_set1_ps(twiddle.re),
                twiddle_im: _mm256_loadu_ps(
                    [
                        -twiddle.im,
                        twiddle.im,
                        -twiddle.im,
                        twiddle.im,
                        -twiddle.im,
                        twiddle.im,
                        -twiddle.im,
                        twiddle.im,
                    ]
                    .as_ptr(),
                ),
            }
        }
    }
}

impl ColumnButterfly3f {
    #[target_feature(enable = "avx2", enable = "fma")]
    #[cfg_attr(feature = "inline_always", inline(always))]
    #[cfg_attr(not(feature = "inline_always"), inline)]
    pub(crate) fn exec(&self, v: [AvxStoreF; 3]) -> [AvxStoreF; 3] {
        let xp = _mm256_add_ps(v[1].v, v[2].v);
        let xn = _mm256_sub_ps(v[1].v, v[2].v);
        let sum = _mm256_add_ps(v[0].v, xp);

        const SH: i32 = shuffle(2, 3, 0, 1);
        let w_1 = _mm256_fmadd_ps(self.twiddle_re, xp, v[0].v);
        let xn_rot = _mm256_shuffle_ps::<SH>(xn, xn);

        let y0 = sum;
        let y1 = _mm256_fmadd_ps(self.twiddle_im, xn_rot, w_1);
        let y2 = _mm256_fnmadd_ps(self.twiddle_im, xn_rot, w_1);
        [AvxStoreF::raw(y0), AvxStoreF::raw(y1), AvxStoreF::raw(y2)]
    }
}
