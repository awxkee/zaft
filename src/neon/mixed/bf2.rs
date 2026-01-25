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
use crate::neon::mixed::neon_store::{NeonStoreD, NeonStoreF, NeonStoreFh};
use std::arch::aarch64::*;

pub(crate) struct ColumnButterfly2d {}
pub(crate) struct ColumnButterfly2f {}

impl ColumnButterfly2d {
    pub(crate) fn new(_: FftDirection) -> ColumnButterfly2d {
        ColumnButterfly2d {}
    }
}

impl ColumnButterfly2f {
    pub(crate) fn new(_: FftDirection) -> ColumnButterfly2f {
        ColumnButterfly2f {}
    }
}

impl ColumnButterfly2d {
    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreD; 2]) -> [NeonStoreD; 2] {
        unsafe {
            let t = vaddq_f64(store[0].v, store[1].v);
            let y1 = vsubq_f64(store[0].v, store[1].v);
            let y0 = t;
            [NeonStoreD { v: y0 }, NeonStoreD { v: y1 }]
        }
    }
}

impl ColumnButterfly2f {
    #[inline(always)]
    pub(crate) fn exec(&self, store: [NeonStoreF; 2]) -> [NeonStoreF; 2] {
        unsafe {
            let t = vaddq_f32(store[0].v, store[1].v);
            let y1 = vsubq_f32(store[0].v, store[1].v);
            let y0 = t;
            [NeonStoreF { v: y0 }, NeonStoreF { v: y1 }]
        }
    }

    #[inline(always)]
    pub(crate) fn exech(&self, store: [NeonStoreFh; 2]) -> [NeonStoreFh; 2] {
        unsafe {
            let t = vadd_f32(store[0].v, store[1].v);
            let y1 = vsub_f32(store[0].v, store[1].v);
            let y0 = t;
            [NeonStoreFh { v: y0 }, NeonStoreFh { v: y1 }]
        }
    }
}
