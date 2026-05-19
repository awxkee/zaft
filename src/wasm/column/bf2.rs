/*
 * // Copyright (c) Radzivon Bartoshyk 5/2026. All rights reserved.
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
use crate::wasm::store::{WasmStoreD, WasmStoreF};

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
    pub(crate) fn exec(&self, store: [WasmStoreD; 2]) -> [WasmStoreD; 2] {
        let t = store[0] + store[1];
        let y1 = store[0] - store[1];
        [t, y1]
    }
}

impl ColumnButterfly2f {
    #[inline(always)]
    pub(crate) fn exec(&self, store: [WasmStoreF; 2]) -> [WasmStoreF; 2] {
        let t = store[0] + store[1];
        let y1 = store[0] - store[1];
        [t, y1]
    }
}
