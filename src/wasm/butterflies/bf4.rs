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
use crate::traits::FftTrigonometry;
use crate::wasm::rotate::{WasmRotate90D, WasmRotate90F};
use crate::wasm::store::{WasmStoreD, WasmStoreF};
use crate::{FftDirection, FftExecutor, ZaftError};
use num_complex::Complex;
use num_traits::{AsPrimitive, Float};

pub(crate) struct WasmButterfly4<T> {
    direction: FftDirection,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Default + Clone + 'static + Copy + FftTrigonometry + Float> WasmButterfly4<T>
where
    f64: AsPrimitive<T>,
{
    pub(crate) fn new(fft_direction: FftDirection) -> Self {
        Self {
            direction: fft_direction,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl FftExecutor<f32> for WasmButterfly4<f32> {
    fn execute(&self, in_place: &mut [Complex<f32>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let rotate90 = WasmRotate90F::new(self.direction);

        for chunk in in_place.as_chunks_mut::<4>().0.iter_mut() {
            let a = WasmStoreF::load_complex(&chunk[0]);
            let b = WasmStoreF::load_complex(&chunk[1]);
            let c = WasmStoreF::load_complex(&chunk[2]);
            let d = WasmStoreF::load_complex(&chunk[3]);

            let t0 = a + c;
            let t1 = a - c;
            let t2 = b + d;
            let mut t3 = b - d;
            t3 = rotate90.rotate(t3);

            (t0 + t2).write_single(&mut chunk[0]);
            (t1 + t3).write_single(&mut chunk[1]);
            (t0 - t2).write_single(&mut chunk[2]);
            (t1 - t3).write_single(&mut chunk[3]);
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute(in_place)
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f32>],
        dst: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let rotate90 = WasmRotate90F::new(self.direction);

        for (dst, src) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<4>().0.iter())
        {
            let a = WasmStoreF::load_complex(&src[0]);
            let b = WasmStoreF::load_complex(&src[1]);
            let c = WasmStoreF::load_complex(&src[2]);
            let d = WasmStoreF::load_complex(&src[3]);

            let t0 = a + c;
            let t1 = a - c;
            let t2 = b + d;
            let mut t3 = b - d;
            t3 = rotate90.rotate(t3);

            (t0 + t2).write_single(&mut dst[0]);
            (t1 + t3).write_single(&mut dst[1]);
            (t0 - t2).write_single(&mut dst[2]);
            (t1 - t3).write_single(&mut dst[3]);
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f32>],
        dst: &mut [Complex<f32>],
        _: &mut [Complex<f32>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
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

impl FftExecutor<f64> for WasmButterfly4<f64> {
    fn execute(&self, in_place: &mut [Complex<f64>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        let rotate90 = WasmRotate90D::new(self.direction);

        for chunk in in_place.as_chunks_mut::<4>().0.iter_mut() {
            let a = WasmStoreD::from_complex(&chunk[0]);
            let b = WasmStoreD::from_complex(&chunk[1]);
            let c = WasmStoreD::from_complex(&chunk[2]);
            let d = WasmStoreD::from_complex(&chunk[3]);

            let t0 = a + c;
            let t1 = a - c;
            let t2 = b + d;
            let mut t3 = b - d;
            t3 = rotate90.rotate(t3);

            (t0 + t2).write_single(&mut chunk[0]);
            (t1 + t3).write_single(&mut chunk[1]);
            (t0 - t2).write_single(&mut chunk[2]);
            (t1 - t3).write_single(&mut chunk[3]);
        }
        Ok(())
    }

    fn execute_with_scratch(
        &self,
        in_place: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute(in_place)
    }

    fn execute_out_of_place_with_scratch(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn execute_out_of_place(
        &self,
        src: &[Complex<f64>],
        dst: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(4) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        let rotate90 = WasmRotate90D::new(self.direction);

        for (dst, src) in dst
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(src.as_chunks::<4>().0.iter())
        {
            let a = WasmStoreD::from_complex(&src[0]);
            let b = WasmStoreD::from_complex(&src[1]);
            let c = WasmStoreD::from_complex(&src[2]);
            let d = WasmStoreD::from_complex(&src[3]);

            let t0 = a + c;
            let t1 = a - c;
            let t2 = b + d;
            let mut t3 = b - d;
            t3 = rotate90.rotate(t3);

            (t0 + t2).write_single(&mut dst[0]);
            (t1 + t3).write_single(&mut dst[1]);
            (t0 - t2).write_single(&mut dst[2]);
            (t1 - t3).write_single(&mut dst[3]);
        }
        Ok(())
    }

    fn execute_destructive_with_scratch(
        &self,
        src: &mut [Complex<f64>],
        dst: &mut [Complex<f64>],
        _: &mut [Complex<f64>],
    ) -> Result<(), ZaftError> {
        self.execute_out_of_place(src, dst)
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        4
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::{test_wasm_butterfly, test_wasm_oof_butterfly};

    wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_node_experimental);

    test_wasm_butterfly!(test_wasm_butterfly4, f32, WasmButterfly4, 4, 1e-5);
    test_wasm_butterfly!(test_wasm_butterfly4_f64, f64, WasmButterfly4, 4, 1e-7);
    test_wasm_oof_butterfly!(test_oof_wasm_butterfly4, f32, WasmButterfly4, 4, 1e-5);
    test_wasm_oof_butterfly!(test_oof_wasm_butterfly4_f64, f64, WasmButterfly4, 4, 1e-9);
}
