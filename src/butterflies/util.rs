/*
 * // Copyright (c) Radzivon Bartoshyk 2/2026. All rights reserved.
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

macro_rules! boring_scalar_butterfly {
    ($bf_name: ident, $size: expr) => {
        impl<T: FftSample> FftExecutor<T> for $bf_name<T>
        where
            f64: AsPrimitive<T>,
        {
            fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
                if !in_place.len().is_multiple_of($size) {
                    return Err(ZaftError::InvalidSizeMultiplier(in_place.len(), $size));
                }

                for chunk in in_place.chunks_exact_mut($size) {
                    use crate::store::InPlaceStore;
                    self.run(&mut InPlaceStore::new(chunk));
                }
                Ok(())
            }

            fn execute_with_scratch(
                &self,
                in_place: &mut [Complex<T>],
                _: &mut [Complex<T>],
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
                src: &[Complex<T>],
                dst: &mut [Complex<T>],
            ) -> Result<(), ZaftError> {
                FftExecutor::execute_out_of_place_with_scratch(self, src, dst, &mut [])
            }

            fn execute_out_of_place_with_scratch(
                &self,
                src: &[Complex<T>],
                dst: &mut [Complex<T>],
                _: &mut [Complex<T>],
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
                src: &mut [Complex<T>],
                dst: &mut [Complex<T>],
                _: &mut [Complex<T>],
            ) -> Result<(), ZaftError> {
                self.execute_out_of_place_with_scratch(src, dst, &mut [])
            }

            fn direction(&self) -> FftDirection {
                self.direction
            }

            #[inline]
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

pub(crate) use boring_scalar_butterfly;
