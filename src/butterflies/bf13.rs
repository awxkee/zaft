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
use crate::mla::fmla;
use crate::util::compute_twiddle;
use crate::{
    CompositeFftExecutor, FftDirection, FftExecutor, FftExecutorOutOfPlace, FftSample,
    R2CFftExecutor, ZaftError,
};
use num_complex::Complex;
use num_traits::AsPrimitive;
use std::sync::Arc;

#[allow(unused)]
pub(crate) struct Butterfly13<T> {
    direction: FftDirection,
    twiddle1: Complex<T>,
    twiddle2: Complex<T>,
    twiddle3: Complex<T>,
    twiddle4: Complex<T>,
    twiddle5: Complex<T>,
    twiddle6: Complex<T>,
}

#[allow(unused)]
impl<T: FftSample> Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    pub fn new(fft_direction: FftDirection) -> Self {
        Butterfly13 {
            direction: fft_direction,
            twiddle1: compute_twiddle(1, 13, fft_direction),
            twiddle2: compute_twiddle(2, 13, fft_direction),
            twiddle3: compute_twiddle(3, 13, fft_direction),
            twiddle4: compute_twiddle(4, 13, fft_direction),
            twiddle5: compute_twiddle(5, 13, fft_direction),
            twiddle6: compute_twiddle(6, 13, fft_direction),
        }
    }
}

impl<T: FftSample> FftExecutor<T> for Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, in_place: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !in_place.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(
                in_place.len(),
                self.length(),
            ));
        }

        for chunk in in_place.chunks_exact_mut(13) {
            let u0 = chunk[0];
            let u1 = chunk[1];
            let u2 = chunk[2];
            let u3 = chunk[3];

            let u4 = chunk[4];
            let u5 = chunk[5];
            let u6 = chunk[6];
            let u7 = chunk[7];

            let u8 = chunk[8];
            let u9 = chunk[9];
            let u10 = chunk[10];
            let u11 = chunk[11];
            let u12 = chunk[12];

            let x112p = u1 + u12;
            let x112n = u1 - u12;
            let x211p = u2 + u11;
            let x211n = u2 - u11;
            let x310p = u3 + u10;
            let x310n = u3 - u10;
            let x49p = u4 + u9;
            let x49n = u4 - u9;
            let x58p = u5 + u8;
            let x58n = u5 - u8;
            let x67p = u6 + u7;
            let x67n = u6 - u7;
            let y0 = u0 + x112p + x211p + x310p + x49p + x58p + x67p;
            chunk[0] = y0;
            let b112re_a = fmla(self.twiddle1.re, x112p.re, u0.re)
                + fmla(self.twiddle2.re, x211p.re, self.twiddle3.re * x310p.re)
                + fmla(self.twiddle4.re, x49p.re, self.twiddle5.re * x58p.re)
                + self.twiddle6.re * x67p.re;
            let b112re_b = fmla(
                self.twiddle1.im,
                x112n.im,
                fmla(
                    self.twiddle2.im,
                    x211n.im,
                    fmla(self.twiddle3.im, x310n.im, self.twiddle4.im * x49n.im)
                        + fmla(self.twiddle5.im, x58n.im, self.twiddle6.im * x67n.im),
                ),
            );
            let b211re_a = fmla(
                self.twiddle2.re,
                x112p.re,
                fmla(self.twiddle4.re, x211p.re, u0.re)
                    + fmla(self.twiddle6.re, x310p.re, self.twiddle5.re * x49p.re)
                    + fmla(self.twiddle3.re, x58p.re, self.twiddle1.re * x67p.re),
            );
            let b211re_b = fmla(
                self.twiddle2.im,
                x112n.im,
                fmla(
                    self.twiddle6.im,
                    x310n.im,
                    fmla(-self.twiddle5.im, x49n.im, self.twiddle4.im * x211n.im),
                ) + fmla(-self.twiddle3.im, x58n.im, -self.twiddle1.im * x67n.im),
            );
            let b310re_a = fmla(self.twiddle3.re, x112p.re, u0.re)
                + fmla(
                    self.twiddle4.re,
                    x310p.re,
                    fmla(self.twiddle1.re, x49p.re, self.twiddle6.re * x211p.re),
                )
                + fmla(self.twiddle2.re, x58p.re, self.twiddle5.re * x67p.re);
            let b310re_b = fmla(
                self.twiddle3.im,
                x112n.im,
                fmla(
                    self.twiddle6.im,
                    x211n.im,
                    fmla(-self.twiddle4.im, x310n.im, -self.twiddle1.im * x49n.im)
                        + fmla(self.twiddle2.im, x58n.im, self.twiddle5.im * x67n.im),
                ),
            );
            let b49re_a = fmla(
                self.twiddle4.re,
                x112p.re,
                fmla(self.twiddle5.re, x211p.re, u0.re)
                    + fmla(self.twiddle1.re, x310p.re, self.twiddle3.re * x49p.re)
                    + fmla(self.twiddle6.re, x58p.re, self.twiddle2.re * x67p.re),
            );
            let b49re_b = fmla(
                self.twiddle4.im,
                x112n.im,
                fmla(-self.twiddle5.im, x211n.im, -self.twiddle1.im * x310n.im)
                    + fmla(self.twiddle3.im, x49n.im, -self.twiddle6.im * x58n.im)
                    + -self.twiddle2.im * x67n.im,
            );
            let b58re_a = fmla(self.twiddle5.re, x112p.re, u0.re)
                + fmla(self.twiddle3.re, x211p.re, self.twiddle2.re * x310p.re)
                + fmla(self.twiddle6.re, x49p.re, self.twiddle1.re * x58p.re)
                + self.twiddle4.re * x67p.re;
            let b58re_b = fmla(
                self.twiddle5.im,
                x112n.im,
                fmla(
                    -self.twiddle3.im,
                    x211n.im,
                    self.twiddle2.im * x310n.im
                        + fmla(
                            -self.twiddle6.im,
                            x49n.im,
                            fmla(-self.twiddle1.im, x58n.im, self.twiddle4.im * x67n.im),
                        ),
                ),
            );
            let b67re_a = fmla(
                self.twiddle6.re,
                x112p.re,
                u0.re
                    + fmla(self.twiddle1.re, x211p.re, self.twiddle5.re * x310p.re)
                    + self.twiddle2.re * x49p.re
                    + fmla(self.twiddle4.re, x58p.re, self.twiddle3.re * x67p.re),
            );
            let b67re_b = fmla(
                self.twiddle6.im,
                x112n.im,
                fmla(
                    -self.twiddle1.im,
                    x211n.im,
                    self.twiddle5.im * x310n.im
                        + fmla(
                            -self.twiddle2.im,
                            x49n.im,
                            fmla(self.twiddle4.im, x58n.im, -self.twiddle3.im * x67n.im),
                        ),
                ),
            );

            let b112im_a = fmla(self.twiddle1.re, x112p.im, u0.im)
                + fmla(self.twiddle2.re, x211p.im, self.twiddle3.re * x310p.im)
                + fmla(
                    self.twiddle4.re,
                    x49p.im,
                    fmla(self.twiddle5.re, x58p.im, self.twiddle6.re * x67p.im),
                );
            let b112im_b = fmla(
                self.twiddle1.im,
                x112n.re,
                fmla(
                    self.twiddle2.im,
                    x211n.re,
                    fmla(self.twiddle3.im, x310n.re, self.twiddle4.im * x49n.re)
                        + fmla(self.twiddle5.im, x58n.re, self.twiddle6.im * x67n.re),
                ),
            );
            let b211im_a = fmla(
                self.twiddle2.re,
                x112p.im,
                fmla(self.twiddle4.re, x211p.im, u0.im)
                    + fmla(self.twiddle6.re, x310p.im, self.twiddle5.re * x49p.im)
                    + fmla(self.twiddle3.re, x58p.im, self.twiddle1.re * x67p.im),
            );
            let b211im_b = fmla(
                self.twiddle2.im,
                x112n.re,
                fmla(self.twiddle4.im, x211n.re, self.twiddle6.im * x310n.re)
                    + fmla(
                        -self.twiddle5.im,
                        x49n.re,
                        fmla(-self.twiddle3.im, x58n.re, -self.twiddle1.im * x67n.re),
                    ),
            );
            let b310im_a = fmla(
                self.twiddle3.re,
                x112p.im,
                fmla(
                    self.twiddle6.re,
                    x211p.im,
                    fmla(
                        self.twiddle4.re,
                        x310p.im,
                        fmla(self.twiddle1.re, x49p.im, u0.im),
                    ) + fmla(self.twiddle2.re, x58p.im, self.twiddle5.re * x67p.im),
                ),
            );
            let b310im_b = fmla(
                self.twiddle3.im,
                x112n.re,
                fmla(self.twiddle6.im, x211n.re, -self.twiddle4.im * x310n.re)
                    + fmla(-self.twiddle1.im, x49n.re, self.twiddle2.im * x58n.re)
                    + self.twiddle5.im * x67n.re,
            );
            let b49im_a = fmla(
                self.twiddle4.re,
                x112p.im,
                fmla(self.twiddle5.re, x211p.im, self.twiddle1.re * x310p.im)
                    + fmla(self.twiddle3.re, x49p.im, u0.im)
                    + fmla(self.twiddle6.re, x58p.im, self.twiddle2.re * x67p.im),
            );
            let b49im_b = fmla(
                self.twiddle4.im,
                x112n.re,
                fmla(-self.twiddle5.im, x211n.re, -self.twiddle1.im * x310n.re)
                    + self.twiddle3.im * x49n.re
                    + fmla(-self.twiddle6.im, x58n.re, -self.twiddle2.im * x67n.re),
            );
            let b58im_a = fmla(
                self.twiddle5.re,
                x112p.im,
                u0.im
                    + fmla(self.twiddle3.re, x211p.im, self.twiddle2.re * x310p.im)
                    + self.twiddle6.re * x49p.im
                    + fmla(self.twiddle1.re, x58p.im, self.twiddle4.re * x67p.im),
            );
            let b58im_b = fmla(
                self.twiddle5.im,
                x112n.re,
                fmla(
                    -self.twiddle3.im,
                    x211n.re,
                    fmla(self.twiddle2.im, x310n.re, -self.twiddle6.im * x49n.re)
                        + fmla(-self.twiddle1.im, x58n.re, self.twiddle4.im * x67n.re),
                ),
            );
            let b67im_a = fmla(
                self.twiddle6.re,
                x112p.im,
                u0.im
                    + fmla(
                        self.twiddle2.re,
                        x49p.im,
                        fmla(self.twiddle1.re, x211p.im, self.twiddle5.re * x310p.im),
                    )
                    + fmla(self.twiddle4.re, x58p.im, self.twiddle3.re * x67p.im),
            );
            let b67im_b = fmla(
                self.twiddle6.im,
                x112n.re,
                fmla(
                    -self.twiddle1.im,
                    x211n.re,
                    fmla(
                        self.twiddle5.im,
                        x310n.re,
                        fmla(
                            -self.twiddle2.im,
                            x49n.re,
                            self.twiddle4.im * x58n.re + -self.twiddle3.im * x67n.re,
                        ),
                    ),
                ),
            );

            chunk[1] = Complex {
                re: b112re_a - b112re_b,
                im: b112im_a + b112im_b,
            };
            chunk[2] = Complex {
                re: b211re_a - b211re_b,
                im: b211im_a + b211im_b,
            };
            chunk[3] = Complex {
                re: b310re_a - b310re_b,
                im: b310im_a + b310im_b,
            };
            chunk[4] = Complex {
                re: b49re_a - b49re_b,
                im: b49im_a + b49im_b,
            };
            chunk[5] = Complex {
                re: b58re_a - b58re_b,
                im: b58im_a + b58im_b,
            };
            chunk[6] = Complex {
                re: b67re_a - b67re_b,
                im: b67im_a + b67im_b,
            };
            chunk[7] = Complex {
                re: b67re_a + b67re_b,
                im: b67im_a - b67im_b,
            };
            chunk[8] = Complex {
                re: b58re_a + b58re_b,
                im: b58im_a - b58im_b,
            };
            chunk[9] = Complex {
                re: b49re_a + b49re_b,
                im: b49im_a - b49im_b,
            };
            chunk[10] = Complex {
                re: b310re_a + b310re_b,
                im: b310im_a - b310im_b,
            };
            chunk[11] = Complex {
                re: b211re_a + b211re_b,
                im: b211im_a - b211im_b,
            };
            chunk[12] = Complex {
                re: b112re_a + b112re_b,
                im: b112im_a - b112im_b,
            };
        }
        Ok(())
    }

    fn direction(&self) -> FftDirection {
        self.direction
    }

    #[inline]
    fn length(&self) -> usize {
        13
    }
}

impl<T: FftSample> FftExecutorOutOfPlace<T> for Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn execute_out_of_place(
        &self,
        src: &[Complex<T>],
        dst: &mut [Complex<T>],
    ) -> Result<(), ZaftError> {
        if !src.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(src.len(), self.length()));
        }
        if !dst.len().is_multiple_of(self.length()) {
            return Err(ZaftError::InvalidSizeMultiplier(dst.len(), self.length()));
        }

        for (dst, src) in dst.chunks_exact_mut(13).zip(src.chunks_exact(13)) {
            let u0 = src[0];
            let u1 = src[1];
            let u2 = src[2];
            let u3 = src[3];

            let u4 = src[4];
            let u5 = src[5];
            let u6 = src[6];
            let u7 = src[7];

            let u8 = src[8];
            let u9 = src[9];
            let u10 = src[10];
            let u11 = src[11];
            let u12 = src[12];

            let x112p = u1 + u12;
            let x112n = u1 - u12;
            let x211p = u2 + u11;
            let x211n = u2 - u11;
            let x310p = u3 + u10;
            let x310n = u3 - u10;
            let x49p = u4 + u9;
            let x49n = u4 - u9;
            let x58p = u5 + u8;
            let x58n = u5 - u8;
            let x67p = u6 + u7;
            let x67n = u6 - u7;
            let y0 = u0 + x112p + x211p + x310p + x49p + x58p + x67p;
            dst[0] = y0;
            let b112re_a = fmla(self.twiddle1.re, x112p.re, u0.re)
                + fmla(self.twiddle2.re, x211p.re, self.twiddle3.re * x310p.re)
                + fmla(self.twiddle4.re, x49p.re, self.twiddle5.re * x58p.re)
                + self.twiddle6.re * x67p.re;
            let b112re_b = fmla(
                self.twiddle1.im,
                x112n.im,
                fmla(
                    self.twiddle2.im,
                    x211n.im,
                    fmla(self.twiddle3.im, x310n.im, self.twiddle4.im * x49n.im)
                        + fmla(self.twiddle5.im, x58n.im, self.twiddle6.im * x67n.im),
                ),
            );
            let b211re_a = fmla(
                self.twiddle2.re,
                x112p.re,
                fmla(self.twiddle4.re, x211p.re, u0.re)
                    + fmla(self.twiddle6.re, x310p.re, self.twiddle5.re * x49p.re)
                    + fmla(self.twiddle3.re, x58p.re, self.twiddle1.re * x67p.re),
            );
            let b211re_b = fmla(
                self.twiddle2.im,
                x112n.im,
                fmla(
                    self.twiddle6.im,
                    x310n.im,
                    fmla(-self.twiddle5.im, x49n.im, self.twiddle4.im * x211n.im),
                ) + fmla(-self.twiddle3.im, x58n.im, -self.twiddle1.im * x67n.im),
            );
            let b310re_a = fmla(self.twiddle3.re, x112p.re, u0.re)
                + fmla(
                    self.twiddle4.re,
                    x310p.re,
                    fmla(self.twiddle1.re, x49p.re, self.twiddle6.re * x211p.re),
                )
                + fmla(self.twiddle2.re, x58p.re, self.twiddle5.re * x67p.re);
            let b310re_b = fmla(
                self.twiddle3.im,
                x112n.im,
                fmla(
                    self.twiddle6.im,
                    x211n.im,
                    fmla(-self.twiddle4.im, x310n.im, -self.twiddle1.im * x49n.im)
                        + fmla(self.twiddle2.im, x58n.im, self.twiddle5.im * x67n.im),
                ),
            );
            let b49re_a = fmla(
                self.twiddle4.re,
                x112p.re,
                fmla(self.twiddle5.re, x211p.re, u0.re)
                    + fmla(self.twiddle1.re, x310p.re, self.twiddle3.re * x49p.re)
                    + fmla(self.twiddle6.re, x58p.re, self.twiddle2.re * x67p.re),
            );
            let b49re_b = fmla(
                self.twiddle4.im,
                x112n.im,
                fmla(-self.twiddle5.im, x211n.im, -self.twiddle1.im * x310n.im)
                    + fmla(self.twiddle3.im, x49n.im, -self.twiddle6.im * x58n.im)
                    + -self.twiddle2.im * x67n.im,
            );
            let b58re_a = fmla(self.twiddle5.re, x112p.re, u0.re)
                + fmla(self.twiddle3.re, x211p.re, self.twiddle2.re * x310p.re)
                + fmla(self.twiddle6.re, x49p.re, self.twiddle1.re * x58p.re)
                + self.twiddle4.re * x67p.re;
            let b58re_b = fmla(
                self.twiddle5.im,
                x112n.im,
                fmla(
                    -self.twiddle3.im,
                    x211n.im,
                    self.twiddle2.im * x310n.im
                        + fmla(
                            -self.twiddle6.im,
                            x49n.im,
                            fmla(-self.twiddle1.im, x58n.im, self.twiddle4.im * x67n.im),
                        ),
                ),
            );
            let b67re_a = fmla(
                self.twiddle6.re,
                x112p.re,
                u0.re
                    + fmla(self.twiddle1.re, x211p.re, self.twiddle5.re * x310p.re)
                    + self.twiddle2.re * x49p.re
                    + fmla(self.twiddle4.re, x58p.re, self.twiddle3.re * x67p.re),
            );
            let b67re_b = fmla(
                self.twiddle6.im,
                x112n.im,
                fmla(
                    -self.twiddle1.im,
                    x211n.im,
                    self.twiddle5.im * x310n.im
                        + fmla(
                            -self.twiddle2.im,
                            x49n.im,
                            fmla(self.twiddle4.im, x58n.im, -self.twiddle3.im * x67n.im),
                        ),
                ),
            );

            let b112im_a = fmla(self.twiddle1.re, x112p.im, u0.im)
                + fmla(self.twiddle2.re, x211p.im, self.twiddle3.re * x310p.im)
                + fmla(
                    self.twiddle4.re,
                    x49p.im,
                    fmla(self.twiddle5.re, x58p.im, self.twiddle6.re * x67p.im),
                );
            let b112im_b = fmla(
                self.twiddle1.im,
                x112n.re,
                fmla(
                    self.twiddle2.im,
                    x211n.re,
                    fmla(self.twiddle3.im, x310n.re, self.twiddle4.im * x49n.re)
                        + fmla(self.twiddle5.im, x58n.re, self.twiddle6.im * x67n.re),
                ),
            );
            let b211im_a = fmla(
                self.twiddle2.re,
                x112p.im,
                fmla(self.twiddle4.re, x211p.im, u0.im)
                    + fmla(self.twiddle6.re, x310p.im, self.twiddle5.re * x49p.im)
                    + fmla(self.twiddle3.re, x58p.im, self.twiddle1.re * x67p.im),
            );
            let b211im_b = fmla(
                self.twiddle2.im,
                x112n.re,
                fmla(self.twiddle4.im, x211n.re, self.twiddle6.im * x310n.re)
                    + fmla(
                        -self.twiddle5.im,
                        x49n.re,
                        fmla(-self.twiddle3.im, x58n.re, -self.twiddle1.im * x67n.re),
                    ),
            );
            let b310im_a = fmla(
                self.twiddle3.re,
                x112p.im,
                fmla(
                    self.twiddle6.re,
                    x211p.im,
                    fmla(
                        self.twiddle4.re,
                        x310p.im,
                        fmla(self.twiddle1.re, x49p.im, u0.im),
                    ) + fmla(self.twiddle2.re, x58p.im, self.twiddle5.re * x67p.im),
                ),
            );
            let b310im_b = fmla(
                self.twiddle3.im,
                x112n.re,
                fmla(self.twiddle6.im, x211n.re, -self.twiddle4.im * x310n.re)
                    + fmla(-self.twiddle1.im, x49n.re, self.twiddle2.im * x58n.re)
                    + self.twiddle5.im * x67n.re,
            );
            let b49im_a = fmla(
                self.twiddle4.re,
                x112p.im,
                fmla(self.twiddle5.re, x211p.im, self.twiddle1.re * x310p.im)
                    + fmla(self.twiddle3.re, x49p.im, u0.im)
                    + fmla(self.twiddle6.re, x58p.im, self.twiddle2.re * x67p.im),
            );
            let b49im_b = fmla(
                self.twiddle4.im,
                x112n.re,
                fmla(-self.twiddle5.im, x211n.re, -self.twiddle1.im * x310n.re)
                    + self.twiddle3.im * x49n.re
                    + fmla(-self.twiddle6.im, x58n.re, -self.twiddle2.im * x67n.re),
            );
            let b58im_a = fmla(
                self.twiddle5.re,
                x112p.im,
                u0.im
                    + fmla(self.twiddle3.re, x211p.im, self.twiddle2.re * x310p.im)
                    + self.twiddle6.re * x49p.im
                    + fmla(self.twiddle1.re, x58p.im, self.twiddle4.re * x67p.im),
            );
            let b58im_b = fmla(
                self.twiddle5.im,
                x112n.re,
                fmla(
                    -self.twiddle3.im,
                    x211n.re,
                    fmla(self.twiddle2.im, x310n.re, -self.twiddle6.im * x49n.re)
                        + fmla(-self.twiddle1.im, x58n.re, self.twiddle4.im * x67n.re),
                ),
            );
            let b67im_a = fmla(
                self.twiddle6.re,
                x112p.im,
                u0.im
                    + fmla(
                        self.twiddle2.re,
                        x49p.im,
                        fmla(self.twiddle1.re, x211p.im, self.twiddle5.re * x310p.im),
                    )
                    + fmla(self.twiddle4.re, x58p.im, self.twiddle3.re * x67p.im),
            );
            let b67im_b = fmla(
                self.twiddle6.im,
                x112n.re,
                fmla(
                    -self.twiddle1.im,
                    x211n.re,
                    fmla(
                        self.twiddle5.im,
                        x310n.re,
                        fmla(
                            -self.twiddle2.im,
                            x49n.re,
                            self.twiddle4.im * x58n.re + -self.twiddle3.im * x67n.re,
                        ),
                    ),
                ),
            );

            dst[1] = Complex {
                re: b112re_a - b112re_b,
                im: b112im_a + b112im_b,
            };
            dst[2] = Complex {
                re: b211re_a - b211re_b,
                im: b211im_a + b211im_b,
            };
            dst[3] = Complex {
                re: b310re_a - b310re_b,
                im: b310im_a + b310im_b,
            };
            dst[4] = Complex {
                re: b49re_a - b49re_b,
                im: b49im_a + b49im_b,
            };
            dst[5] = Complex {
                re: b58re_a - b58re_b,
                im: b58im_a + b58im_b,
            };
            dst[6] = Complex {
                re: b67re_a - b67re_b,
                im: b67im_a + b67im_b,
            };
            dst[7] = Complex {
                re: b67re_a + b67re_b,
                im: b67im_a - b67im_b,
            };
            dst[8] = Complex {
                re: b58re_a + b58re_b,
                im: b58im_a - b58im_b,
            };
            dst[9] = Complex {
                re: b49re_a + b49re_b,
                im: b49im_a - b49im_b,
            };
            dst[10] = Complex {
                re: b310re_a + b310re_b,
                im: b310im_a - b310im_b,
            };
            dst[11] = Complex {
                re: b211re_a + b211re_b,
                im: b211im_a - b211im_b,
            };
            dst[12] = Complex {
                re: b112re_a + b112re_b,
                im: b112im_a - b112im_b,
            };
        }
        Ok(())
    }
}

impl<T: FftSample> CompositeFftExecutor<T> for Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn into_fft_executor(self: Arc<Self>) -> Arc<dyn FftExecutor<T> + Send + Sync> {
        self
    }
}

impl<T: FftSample> R2CFftExecutor<T> for Butterfly13<T>
where
    f64: AsPrimitive<T>,
{
    fn execute(&self, input: &[T], output: &mut [Complex<T>]) -> Result<(), ZaftError> {
        if !input.len().is_multiple_of(13) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 13));
        }
        if !output.len().is_multiple_of(7) {
            return Err(ZaftError::InvalidSizeMultiplier(input.len(), 7));
        }

        for (input, complex) in input.chunks_exact(13).zip(output.chunks_exact_mut(7)) {
            let u0 = input[0];
            let u1 = input[1];
            let u2 = input[2];
            let u3 = input[3];

            let u4 = input[4];
            let u5 = input[5];
            let u6 = input[6];
            let u7 = input[7];

            let u8 = input[8];
            let u9 = input[9];
            let u10 = input[10];
            let u11 = input[11];
            let u12 = input[12];

            let x112p = u1 + u12;
            let x112n = u1 - u12;
            let x211p = u2 + u11;
            let x211n = u2 - u11;
            let x310p = u3 + u10;
            let x310n = u3 - u10;
            let x49p = u4 + u9;
            let x49n = u4 - u9;
            let x58p = u5 + u8;
            let x58n = u5 - u8;
            let x67p = u6 + u7;
            let x67n = u6 - u7;
            let y0 = u0 + x112p + x211p + x310p + x49p + x58p + x67p;
            complex[0] = Complex::new(y0, T::zero());
            let b112re_a = fmla(self.twiddle1.re, x112p, u0)
                + fmla(self.twiddle2.re, x211p, self.twiddle3.re * x310p)
                + fmla(self.twiddle4.re, x49p, self.twiddle5.re * x58p)
                + self.twiddle6.re * x67p;
            let b211re_a = fmla(
                self.twiddle2.re,
                x112p,
                fmla(self.twiddle4.re, x211p, u0)
                    + fmla(self.twiddle6.re, x310p, self.twiddle5.re * x49p)
                    + fmla(self.twiddle3.re, x58p, self.twiddle1.re * x67p),
            );
            let b310re_a = fmla(self.twiddle3.re, x112p, u0)
                + fmla(
                    self.twiddle4.re,
                    x310p,
                    fmla(self.twiddle1.re, x49p, self.twiddle6.re * x211p),
                )
                + fmla(self.twiddle2.re, x58p, self.twiddle5.re * x67p);
            let b49re_a = fmla(
                self.twiddle4.re,
                x112p,
                fmla(self.twiddle5.re, x211p, u0)
                    + fmla(self.twiddle1.re, x310p, self.twiddle3.re * x49p)
                    + fmla(self.twiddle6.re, x58p, self.twiddle2.re * x67p),
            );
            let b58re_a = fmla(self.twiddle5.re, x112p, u0)
                + fmla(self.twiddle3.re, x211p, self.twiddle2.re * x310p)
                + fmla(self.twiddle6.re, x49p, self.twiddle1.re * x58p)
                + self.twiddle4.re * x67p;
            let b67re_a = fmla(
                self.twiddle6.re,
                x112p,
                u0 + fmla(self.twiddle1.re, x211p, self.twiddle5.re * x310p)
                    + self.twiddle2.re * x49p
                    + fmla(self.twiddle4.re, x58p, self.twiddle3.re * x67p),
            );
            let b112im_b = fmla(
                self.twiddle1.im,
                x112n,
                fmla(
                    self.twiddle2.im,
                    x211n,
                    fmla(self.twiddle3.im, x310n, self.twiddle4.im * x49n)
                        + fmla(self.twiddle5.im, x58n, self.twiddle6.im * x67n),
                ),
            );
            let b211im_b = fmla(
                self.twiddle2.im,
                x112n,
                fmla(self.twiddle4.im, x211n, self.twiddle6.im * x310n)
                    + fmla(
                        -self.twiddle5.im,
                        x49n,
                        fmla(-self.twiddle3.im, x58n, -self.twiddle1.im * x67n),
                    ),
            );
            let b310im_b = fmla(
                self.twiddle3.im,
                x112n,
                fmla(self.twiddle6.im, x211n, -self.twiddle4.im * x310n)
                    + fmla(-self.twiddle1.im, x49n, self.twiddle2.im * x58n)
                    + self.twiddle5.im * x67n,
            );
            let b49im_b = fmla(
                self.twiddle4.im,
                x112n,
                fmla(-self.twiddle5.im, x211n, -self.twiddle1.im * x310n)
                    + self.twiddle3.im * x49n
                    + fmla(-self.twiddle6.im, x58n, -self.twiddle2.im * x67n),
            );
            let b58im_b = fmla(
                self.twiddle5.im,
                x112n,
                fmla(
                    -self.twiddle3.im,
                    x211n,
                    fmla(self.twiddle2.im, x310n, -self.twiddle6.im * x49n)
                        + fmla(-self.twiddle1.im, x58n, self.twiddle4.im * x67n),
                ),
            );
            let b67im_b = fmla(
                self.twiddle6.im,
                x112n,
                fmla(
                    -self.twiddle1.im,
                    x211n,
                    fmla(
                        self.twiddle5.im,
                        x310n,
                        fmla(
                            -self.twiddle2.im,
                            x49n,
                            self.twiddle4.im * x58n + -self.twiddle3.im * x67n,
                        ),
                    ),
                ),
            );

            complex[1] = Complex {
                re: b112re_a,
                im: b112im_b,
            };
            complex[2] = Complex {
                re: b211re_a,
                im: b211im_b,
            };
            complex[3] = Complex {
                re: b310re_a,
                im: b310im_b,
            };
            complex[4] = Complex {
                re: b49re_a,
                im: b49im_b,
            };
            complex[5] = Complex {
                re: b58re_a,
                im: b58im_b,
            };
            complex[6] = Complex {
                re: b67re_a,
                im: b67im_b,
            };
        }
        Ok(())
    }

    fn real_length(&self) -> usize {
        13
    }

    fn complex_length(&self) -> usize {
        7
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterflies::{test_butterfly, test_oof_butterfly};
    use crate::r2c::test_r2c_butterfly;

    test_r2c_butterfly!(test_r2c_butterfly13, f32, Butterfly13, 13, 1e-5);
    test_butterfly!(test_butterfly13, f32, Butterfly13, 13, 1e-5);
    test_oof_butterfly!(test_oof_butterfly13, f32, Butterfly13, 13, 1e-5);
}
