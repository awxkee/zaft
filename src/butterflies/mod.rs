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
use num_complex::Complex;
use std::ops::Neg;

mod bf1;
mod bf10;
mod bf11;
mod bf12;
mod bf13;
mod bf15;
mod bf16;
mod bf17;
mod bf2;
mod bf3;
mod bf4;
mod bf5;
mod bf6;
mod bf7;
mod bf8;
mod bf9;
mod fast_bf8;
pub mod short_butterflies;

pub(crate) use bf1::Butterfly1;
#[allow(unused)]
pub(crate) use bf2::Butterfly2;
#[allow(unused)]
pub(crate) use bf3::Butterfly3;
#[allow(unused)]
pub(crate) use bf4::Butterfly4;
#[allow(unused)]
pub(crate) use bf5::Butterfly5;
#[allow(unused)]
pub(crate) use bf6::Butterfly6;
#[allow(unused)]
pub(crate) use bf7::Butterfly7;
#[allow(unused)]
pub(crate) use bf8::Butterfly8;
#[allow(unused)]
pub(crate) use bf9::Butterfly9;
#[allow(unused)]
pub(crate) use bf10::Butterfly10;
#[allow(unused)]
pub(crate) use bf11::Butterfly11;
#[allow(unused)]
pub(crate) use bf12::Butterfly12;
#[allow(unused)]
pub(crate) use bf13::Butterfly13;
#[allow(unused)]
pub(crate) use bf15::Butterfly15;
#[allow(unused)]
pub(crate) use bf16::Butterfly16;
#[allow(unused)]
pub(crate) use bf17::Butterfly17;

pub(crate) fn rotate_90<T: Copy + Neg<Output = T>>(
    value: Complex<T>,
    direction: FftDirection,
) -> Complex<T> {
    match direction {
        FftDirection::Forward => Complex {
            re: value.im,
            im: -value.re,
        },
        FftDirection::Inverse => Complex {
            re: -value.im,
            im: value.re,
        },
    }
}
