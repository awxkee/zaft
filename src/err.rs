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
use std::error::Error;
use std::fmt::Formatter;

#[derive(Clone, Debug)]
pub enum ZaftError {
    OutOfMemory(usize),
    InvalidInPlaceLength(usize, usize),
    InvalidPointerSize(u128),
    InvalidSizeMultiplier(usize, usize),
    ZeroSizedFft,
    ScratchBufferIsTooSmall(usize, usize),
}

impl Error for ZaftError {}

impl std::fmt::Display for ZaftError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ZaftError::OutOfMemory(length) => {
                f.write_fmt(format_args!("Cannot allocate {length} bytes to vector",))
            }
            ZaftError::InvalidInPlaceLength(s0, s1) => f.write_fmt(format_args!(
                "In-place length expected to be {s0}, but it was {s1}"
            )),
            ZaftError::InvalidPointerSize(s0) => {
                f.write_fmt(format_args!("Pointer size invalid, it's too big {s0}"))
            }
            ZaftError::InvalidSizeMultiplier(s0, s1) => f.write_fmt(format_args!(
                "Size {s0} is assumed to be multiplier of {s1} to execute many DFT, but it wasn't"
            )),
            ZaftError::ZeroSizedFft => f.write_str("Cannot execute FFT on zero-sized buffers"),
            ZaftError::ScratchBufferIsTooSmall(current, required) => f.write_fmt(format_args!(
                "Scratch buffer size must be at least {required} but it is {current}"
            )),
        }
    }
}

macro_rules! try_vec {
    () => {
        Vec::new()
    };
    ($elem:expr; $n:expr) => {{
        let mut v = Vec::new();
        v.try_reserve_exact($n)
            .map_err(|_| crate::err::ZaftError::OutOfMemory($n))?;
        v.resize($n, $elem);
        v
    }};
}

pub(crate) use try_vec;
