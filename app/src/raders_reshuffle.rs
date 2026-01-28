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

pub(crate) fn raders_reshuffle(input: &[usize], output: &[usize], r2c: bool) -> (String, String) {
    let mut input_str = String::new();
    let mut out_str = String::new();
    if r2c {
        for i in 0..input.len() {
            input_str = input_str
                + &format!(
                    "scratch[{i}] = Complex::new(buffer[{}], T::zero());\n",
                    input[i]
                );
        }
    } else {
        for i in 0..input.len() {
            input_str = input_str + &format!("scratch[{i}] = buffer[{}];\n", input[i]);
        }
    }
    if r2c {
        let complex_length = output.len() / 2 + 1;
        for i in 0..output.len() {
            if output[i] + 1 >= complex_length {
                continue;
            }
            out_str = out_str + &format!("complex[{}] = scratch[{i}].conj();\n", output[i] + 1);
        }
    } else {
        for i in 0..output.len() {
            out_str = out_str + &format!("buffer[{}] = scratch[{i}].conj();\n", output[i]);
        }
    }
    (input_str, out_str)
}
