/*
 * // Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

use crate::ZaftError;

fn baseline_len(min_len: usize) -> Result<usize, ZaftError> {
    if min_len == 0 {
        return Err(ZaftError::ZeroSizedFft);
    }
    let pow2 = min_len
        .checked_next_power_of_two()
        .ok_or(ZaftError::Overflow)?;
    let factor3 = pow2 / 4 * 3;
    Ok(if factor3 >= min_len { factor3 } else { pow2 })
}

// For each odd part, only its smallest suitable power-of-two multiple is needed.
// Keep at least four-way divisibility and never exceed the old inner FFT length.
fn visit_candidates(
    min_len: usize,
    max_len: usize,
    mut visit: impl FnMut(usize, u32, u32, u32, u32),
) {
    let max_odd = max_len / 4;
    let (mut factor7, mut d) = (1usize, 0);
    while factor7 <= max_odd {
        let (mut factor5, mut c) = (factor7, 0);
        loop {
            let (mut odd, mut b) = (factor5, 0);
            loop {
                let pow2 = min_len.div_ceil(odd).max(4).next_power_of_two();
                if let Some(len) = odd.checked_mul(pow2).filter(|&len| len <= max_len) {
                    visit(len, pow2.trailing_zeros(), b, c, d);
                }
                if odd > max_odd / 3 {
                    break;
                }
                odd *= 3;
                b += 1;
            }
            if factor5 > max_odd / 5 {
                break;
            }
            factor5 *= 5;
            c += 1;
        }
        if factor7 > max_odd / 7 {
            break;
        }
        factor7 *= 7;
        d += 1;
    }
}

fn estimated_cost(len: usize, a: u32, b: u32, c: u32, d: u32) -> u128 {
    // Relative radix work plus linear convolution passes.
    // Long power-of-three components with a short power-of-two component tend
    // to require less favorable mixed-radix decompositions.
    len as u128 * (20 * a + 30 * b + 50 * c + 60 * d + 20 + 20 * b.saturating_sub(a)) as u128
}

pub(crate) fn choose_bluestein_inner_len(min_len: usize) -> Result<usize, ZaftError> {
    let baseline = baseline_len(min_len)?;
    // Small plans depend too much on individual butterflies for this radix model.
    if baseline <= 8192 {
        return Ok(baseline);
    }
    let baseline_cost = estimated_cost(
        baseline,
        baseline.trailing_zeros(),
        u32::from(!baseline.is_power_of_two()),
        0,
        0,
    );
    let mut best = (baseline_cost, baseline);
    visit_candidates(min_len, baseline, |len, a, b, c, d| {
        // Radix 7 is competitive in larger combinations containing radix 5.
        // Small or repeated radix-7 stages need a more conservative estimate.
        if d > 1 || (d == 1 && (len < 16384 || c == 0)) {
            return;
        }
        best = best.min((estimated_cost(len, a, b, c, d), len));
    });
    Ok(if best.0 * 100 < baseline_cost * 85 {
        best.1
    } else {
        baseline
    })
}
