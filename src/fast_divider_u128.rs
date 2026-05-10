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

use std::ops::{Div, Rem};

/// A 256-bit unsigned integer represented as (hi: u128, lo: u128).
/// `value = hi * 2^128 + lo`
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct U256 {
    hi: u128,
    lo: u128,
}

impl U256 {
    #[inline]
    const fn from_u128(v: u128) -> Self {
        U256 { hi: 0, lo: v }
    }

    /// Shift left by one bit.
    #[inline]
    fn shl1(self) -> Self {
        U256 {
            hi: (self.hi << 1) | (self.lo >> 127),
            lo: self.lo << 1,
        }
    }

    /// Check whether `self >= rhs` (unsigned).
    #[inline]
    fn ge(self, rhs: Self) -> bool {
        self.hi > rhs.hi || (self.hi == rhs.hi && self.lo >= rhs.lo)
    }

    /// Saturating subtraction: returns `self - rhs` (assumes `self >= rhs`).
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let (lo, borrow) = self.lo.overflowing_sub(rhs.lo);
        let hi = self.hi - rhs.hi - borrow as u128;
        U256 { hi, lo }
    }

    // Shift right by one bit.
    // #[inline]
    // fn shr1(self) -> Self {
    //     U256 {
    //         lo: (self.lo >> 1) | (self.hi << 127),
    //         hi: self.hi >> 1,
    //     }
    // }
}

/// Compute `(quotient, remainder)` for `numerator / denominator` in 256-bit arithmetic.
///
/// This is a classical binary long-division, sufficient for the single call needed
/// during `DividerU128` construction (not a hot path).
fn u256_div_rem(numerator: U256, denominator: u128) -> (u128, u128) {
    // We only need the low 128 bits of the quotient (the algorithm guarantees it fits).
    let denom = U256::from_u128(denominator);

    // Find the highest bit of the denominator within the 256-bit numerator context.
    // We iterate 256 times for a general divisor.  Because `denom.hi == 0` always,
    // we only need 128 iterations, but 256 is safe and simple.
    let mut quotient: u128 = 0;
    let mut remainder = U256::from_u128(0);

    // Process bits from MSB of numerator downward.
    for shift in (0u32..256).rev() {
        // Extract bit `shift` of numerator.
        let bit = if shift >= 128 {
            (numerator.hi >> (shift - 128)) & 1
        } else {
            (numerator.lo >> shift) & 1
        };

        remainder = remainder.shl1();
        remainder.lo |= bit;

        if remainder.ge(denom) {
            remainder = remainder.sub(denom);
            if shift < 128 {
                quotient |= 1u128 << shift;
            }
            // If shift >= 128 the quotient bit is in the upper half, which we
            // assert is zero (the algorithm guarantees the true quotient fits in u128).
        }
    }

    (quotient, remainder.lo)
}

/// Branchfree magic-number fast divider for `u128`.
///
/// Follows the same "libdivide" style used by `DividerU64` / `DividerU32` above,
/// extended to 128 bits.  The division algorithm is:
///
/// ```text
/// q  = mulhi_128(x, magic) >> 0  (i.e. the upper 128 bits of x * magic)
/// t  = ((x - q) >> 1) + q
/// result = t >> shift
/// ```
///
/// `magic` and `shift` (`more`) are pre-computed in `new()` via a single
/// 256-bit division.
#[derive(Copy, Clone, Debug)]
pub(crate) struct DividerU128 {
    magic: u128,
    more: u8, // shift amount, 0..127
    divisor: u128,
}

#[inline]
pub(crate) fn mulhi_u128(a: u128, b: u128) -> u128 {
    let a_lo = a as u64 as u128;
    let a_hi = (a >> 64) as u64 as u128;
    let b_lo = b as u64 as u128;
    let b_hi = (b >> 64) as u64 as u128;

    let lo_lo = a_lo * b_lo;
    let lo_hi = a_lo * b_hi;
    let hi_lo = a_hi * b_lo;
    let hi_hi = a_hi * b_hi;

    let carry = (lo_lo >> 64)
        .wrapping_add(lo_hi & 0xffff_ffff_ffff_ffff)
        .wrapping_add(hi_lo & 0xffff_ffff_ffff_ffff);
    let mid = (lo_hi >> 64)
        .wrapping_add(hi_lo >> 64)
        .wrapping_add(carry >> 64);

    hi_hi.wrapping_add(mid)
}

impl DividerU128 {
    pub(crate) fn new(divisor: u128) -> Self {
        assert_ne!(divisor, 0, "Divisor must not be zero");
        assert_ne!(divisor, 1, "Divisor must not be 1");

        let floor_log_2_d: u32 = 127 - divisor.leading_zeros();

        // Power-of-two fast path: shift only.
        if (divisor & (divisor - 1)) == 0 {
            return DividerU128 {
                magic: 0,
                more: (floor_log_2_d.wrapping_sub(1) as u8) & 0x7F,
                divisor,
            };
        }

        // Build the 256-bit numerator  2^(floor_log_2_d) << 128 as a U256.
        let num = {
            // `floor_log_2_d` is in 0..127 for non-power-of-two 128-bit divisors.
            // Shift a single 1-bit into the correct position across the two halves.
            if floor_log_2_d < 128 {
                U256 {
                    hi: 1u128 << floor_log_2_d,
                    lo: 0,
                }
            } else {
                // floor_log_2_d can be at most 127, so this branch is unreachable,
                // but we keep it for clarity.
                unreachable!("floor_log_2_d >= 128 for a 128-bit divisor");
            }
        };

        let (proposed_m1, rem1) = u256_div_rem(num, divisor);

        debug_assert!(rem1 > 0 && rem1 < divisor);

        let mut proposed_m = proposed_m1;
        let rem: u128 = rem1;

        // Double proposed_m; correct if twice_rem overflows or exceeds divisor.
        proposed_m = proposed_m.wrapping_add(proposed_m);
        let twice_rem = rem.wrapping_add(rem);
        if twice_rem >= divisor || twice_rem < rem {
            proposed_m = proposed_m.wrapping_add(1);
        }

        let more = floor_log_2_d as u8; // branchfree: shift is simply floor_log_2_d (no flag bits needed)
        let magic = 1u128.wrapping_add(proposed_m);

        DividerU128 {
            magic,
            more,
            divisor,
        }
    }
}

impl Div<DividerU128> for u128 {
    type Output = u128;

    #[inline]
    fn div(self, denom: DividerU128) -> Self::Output {
        let q = mulhi_u128(self, denom.magic);
        let t = ((self.wrapping_sub(q)) >> 1).wrapping_add(q);
        t >> denom.more
    }
}

impl Rem<DividerU128> for u128 {
    type Output = u128;

    #[inline]
    fn rem(self, divider: DividerU128) -> Self {
        let q = self / divider;
        self - q * divider.divisor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hint::black_box;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Rng {
        state: u64,
    }

    impl Rng {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }
        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }
        fn next_u128(&mut self) -> u128 {
            let lo = self.next_u64() as u128;
            let hi = self.next_u64() as u128;
            (hi << 64) | lo
        }
    }

    #[test]
    fn test_divider_u128_edge_cases() {
        let divisors: &[u128] = &[
            2,
            3,
            5,
            7,
            10,
            16,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            1_000,
            10_000,
            65_535,
            100_000,
            1_000_000,
            u32::MAX as u128,
            u64::MAX as u128,
            u128::MAX / 3,
            u128::MAX / 2,
            u128::MAX - 1,
            u128::MAX,
        ];

        let values: &[u128] = &[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            15,
            16,
            31,
            32,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            1000,
            10_000,
            1_000_000,
            u64::MAX as u128,
            u128::MAX / 3,
            u128::MAX / 2,
            u128::MAX - 1,
            u128::MAX,
        ];

        for &d in divisors {
            if d <= 1 {
                continue;
            }

            let divider = DividerU128::new(d);

            for &x in values {
                let fast = x / divider;
                let exact = x / d;
                assert_eq!(
                    fast, exact,
                    "Div mismatch: x={x}, d={d}, magic={}, shift={}",
                    divider.magic, divider.more
                );
            }

            for &x in values {
                let fast = x % divider;
                let exact = x % d;
                assert_eq!(
                    fast, exact,
                    "Rem mismatch: x={x}, d={d}, magic={}, shift={}",
                    divider.magic, divider.more
                );
            }

            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let mut rng = Rng::new((now.as_millis() & 0xffff_ffff_ffff_ffff) as u64);

            for _ in 0..500 {
                let x = rng.next_u128();
                let fast = x / divider;
                let exact = x / d;
                assert_eq!(
                    fast, exact,
                    "Random div mismatch: x={x}, d={d}, magic={}, shift={}",
                    divider.magic, divider.more
                );
            }

            for _ in 0..500 {
                let x = rng.next_u128();
                let fast = x % divider;
                let exact = x % d;
                assert_eq!(
                    fast, exact,
                    "Random rem mismatch: x={x}, d={d}, magic={}, shift={}",
                    divider.magic, divider.more
                );
            }
        }
    }

    #[test]
    fn test_basic() {
        let divisor = DividerU128::new(3);
        assert_eq!(black_box(9u128) / black_box(divisor), 3);
        assert_eq!(black_box(10u128) % black_box(divisor), 1);
    }

    #[test]
    fn test_power_of_two() {
        let divisor = DividerU128::new(16);
        assert_eq!(black_box(128u128) / black_box(divisor), 8);
        assert_eq!(black_box(130u128) % black_box(divisor), 2);
    }

    #[test]
    fn test_large_values() {
        let d: u128 = u64::MAX as u128 + 7;
        let divider = DividerU128::new(d);
        let x = u128::MAX;
        assert_eq!(x / divider, x / d);
        assert_eq!(x % divider, x % d);
    }
}
