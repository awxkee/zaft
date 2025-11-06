/*
 * // Copyright (c) Radzivon Bartoshyk 11/2025. All rights reserved.
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
use num_traits::Euclid;
use std::ops::{Div, Rem};

#[derive(Copy, Clone)]
#[allow(unused)]
pub(crate) struct DividerU64 {
    magic: u64,
    more: u8,
    divisor: u64,
}

#[derive(Copy, Clone)]
#[allow(unused)]
pub(crate) struct DividerU32 {
    magic: u32,
    more: u8,
    divisor: u32,
}

#[derive(Copy, Clone)]
pub(crate) struct DividerU16 {
    magic: u16,
    more: u8,
    divisor: u16,
}

impl DividerU16 {
    pub(crate) fn new(divisor: u16) -> Self {
        assert_ne!(divisor, 0, "Divisor must not be zero");
        assert_ne!(divisor, 1, "Divisor must not be 1");

        let floor_log_2_d: u16 = 15 - divisor.leading_zeros() as u16;

        // Power of 2
        if (divisor & (divisor - 1)) == 0 {
            // We need to subtract 1 from the shift value in case of an unsigned
            // branchfree divider because there is a hardcoded right shift by 1
            // in its division algorithm. Because of this we also need to add back
            // 1 in its recovery algorithm.
            DividerU16 {
                magic: 0,
                more: (floor_log_2_d.wrapping_sub(1) as u8) & 0x1F,
                divisor,
            }
        } else {
            let num = (1u32 << floor_log_2_d) << 16;
            let (proposed_m1, rem1) = num.div_rem_euclid(&(divisor as u32));

            debug_assert!(rem1 > 0 && rem1 < divisor as u32);
            let mut proposed_m: u16 = proposed_m1 as u16;
            let rem: u16 = rem1 as u16;

            // This power works if e < 2**floor_log_2_d.
            // We have to use the general 33-bit algorithm.  We need to compute
            // (2**power) / d. However, we already have (2**(power-1))/d and
            // its remainder.  By doubling both, and then correcting the
            // remainder, we can compute the larger division.
            // don't care about overflow here - in fact, we expect it
            proposed_m = proposed_m.wrapping_add(proposed_m);
            let twice_rem = rem.wrapping_add(rem);
            if twice_rem >= divisor || twice_rem < rem {
                proposed_m = proposed_m.wrapping_add(1);
            }
            let more = ((floor_log_2_d | 0x40) as u8) & 0x1F;
            let magic = 1u16.wrapping_add(proposed_m);
            // result.more's shift should in general be ceil_log_2_d. But if we
            // used the smaller power, we subtract one from the shift because we're
            // using the smaller power. If we're using the larger power, we
            // subtract one from the shift because it's taken care of by the add
            // indicator. So floor_log_2_d happens to be correct in both cases.
            DividerU16 {
                more,
                magic,
                divisor,
            }
        }
    }

    #[inline]
    pub(crate) fn divisor(&self) -> u16 {
        self.divisor
    }
}

macro_rules! impl_div_rem {
    ($for_type: ident, $int_type: ident) => {
        impl $for_type {
            #[inline]
            pub(crate) fn div_rem(x: $int_type, divisor: Self) -> ($int_type, $int_type) {
                let q = x / divisor;
                let rem = x - q * divisor.divisor;
                (q, rem)
            }
        }
    };
}

impl_div_rem!(DividerU16, u16);
// impl_div_rem!(DividerU32, u32);
// impl_div_rem!(DividerU64, u64);

impl DividerUsize {
    #[inline]
    pub(crate) fn div_rem(x: usize, divisor: Self) -> (usize, usize) {
        let q = x / divisor;
        let rem = x - q * divisor.divisor();
        (q, rem)
    }
}

impl DividerU32 {
    #[allow(unused)]
    pub(crate) fn new(divisor: u32) -> Self {
        assert_ne!(divisor, 0, "Divisor must not be zero");
        assert_ne!(divisor, 1, "Divisor must not be 1");

        let floor_log_2_d: u32 = 31 - divisor.leading_zeros();

        // Power of 2
        if (divisor & (divisor - 1)) == 0 {
            // We need to subtract 1 from the shift value in case of an unsigned
            // branchfree divider because there is a hardcoded right shift by 1
            // in its division algorithm. Because of this we also need to add back
            // 1 in its recovery algorithm.
            DividerU32 {
                magic: 0,
                more: (floor_log_2_d.wrapping_sub(1) as u8) & 0x1F,
                divisor,
            }
        } else {
            let num = (1u64 << floor_log_2_d) << 32;
            let (proposed_m1, rem1) = num.div_rem_euclid(&(divisor as u64));

            debug_assert!(rem1 > 0 && rem1 < divisor as u64);
            let mut proposed_m: u32 = proposed_m1 as u32;
            let rem: u32 = rem1 as u32;

            // This power works if e < 2**floor_log_2_d.
            // We have to use the general 33-bit algorithm.  We need to compute
            // (2**power) / d. However, we already have (2**(power-1))/d and
            // its remainder.  By doubling both, and then correcting the
            // remainder, we can compute the larger division.
            // don't care about overflow here - in fact, we expect it
            proposed_m = proposed_m.wrapping_add(proposed_m);
            let twice_rem = rem.wrapping_add(rem);
            if twice_rem >= divisor || twice_rem < rem {
                proposed_m = proposed_m.wrapping_add(1);
            }
            let more = ((floor_log_2_d | 0x40) as u8) & 0x1F;
            let magic = 1u32.wrapping_add(proposed_m);
            // result.more's shift should in general be ceil_log_2_d. But if we
            // used the smaller power, we subtract one from the shift because we're
            // using the smaller power. If we're using the larger power, we
            // subtract one from the shift because it's taken care of by the add
            // indicator. So floor_log_2_d happens to be correct in both cases.
            DividerU32 {
                more,
                magic,
                divisor,
            }
        }
    }

    #[inline]
    #[allow(unused)]
    pub(crate) fn divisor(&self) -> u32 {
        self.divisor
    }
}

impl DividerU64 {
    #[allow(unused)]
    pub(crate) fn new(divisor: u64) -> Self {
        assert_ne!(divisor, 0, "Divisor must not be zero");
        assert_ne!(divisor, 1, "Divisor must not be 1");

        let floor_log_2_d: u64 = 63 - divisor.leading_zeros() as u64;

        // Power of 2
        if (divisor & (divisor - 1)) == 0 {
            // We need to subtract 1 from the shift value in case of an unsigned
            // branchfree divider because there is a hardcoded right shift by 1
            // in its division algorithm. Because of this we also need to add back
            // 1 in its recovery algorithm.
            DividerU64 {
                magic: 0,
                more: (floor_log_2_d.wrapping_sub(1) as u8) & 0x3F,
                divisor,
            }
        } else {
            let num = ((1u64 << floor_log_2_d) as u128) << 64;
            let (proposed_m1, rem1) = num.div_rem_euclid(&(divisor as u128));

            debug_assert!(rem1 > 0 && rem1 < divisor as u128);
            let mut proposed_m: u64 = proposed_m1 as u64;
            let rem: u64 = rem1 as u64;

            // This power works if e < 2**floor_log_2_d.
            // We have to use the general 33-bit algorithm.  We need to compute
            // (2**power) / d. However, we already have (2**(power-1))/d and
            // its remainder.  By doubling both, and then correcting the
            // remainder, we can compute the larger division.
            // don't care about overflow here - in fact, we expect it
            proposed_m = proposed_m.wrapping_add(proposed_m);
            let twice_rem = rem.wrapping_add(rem);
            if twice_rem >= divisor || twice_rem < rem {
                proposed_m = proposed_m.wrapping_add(1);
            }
            let more = ((floor_log_2_d | 0x40) as u8) & 0x3F;
            let magic = 1u64.wrapping_add(proposed_m);
            // result.more's shift should in general be ceil_log_2_d. But if we
            // used the smaller power, we subtract one from the shift because we're
            // using the smaller power. If we're using the larger power, we
            // subtract one from the shift because it's taken care of by the add
            // indicator. So floor_log_2_d happens to be correct in both cases.
            DividerU64 {
                more,
                magic,
                divisor,
            }
        }
    }

    #[inline]
    #[allow(unused)]
    pub(crate) fn divisor(&self) -> u64 {
        self.divisor
    }
}

impl Div<DividerU32> for u32 {
    type Output = u32;

    #[inline]
    fn div(self, denom: DividerU32) -> Self::Output {
        let q = ((self as u64 * denom.magic as u64) >> 32) as u32;
        let t = ((self.wrapping_sub(q)) >> 1).wrapping_add(q);
        t >> denom.more
    }
}

impl Div<DividerU64> for u64 {
    type Output = u64;

    #[inline]
    fn div(self, denom: DividerU64) -> Self::Output {
        let q = ((self as u128 * denom.magic as u128) >> 64) as u64;
        let t = ((self.wrapping_sub(q)) >> 1).wrapping_add(q);
        t >> denom.more
    }
}

impl Div<DividerU16> for u16 {
    type Output = u16;

    #[inline]
    fn div(self, denom: DividerU16) -> Self::Output {
        let q = ((self as u32 * denom.magic as u32) >> 16) as u16;
        let t = ((self.wrapping_sub(q)) >> 1).wrapping_add(q);
        t >> denom.more
    }
}

impl Rem<DividerU32> for u32 {
    type Output = u32;
    #[inline]
    fn rem(self, divider: DividerU32) -> Self {
        let q = self / divider;
        self - q * divider.divisor
    }
}

impl Rem<DividerU64> for u64 {
    type Output = u64;
    #[inline]
    fn rem(self, divider: DividerU64) -> Self {
        let q = self / divider;
        self - q * divider.divisor
    }
}

impl Rem<DividerU16> for u16 {
    type Output = u16;
    #[inline]
    fn rem(self, divider: DividerU16) -> Self {
        let q = self / divider;
        self - q * divider.divisor
    }
}

#[derive(Copy, Clone)]
pub(crate) enum DividerUsize {
    #[cfg(target_pointer_width = "32")]
    U32(DividerU32),
    #[cfg(target_pointer_width = "64")]
    U64(DividerU64),
}

impl DividerUsize {
    #[inline(always)]
    pub(crate) fn new(divisor: usize) -> Self {
        #[cfg(target_pointer_width = "32")]
        {
            Self::U32(DividerU32::new(divisor as u32))
        }

        #[cfg(target_pointer_width = "64")]
        {
            Self::U64(DividerU64::new(divisor as u64))
        }
    }

    #[inline(always)]
    pub fn divisor(&self) -> usize {
        match self {
            #[cfg(target_pointer_width = "32")]
            Self::U32(d) => d.divisor as usize,
            #[cfg(target_pointer_width = "64")]
            Self::U64(d) => d.divisor as usize,
        }
    }
}

impl Div<DividerUsize> for usize {
    type Output = usize;

    #[inline(always)]
    fn div(self, denom: DividerUsize) -> Self::Output {
        match denom {
            #[cfg(target_pointer_width = "32")]
            DividerUsize::U32(d) => (self as u32 / d) as usize,
            #[cfg(target_pointer_width = "64")]
            DividerUsize::U64(d) => (self as u64 / d) as usize,
        }
    }
}

impl Rem<DividerUsize> for usize {
    type Output = usize;

    #[inline(always)]
    fn rem(self, denom: DividerUsize) -> Self::Output {
        match denom {
            #[cfg(target_pointer_width = "32")]
            DividerUsize::U32(d) => (self as u32 % d) as usize,
            #[cfg(target_pointer_width = "64")]
            DividerUsize::U64(d) => (self as u64 % d) as usize,
        }
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
        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 32) as u32
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.state
        }
    }

    #[test]
    fn test_divider_u32_edge_cases() {
        let divisors = [
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
            u32::MAX / 2,
            u32::MAX - 1,
            u32::MAX,
        ];

        let values = [
            0u32,
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
            u32::MAX / 3,
            u32::MAX / 2,
            u32::MAX - 1,
            u32::MAX,
        ];

        for &d in &divisors {
            // Skip division by zero (we assert in constructor)
            if d == 0 {
                continue;
            }

            let divider = DividerU32::new(d);

            for &x in &values {
                let fast = x / divider;
                let exact = x / d;

                assert_eq!(
                    fast, exact,
                    "Mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            for &x in &values {
                let fast = x % divider;
                let exact = x % d;

                assert_eq!(
                    fast, exact,
                    "Mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let mut rng = Rng::new((now.as_millis() & 0xffff_ffff_ffff_ffff) as u64);
            // Fuzz 1000 random samples
            for _ in 0..1000 {
                let x = rng.next_u32();
                let fast = x / divider;
                let exact = x / d;
                assert_eq!(
                    fast, exact,
                    "Random mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            for _ in 0..1000 {
                let x = rng.next_u32();
                let fast = x % divider;
                let exact = x % d;
                assert_eq!(
                    fast, exact,
                    "Random mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }
        }
    }

    #[test]
    fn test_divider_u64_edge_cases() {
        let divisors = [
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
            u64::MAX / 2,
            u64::MAX - 1,
            u64::MAX,
        ];

        let values = [
            0u64,
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
            u64::MAX / 3,
            u64::MAX / 2,
            u64::MAX - 1,
            u64::MAX,
        ];

        for &d in &divisors {
            if d == 0 {
                continue;
            }

            let divider = DividerU64::new(d);

            for &x in &values {
                let fast = x / divider;
                let exact = x / d;

                assert_eq!(
                    fast, exact,
                    "Mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            for &x in &values {
                let fast = x % divider;
                let exact = x % d;

                assert_eq!(
                    fast, exact,
                    "Mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let mut rng = Rng::new((now.as_millis() & 0xffff_ffff_ffff_ffff) as u64);

            // Fuzz random samples
            for _ in 0..500 {
                let x = rng.next_u64();
                let fast = x / divider;
                let exact = x / d;
                assert_eq!(
                    fast, exact,
                    "Random mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            for _ in 0..500 {
                let x = rng.next_u64();
                let fast = x % divider;
                let exact = x % d;
                assert_eq!(
                    fast, exact,
                    "Random mismatch for x = {x}, divisor = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }
        }
    }

    #[test]
    fn test() {
        let divisor = DividerU32::new(3);
        assert_eq!(
            black_box(3) / black_box(divisor),
            black_box(3) / black_box(3)
        );
    }

    #[test]
    fn test_divider_u16_edge_cases() {
        let divisors = [
            2, 3, 4, 5, 7, 10, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 512,
            1023, 1024, 2048, 4096, 8191, 8192, 16383, 16384, 32767, 32768, 65535,
        ];

        let values = [
            0u16, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257,
            511, 512, 1023, 1024, 2047, 2048, 4095, 4096, 8191, 8192, 16383, 16384, 32767, 32768,
            65534, 65535,
        ];

        for &d in &divisors {
            if d == 0 {
                continue;
            }

            let divider = DividerU16::new(d);

            for &x in &values {
                let fast = x / divider;
                let exact = x / d;

                assert_eq!(
                    fast, exact,
                    "Mismatch in division: x = {x}, d = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            for &x in &values {
                let fast = x % divider;
                let exact = x % d;

                assert_eq!(
                    fast, exact,
                    "Mismatch in modulo: x = {x}, d = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            // Randomized fuzz testing
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let mut rng = Rng::new((now.as_millis() & 0xffff_ffff) as u64);

            for _ in 0..2000 {
                let x = (rng.next_u32() & 0xFFFF) as u16;
                let fast = x / divider;
                let exact = x / d;
                assert_eq!(
                    fast, exact,
                    "Random mismatch in division: x = {x}, d = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }

            for _ in 0..2000 {
                let x = (rng.next_u32() & 0xFFFF) as u16;
                let fast = x % divider;
                let exact = x % d;
                assert_eq!(
                    fast, exact,
                    "Random mismatch in modulo: x = {x}, d = {d}, magic = {}, shift = {}",
                    divider.magic, divider.more
                );
            }
        }
    }
}
