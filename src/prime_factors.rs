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
use crate::util::{
    is_power_of_eleven, is_power_of_five, is_power_of_seven, is_power_of_six, is_power_of_ten,
    is_power_of_thirteen, is_power_of_three, is_power_of_twelve,
};

/// Uses deterministic Miller-Rabin primality testing and Pollard-Brent factorization,
/// both valid for the full `u64` range. Small factors are stripped first so common FFT
/// dimensions retain a very cheap fast path.
pub(crate) fn prime_factors(mut n: u64) -> Vec<u64> {
    static SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

    let mut factors = Vec::new();
    if n < 2 {
        return factors;
    }

    // This is faster than Miller-Rabin/Pollard for the overwhelmingly common small-radix cases.
    for prime in SMALL_PRIMES {
        while n.is_multiple_of(prime) {
            factors.push(prime);
            n /= prime;
        }
    }

    let mut pending = Vec::new();
    if n > 1 {
        pending.push(n);
    }

    while let Some(value) = pending.pop() {
        if is_prime_u64(value) {
            factors.push(value);
            continue;
        }

        let divisor = pollard_brent(value);
        pending.push(divisor);
        pending.push(value / divisor);
    }

    factors.sort_unstable();
    factors
}

#[inline]
fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

#[inline]
fn mul_mod_u64(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

#[inline]
fn add_mod_u64(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 + b as u128) % modulus as u128) as u64
}

#[inline]
fn rho_step(value: u64, constant: u64, modulus: u64) -> u64 {
    add_mod_u64(mul_mod_u64(value, value, modulus), constant, modulus)
}

/// Deterministic Miller-Rabin for all `u64` values.
fn is_prime_u64(n: u64) -> bool {
    static SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    // This seven-base set is deterministic over the complete unsigned 64-bit domain.
    static WITNESSES: [u64; 7] = [2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022];

    if n < 2 {
        return false;
    }
    for prime in SMALL_PRIMES {
        if n.is_multiple_of(prime) {
            return n == prime;
        }
    }

    let powers_of_two = (n - 1).trailing_zeros();
    let odd_part = (n - 1) >> powers_of_two;

    'witness: for witness in WITNESSES {
        let base = witness % n;
        if base == 0 {
            continue;
        }

        let mut value = modular_exponent(base, odd_part, n);
        if value == 1 || value == n - 1 {
            continue;
        }

        for _ in 1..powers_of_two {
            value = mul_mod_u64(value, value, n);
            if value == n - 1 {
                continue 'witness;
            }
        }
        return false;
    }

    true
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Pollard's rho with Brent cycle detection and batched GCDs.
///
/// The caller only supplies odd composite values which have no factor <= 37.
fn pollard_brent(n: u64) -> u64 {
    const GCD_BATCH: usize = 128;

    debug_assert!(n > 1 && !n.is_multiple_of(2) && !is_prime_u64(n));

    let mut state = n ^ 0x243f_6a88_85a3_08d3;
    loop {
        // SplitMix gives deterministic but decorrelated retries, avoiding a `rand` dependency.
        let mut y = 2 + splitmix64(&mut state) % (n - 3);
        let constant = 1 + splitmix64(&mut state) % (n - 1);
        let mut cycle_len = 1usize;
        let mut gcd = 1u64;
        let mut x = 0u64;
        let mut saved_y = 0u64;

        while gcd == 1 {
            x = y;
            for _ in 0..cycle_len {
                y = rho_step(y, constant, n);
            }

            let mut offset = 0usize;
            while offset < cycle_len && gcd == 1 {
                saved_y = y;
                let batch_len = (cycle_len - offset).min(GCD_BATCH);
                let mut product = 1u64;

                for _ in 0..batch_len {
                    y = rho_step(y, constant, n);
                    product = mul_mod_u64(product, x.abs_diff(y), n);
                }

                gcd = gcd_u64(product, n);
                offset += batch_len;
            }

            let Some(next_cycle_len) = cycle_len.checked_mul(2) else {
                gcd = n;
                break;
            };
            cycle_len = next_cycle_len;
        }

        if gcd == n {
            loop {
                saved_y = rho_step(saved_y, constant, n);
                gcd = gcd_u64(x.abs_diff(saved_y), n);
                if gcd != 1 {
                    break;
                }
            }
        }

        if gcd != n {
            return gcd;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn primitive_root(prime: u64) -> Option<u64> {
    let test_exponents: Vec<u64> = prime_factorization(prime - 1)
        .into_iter()
        .map(|(factor, _)| (prime - 1) / factor)
        .collect();
    'next: for potential_root in 2..prime {
        // For each distinct factor, if potential_root^((p - 1) / factor) mod p is 1,
        // reject it.
        for &exponent in &test_exponents {
            if modular_exponent(potential_root, exponent, prime) == 1 {
                continue 'next;
            }
        }

        return Some(potential_root);
    }
    None
}

fn modular_exponent(mut base: u64, mut exponent: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;

    while exponent != 0 {
        if exponent & 1 != 0 {
            result = mul_mod_u64(result, base, modulus);
        }
        exponent >>= 1;
        base = mul_mod_u64(base, base, modulus);
    }
    result
}

/// Return the prime factorization as (prime, exponent) pairs.
/// Example: `prime_factorization(360) -> [(2,3), (3,2), (5,1)]`.
pub(crate) fn prime_factorization(n: u64) -> Vec<(u64, u32)> {
    let factors = prime_factors(n);
    let mut out = Vec::new();
    let mut iter = factors.into_iter();
    if let Some(mut cur) = iter.next() {
        let mut cnt: u32 = 1;
        for f in iter {
            if f == cur {
                cnt += 1;
            } else {
                out.push((cur, cnt));
                cur = f;
                cnt = 1;
            }
        }
        out.push((cur, cnt));
    }
    out
}

#[derive(Clone, Ord, PartialOrd, Eq, PartialEq, Debug)]
pub(crate) struct PrimeFactors {
    pub(crate) n: u64,
    pub(crate) is_power_of_two: bool,
    pub(crate) is_power_of_three: bool,
    pub(crate) is_power_of_five: bool,
    pub(crate) is_power_of_six: bool,
    pub(crate) is_power_of_seven: bool,
    pub(crate) is_power_of_ten: bool,
    pub(crate) is_power_of_eleven: bool,
    pub(crate) is_power_of_thirteen: bool,
    pub(crate) is_power_of_twelve: bool,
    pub(crate) factorization: Vec<(u64, u32)>,
}

impl PrimeFactors {
    pub(crate) fn from_number(n: u64) -> PrimeFactors {
        let is_power_of_three = is_power_of_three(n);
        let is_power_of_two = n.is_power_of_two();
        let is_power_of_six = is_power_of_six(n);
        let is_power_of_five = is_power_of_five(n);
        let is_power_of_seven = is_power_of_seven(n);
        let is_power_of_eleven = is_power_of_eleven(n);
        let is_power_of_ten = is_power_of_ten(n);
        let factorization = prime_factorization(n);
        PrimeFactors {
            n,
            is_power_of_two,
            is_power_of_five,
            is_power_of_six,
            is_power_of_three,
            is_power_of_seven,
            is_power_of_eleven,
            is_power_of_ten,
            is_power_of_twelve: is_power_of_twelve(n),
            is_power_of_thirteen: is_power_of_thirteen(n),
            factorization,
        }
    }

    pub(crate) fn may_be_represented_in_mixed_radix(&self) -> bool {
        self.factorization.len() > 1 || self.factorization[0].1 != 1
    }

    pub(crate) fn is_prime(&self) -> bool {
        self.factorization.len() == 1
            && self.factorization[0].0 == self.n
            && self.factorization[0].1 == 1
    }

    pub(crate) fn is_power_of_two_and_three(&self) -> bool {
        if self.factorization.len() == 2 {
            let is_any_two = self.factorization.iter().any(|p| p.0 == 2);
            let is_any_three = self.factorization.iter().any(|p| p.0 == 3);
            return is_any_two && is_any_three;
        }
        false
    }

    pub(crate) fn has_power_of_two_and_three(&self) -> bool {
        let is_any_two = self.factorization.iter().any(|p| p.0 == 2);
        let is_any_three = self.factorization.iter().any(|p| p.0 == 3);
        is_any_two && is_any_three
    }

    pub(crate) fn is_power_of_two_and_five(&self) -> bool {
        if self.factorization.len() == 2 {
            let is_any_two = self.factorization.iter().any(|p| p.0 == 2);
            let is_any_five = self.factorization.iter().any(|p| p.0 == 5);
            return is_any_two && is_any_five;
        }
        false
    }

    pub(crate) fn is_power_of_three_and_five(&self) -> bool {
        if self.factorization.len() == 2 {
            let is_any_three = self.factorization.iter().any(|p| p.0 == 3);
            let is_any_five = self.factorization.iter().any(|p| p.0 == 5);
            return is_any_three && is_any_five;
        }
        false
    }

    pub(crate) fn is_power_of_two_and_seven(&self) -> bool {
        if self.factorization.len() == 2 {
            let is_any_two = self.factorization.iter().any(|p| p.0 == 2);
            let is_any_seven = self.factorization.iter().any(|p| p.0 == 7);
            return is_any_two && is_any_seven;
        }
        false
    }

    pub(crate) fn has_power_of_five_and_seven(&self) -> bool {
        let is_any_five = self.factorization.iter().any(|p| p.0 == 5);
        let is_any_seven = self.factorization.iter().any(|p| p.0 == 7);
        is_any_five && is_any_seven
    }

    pub(crate) fn is_power_of_five_and_seven(&self) -> bool {
        if self.factorization.len() == 2 {
            let is_any_five = self.factorization.iter().any(|p| p.0 == 5);
            let is_any_seven = self.factorization.iter().any(|p| p.0 == 7);
            return is_any_five && is_any_seven;
        }
        false
    }

    pub(crate) fn factor_of_7(&self) -> u32 {
        self.factorization
            .iter()
            .find(|p| p.0 == 7)
            .map(|x| x.1)
            .unwrap_or(0)
    }

    pub(crate) fn factor_of_5(&self) -> u32 {
        self.factorization
            .iter()
            .find(|p| p.0 == 5)
            .map(|x| x.1)
            .unwrap_or(0)
    }

    pub(crate) fn factor_of_2(&self) -> u32 {
        self.factorization
            .iter()
            .find(|p| p.0 == 2)
            .map(|x| x.1)
            .unwrap_or(0)
    }

    pub(crate) fn factor_of_3(&self) -> u32 {
        self.factorization
            .iter()
            .find(|p| p.0 == 3)
            .map(|x| x.1)
            .unwrap_or(0)
    }
}

pub(crate) fn split_factors_closest(factors: &[(u64, u32)]) -> (u64, u64) {
    let total = factors.iter().fold(1u64, |product, &(prime, exponent)| {
        let power = prime
            .checked_pow(exponent)
            .expect("prime factor power exceeds u64");
        product
            .checked_mul(power)
            .expect("prime factorization product exceeds u64")
    });

    // Enumerate each distinct divisor once. The old implementation expanded repeated
    // primes and visited 2^sum(exponents) subsets; this visits product(exponent + 1)
    // states instead. Restricting products to <= sqrt(total) lets the closest pair be
    // found by maximizing its smaller member.
    fn visit_divisors(
        factors: &[(u64, u32)],
        index: usize,
        product: u64,
        total: u64,
        best: &mut u64,
    ) {
        if product > total / product {
            return;
        }
        if index == factors.len() {
            *best = (*best).max(product);
            return;
        }

        let (prime, exponent) = factors[index];
        let mut next_product = product;
        for power in 0..=exponent {
            if next_product > total / next_product {
                break;
            }
            visit_divisors(factors, index + 1, next_product, total, best);

            if power != exponent {
                let Some(value) = next_product.checked_mul(prime) else {
                    break;
                };
                next_product = value;
            }
        }
    }

    let mut smaller = 1u64;
    visit_divisors(factors, 0, 1, total, &mut smaller);

    // Preserve the previous API's larger-first ordering.
    (total / smaller, smaller)
}

pub(crate) fn can_be_two_factors(factors: &[(u64, u32)]) -> Option<(u64, u64)> {
    // Allowed numbers
    static ALLOWED: [u64; 24] = [
        2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 23, 25, 27, 29, 31,
    ];

    // Compute the total number from prime factorization
    let total: u64 = factors.iter().fold(1u64, |acc, &(p, exp)| {
        acc.checked_mul(p.pow(exp)).unwrap_or(0)
    });
    if total == 0 {
        return None;
    }

    // Try all pairs of allowed numbers
    for &a in ALLOWED.iter() {
        if !total.is_multiple_of(a) {
            continue;
        }
        let b = total / a;
        if ALLOWED.contains(&b) {
            return Some((a, b));
        }
    }

    None
}

pub(crate) fn try_greedy_pure_power_split(factors: &[(u64, u32)]) -> Option<(u64, u64)> {
    // Preferred bases (note: 4 is composite, but we allow it as "preferred")
    static PREF_BASES: [u64; 8] = [2, 3, 4, 5, 7, 10, 11, 13];
    let number = factors.iter().map(|x| x.0.pow(x.1)).product::<u64>();

    // Recursive helper to find max power of `base` that divides `n`
    fn max_power(n: u64, base: u64) -> (u64, u64) {
        let mut power = 1;
        let mut rem = n;
        while rem.is_multiple_of(base) {
            rem /= base;
            power *= base;
        }
        (power, rem)
    }

    for &base in PREF_BASES.iter().rev() {
        let (p, a) = max_power(number, base);
        if p > 1 {
            return Some((p, a));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small() {
        assert_eq!(prime_factors(1), Vec::<u64>::new());
        assert_eq!(prime_factors(2), vec![2]);
        assert_eq!(prime_factors(3), vec![3]);
        assert_eq!(prime_factors(4), vec![2, 2]);
        assert_eq!(prime_factors(18), vec![2, 3, 3]);
        assert_eq!(prime_factorization(1296), vec![(2, 4), (3, 4)]);
        assert_eq!(prime_factorization(360), vec![(2, 3), (3, 2), (5, 1)]);
        assert_eq!(prime_factorization(20), vec![(2, 2), (5, 1)]);
        assert_eq!(prime_factorization(97), vec![(97, 1)]);
        assert_eq!(prime_factorization(36), vec![(2, 2), (3, 2)]);
        assert_eq!(prime_factorization(36 * 6), vec![(2, 3), (3, 3)]);
    }

    #[test]
    fn test_large_prime() {
        let p = 4_294_967_291u64; // this is prime
        assert_eq!(prime_factors(p), vec![p]);
        assert_eq!(
            prime_factors(18_446_744_073_709_551_557),
            vec![18_446_744_073_709_551_557]
        );
        assert_eq!(prime_factorization(p), vec![(p, 1)]);
        assert_eq!(prime_factorization(2028), vec![(2, 2), (3, 1), (13, 2)]);
        assert_eq!(prime_factorization(900), vec![(2, 2), (3, 2), (5, 2)]);
        assert_eq!(prime_factorization(121), vec![(11, 2)]);
        assert_eq!(prime_factorization(1312), vec![(2, 5), (41, 1)]);
        assert_eq!(prime_factorization(1201), vec![(1201, 1)]);
        assert_eq!(prime_factorization(1200), vec![(2, 4), (3, 1), (5, 2)]);
        assert_eq!(prime_factorization(1295), vec![(5, 1), (7, 1), (37, 1)]);
        assert_eq!(prime_factorization(1859), vec![(11, 1), (13, 2)]);
    }

    #[test]
    fn test_full_width_factorization() {
        let p = 4_294_967_279u64;
        let q = 4_294_967_291u64;
        assert_eq!(prime_factors(p * q), vec![p, q]);
        assert_eq!(
            prime_factors(u64::MAX),
            vec![3, 5, 17, 257, 641, 65_537, 6_700_417]
        );

        // A strong pseudoprime for several smaller witness sets.
        assert!(!is_prime_u64(341_550_071_728_321));
    }

    #[test]
    fn test_factors_splitting() {
        assert_eq!(
            split_factors_closest(&vec![(2, 4), (3, 1), (5, 2)]),
            (40, 30)
        );
        assert_eq!(
            try_greedy_pure_power_split(&vec![(2, 2), (5, 3)]), // 500
            Some((100, 5))
        );
        assert_eq!(
            try_greedy_pure_power_split(&vec![(2, 2), (3, 1), (13, 2)]),
            Some((169, 12))
        );
        assert_eq!(
            try_greedy_pure_power_split(&vec![(2, 4), (3, 1), (5, 2)]),
            Some((100, 12))
        );
        assert_eq!(
            split_factors_closest(&vec![(2, 2), (3, 1), (13, 2)]),
            (52, 39)
        );
        assert_eq!(
            split_factors_closest(&[(2, 63)]),
            (4_294_967_296, 2_147_483_648)
        );
    }

    #[test]
    fn test_composite() {
        let n = 2u64.pow(10) * 3u64.pow(6) * 7u64;
        assert_eq!(prime_factorization(n), vec![(2, 10), (3, 6), (7, 1)]);
    }
}
