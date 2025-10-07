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
    is_power_of_eleven, is_power_of_five, is_power_of_seven, is_power_of_six, is_power_of_thirteen,
    is_power_of_three,
};
use num_traits::{One, PrimInt, Zero};

/// Return the prime factors of `n` as a Vec with multiplicity.
/// For example: `prime_factors(360) -> [2,2,2,3,3,5]`.
/// Special cases:
///  - n == 0 -> returns empty vec (undefined factorization)
///  - n == 1 -> returns empty vec
pub(crate) fn prime_factors(mut n: u64) -> Vec<u64> {
    let mut res = Vec::new();
    if n < 2 {
        return res;
    }

    // factor out 2s
    while (n & 1) == 0 {
        res.push(2);
        n >>= 1;
    }

    // factor out 3s
    while n % 3 == 0 {
        res.push(3);
        n /= 3;
    }

    // trial divide by 6k - 1 and 6k + 1
    let mut p: u64 = 5;
    while (p as u128) * (p as u128) <= n as u128 {
        while n % p == 0 {
            res.push(p);
            n /= p;
        }
        let q = p + 2; // p = 6k-1, q = 6k+1
        while n % q == 0 {
            res.push(q);
            n /= q;
        }
        p += 6;
    }

    // if remaining n > 1 it's prime
    if n > 1 {
        res.push(n);
    }
    res
}

pub(crate) fn primitive_root(prime: u64) -> Option<u64> {
    let test_exponents: Vec<u64> = prime_factors(prime - 1)
        .iter()
        .map(|factor| (prime - 1) / factor)
        .collect();
    'next: for potential_root in 2..prime {
        // for each distinct factor, if potential_root^(p-1)/factor mod p is 1, reject it
        for exp in &test_exponents {
            if modular_exponent(potential_root, *exp, prime) == 1 {
                continue 'next;
            }
        }

        // if we reach this point, it means this root was not rejected, so return it
        return Some(potential_root);
    }
    None
}

/// computes base^exponent % modulo using the standard exponentiation by squaring algorithm
pub(crate) fn modular_exponent<T: PrimInt>(mut base: T, mut exponent: T, modulo: T) -> T {
    let one = T::one();

    let mut result = one;

    while exponent > Zero::zero() {
        if exponent & one == one {
            result = result * base % modulo;
        }
        exponent = exponent >> One::one();
        base = (base * base) % modulo;
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
    pub(crate) is_power_of_eleven: bool,
    pub(crate) is_power_of_thirteen: bool,
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
        let factorization = prime_factorization(n);
        PrimeFactors {
            n,
            is_power_of_two,
            is_power_of_five,
            is_power_of_six,
            is_power_of_three,
            is_power_of_seven,
            is_power_of_eleven,
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
}

pub(crate) fn split_factors_closest(factors: &[(u64, u32)]) -> (u64, u64) {
    // Step 1: expand the factors into a flat list of primes
    let mut primes = Vec::new();
    for &(p, exp) in factors {
        for _ in 0..exp {
            primes.push(p);
        }
    }

    let total: u64 = primes.iter().product();

    // Step 2: recursive helper to try all subset products
    fn dfs(
        primes: &[u64],
        index: usize,
        prod: u64,
        total: u64,
        best: &mut u64,
        best_prod: &mut u64,
    ) {
        if index == primes.len() {
            let other = total / prod;
            let diff = if prod > other {
                prod - other
            } else {
                other - prod
            };
            if diff < *best {
                *best = diff;
                *best_prod = prod;
            }
            return;
        }

        // include current prime in prod
        dfs(
            primes,
            index + 1,
            prod * primes[index],
            total,
            best,
            best_prod,
        );
        // exclude current prime from prod
        dfs(primes, index + 1, prod, total, best, best_prod);
    }

    let mut best_diff = u64::MAX;
    let mut best_prod = 1;

    dfs(&primes, 0, 1, total, &mut best_diff, &mut best_prod);

    let n1 = best_prod;
    let n2 = total / best_prod;
    (n1, n2)
}

pub(crate) fn try_greedy_pure_power_split(factors: &[(u64, u32)]) -> Option<(u64, u64)> {
    // Preferred bases (note: 4 is composite, but we allow it as "preferred")
    const PREF_BASES: [u64; 7] = [2, 3, 4, 5, 7, 11, 13];
    use std::collections::HashMap;

    // Build prime -> exponent counts from input
    let mut counts: HashMap<u64, u32> = HashMap::new();
    for &(p, exp) in factors {
        *counts.entry(p).or_insert(0) += exp;
    }

    // Compute total as u128 (safer for intermediate multiplications)
    let mut total_u128: u128 = 1;
    for (&p, &exp) in &counts {
        for _ in 0..exp {
            total_u128 = total_u128.saturating_mul(p as u128);
        }
    }
    if total_u128 == 0 {
        return None;
    } // guard, though not expected

    // Helper: factor small base into its prime factors -> map prime->exponent
    fn factor_base(base: u64) -> HashMap<u64, u32> {
        let mut n = base;
        let mut out = HashMap::new();
        let mut d = 2;
        while d * d <= n {
            while n % d == 0 {
                *out.entry(d).or_insert(0) += 1;
                n /= d;
            }
            d += 1;
        }
        if n > 1 {
            *out.entry(n).or_insert(0) += 1;
        }
        out
    }

    // Helper: compute base^k as u128 (returns None on overflow)
    fn pow_u128(base: u64, exp: u32) -> Option<u128> {
        let mut acc: u128 = 1;
        for _ in 0..exp {
            acc = acc.checked_mul(base as u128)?;
        }
        Some(acc)
    }

    // For each preferred base, compute maximum k such that base^k can be made from counts
    // (i.e., for each prime q dividing base with exponent e_base, we need e_base * k <= counts[q]).
    let mut best_value: u128 = 1;
    for &base in &PREF_BASES {
        let base_factor = factor_base(base);
        if base_factor.is_empty() {
            continue;
        }

        // Compute k_max for this base
        let mut k_max: u32 = u32::MAX;
        for (&q, &e_base) in &base_factor {
            let available = counts.get(&q).copied().unwrap_or(0);
            if available == 0 {
                k_max = 0;
                break;
            } else {
                let k_for_q = available / e_base;
                k_max = k_max.min(k_for_q);
            }
        }
        if k_max == 0 || k_max == u32::MAX {
            continue;
        }

        // We want the largest value base^k (k from k_max down to 1)
        // but computing base^k_max should be fine: pick that
        if let Some(val) = pow_u128(base, k_max) {
            // ensure val > 1 and val divides the total (it will by construction)
            if val > 1 && total_u128 % val == 0 {
                if val > best_value {
                    best_value = val;
                }
            }
        }
    }

    // If we found only 1 (no pure power >1), return None
    if best_value <= 1 {
        return None;
    }

    // Convert back to u64 if possible (if overflowed the original u64 product, bail)
    let total_u64 = match u64::try_from(total_u128) {
        Ok(t) => t,
        Err(_) => return None, // can't represent in u64, let caller handle
    };
    let best_u64 = match u64::try_from(best_value) {
        Ok(b) => b,
        Err(_) => return None,
    };

    // Final sanity: best divides total
    if total_u64 % best_u64 != 0 {
        return None;
    }

    Some((best_u64, total_u64 / best_u64))
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
    fn test_factors_splitting() {
        assert_eq!(
            split_factors_closest(&vec![(2, 4), (3, 1), (5, 2)]),
            (40, 30)
        );
        assert_eq!(
            try_greedy_pure_power_split(&vec![(2, 2), (3, 1), (13, 2)]),
            Some((169, 12))
        );
        assert_eq!(
            try_greedy_pure_power_split(&vec![(2, 4), (3, 1), (5, 2)]),
            Some((25, 48))
        );
        assert_eq!(
            split_factors_closest(&vec![(2, 2), (3, 1), (13, 2)]),
            (52, 39)
        );
    }

    #[test]
    fn test_composite() {
        let n = 2u64.pow(10) * 3u64.pow(6) * 7u64;
        assert_eq!(prime_factorization(n), vec![(2, 10), (3, 6), (7, 1)]);
    }
}
