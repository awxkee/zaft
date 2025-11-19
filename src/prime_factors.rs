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
    is_power_of_thirteen, is_power_of_three,
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
    pub(crate) is_power_of_ten: bool,
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

    pub(crate) fn factor_of_11(&self) -> u32 {
        self.factorization
            .iter()
            .find(|p| p.0 == 11)
            .map(|x| x.1)
            .unwrap_or(0)
    }

    pub(crate) fn factor_of_13(&self) -> u32 {
        self.factorization
            .iter()
            .find(|p| p.0 == 13)
            .map(|x| x.1)
            .unwrap_or(0)
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
            let diff = prod.abs_diff(other);
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

pub(crate) fn can_be_two_factors(factors: &[(u64, u32)]) -> Option<(u64, u64)> {
    // Allowed numbers
    const ALLOWED: [u64; 24] = [
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
        if total % a != 0 {
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
    const PREF_BASES: [u64; 8] = [2, 3, 4, 5, 7, 10, 11, 13];
    let number = factors.iter().map(|x| x.0.pow(x.1)).product::<u64>();

    // Recursive helper to find max power of `base` that divides `n`
    fn max_power(n: u64, base: u64) -> (u64, u64) {
        let mut power = 1;
        let mut rem = n;
        while rem % base == 0 {
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
    }

    #[test]
    fn test_composite() {
        let n = 2u64.pow(10) * 3u64.pow(6) * 7u64;
        assert_eq!(prime_factorization(n), vec![(2, 10), (3, 6), (7, 1)]);
    }
}
