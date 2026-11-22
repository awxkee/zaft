use super::*;
use crate::{FftDirection, Zaft};

macro_rules! test_reconstruction {
    ($name:ident, $ty:ty, $nan:expr, $(($plan:ty, $radix:expr)),+ $(,)?) => {
        #[test]
        fn $name() {
            let values = [
                0.0, -0.0, <$ty>::INFINITY, <$ty>::NEG_INFINITY,
                <$ty>::from_bits($nan), <$ty>::MIN_POSITIVE,
                <$ty>::from_bits(1), -42.75,
            ];
            $(for width in (1..=65).chain([127, 128, 129]) {
                let plan = <$plan>::new(Zaft::strategy(width, FftDirection::Forward).unwrap()).unwrap();
                let input: Vec<_> = (0..plan.second_stage_len)
                    .map(|i| Complex::new(values[i % values.len()], values[(i / values.len()) % values.len()]))
                    .collect();
                let guard = Complex::new(12345.0, -6789.0);
                let mut output = vec![guard; plan.complex_length() + 2];
                let mut reference = vec![guard; plan.complex_length()];

                // Scatter retained bins to their original frequencies, folding
                // bins above N/2 by Hermitian symmetry. Row zero's upper half
                // duplicates its lower half and was never used by the old pass.
                for (row, values) in input.chunks_exact(width).enumerate() {
                    for (column, &value) in values.iter().enumerate() {
                        let frequency = column * $radix + row;
                        if frequency < reference.len() {
                            reference[frequency] = value;
                        } else if row != 0 {
                            reference[plan.real_length() - frequency] = value.conj();
                        }
                    }
                }
                let len = output.len();
                unsafe { plan.transpose_spectrum(&input, &mut output[1..len - 1]); }
                for (index, (actual, expected)) in output[1..len - 1].iter().zip(&reference).enumerate() {
                    assert_eq!(actual.re.to_bits(), expected.re.to_bits(), "real {}, width={width}, bin={index}", stringify!($plan));
                    assert_eq!(actual.im.to_bits(), expected.im.to_bits(), "imag {}, width={width}, bin={index}", stringify!($plan));
                }
                assert_eq!((output[0], output[len - 1]), (guard, guard));
            })+
        }
    };
}

macro_rules! test_batches {
    ($name:ident, $ty:ty, $tolerance:expr, $feature:tt, $($plan:ty),+ $(,)?) => {
        #[test]
        fn $name() {
            if !std::arch::is_aarch64_feature_detected!($feature) {
                return;
            }
            $(for width in (1..=65).chain([105, 315, 509, 729]) {
                let plan = <$plan>::new(Zaft::strategy(width, FftDirection::Forward).unwrap()).unwrap();
                let n = plan.real_length();
                let m = plan.complex_length();
                let guard = Complex::new(12345.0, -6789.0);
                let input: Vec<$ty> = (0..n * 3 + 2)
                    .map(|i| ((i * 17 + 3) % 97) as $ty / 97.0 - 0.5)
                    .collect();
                let original = input.clone();
                let input_slice = &input[1..input.len() - 1];
                let mut output = vec![guard; m * 3 + 2];
                let need = plan.complex_scratch_length();
                let mut scratch = vec![guard; need + 2];
                let mut reference: Vec<_> = input_slice.iter().map(|&v| Complex::new(v, 0.0)).collect();
                Zaft::strategy(n, FftDirection::Forward).unwrap().execute(&mut reference).unwrap();
                plan.execute_with_scratch(input_slice, &mut output[1..m * 3 + 1], &mut scratch[1..need + 1]).unwrap();
                for (expected, actual) in reference.chunks_exact(n).zip(output[1..m * 3 + 1].chunks_exact(m)) {
                    let scale = expected.iter().map(|z| z.norm()).fold(1.0, <$ty>::max);
                    for (index, (a, b)) in expected.iter().zip(actual).enumerate() {
                        assert!((*a - *b).norm() <= $tolerance * scale,
                            "{}, width={width}, bin={index}: {a:?} != {b:?}", stringify!($plan));
                    }
                }
                assert_eq!(input, original);
                assert_eq!((output[0], output[m * 3 + 1], scratch[0], scratch[need + 1]), (guard, guard, guard, guard));
                assert!(plan.execute_with_scratch(input_slice, &mut output[1..m * 3 + 1], &mut scratch[1..need]).is_err());
                assert!(plan.execute_with_scratch(&input_slice[..n * 3 - 1], &mut output[1..m * 3 + 1], &mut scratch[1..need + 1]).is_err());
                assert!(plan.execute_with_scratch(input_slice, &mut output[1..m * 2 + 1], &mut scratch[1..need + 1]).is_err());
            })+
        }
    };
}

test_reconstruction!(
    fused_reconstruction_f32,
    f32,
    0x7fc0_1234,
    (NeonR2CMixedRadix3f, 3),
    (NeonR2CMixedRadix5f, 5),
    (NeonR2CMixedRadix7f, 7),
    (NeonR2CMixedRadix9f, 9),
    (NeonR2CMixedRadix11f, 11),
    (NeonR2CMixedRadix13f, 13),
);
test_reconstruction!(
    fused_reconstruction_f64,
    f64,
    0x7ff8_0000_0000_1234,
    (NeonR2CMixedRadix3d, 3),
    (NeonR2CMixedRadix5d, 5),
    (NeonR2CMixedRadix7d, 7),
    (NeonR2CMixedRadix9d, 9),
    (NeonR2CMixedRadix11d, 11),
    (NeonR2CMixedRadix13d, 13),
);
test_batches!(
    fused_batches_f32,
    f32,
    2e-6,
    "neon",
    NeonR2CMixedRadix3f,
    NeonR2CMixedRadix5f,
    NeonR2CMixedRadix7f,
    NeonR2CMixedRadix9f,
    NeonR2CMixedRadix11f,
    NeonR2CMixedRadix13f,
);
test_batches!(
    fused_batches_f64,
    f64,
    1e-12,
    "neon",
    NeonR2CMixedRadix3d,
    NeonR2CMixedRadix5d,
    NeonR2CMixedRadix7d,
    NeonR2CMixedRadix9d,
    NeonR2CMixedRadix11d,
    NeonR2CMixedRadix13d,
);
#[cfg(feature = "fcma")]
test_batches!(
    fused_batches_fcma_f32,
    f32,
    2e-6,
    "fcma",
    NeonFcmaR2CMixedRadix3f,
    NeonFcmaR2CMixedRadix5f,
    NeonFcmaR2CMixedRadix7f,
    NeonFcmaR2CMixedRadix9f,
    NeonFcmaR2CMixedRadix11f,
    NeonFcmaR2CMixedRadix13f,
);
#[cfg(feature = "fcma")]
test_batches!(
    fused_batches_fcma_f64,
    f64,
    1e-12,
    "fcma",
    NeonFcmaR2CMixedRadix3d,
    NeonFcmaR2CMixedRadix5d,
    NeonFcmaR2CMixedRadix7d,
    NeonFcmaR2CMixedRadix9d,
    NeonFcmaR2CMixedRadix11d,
    NeonFcmaR2CMixedRadix13d,
);
