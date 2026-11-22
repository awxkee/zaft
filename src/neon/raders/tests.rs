use super::{NeonRadersIndicer, RadersIndicer};
use crate::FftSample;
use num_complex::Complex;
use num_traits::AsPrimitive;

fn check_output_gather<T: FftSample>(indicer: &dyn RadersIndicer<T>)
where
    f64: AsPrimitive<T>,
{
    // Exercise every SIMD tail length and unaligned source/destination slices.
    for len in 0..40 {
        let source: Vec<_> = (0..len + 2)
            .map(|i| Complex::new((i as f64 + 0.25).as_(), (i as f64 + 0.5).as_()))
            .collect();
        let scratch = &source[1..len + 1];
        let indices: Vec<_> = (0..len).map(|i| ((i + len / 3 + 1) % len) as u32).collect();
        let guard = Complex::new(4321.0.as_(), (-8765.0).as_());
        let mut output = vec![guard; len + 2];
        indicer.output_indices(&mut output[1..len + 1], scratch, &indices);
        assert_eq!(output[0], guard);
        assert_eq!(output[len + 1], guard);
        for (dst, &index) in output[1..len + 1].iter().zip(&indices) {
            assert_eq!(*dst, scratch[index as usize].conj(), "length {len}");
        }
    }
}

#[test]
fn output_gather_f32() {
    check_output_gather::<f32>(&NeonRadersIndicer);
}

#[test]
fn output_gather_f64() {
    check_output_gather::<f64>(&NeonRadersIndicer);
}

#[cfg(feature = "sve")]
#[test]
fn output_gather_sve_f32() {
    if std::arch::is_aarch64_feature_detected!("sve")
        && std::arch::is_aarch64_feature_detected!("sve2")
    {
        check_output_gather::<f32>(&crate::sve::SveLutGather);
    }
}
