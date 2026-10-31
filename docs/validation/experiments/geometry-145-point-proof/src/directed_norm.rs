//! Fast conservative upper norms; unusual exponents retain the MP fallback.
use astro_float::{BigFloat as B, RoundingMode as R, Sign, WORD_BIT_SIZE};

fn prefix(high: u64, next: u64, bits: usize) -> Option<u64> {
    match bits {
        64 => Some(high),
        32 if high <= u32::MAX as u64 && next <= u32::MAX as u64 => Some((high << 32) | next),
        _ => None,
    }
}
fn magnitude_up(value: &B) -> Option<f64> {
    let (words, _, _, exponent, _) = value.as_raw_parts()?;
    if value.is_zero() {
        return Some(0.);
    }
    // Keep every square, sum and root normal and far from overflow. Non-normal
    // AstroFloat mantissas are handled by the original multiprecision path.
    if !(-500..=500).contains(&exponent) {
        return None;
    }
    let high = *words.last()? as u64;
    let next = words
        .len()
        .checked_sub(2)
        .and_then(|i| words.get(i))
        .copied()
        .unwrap_or(0) as u64;
    let top = prefix(high, next, WORD_BIT_SIZE)?;
    if top >> 63 != 1 {
        return None;
    }
    // Stored magnitude = mantissa * 2^(exponent - storage_bits). Keeping its
    // leading53bits truncates downward; the successor encloses every tail bit.
    let fraction = (top >> 11) & ((1u64 << 52) - 1);
    let truncated = f64::from_bits((((exponent + 1022) as u64) << 52) | fraction);
    Some(truncated.next_up())
}
fn sqrt_up(value: f64) -> f64 {
    value.sqrt().next_up()
}
pub(super) fn upper(re: &B, im: &B, p: usize) -> Option<B> {
    if re.is_nan() || re.is_inf() || im.is_nan() || im.is_inf() {
        return None;
    }
    if re.is_zero() || im.is_zero() {
        let mut result = if re.is_zero() { im.clone() } else { re.clone() };
        if result.sign() == Some(Sign::Neg) {
            result.inv_sign();
        }
        result.set_precision(p, R::Up).ok()?;
        return Some(result);
    }
    let a = magnitude_up(re)?;
    let b = magnitude_up(im)?;
    let aa = (a * a).next_up();
    let bb = (b * b).next_up();
    let sum = (aa + bb).next_up();
    let bound = sqrt_up(sum);
    if !bound.is_finite() || bound <= 0. {
        return None;
    }
    let result = B::from_f64(bound, p);
    (!result.is_nan() && !result.is_inf()).then_some(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mantissa_tail_and_both_word_layouts_are_enclosed() {
        let value = B::from(1u8).add(&B::from_f64(2f64.powi(-100), 256), 256, R::ToEven);
        let upper = B::from_f64(magnitude_up(&value).unwrap(), 256);
        assert!(upper.cmp(&value).is_some_and(|v| v >= 0));
        assert_eq!(prefix(0x80000000, 0, 32), Some(1u64 << 63));
        assert_eq!(prefix(0x80000000, 0x12345678, 32), Some(0x8000000012345678));
        assert_eq!(prefix(0x8000000012345678, 0, 64), Some(0x8000000012345678));
        assert_eq!(prefix(u64::MAX, 0, 32), None);
        assert_eq!(prefix(1, 0, 16), None);
    }
    #[test]
    fn fast_upper_encloses_independent_512_bit_norms() {
        for exponent in [-490, -100, 0, 100, 490] {
            let scale = B::from_f64(2f64.powi(exponent), 256);
            for k in 1..=128u64 {
                let mut re = B::from_word(k, 256)
                    .div(&B::from(37u8), 256, R::ToEven)
                    .mul(&scale, 256, R::ToEven);
                if k % 2 == 0 {
                    re.inv_sign();
                }
                let im = B::from_word(k * k + 1, 256)
                    .div(&B::from(53u8), 256, R::ToEven)
                    .mul(&scale, 256, R::ToEven);
                let result = upper(&re, &im, 128).unwrap();
                let reference = re
                    .mul(&re, 512, R::Up)
                    .add(&im.mul(&im, 512, R::Up), 512, R::Up)
                    .sqrt(512, R::Up);
                assert!(result.cmp(&reference).is_some_and(|v| v >= 0));
            }
        }
    }
    #[test]
    fn zero_axes_and_extreme_exponents_preserve_fallback() {
        let zero = B::new(128);
        assert!(upper(&zero, &zero, 128).unwrap().is_zero());
        let x = B::from_f64(-0.3, 128);
        assert_eq!(
            upper(&x, &zero, 128).unwrap().cmp(&B::from_f64(0.3, 128)),
            Some(0)
        );
        for exponent in [-800, 800] {
            let x = B::from_f64(2f64.powi(exponent), 128);
            assert!(upper(&x, &x, 128).is_none());
            let actual = super::super::Complex {
                re: x.clone(),
                im: x.clone(),
            }
            .norm_upper(128);
            let reference = x
                .mul(&x, 512, R::Up)
                .mul(&B::from(2u8), 512, R::Up)
                .sqrt(512, R::Up);
            assert!(actual.cmp(&reference).is_some_and(|v| v >= 0));
        }
        for value in [f64::NAN, f64::INFINITY] {
            assert!(upper(&B::from_f64(value, 128), &zero, 128).is_none());
        }
    }
    #[test]
    fn exponent_guard_endpoints_are_outward_and_neighbors_fall_back() {
        // AstroFloat's normalized exponent is one greater than the power
        // for an exact power of two.
        for exponent in [-500, 500] {
            let x = B::from_f64(2f64.powi(exponent - 1), 256);
            let actual = upper(&x, &x, 128).unwrap();
            let reference = x
                .mul(&x, 512, R::Up)
                .mul(&B::from(2u8), 512, R::Up)
                .sqrt(512, R::Up);
            assert!(actual.cmp(&reference).is_some_and(|v| v >= 0));
        }
        for exponent in [-501, 501] {
            let x = B::from_f64(2f64.powi(exponent - 1), 256);
            assert!(upper(&x, &x, 128).is_none());
            let actual = super::super::Complex {
                re: x.clone(),
                im: x.clone(),
            }
            .norm_upper(128);
            let reference = x
                .mul(&x, 512, R::Up)
                .mul(&B::from(2u8), 512, R::Up)
                .sqrt(512, R::Up);
            assert!(actual.cmp(&reference).is_some_and(|v| v >= 0));
        }
    }
    #[test]
    fn square_root_bound_is_outward() {
        let actual = B::from_f64(sqrt_up(3.), 512);
        let reference = B::from(3u8).sqrt(512, R::Up);
        assert!(actual.cmp(&reference).is_some_and(|v| v >= 0));
    }
}
