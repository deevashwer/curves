use ark_std::rand::Rng;
use ark_std::test_rng;

use crate::*;

use ark_algebra_test_templates::fields::*;

#[test]
fn test_fr() {
    let mut rng = test_rng();
    let a: Fr = rng.gen();
    let b: Fr = rng.gen();
    field_test(a, b);
    sqrt_field_test(a);
    from_str_test::<Fr>();
    // primefield_test will fail because FftField is not implemented for Fr.
    // primefield_test::<Fr>();
}

#[test]
fn test_fq() {
    let mut rng = test_rng();
    let a: Fq = rng.gen();
    let b: Fq = rng.gen();
    field_test(a, b);
    sqrt_field_test(a);
    from_str_test::<Fq>();
    // primefield_test will fail because FftField is not implemented for Fq.
    // primefield_test::<Fq>();
}
