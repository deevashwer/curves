// use ark_ff::{field_new, SquareRootField, PrimeField};
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
    primefield_test::<Fr>();
}

#[test]
fn test_fq() {
    let mut rng = test_rng();
    let a: Fq = rng.gen();
    let b: Fq = rng.gen();
    field_test(a, b);
    // let neg_sixteen = field_new!(Fq, "-16");
    // println!("neg_sixteen: {:?}", neg_sixteen);
    // let neg_sixteen_sqrt = neg_sixteen.sqrt().unwrap();
    // println!("neg_sixteen_sqrt: {:?}", neg_sixteen_sqrt);
    // println!("neg_sixteen_sqrt: {:?}", neg_sixteen_sqrt.into_repr().to_string());
    // let sqrt_neg_sixteen = field_new!(Fq, "17631683881184975370165255887551781615748388533673675138860");
    // let neg_sixteen_expected = sqrt_neg_sixteen * sqrt_neg_sixteen;
    // println!("neg_sixteen_expected: {:?}", neg_sixteen_expected);
    sqrt_field_test(a);
    primefield_test::<Fq>();
}
