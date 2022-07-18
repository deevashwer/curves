use core::{ops::{Add, Sub, Mul, Div, AddAssign, SubAssign, MulAssign, DivAssign, Neg}, cmp::Ordering, fmt::{Display, Formatter, Result as FmtResult}, str::FromStr};

use ark_ff::{Field, SquareRootField, BigInteger, BigInteger256, Zero, One, PrimeField, FpParameters, FftParameters, ToBytes, FromBytes, FftField};
use derivative::Derivative;
use k256::{FieldElement, elliptic_curve::{generic_array::GenericArray}};
use ark_serialize::{Flags, CanonicalSerializeWithFlags, SerializationError, buffer_byte_size, EmptyFlags, CanonicalSerialize, CanonicalDeserializeWithFlags, CanonicalDeserialize};
use ark_std::io::{Result as IoResult, Read, Write};
use ark_std::hash::Hash;
use num_bigint::BigUint;

// pub trait ModelParameters: Send + Sync + 'static {
//     type BaseField: Field + SquareRootField;
//     type ScalarField: PrimeField + SquareRootField + Into<<Self::ScalarField as PrimeField>::BigInt>;
// }

#[derive(Derivative)]
#[derivative(
    Default(bound = ""),
    // Hash(bound = ""),
    Clone(bound = ""),
    Copy(bound = ""),
    Debug(bound = ""),
    // PartialEq(bound = ""),
    // Eq(bound = "")
)]
pub struct Fq {
    pub value: FieldElement,
}

impl Fq {
    pub fn is_valid(&self) -> bool {
        self < &Self::from(<FqParameters as FpParameters>::MODULUS)
    }
}

impl Zero for Fq {
    #[inline]
    fn zero() -> Self {
        Self { value: FieldElement::ZERO }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        bool::from(self.value.is_zero())
    }
}

impl One for Fq {
    #[inline]
    fn one() -> Self {
        Self { value: FieldElement::ONE }
    }

    #[inline]
    fn is_one(&self) -> bool {
        self == &Self::one()
    }
}

impl Add for Fq {
    type Output = Self;

    #[inline]
    fn add(mut self, other: Fq) -> Self {
        self.add_assign(other);
        self
    }
}

impl<'a> Add<&'a Fq> for Fq {
    type Output = Self;

    #[inline]
    fn add(mut self, other: &Self) -> Self {
        self.add_assign(other);
        self
    }
}

impl Sub for Fq {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: Fq) -> Self {
        self.sub_assign(other);
        self
    }
}

impl<'a> Sub<&'a Fq> for Fq {
    type Output = Self;

    #[inline]
    fn sub(mut self, other: &Self) -> Self {
        self.sub_assign(other);
        self
    }
}

impl Mul for Fq {
    type Output = Self;

    #[inline]
    fn mul(mut self, other: Fq) -> Self {
        self.mul_assign(other);
        self
    }
}

impl<'a> Mul<&'a Fq> for Fq {
    type Output = Self;

    #[inline]
    fn mul(mut self, other: &Self) -> Self {
        self.mul_assign(other);
        self
    }
}

impl Div for Fq {
    type Output = Self;

    #[inline]
    fn div(mut self, other: Fq) -> Self {
        self.div_assign(other);
        self
    }
}

impl<'a> Div<&'a Fq> for Fq {
    type Output = Self;

    #[inline]
    fn div(mut self, other: &Self) -> Self {
        self.div_assign(other);
        self
    }
}

impl AddAssign for Fq {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.value.add_assign(other.value);
    }
}

impl<'a> AddAssign<&'a Self> for Fq {
    fn add_assign(&mut self, other: &'a Self) {
        self.value.add_assign(other.value)
    }
}

impl SubAssign for Fq {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.value.sub_assign(other.value);
    }
}

impl<'a> SubAssign<&'a Self> for Fq {
    fn sub_assign(&mut self, other: &'a Self) {
        self.value.sub_assign(other.value);
    }
}

impl MulAssign for Fq {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.value.mul_assign(other.value);
    }
}

impl<'a> MulAssign<&'a Self> for Fq {
    fn mul_assign(&mut self, other: &'a Self) {
        self.value.mul_assign(other.value);
    }
}

impl DivAssign for Fq {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.value.mul_assign(other.value.invert().unwrap());
    }
}

impl<'a> DivAssign<&'a Self> for Fq {
    fn div_assign(&mut self, other: &'a Self) {
        self.value.mul_assign(other.value.invert().unwrap());
    }
}

impl zeroize::Zeroize for Fq {
    fn zeroize(&mut self) {
        self.value.zeroize();
    }
}

impl From<bool> for Fq {
    fn from(other: bool) -> Self {
        Self::from_repr(<Fq as PrimeField>::BigInt::from(u64::from(other))).unwrap()
    }
}

impl From<u8> for Fq {
    fn from(other: u8) -> Self {
        Self::from_repr(<Fq as PrimeField>::BigInt::from(u64::from(other))).unwrap()
    }
}

impl From<u16> for Fq {
    fn from(other: u16) -> Self {
        Self::from_repr(<Fq as PrimeField>::BigInt::from(u64::from(other))).unwrap()
    }
}

impl From<u32> for Fq {
    fn from(other: u32) -> Self {
        Self::from_repr(<Fq as PrimeField>::BigInt::from(u64::from(other))).unwrap()
    }
}

impl From<u64> for Fq {
    fn from(other: u64) -> Self {
        Self::from_repr(<Fq as PrimeField>::BigInt::from(u64::from(other))).unwrap()
    }
}

impl From<u128> for Fq {
    fn from(other: u128) -> Self {
        let mut default_int = <Fq as PrimeField>::BigInt::default();
        let upper = (other >> 64) as u64;
        let lower = ((other << 64) >> 64) as u64;
        // This is equivalent to the following, but satisfying the compiler:
        // default_int.0[0] = lower;
        // default_int.0[1] = upper;
        let limbs = [lower, upper];
        for (cur, other) in default_int.0.iter_mut().zip(&limbs) {
            *cur = *other;
        }
        Self::from_repr(default_int).unwrap()
    }
}

impl core::iter::Product<Self> for Fq {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), core::ops::Mul::mul)
    }
}

impl<'a> core::iter::Product<&'a Self> for Fq {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Mul::mul)
    }
}

impl core::iter::Sum<Self> for Fq {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), core::ops::Add::add)
    }
}

impl<'a> core::iter::Sum<&'a Self> for Fq {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), Add::add)
    }
}

impl CanonicalSerializeWithFlags for Fq {
    fn serialize_with_flags<W: ark_std::io::Write, F: Flags>(
        &self,
        mut writer: W,
        flags: F,
    ) -> Result<(), SerializationError> {
        // All reasonable `Flags` should be less than 8 bits in size
        // (256 values are enough for anyone!)
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }

        // Calculate the number of bytes required to represent a field element
        // serialized with `flags`. If `F::BIT_SIZE < 8`,
        // this is at most `$byte_size + 1`
        let output_byte_size = buffer_byte_size(<FqParameters as FpParameters>::MODULUS_BITS as usize + F::BIT_SIZE);

        // Write out `self` to a temporary buffer.
        // The size of the buffer is $byte_size + 1 because `F::BIT_SIZE`
        // is at most 8 bits.
        const BYTE_SIZE: usize = <FqParameters as FpParameters>::MODULUS_BITS as usize / 8;
        let mut bytes = [0u8; BYTE_SIZE + 1];
        self.write(&mut bytes[..BYTE_SIZE])?;

        // Mask out the bits of the last byte that correspond to the flag.
        bytes[output_byte_size - 1] |= flags.u8_bitmask();

        writer.write_all(&bytes[..output_byte_size])?;
        Ok(())
    }

    // Let `m = 8 * n` for some `n` be the smallest multiple of 8 greater
    // than `P::MODULUS_BITS`.
    // If `(m - P::MODULUS_BITS) >= F::BIT_SIZE` , then this method returns `n`;
    // otherwise, it returns `n + 1`.
    fn serialized_size_with_flags<F: Flags>(&self) -> usize {
        buffer_byte_size(<FqParameters as FpParameters>::MODULUS_BITS as usize + F::BIT_SIZE)
    }
}

impl CanonicalSerialize for Fq {
    #[inline]
    fn serialize<W: ark_std::io::Write>(
        &self,
        writer: W,
    ) -> Result<(), SerializationError> {
        self.serialize_with_flags(writer, EmptyFlags)
    }

    #[inline]
    fn serialized_size(&self) -> usize {
        self.serialized_size_with_flags::<EmptyFlags>()
    }
}

impl CanonicalDeserializeWithFlags for Fq {
    fn deserialize_with_flags<R: ark_std::io::Read, F: Flags>(
        mut reader: R,
    ) -> Result<(Self, F), SerializationError> {
        // All reasonable `Flags` should be less than 8 bits in size
        // (256 values are enough for anyone!)
        if F::BIT_SIZE > 8 {
            return Err(SerializationError::NotEnoughSpace);
        }
        // Calculate the number of bytes required to represent a field element
        // serialized with `flags`. If `F::BIT_SIZE < 8`,
        // this is at most `$byte_size + 1`
        let output_byte_size = buffer_byte_size(<FqParameters as FpParameters>::MODULUS_BITS as usize + F::BIT_SIZE);

        const BYTE_SIZE: usize = <FqParameters as FpParameters>::MODULUS_BITS as usize / 8;
        let mut masked_bytes = [0; BYTE_SIZE + 1];
        reader.read_exact(&mut masked_bytes[..output_byte_size])?;

        let flags = F::from_u8_remove_flags(&mut masked_bytes[output_byte_size - 1])
            .ok_or(SerializationError::UnexpectedFlags)?;

        Ok((Self::read(&masked_bytes[..])?, flags))
    }
}

impl CanonicalDeserialize for Fq {
    fn deserialize<R: ark_std::io::Read>(reader: R) -> Result<Self, SerializationError> {
        Self::deserialize_with_flags::<R, EmptyFlags>(reader).map(|(r, _)| r)
    }
}

impl ToBytes for Fq {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        self.into_repr().write(writer)
    }
}

impl FromBytes for Fq {
    #[inline]
    fn read<R: Read>(reader: R) -> IoResult<Self> {
        BigInteger256::read(reader).and_then(|b|
            match Fq::from_repr(b) {
                Some(f) => Ok(f),
                None => Err(ark_std::io::Error::new(ark_std::io::ErrorKind::Other, "FromBytes::read failed")),
            })
    }
}

impl Neg for Fq {
    type Output = Self;
    #[inline]
    #[must_use]
    fn neg(self) -> Self {
        Self { value: self.value.neg().normalize_weak() } 
    }
}

impl PartialEq for Fq {
    fn eq(&self, other: &Self) -> bool {
        self.value.normalize().eq(&other.value.normalize())
        // self.into_repr().eq(&other.into_repr())
    }
}
impl Eq for Fq {}

impl Ord for Fq {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_repr().cmp(&other.into_repr())
    }
}

impl PartialOrd for Fq {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Display for Fq {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, stringify!("Secp_256k1 Fq({})"), self.into_repr())
    }
}

impl ark_std::rand::distributions::Distribution<Fq>
    for ark_std::rand::distributions::Standard
{
    #[inline]
    fn sample<R: ark_std::rand::Rng + ?Sized>(&self, rng: &mut R) -> Fq {
        let value = <FieldElement as k256::elliptic_curve::Field>::random(rng);
        Fq { value }
    }
}

impl Hash for Fq {
    fn hash<H: core::hash::Hasher>(&self, _state: &mut H) {
        todo!()
    }
}

impl Field for Fq {
    type BasePrimeField = Self;

    fn extension_degree() -> u64 {
        1
    }

    fn from_base_prime_field_elems(elems: &[Self::BasePrimeField]) -> Option<Self> {
        if elems.len() != (Self::extension_degree() as usize) {
            return None;
        }
        Some(elems[0])
    }

    #[inline]
    fn double(&self) -> Self {
        let mut temp = *self;
        temp.double_in_place();
        temp
    }

    #[inline]
    fn double_in_place(&mut self) -> &mut Self {
        self.value = self.value.double();
        self
    }

    #[inline]
    fn characteristic() -> &'static [u64] {
        <FqParameters as FpParameters>::MODULUS.as_ref()
    }

    fn from_random_bytes_with_flags<F: Flags>(_bytes: &[u8]) -> Option<(Self, F)> {
        todo!()
    }

    #[inline]
    fn square(&self) -> Self {
        let mut temp = *self;
        temp.square_in_place();
        temp
    }

    fn square_in_place(&mut self) -> &mut Self {
        self.value = self.value.square();
        self
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        let mut temp = *self;
        let ret = temp.inverse_in_place();
        if let Some(_) = ret {
            Some(temp)
        } else {
            None
        }
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        let invert_value = self.value.invert();
        if bool::from(invert_value.is_some()) {
            self.value = invert_value.unwrap();
            Some(self)
        } else {
            None
        }
    }

    fn frobenius_map(&mut self, _power: usize) { }
}

impl SquareRootField for Fq {
    fn legendre(&self) -> ark_ff::LegendreSymbol {
        use ark_ff::fields::LegendreSymbol::*;

        // s = self^((MODULUS - 1) // 2)
        let s = self.pow(<FqParameters as FpParameters>::MODULUS_MINUS_ONE_DIV_TWO);
        if s.is_zero() {
            Zero
        } else if s.is_one() {
            QuadraticResidue
        } else {
            QuadraticNonResidue
        }
    }

    #[inline]
    fn sqrt(&self) -> Option<Self> {
        let mut temp = *self;
        let ret = temp.sqrt_in_place();
        if let Some(_) = ret {
            Some(temp)
        } else {
            None
        }
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        let sqrt_value = self.value.sqrt();
        if bool::from(sqrt_value.is_some()) {
            self.value = sqrt_value.unwrap();
            Some(self)
        } else {
            None
        }
    }
}

pub struct FqParameters;

impl FftParameters for FqParameters {
    type BigInt = BigInteger256;
    const TWO_ADICITY: u32 = 1;
    const TWO_ADIC_ROOT_OF_UNITY: Self::BigInt = todo!();
    const SMALL_SUBGROUP_BASE: Option<u32> = None;
    const SMALL_SUBGROUP_BASE_ADICITY: Option<u32> = None;
    const LARGE_SUBGROUP_ROOT_OF_UNITY: Option<Self::BigInt> = None;
}

impl FftField for Fq {
    type FftParams = FqParameters;

    fn two_adic_root_of_unity() -> Self {
        todo!()
    }

    fn large_subgroup_root_of_unity() -> Option<Self> {
        todo!()
    }

    fn multiplicative_generator() -> Self {
        Fq::from(<FqParameters as FpParameters>::GENERATOR)
    }
}

impl FpParameters for FqParameters {
    const MODULUS: Self::BigInt = BigInteger256([ 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xfffffffefffffc2f ]);
    const MODULUS_BITS: u32 = 256;
    const REPR_SHAVE_BITS: u32 = 0;
    const R: Self::BigInt = todo!();
    const R2: Self::BigInt = todo!();
    const INV: u64 = todo!();
    const GENERATOR: Self::BigInt = todo!();//BigInteger256::from(3u64);
    const CAPACITY: u32 = 255;
    const T: Self::BigInt = todo!();
    const T_MINUS_ONE_DIV_TWO: Self::BigInt = todo!();
    const MODULUS_MINUS_ONE_DIV_TWO: Self::BigInt = BigInteger256( [0x7fffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffff7ffffe17] );
}

impl From<num_bigint::BigUint> for Fq {
    #[inline]
    fn from(val: num_bigint::BigUint) -> Fq {
        Fq::from_repr(val.try_into().unwrap()).unwrap()
    }
}

impl Into<num_bigint::BigUint> for Fq {
    #[inline]
    fn into(self) -> num_bigint::BigUint {
        self.into_repr().into()
    }
}

impl From<BigInteger256> for Fq {
    #[inline]
    fn from(val: BigInteger256) -> Fq {
        Fq::from_repr(val).unwrap()
    }
}

impl Into<BigInteger256> for Fq {
    #[inline]
    fn into(self) -> BigInteger256 {
        self.into_repr()
    }
}

impl FromStr for Fq {
    type Err = ();

    /// Interpret a string of numbers as a (congruent) prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(());
        }

        if s == "0" {
            return Ok(Self::zero());
        }

        let mut res = Self::zero();

        let ten = Self::from(<Self as PrimeField>::BigInt::from(10));

        let mut first_digit = true;

        for c in s.chars() {
            match c.to_digit(10) {
                Some(c) => {
                    if first_digit {
                        if c == 0 {
                            return Err(());
                        }

                        first_digit = false;
                    }

                    res.mul_assign(&ten);
                    let digit = Self::from(u64::from(c));
                    res.add_assign(&digit);
                },
                None => {
                    return Err(());
                },
            }
        }
        if !res.is_valid() {
            Err(())
        } else {
            Ok(res)
        }
    }
}

impl PrimeField for Fq {
    type Params = FqParameters;

    type BigInt = BigInteger256;

    fn from_repr(repr: Self::BigInt) -> Option<Self> {
        let bytes = repr.to_bytes_be();
        assert!(bytes.len() <= 32);
        let bytes_generic = GenericArray::from_slice(&bytes[..32]);
        let value = FieldElement::from_bytes(&bytes_generic);
        if bool::from(value.is_some()) {
            Some(Self { value: value.unwrap() })
        } else {
            None
        }
    }

    fn into_repr(&self) -> Self::BigInt {
        let bytes = self.value.to_bytes();
        let bigint = BigUint::from_bytes_be(&bytes).try_into().unwrap();
        bigint
    }
}