use ahash::RandomState as AHashState;
use rustc_hash::FxHasher;
use siphasher::sip::SipHasher13;
use std::hash::{BuildHasher, Hasher};
use stringzilla::sz;
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// Trait for hash functions that can be tested
pub trait HashFunction: Send + Sync {
    fn name(&self) -> &'static str;
    fn bits(&self) -> u32;
    fn hash(&self, data: &[u8]) -> u64;
}

/// StringZilla hash function
pub struct StringZillaHash {
    seed: u64,
}

impl StringZillaHash {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl HashFunction for StringZillaHash {
    fn name(&self) -> &'static str {
        "StringZilla"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        sz::hash_with_seed(data, self.seed)
    }
}

/// SipHash using siphasher crate for proper seeding
pub struct SipHashFunction {
    hasher: SipHasher13,
}

impl SipHashFunction {
    pub fn with_seed(seed: u64) -> Self {
        // Derive a 16-byte key from the 64-bit seed
        let mut key = [0u8; 16];
        key[0..8].copy_from_slice(&seed.to_le_bytes());
        key[8..16].copy_from_slice(&seed.wrapping_add(0x9e3779b97f4a7c15).to_le_bytes());
        Self {
            hasher: SipHasher13::new_with_key(&key),
        }
    }
}

impl HashFunction for SipHashFunction {
    fn name(&self) -> &'static str {
        "SipHash"
    }
    fn bits(&self) -> u32 {
        64
    }

    fn hash(&self, data: &[u8]) -> u64 {
        // Clone the hasher to avoid mutation, since we need to reset state
        let mut hasher = self.hasher.clone();
        hasher.write(data);
        hasher.finish()
    }
}

/// aHash function
pub struct AHashFunction {
    builder: AHashState,
}

impl AHashFunction {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            builder: AHashState::with_seed(seed as usize),
        }
    }
}

impl HashFunction for AHashFunction {
    fn name(&self) -> &'static str {
        "aHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        self.builder.hash_one(data)
    }
}

/// xxHash3 function
pub struct XXHash3Function {
    seed: u64,
}

impl XXHash3Function {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl HashFunction for XXHash3Function {
    fn name(&self) -> &'static str {
        "xxHash3"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        xxh3_64_with_seed(data, self.seed)
    }
}

/// gxHash function
pub struct GxHashFunction {
    seed: u64,
}

impl GxHashFunction {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl HashFunction for GxHashFunction {
    fn name(&self) -> &'static str {
        "gxHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        gxhash::gxhash64(data, self.seed as i64)
    }
}

/// CRC32 function with seeding support
pub struct Crc32Function {
    hasher: crc32fast::Hasher,
}

impl Crc32Function {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            hasher: crc32fast::Hasher::new_with_initial(seed as u32),
        }
    }
}

impl HashFunction for Crc32Function {
    fn name(&self) -> &'static str {
        "Crc32"
    }
    fn bits(&self) -> u32 {
        32
    }
    fn hash(&self, data: &[u8]) -> u64 {
        // Clone the hasher to get a fresh state
        let mut hasher = self.hasher.clone();
        hasher.update(data);
        hasher.finalize() as u64
    }
}

/// Murmur3 function
pub struct Murmur3Function {
    seed: u32,
}

impl Murmur3Function {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed: seed as u32 }
    }
}

impl HashFunction for Murmur3Function {
    fn name(&self) -> &'static str {
        "MurMur3"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        let (h1, _h2) = mur3::murmurhash3_x64_128(data, self.seed);
        h1
    }
}

/// FarmHash function with seeding support
pub struct FarmHashFunction {
    seed: u64,
}

impl FarmHashFunction {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl HashFunction for FarmHashFunction {
    fn name(&self) -> &'static str {
        "FarmHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        farmhash::hash64_with_seed(data, self.seed)
    }
}

/// Blake3 function (XOR-folded from 256 bits to 64 bits) with keyed hashing support
pub struct Blake3Function {
    key: [u8; 32],
}

impl Blake3Function {
    pub fn with_seed(seed: u64) -> Self {
        // Derive a 32-byte key from the 64-bit seed
        let mut key = [0u8; 32];
        for i in 0..4 {
            let offset = i * 8;
            let derived_seed = seed.wrapping_add((i as u64) * 0x9e3779b97f4a7c15);
            key[offset..offset + 8].copy_from_slice(&derived_seed.to_le_bytes());
        }
        Self { key }
    }

    fn fold_hash_to_u64(hash_bytes: &[u8; 32]) -> u64 {
        // XOR-fold 256 bits into 64 bits by XORing four 64-bit chunks
        let chunk1 = u64::from_le_bytes([
            hash_bytes[0],
            hash_bytes[1],
            hash_bytes[2],
            hash_bytes[3],
            hash_bytes[4],
            hash_bytes[5],
            hash_bytes[6],
            hash_bytes[7],
        ]);
        let chunk2 = u64::from_le_bytes([
            hash_bytes[8],
            hash_bytes[9],
            hash_bytes[10],
            hash_bytes[11],
            hash_bytes[12],
            hash_bytes[13],
            hash_bytes[14],
            hash_bytes[15],
        ]);
        let chunk3 = u64::from_le_bytes([
            hash_bytes[16],
            hash_bytes[17],
            hash_bytes[18],
            hash_bytes[19],
            hash_bytes[20],
            hash_bytes[21],
            hash_bytes[22],
            hash_bytes[23],
        ]);
        let chunk4 = u64::from_le_bytes([
            hash_bytes[24],
            hash_bytes[25],
            hash_bytes[26],
            hash_bytes[27],
            hash_bytes[28],
            hash_bytes[29],
            hash_bytes[30],
            hash_bytes[31],
        ]);

        chunk1 ^ chunk2 ^ chunk3 ^ chunk4
    }
}

impl HashFunction for Blake3Function {
    fn name(&self) -> &'static str {
        "Blake3"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        let hash = blake3::keyed_hash(&self.key, data);
        let bytes = hash.as_bytes();
        Self::fold_hash_to_u64(bytes)
    }
}

/// FxHash function with seeding support
pub struct FxHashFunction {
    hasher: FxHasher,
}

impl FxHashFunction {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            hasher: FxHasher::with_seed(seed as usize),
        }
    }
}

impl HashFunction for FxHashFunction {
    fn name(&self) -> &'static str {
        "FxHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        // Clone the hasher to avoid mutation, since we need to reset state
        let mut hasher = self.hasher.clone();
        hasher.write(data);
        hasher.finish()
    }
}

/// FoldHash function
pub struct FoldHashFunction {
    builder: foldhash::fast::FixedState,
}

impl FoldHashFunction {
    pub fn with_seed(seed: u64) -> Self {
        Self {
            builder: foldhash::fast::FixedState::with_seed(seed),
        }
    }
}

impl HashFunction for FoldHashFunction {
    fn name(&self) -> &'static str {
        "FoldHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        self.builder.hash_one(data)
    }
}

/// SeaHash function
pub struct SeaHashFunction {
    seed: u64,
}

impl SeaHashFunction {
    pub fn with_seed(seed: u64) -> Self {
        Self { seed }
    }
}

impl HashFunction for SeaHashFunction {
    fn name(&self) -> &'static str {
        "SeaHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        seahash::hash_seeded(data, self.seed, self.seed, self.seed, self.seed)
    }
}

/// Classical 32-bit Karp-Rabin rolling hash with seeded additive factor
pub struct RabinKarp32Function {
    additive_factor: u64,
}

impl RabinKarp32Function {
    const BASE: u64 = 256;
    const MODULUS: u64 = 1_000_000_007; // large prime commonly used for Rabin-Karp

    pub fn with_seed(seed: u64) -> Self {
        Self {
            additive_factor: seed,
        }
    }

    fn compute(&self, data: &[u8]) -> u32 {
        let mut acc = 0u64;
        for &byte in data {
            // Add the seed as an additive factor to each byte value
            let seeded_byte = byte as u64 + self.additive_factor;
            acc = (acc * Self::BASE + seeded_byte) % Self::MODULUS;
        }
        acc as u32
    }
}

impl HashFunction for RabinKarp32Function {
    fn name(&self) -> &'static str {
        "RabinKarp32"
    }
    fn bits(&self) -> u32 {
        32
    }
    fn hash(&self, data: &[u8]) -> u64 {
        self.compute(data) as u64
    }
}

/// Central hash function registry - single source of truth
pub fn create_all_hash_functions(seed: u64) -> Vec<Box<dyn HashFunction>> {
    vec![
        Box::new(StringZillaHash::with_seed(seed)),
        Box::new(SipHashFunction::with_seed(seed)),
        Box::new(AHashFunction::with_seed(seed)),
        Box::new(XXHash3Function::with_seed(seed)),
        Box::new(GxHashFunction::with_seed(seed)),
        Box::new(Crc32Function::with_seed(seed)),
        Box::new(Murmur3Function::with_seed(seed)),
        Box::new(FarmHashFunction::with_seed(seed)),
        Box::new(Blake3Function::with_seed(seed)),
        Box::new(FxHashFunction::with_seed(seed)),
        Box::new(FoldHashFunction::with_seed(seed)),
        Box::new(SeaHashFunction::with_seed(seed)),
        Box::new(RabinKarp32Function::with_seed(seed)),
    ]
}
