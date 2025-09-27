use ahash::RandomState as AHashState;
use blake3;
use cityhash;
use crc32fast;
use gxhash;
use murmur3;
use std::collections::hash_map::RandomState;
use std::hash::BuildHasher;
use std::io::Cursor;
use stringzilla::sz;
use xxhash_rust::xxh3::xxh3_64;

/// Trait for hash functions that can be tested
pub trait HashFunction: Send + Sync {
    fn name(&self) -> &'static str;
    fn bits(&self) -> u32;
    fn hash(&self, data: &[u8]) -> u64;
}

/// StringZilla hash function
pub struct StringZillaHash;

impl HashFunction for StringZillaHash {
    fn name(&self) -> &'static str {
        "StringZilla"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        sz::hash(data)
    }
}

/// SipHash (standard library default)
pub struct SipHashFunction {
    builder: RandomState,
}

impl SipHashFunction {
    pub fn new() -> Self {
        Self {
            builder: RandomState::new(),
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
        self.builder.hash_one(data)
    }
}

/// aHash function
pub struct AHashFunction {
    builder: AHashState,
}

impl AHashFunction {
    pub fn new() -> Self {
        Self {
            builder: AHashState::with_seed(42),
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
pub struct XXHash3Function;

impl HashFunction for XXHash3Function {
    fn name(&self) -> &'static str {
        "xxHash3"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        xxh3_64(data)
    }
}

/// gxHash function
pub struct GxHashFunction;

impl HashFunction for GxHashFunction {
    fn name(&self) -> &'static str {
        "gxHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        gxhash::gxhash64(data, 42)
    }
}

/// CRC32 function
pub struct Crc32Function;

impl HashFunction for Crc32Function {
    fn name(&self) -> &'static str {
        "Crc32"
    }
    fn bits(&self) -> u32 {
        32
    }
    fn hash(&self, data: &[u8]) -> u64 {
        crc32fast::hash(data) as u64
    }
}

/// Murmur3 function
pub struct Murmur3Function;

impl HashFunction for Murmur3Function {
    fn name(&self) -> &'static str {
        "MurMur3"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        let mut cursor = Cursor::new(data);
        murmur3::murmur3_x64_128(&mut cursor, 0).unwrap() as u64
    }
}

/// CityHash function
pub struct CityHashFunction;

impl HashFunction for CityHashFunction {
    fn name(&self) -> &'static str {
        "CityHash"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        cityhash::city_hash_64(data)
    }
}

/// Blake3 function (first 64 bits)
pub struct Blake3Function;

impl HashFunction for Blake3Function {
    fn name(&self) -> &'static str {
        "Blake3"
    }
    fn bits(&self) -> u32 {
        64
    }
    fn hash(&self, data: &[u8]) -> u64 {
        let hash = blake3::hash(data);
        let bytes = hash.as_bytes();
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }
}

/// Get all available hash functions
pub fn get_all_hash_functions() -> Vec<Box<dyn HashFunction>> {
    vec![
        Box::new(StringZillaHash),
        Box::new(SipHashFunction::new()),
        Box::new(AHashFunction::new()),
        Box::new(XXHash3Function),
        Box::new(GxHashFunction),
        Box::new(Crc32Function),
        Box::new(Murmur3Function),
        Box::new(CityHashFunction),
        Box::new(Blake3Function),
    ]
}

/// Get hash functions by name (case-insensitive)
pub fn get_hash_functions_by_names(names: &[String]) -> Vec<Box<dyn HashFunction>> {
    let all_functions = get_all_hash_functions();
    let names_lower: Vec<String> = names.iter().map(|s| s.to_lowercase()).collect();

    all_functions
        .into_iter()
        .filter(|f| names_lower.contains(&f.name().to_lowercase()))
        .collect()
}

/// Get available hash function names
pub fn get_available_hash_names() -> Vec<&'static str> {
    get_all_hash_functions().iter().map(|f| f.name()).collect()
}
