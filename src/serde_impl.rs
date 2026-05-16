use serde::{
    de::{SeqAccess, Visitor},
    ser::SerializeSeq,
    Deserialize, Deserializer, Serialize, Serializer
};

use crate::CompactVec;

impl Serialize for CompactVec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.len))?;
        for val in self.iter() {
            seq.serialize_element(&val)?;
        }
        seq.end()
    }
}

impl<'de> Deserialize<'de> for CompactVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct CompactVecVisitor;

        impl<'de> Visitor<'de> for CompactVecVisitor {
            type Value = CompactVec;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a sequence of u32")
            }

            fn visit_seq<S>(self, mut seq: S) -> Result<Self::Value, S::Error>
            where
                S: SeqAccess<'de>,
            {
                let mut cv = CompactVec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(value) = seq.next_element()? {
                    cv.push(value);
                }
                Ok(cv)
            }
        }

        deserializer.deserialize_seq(CompactVecVisitor)
    }
}

#[cfg(test)]
mod tests {
    use rmp_serde::{decode, encode};

    use crate::CompactVec;

    #[test]
    fn test_messagepack_roundtrip_basic() {
        let mut cv = CompactVec::new();
        cv.push(10);
        cv.push(500);
        cv.push(100_000); // Forces U24

        // Serialize to MessagePack bytes
        let buf = encode::to_vec(&cv).expect("Encoding failed");

        // Deserialize back
        let decoded: CompactVec = decode::from_slice(&buf).expect("Decoding failed");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded.get(0), Some(10));
        assert_eq!(decoded.get(1), Some(500));
        assert_eq!(decoded.get(2), Some(100_000));
        assert_eq!(decoded.width_bits(), 24);
    }

    #[test]
    fn test_messagepack_compatibility_with_vec() {
        let mut cv = CompactVec::new();
        cv.push(1);
        cv.push(2);
        cv.push(3);

        // Serialize CompactVec
        let cv_bytes = encode::to_vec(&cv).expect("Encoding failed");

        // Deserialize into a standard Vec<u32>
        let v: Vec<u32> = decode::from_slice(&cv_bytes).expect("Decoding into Vec failed");
        assert_eq!(v, vec![1, 2, 3]);

        // Serialize standard Vec<u32>
        let v_bytes = encode::to_vec(&v).expect("Encoding Vec failed");

        // Deserialize into CompactVec
        let decoded_cv: CompactVec = decode::from_slice(&v_bytes).expect("Decoding into CV failed");
        assert_eq!(decoded_cv.get(0), Some(1));
        assert_eq!(decoded_cv.width_bits(), 8); // Should correctly start at U8
    }

    #[test]
    fn test_serialization_empty() {
        let cv = CompactVec::new();
        let buf = encode::to_vec(&cv).expect("Encoding failed");
        let decoded: CompactVec = decode::from_slice(&buf).expect("Decoding failed");
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_large_u32_roundtrip() {
        let mut cv = CompactVec::new();
        cv.push(u32::MAX);

        let buf = encode::to_vec(&cv).expect("Encoding failed");
        let decoded: CompactVec = decode::from_slice(&buf).expect("Decoding failed");

        assert_eq!(decoded.get(0), Some(u32::MAX));
        assert_eq!(decoded.width_bits(), 32);
    }
}
