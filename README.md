# compactvec

[![Crates.io](https://img.shields.io/crates/v/compactvec)](https://crates.io/crates/compactvec)
[![Docs.rs](https://docs.rs/compactvec/badge.svg)](https://docs.rs/compactvec)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A `Vec<u32>`-compatible container that stores unsigned integers in the **smallest possible byte width**, automatically upgrading in place when a new value demands more space.

---

## Why `CompactVec`?

`Vec<u32>` always uses 4 bytes per element. When your data is dominated by small values — indices, counts, IDs — three of those bytes are often wasted. `CompactVec` picks the tightest representation that fits all current elements:

| All values fit in | Storage | Bytes / element | Savings  |
| ----------------- | ------- | --------------- | -------- |
| 0 – 255           | `u8`    | 1               | **75 %** |
| 0 – 65 535        | `u16`   | 2               | **50 %** |
| 0 – 16 777 215    | `u24`   | 3               | **25 %** |
| 0 – 4 294 967 295 | `u32`   | 4               | —        |

A vector of one million token-IDs that happen to stay below 256 consumes **1 MB** instead of **4 MB**.

The unusual 24-bit tier means that data up to ~16 M (common in file offsets, colour values, and medium-sized indices) still saves 25 % over the naive approach, rather than jumping straight to full u32 cost.

---

## Quick start

```toml
[dependencies]
compactvec = "0.1"
# Optional: serde support
# compactvec = { version = "0.1", features = ["serde"] }
```

```rust
use compactvec::CompactVec;

let mut cv = CompactVec::new();

// Starts at 1 byte per element.
for id in 0u32..=255 {
    cv.push(id);
}
assert_eq!(cv.width_bits(), 8);

// Exceeding 255 triggers an automatic upgrade; all existing elements
// are re-encoded in place — zero user intervention required.
cv.push(256);
assert_eq!(cv.width_bits(), 16);

// Standard Vec-like access.
assert_eq!(cv.get(0), Some(0));
assert_eq!(cv.len(), 257);
```

---

## Feature overview

### Automatic width management

`CompactVec` tracks the minimum width needed to represent every element it holds. Pushes and sets that stay within the current range are **O(1)** and cheap. When a wider value arrives, all elements are re-encoded to the new width before the append — an amortised cost that is still **O(1)** per push because capacity is doubled alongside the upgrade.

```rust
let mut cv = CompactVec::new();
cv.push(1);          // U8  – 1 byte
cv.push(1_000);      // U16 – 2 bytes each (re-encodes previous element)
cv.push(100_000);    // U24 – 3 bytes each
cv.push(u32::MAX);   // U32 – 4 bytes each
```

### O(1) random access

Because elements have a uniform stride within each width tier, random access is a single multiply-and-dereference — identical performance to a standard slice at the same element width.

```rust
let val: Option<u32> = cv.get(42);
// Or, in tight loops where you've already bounds-checked:
let val: u32 = unsafe { cv.get_unchecked(42) };
```

### `shrink_to_fit`

Widths only grow during normal operation. After removing large elements, call `shrink_to_fit()` to re-scan the remaining values and potentially downgrade to a narrower width, releasing memory:

```rust
let mut cv: CompactVec = vec![1u32, 100_000].into_iter().collect();
assert_eq!(cv.width_bits(), 24);

cv.pop(); // remove 100_000

cv.shrink_to_fit();
assert_eq!(cv.width_bits(), 8);  // only 1 remains; U8 is enough
assert_eq!(cv.capacity(), 1);
```

### `serde` support (optional)

Enable the `serde` feature to get `Serialize` / `Deserialize` implementations. The wire format is identical to `Vec<u32>`, so a `CompactVec` serialised to JSON or MessagePack can be deserialised directly into a `Vec<u32>` and vice versa.

```rust
// Serialize
let buf = rmp_serde::encode::to_vec(&cv)?;

// Deserialize
let cv: CompactVec = rmp_serde::decode::from_slice(&buf)?;
```

---

## Compared with alternatives

|                       | `Vec<u32>` | `CompactVec`     | `smallvec`    | bit-packed vecs   |
| --------------------- | ---------- | ---------------- | ------------- | ----------------- |
| Memory (small values) | 4 B/elem   | **1–3 B/elem**   | 4 B/elem      | < 1 B/elem        |
| Random access         | O(1)       | **O(1)**         | O(1)          | O(1) with bit ops |
| Push (amortised)      | O(1)       | **O(1)**         | O(1)          | O(1)              |
| Value range           | full u32   | **full u32**     | full u32      | limited           |
| Upgrade cost          | —          | one-time realloc | —             | —                 |
| Extra complexity      | none       | **minimal**      | heap-or-stack | significant       |

`CompactVec` occupies the sweet spot between a raw `Vec<u32>` (simple, wasteful) and fully bit-packed structures (memory-optimal but complex and slow for random access).

---

## Safety

The internal buffer is a flat heap allocation managed with the global allocator directly (`alloc::alloc` / `alloc::realloc` / `alloc::dealloc`). All reads and writes use **unaligned** accessors so no alignment requirements are imposed on the buffer pointer. The `unsafe` surface is concentrated in `read_val` and `write_val`, both of which are guarded by preconditions checked at every call site.

`CompactVec` implements `Send` and `Sync` because it exclusively owns its allocation.

---

## License

MIT. See [LICENSE](LICENSE.md).
