//! # compact-vec
//!
//! A [`CompactVec`] is a `Vec<u32>`-compatible container that automatically stores
//! unsigned integers in the smallest possible byte width (1, 2, 3, or 4 bytes per
//! element), upgrading in place when a newly pushed value exceeds the current
//! maximum representable value.
//!
//! ## Quick example
//!
//! ```rust
//! use compact_vec::CompactVec;
//!
//! let mut cv = CompactVec::new();
//! cv.push(0);          // stored as u8  – 1 byte
//! cv.push(255);        // stored as u8  – 1 byte
//! cv.push(256);        // triggers upgrade: all elements re-encoded as u16
//! assert_eq!(cv.width_bits(), 16);
//! assert_eq!(cv.get(0), Some(0));
//! assert_eq!(cv.get(2), Some(256));
//! ```
//!
//! ## Memory savings
//!
//! | Value range      | Width  | Savings vs `Vec<u32>` |
//! |-----------------|--------|-----------------------|
//! | 0 – 255         | 8 bit  | 75 %                  |
//! | 256 – 65 535    | 16 bit | 50 %                  |
//! | 65 536 – 16 M   | 24 bit | 25 %                  |
//! | 16 M – 4 G      | 32 bit | 0 %                   |
//!
//! ## Feature flags
//!
//! | Flag      | Effect                                     |
//! |-----------|--------------------------------------------|
//! | `serde`   | Implements `Serialize` / `Deserialize`     |

#[cfg(feature = "serde")]
mod serde_impl;

use std::{
    alloc::{self, Layout},
    ptr::NonNull,
};

/// Discriminant for the current per-element storage width of a [`CompactVec`].
///
/// The numeric value of each variant equals its byte size, so `BitWidth as
/// usize` directly gives the `elem_size`.
///
/// # Ordering
///
/// `BitWidth` implements `Ord`: narrower widths compare *less than* wider ones,
/// so code can use `<`, `>`, and `max()` without any extra mapping.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BitWidth {
    U8 = 1,
    U16 = 2,
    U24 = 3,
    U32 = 4,
}

impl BitWidth {
    /// Returns the number of bytes each element occupies at this width.
    #[inline(always)]
    const fn elem_size(&self) -> usize {
        match self {
            BitWidth::U8 => 1,
            BitWidth::U16 => 2,
            BitWidth::U24 => 3,
            BitWidth::U32 => 4,
        }
    }

    /// Returns the narrowest [`BitWidth`] capable of representing `v`.
    #[inline(always)]
    const fn width_for(v: u32) -> Self {
        if v <= u8::MAX as u32 {
            BitWidth::U8
        } else if v <= u16::MAX as u32 {
            BitWidth::U16
        } else if v <= 0xFFFFFF {
            BitWidth::U24
        } else {
            BitWidth::U32
        }
    }
}

impl Ord for BitWidth {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for BitWidth {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some((*self as u8).cmp(&(*other as u8)))
    }
}

/// A `Vec<u32>`-compatible container that packs integers into the smallest
/// possible byte width, upgrading automatically when a new value demands more
/// space.
///
/// # Invariants
///
/// The following invariants are maintained at all times:
/// * `len ≤ cap`.
/// * If `cap == 0`, `ptr` is a [`NonNull::dangling()`] placeholder and must
///   never be dereferenced.
/// * If `cap > 0`, `ptr` points to a valid heap allocation of exactly
///   `cap * width.elem_size()` bytes obtained from the global allocator.
/// * Every value stored at indices `0..len` fits within `width` (i.e. is
///   ≤ `self.max_value()`).
///
/// # Memory model
///
/// Elements are stored in a flat, byte-addressed buffer using little-endian
/// byte order (U16, U24, U32 all use unaligned LE reads/writes). The buffer
/// contains no padding; element `i` starts at byte offset `i * elem_size`.
///
/// # Upgrade semantics
///
/// Width upgrades are irreversible during normal operation — pushing a large
/// value upgrades the vector for the lifetime of that instance. Call
/// [`shrink_to_fit`](CompactVec::shrink_to_fit) to reclaim width after
/// removing large values.
pub struct CompactVec {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize,
    width: BitWidth,
}

impl Clone for CompactVec {
    /// Returns a deep copy of the vector, preserving both length *and* capacity.
    ///
    /// The clone allocates exactly `self.capacity()` slots at `self.width()`,
    /// then copies the live elements.
    fn clone(&self) -> Self {
        let mut new_cv = CompactVec::new();

        if self.len > 0 {
            let es = self.width.elem_size();
            // CRITICAL: width must be set *before* grow_alloc so that
            // grow_alloc computes the correct byte count (cap * es).
            // Previously, setting width after grow_alloc caused the
            // allocation to use elem_size=1 (U8 default) while the
            // subsequent copy used the original elem_size, producing an
            // out-of-bounds write.
            new_cv.width = self.width; // must precede grow_alloc
            new_cv.grow_alloc(self.cap);

            // SAFETY: Regions cannot overlap (separate heap allocations).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.ptr.as_ptr(),
                    new_cv.ptr.as_ptr(),
                    self.len * es,
                );
            }
            new_cv.len = self.len;
        }

        new_cv
    }

    /// Reuses the existing allocation when it is large enough *and* the
    /// widths match, avoiding a reallocation in the common case.
    fn clone_from(&mut self, source: &Self) {
        if self.cap >= source.len && self.width == source.width {
            let es = self.width.elem_size();
            // SAFETY: If source.len == 0 the copy is a no-op; dangling pointers
            // are acceptable for zero-byte copies (NonNull ≠ null, so the
            // precondition is met).
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.ptr.as_ptr(),
                    self.ptr.as_ptr(),
                    source.len * es,
                );
            }
            self.len = source.len;
        } else {
            *self = source.clone();
        }
    }
}

// SAFETY: CompactVec owns its allocation exclusively; no other object holds a
// reference to the same memory.
unsafe impl Send for CompactVec {}
unsafe impl Sync for CompactVec {}

impl CompactVec {
    /// Creates a new, empty `CompactVec` with no heap allocation.
    ///
    /// The initial storage width is [`BitWidth::U8`]. The first
    /// [`push`](CompactVec::push) that exceeds 255 triggers both an allocation
    /// and a width upgrade.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv = CompactVec::new();
    /// assert!(cv.is_empty());
    /// assert_eq!(cv.capacity(), 0);
    /// ```
    pub const fn new() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
            width: BitWidth::U8,
        }
    }

    /// Creates a new, empty `CompactVec` pre-allocated for at least `cap`
    /// elements at [`BitWidth::U8`].
    ///
    /// If `cap` is 0, this is equivalent to [`new`](CompactVec::new) and no
    /// heap allocation takes place. The width will upgrade automatically when
    /// a value that cannot be represented by the current width is pushed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv = CompactVec::with_capacity(64);
    /// assert!(cv.capacity() >= 64);
    /// for i in 0u32..64 {
    ///     cv.push(i % 128);
    /// }
    /// // No reallocation occurred because every value fits in u8.
    /// assert_eq!(cv.width_bits(), 8);
    /// ```
    pub fn with_capacity(cap: usize) -> Self {
        let mut cv = Self::new();
        if cap > 0 {
            cv.grow_alloc(cap);
        }
        cv
    }

    /// Returns the current storage width.
    ///
    /// The width only increases during the lifetime of a `CompactVec` unless
    /// [`shrink_to_fit`](CompactVec::shrink_to_fit) is called.
    pub const fn width(&self) -> BitWidth {
        self.width
    }

    /// Returns the number of elements in the vector.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv = CompactVec::new();
    /// assert_eq!(cv.len(), 0);
    /// cv.push(1);
    /// assert_eq!(cv.len(), 1);
    /// ```
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }
    /// Returns `true` if the vector contains no elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv = CompactVec::new();
    /// assert!(cv.is_empty());
    /// cv.push(0);
    /// assert!(!cv.is_empty());
    /// ```
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the number of elements the vector can hold without reallocating
    /// *at the current width*.
    ///
    /// Note that a subsequent [`push`](CompactVec::push) of a value that
    /// requires a wider representation will trigger both a width upgrade and a
    /// reallocation regardless of capacity headroom.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv = CompactVec::with_capacity(10);
    /// assert!(cv.capacity() >= 10);
    /// ```
    #[inline]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns the largest value that can be stored at the current width
    /// without triggering an upgrade.
    ///
    /// Pushing a value greater than this will cause an upgrade (and
    /// potentially a reallocation).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv = CompactVec::new(); // BitWidth::U8
    /// assert_eq!(cv.max_value(), 255);
    /// ```
    #[inline]
    pub const fn max_value(&self) -> u32 {
        match self.width {
            BitWidth::U8 => u8::MAX as u32,
            BitWidth::U16 => u16::MAX as u32,
            BitWidth::U24 => 0xFFFFFF,
            BitWidth::U32 => u32::MAX,
        }
    }

    /// Returns the storage width in *bits* (`8`, `16`, `24`, or `32`).
    ///
    /// Convenience wrapper around [`width`](CompactVec::width) for use in
    /// assertions and diagnostics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv = CompactVec::new();
    /// assert_eq!(cv.width_bits(), 8);
    /// cv.push(300);
    /// assert_eq!(cv.width_bits(), 16);
    /// ```
    #[inline]
    pub const fn width_bits(&self) -> u32 {
        match self.width {
            BitWidth::U8 => 8,
            BitWidth::U16 => 16,
            BitWidth::U24 => 24,
            BitWidth::U32 => 32,
        }
    }

    /// Returns the element at `idx`, or `None` if `idx ≥ len`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv = CompactVec::new();
    /// assert_eq!(cv.get(0), None);
    /// cv.push(42);
    /// assert_eq!(cv.get(0), Some(42));
    /// assert_eq!(cv.get(1), None);
    /// ```
    #[inline]
    pub fn get(&self, idx: usize) -> Option<u32> {
        if idx >= self.len {
            return None;
        }
        // SAFETY: idx < len, buffer is valid.
        Some(unsafe { self.read_at(idx) })
    }

    /// Returns the element at `idx` without bounds checking.
    ///
    /// # Safety
    ///
    /// `idx` must be less than `self.len()`. Violating this causes undefined
    /// behaviour (out-of-bounds read into the internal buffer).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv: CompactVec = vec![10u32, 20, 300].into_iter().collect();
    /// // SAFETY: indices 0–2 are within bounds.
    /// let v = unsafe { cv.get_unchecked(2) };
    /// assert_eq!(v, 300);
    /// ```
    #[inline]
    pub unsafe fn get_unchecked(&self, idx: usize) -> u32 {
        // SAFETY: caller guarantees idx < len.
        unsafe { self.read_at(idx) }
    }

    /// Appends `value` to the end of the vector.
    ///
    /// If `value` exceeds [`max_value`](CompactVec::max_value), all existing
    /// elements are re-encoded at the new (wider) storage width before the
    /// append. Capacity is doubled when necessary (starting at 4), so the
    /// amortised cost is O(1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv = CompactVec::new();
    /// cv.push(100);   // stored as u8
    /// cv.push(1000);  // upgrade to u16; 100 re-encoded
    /// assert_eq!(cv.width_bits(), 16);
    /// assert_eq!(cv.get(0), Some(100));
    /// assert_eq!(cv.get(1), Some(1000));
    /// ```
    #[inline]
    pub fn push(&mut self, value: u32) {
        let needed = BitWidth::width_for(value);
        let is_full = self.len == self.cap;

        if self.needs_upgrade(needed) {
            let target_cap = if is_full {
                if self.cap == 0 { 4 } else { self.cap * 2 }
            } else {
                self.cap
            };
            self.upgrade_to(needed, target_cap);
        } else if is_full {
            let new_cap = if self.cap == 0 { 4 } else { self.cap * 2 };
            self.grow_alloc(new_cap);
        }

        unsafe {
            self.write_at(self.len, value);
        }
        self.len += 1;
    }

    /// Replaces the element at `idx` with `value`.
    ///
    /// If `value` requires a wider storage than the current width, all
    /// elements (including those at other indices) are re-encoded before the
    /// write.
    ///
    /// # Panics
    ///
    /// Panics if `idx ≥ self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = vec![1u32, 2, 3].into_iter().collect();
    /// assert_eq!(cv.width_bits(), 8);
    /// cv.set(1, 100_000); // forces upgrade to U24
    /// assert_eq!(cv.width_bits(), 24);
    /// assert_eq!(cv.get(1), Some(100_000));
    /// assert_eq!(cv.get(0), Some(1)); // other elements preserved
    /// ```
    pub fn set(&mut self, idx: usize, value: u32) {
        assert!(idx < self.len, "index out of bounds");
        let needed = BitWidth::width_for(value);
        if self.needs_upgrade(needed) {
            self.upgrade(needed);
        }
        // SAFETY: idx < len, buffer is valid.
        unsafe {
            self.write_at(idx, value);
        }
    }

    /// Removes and returns the last element, or `None` if the vector is empty.
    ///
    /// This operation is O(1) and never reallocates. The storage width is
    /// not reduced; call [`shrink_to_fit`](CompactVec::shrink_to_fit) to
    /// potentially downgrade.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = vec![1u32, 2, 3].into_iter().collect();
    /// assert_eq!(cv.pop(), Some(3));
    /// assert_eq!(cv.pop(), Some(2));
    /// assert_eq!(cv.pop(), Some(1));
    /// assert_eq!(cv.pop(), None);
    /// ```
    #[inline]
    pub fn pop(&mut self) -> Option<u32> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;
        Some(unsafe { self.read_at(self.len) })
    }

    /// Removes the element at `index`, shifting all subsequent elements one
    /// position to the left. Returns the removed value.
    ///
    /// This is an O(n) operation. Prefer [`pop`](CompactVec::pop) when you
    /// only need the last element.
    ///
    /// # Panics
    ///
    /// Panics if `index ≥ self.len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = vec![10u32, 20, 30].into_iter().collect();
    /// let removed = cv.remove(1);
    /// assert_eq!(removed, 20);
    /// assert_eq!(cv.len(), 2);
    /// assert_eq!(cv.get(1), Some(30)); // 30 shifted left
    /// ```
    pub fn remove(&mut self, index: usize) -> u32 {
        assert!(index < self.len, "index out of bounds");

        let removed_val = unsafe { self.read_at(index) };

        if index < self.len - 1 {
            let es = self.width.elem_size();
            unsafe {
                let ptr = self.ptr.as_ptr();
                let dst = ptr.add(index * es);
                let src = dst.add(es);
                let count = (self.len - index - 1) * es;

                // Use copy (memmove) because source and destination overlap
                std::ptr::copy(src, dst, count);
            }
        }

        self.len -= 1;
        removed_val
    }

    /// Clears the vector, removing all elements, but retaining the current
    /// allocation and storage width.
    ///
    /// Use this when you plan to refill the vector with values of similar
    /// magnitude. If the next batch is expected to be much smaller, prefer
    /// [`clear_and_shrink_width`](CompactVec::clear_and_shrink_width).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = (0u32..10).collect();
    /// let cap = cv.capacity();
    /// cv.clear();
    /// assert!(cv.is_empty());
    /// assert_eq!(cv.capacity(), cap); // allocation retained
    /// ```
    pub const fn clear(&mut self) {
        self.len = 0;
    }

    /// Shortens the vector to at most `new_len` elements.
    ///
    /// If `new_len ≥ self.len()` this is a no-op. No memory is
    /// freed; call [`shrink_to_fit`](Self::shrink_to_fit) afterwards
    /// if you need to reclaim space or downgrade the storage width.
    ///
    /// This operation is O(1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = (0u32..10).collect();
    /// cv.truncate(4);
    /// assert_eq!(cv.len(), 4);
    /// assert_eq!(cv.get(3), Some(3));
    /// assert_eq!(cv.get(4), None);
    ///
    /// cv.truncate(100); // larger than len — no-op
    /// assert_eq!(cv.len(), 4);
    /// ```
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.len {
            self.len = new_len;
        }
    }

    /// Removes **consecutive** duplicate elements.
    ///
    /// If the vector is already sorted this removes all duplicates.
    /// For arbitrary data it removes only adjacent duplicates (same
    /// semantics as [`std::vec::Vec::dedup`]).
    ///
    /// This is an in-place, O(n) operation with no allocation.
    /// The storage width is never reduced; call
    /// [`shrink_to_fit`](Self::shrink_to_fit) if desired.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = vec![1u32, 1, 2, 3, 3, 3, 4].into_iter().collect();
    /// cv.dedup();
    /// let v: Vec<u32> = cv.iter().collect();
    /// assert_eq!(v, &[1, 2, 3, 4]);
    ///
    /// // Only consecutive — non-adjacent duplicates are kept:
    /// let mut cv2: CompactVec = vec![1u32, 2, 1].into_iter().collect();
    /// cv2.dedup();
    /// assert_eq!(cv2.len(), 3);
    /// ```
    pub fn dedup(&mut self) {
        if self.len < 2 {
            return;
        }
        // Invariant: elements [0..write) hold the deduplicated prefix;
        // `prev` is the last value written there.
        let mut write = 1usize;
        // SAFETY: 0 < len ≤ cap.
        let mut prev = unsafe { self.read_at(0) };

        for read in 1..self.len {
            // SAFETY: read < len ≤ cap.
            let curr = unsafe { self.read_at(read) };
            if curr != prev {
                if write != read {
                    // SAFETY: write < read < len ≤ cap; curr fits current width.
                    unsafe { self.write_at(write, curr) };
                }
                prev = curr;
                write += 1;
            }
        }
        self.len = write;
    }

    /// Retains only the elements for which `f` returns `true`.
    ///
    /// Elements are visited in order; those for which `f` returns
    /// `false` are removed and the remaining elements are shifted
    /// left. The storage width is never reduced automatically.
    ///
    /// This is an O(n) operation with no allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = (0u32..10).collect();
    /// cv.retain(|v| v % 2 == 0); // keep even numbers
    /// let v: Vec<u32> = cv.iter().collect();
    /// assert_eq!(v, &[0, 2, 4, 6, 8]);
    /// ```
    pub fn retain<F: FnMut(u32) -> bool>(&mut self, mut f: F) {
        let mut write_at = 0usize;

        for read_at in 0..self.len {
            // SAFETY: read < len ≤ cap.
            let v = unsafe { self.read_at(read_at) };
            if f(v) {
                if write_at != read_at {
                    // SAFETY: write < read < len ≤ cap; v fits current width.
                    unsafe { self.write_at(write_at, v) };
                }
                write_at += 1;
            }
        }
        self.len = write_at;
    }

    /// Appends all elements of `other` to `self`.
    ///
    /// This is a specialised, efficient alternative to
    /// [`extend`](Self::extend) that avoids per-element overhead:
    ///
    /// * The required storage width and total capacity are computed
    ///   **upfront**, so at most one reallocation (and one width
    ///   upgrade) ever occurs regardless of `other`'s length.
    /// * When both vectors already share the same width, the payload
    ///   is copied with a single `memcpy`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut a: CompactVec = vec![1u32, 2].into_iter().collect();
    /// let     b: CompactVec = vec![300u32, 400].into_iter().collect(); // U16
    ///
    /// a.extend_from_compact_vec(&b);
    /// // `a` upgraded to U16 in one step.
    /// assert_eq!(a.width_bits(), 16);
    /// assert_eq!(a.len(), 4);
    /// assert_eq!(a.get(2), Some(300));
    /// assert_eq!(a.get(3), Some(400));
    /// ```
    pub fn extend_from_compact_vec(&mut self, other: &CompactVec) {
        if other.is_empty() {
            return;
        }

        let target_width = self.width.max(other.width);
        let needed_len = self.len + other.len;
        // Round up to next power-of-two capacity, but never shrink below
        // what we already have.
        let new_cap = needed_len
            .checked_next_power_of_two()
            .unwrap_or(needed_len)
            .max(self.cap);

        if self.needs_upgrade(target_width) {
            // upgrade_to re-encodes existing elements AND grows the buffer.
            self.upgrade_to(target_width, new_cap);
        } else if needed_len > self.cap {
            self.grow_alloc(new_cap);
        }
        // Post-condition: self.width == target_width, self.cap ≥ needed_len.

        if other.width == self.width {
            // Widths match — bulk memcpy of the entire other buffer.
            let es = self.width.elem_size();
            // SAFETY:
            //   src:  other.ptr, valid for other.len * es bytes.
            //   dst:  self.ptr + self.len * es, within self.cap * es allocation.
            //   Regions are disjoint (separate heap allocations).
            unsafe {
                core::ptr::copy_nonoverlapping(
                    other.ptr.as_ptr(),
                    self.ptr.as_ptr().add(self.len * es),
                    other.len * es,
                );
            }
        } else {
            // other.width < self.width — upcast element-by-element.
            for i in 0..other.len {
                // SAFETY: i < other.len ≤ other.cap.
                let v = unsafe { other.read_at(i) };
                // SAFETY: self.len + i < needed_len ≤ self.cap; v fits target_width.
                unsafe { self.write_at(self.len + i, v) };
            }
        }
        self.len += other.len;
    }

    /// Clears the vector and resets the storage width to [`BitWidth::U8`],
    /// freeing the current allocation.
    ///
    /// This is the "hard reset" variant: both `len` and `cap` are zeroed and
    /// the backing memory is released. The next [`push`](CompactVec::push)
    /// will allocate afresh.
    ///
    /// Contrast with [`clear`](CompactVec::clear), which is O(1) and retains
    /// the allocation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = vec![u32::MAX].into_iter().collect();
    /// assert_eq!(cv.width_bits(), 32);
    /// cv.clear_and_shrink_width();
    /// assert_eq!(cv.capacity(), 0);
    /// assert_eq!(cv.width_bits(), 8); // back to narrowest
    /// cv.push(1);                     // fresh allocation at U8
    /// assert_eq!(cv.get(0), Some(1));
    /// ```
    pub fn clear_and_shrink_width(&mut self) {
        if self.cap > 0 {
            let layout = Layout::array::<u8>(self.cap * self.width.elem_size()).unwrap();
            unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
            self.ptr = NonNull::dangling();
            self.cap = 0;
        }
        self.len = 0;
        self.width = BitWidth::U8;
    }

    /// Shrinks the allocation and, if possible, the storage width to exactly
    /// fit the current contents.
    ///
    /// After this call `capacity() == len()` and `width()` is the narrowest
    /// [`BitWidth`] capable of representing every element still in the vector.
    ///
    /// If the vector is empty the backing memory is released entirely and the
    /// width is reset to [`BitWidth::U8`].
    ///
    /// This is an O(n) operation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv = CompactVec::new();
    /// cv.push(1);
    /// cv.push(100_000); // forces U24
    /// cv.pop();         // remove the wide element
    ///
    /// cv.shrink_to_fit();
    /// assert_eq!(cv.width_bits(), 8); // only '1' remains; U8 suffices
    /// assert_eq!(cv.capacity(), 1);
    /// ```
    pub fn shrink_to_fit(&mut self) {
        // Determine the minimum width needed for current elements.
        let mut min_needed = BitWidth::U8;
        for i in 0..self.len {
            let val = unsafe { self.read_at(i) };
            let width = BitWidth::width_for(val);
            if width > min_needed {
                min_needed = width;
            }
            if min_needed == BitWidth::U32 {
                break;
            }
        }

        if self.cap > self.len || min_needed < self.width {
            if self.len == 0 {
                // If empty, just drop the allocation entirely.
                if self.cap > 0 {
                    let es = self.width.elem_size();
                    let layout = Layout::array::<u8>(self.cap * es).unwrap();
                    unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
                    self.ptr = NonNull::dangling();
                    self.cap = 0;
                    self.width = BitWidth::U8;
                }
            } else {
                self.upgrade_to(min_needed, self.len);
            }
        }
    }

    /// Collects all elements into a `Vec<u32>`, consuming `self`.
    ///
    /// The returned `Vec` has an independent allocation; the compact storage
    /// is released.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv: CompactVec = vec![1u32, 2, 3].into_iter().collect();
    /// assert_eq!(cv.to_vec(), vec![1, 2, 3]);
    /// ```
    #[inline]
    pub fn to_vec(self) -> Vec<u32> {
        self.iter().collect()
    }

    /// Returns an iterator over all elements in insertion order.
    ///
    /// Each call to [`next`](Iterator::next) decodes and returns one `u32`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv: CompactVec = vec![10u32, 200, 3000].into_iter().collect();
    /// let sum: u32 = cv.iter().sum();
    /// assert_eq!(sum, 3210);
    /// ```
    pub fn iter(&self) -> Iter<'_> {
        Iter {
            cv: self,
            start: 0,
            end: self.len,
        }
    }

    /// Searches the vector for `value` using binary search.
    ///
    /// The vector **must** be sorted in ascending order. If it is
    /// not, the return value is unspecified (but never unsafe).
    ///
    /// Returns `Ok(index)` when `value` is found. If multiple
    /// occurrences exist, any one of their indices may be returned.
    /// Returns `Err(index)` when `value` is absent, where `index`
    /// is the position at which `value` could be inserted to
    /// maintain sort order.
    ///
    /// This is an O(log n) operation.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv: CompactVec = vec![10u32, 20, 30, 40, 50].into_iter().collect();
    ///
    /// assert_eq!(cv.binary_search(30), Ok(2));
    /// assert_eq!(cv.binary_search(25), Err(2)); // would insert at index 2
    /// assert_eq!(cv.binary_search(0),  Err(0)); // before all elements
    /// assert_eq!(cv.binary_search(99), Err(5)); // after all elements
    /// ```
    pub fn binary_search(&self, value: u32) -> Result<usize, usize> {
        use core::cmp::Ordering;
        let mut lo = 0usize;
        let mut hi = self.len;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            // SAFETY: lo ≤ mid < hi ≤ len ≤ cap.
            let v = unsafe { self.read_at(mid) };
            match v.cmp(&value) {
                Ordering::Equal => return Ok(mid),
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
            }
        }
        Err(lo)
    }

    /// Sorts the vector in ascending order using an unstable algorithm.
    ///
    /// For U8 storage the raw byte slice is sorted directly (zero
    /// extra allocation). For wider widths the values are decoded into
    /// a temporary `Vec<u32>`, sorted, then written back — O(n) extra
    /// space, O(n log n) time. The storage width is never changed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = vec![3u32, 1, 4, 1, 5, 9].into_iter().collect();
    /// cv.sort_unstable();
    /// let v: Vec<u32> = cv.iter().collect();
    /// assert_eq!(v, &[1, 1, 3, 4, 5, 9]);
    /// ```
    pub fn sort_unstable(&mut self) {
        if self.len < 2 {
            return;
        }
        match self.width {
            BitWidth::U8 => {
                // u8 values are numerically ordered by their raw byte value —
                // no decoding necessary.
                // SAFETY: ptr valid for self.len bytes; slice lives ≤ &mut self.
                let slice = unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) };
                slice.sort_unstable();
            }
            _ => {
                // Decode → sort → re-encode.
                // O(n) allocation; avoids implementing pdqsort on raw bytes.
                let mut tmp: Vec<u32> = (0..self.len).map(|i| unsafe { self.read_at(i) }).collect();
                tmp.sort_unstable();
                for (i, v) in tmp.into_iter().enumerate() {
                    // SAFETY: i < self.len ≤ self.cap; v fits width (unchanged).
                    unsafe { self.write_at(i, v) };
                }
            }
        }
    }

    /// Sorts the vector in ascending order using a stable algorithm.
    ///
    /// Preserves the relative order of equal elements. Otherwise
    /// identical to [`sort_unstable`](Self::sort_unstable); see its
    /// documentation for performance characteristics.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let mut cv: CompactVec = vec![3u32, 1, 4, 1, 5].into_iter().collect();
    /// cv.sort();
    /// let v: Vec<u32> = cv.iter().collect();
    /// assert_eq!(v, &[1, 1, 3, 4, 5]);
    /// ```
    pub fn sort(&mut self) {
        if self.len < 2 {
            return;
        }
        match self.width {
            BitWidth::U8 => {
                // SAFETY: same as sort_unstable U8 arm.
                let slice = unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) };
                slice.sort();
            }
            _ => {
                let mut tmp: Vec<u32> = (0..self.len).map(|i| unsafe { self.read_at(i) }).collect();
                tmp.sort();
                for (i, v) in tmp.into_iter().enumerate() {
                    unsafe { self.write_at(i, v) };
                }
            }
        }
    }

    /// Returns the number of bytes currently **occupied** by live elements
    /// (`len() * elem_size`).
    ///
    /// Useful for I/O budgeting and progress reporting.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv: CompactVec = vec![0u32; 4].into_iter().collect();
    /// assert_eq!(cv.width_bits(), 8);
    /// assert_eq!(cv.byte_len(), 4); // 4 × 1 byte
    ///
    /// let mut cv2 = CompactVec::new();
    /// cv2.push(300); // U16
    /// cv2.push(400);
    /// assert_eq!(cv2.byte_len(), 4); // 2 × 2 bytes
    /// ```
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * self.width.elem_size()
    }

    /// Returns the total number of bytes in the backing allocation
    /// (`capacity() * elem_size`).
    ///
    /// Includes unused headroom beyond `len()`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv = CompactVec::with_capacity(8); // 8 × 1 = 8 bytes allocated
    /// assert_eq!(cv.byte_capacity(), 8);
    /// ```
    #[inline]
    pub fn byte_capacity(&self) -> usize {
        self.cap * self.width.elem_size()
    }

    /// Builds a `CompactVec` from a **sorted** (ascending) iterator in
    /// one allocation with no intermediate width upgrades.
    ///
    /// Because the last element of a sorted sequence is its maximum,
    /// the required storage width is known before any element is encoded.
    /// This avoids the per-upgrade reallocations that
    /// [`from_iter`](core::iter::FromIterator) may incur.
    ///
    /// The input is collected into a temporary `Vec<u32>` so that its
    /// maximum can be inspected before choosing the width. For a sorted
    /// slice, prefer the zero-extra-allocation
    /// [`from_sorted_slice`](Self::from_sorted_slice).
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if the input is not sorted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv = CompactVec::from_sorted_iter(0u32..256);
    /// // Width chosen upfront based on the last element (255 → U8).
    /// assert_eq!(cv.width_bits(), 8);
    /// assert_eq!(cv.len(), 256);
    ///
    /// // Large max → U32, but chosen in one shot.
    /// let cv2 = CompactVec::from_sorted_iter([1u32, 100, u32::MAX]);
    /// assert_eq!(cv2.width_bits(), 32);
    /// ```
    pub fn from_sorted_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        // Collect once to learn the length and maximum value.
        // The temporary Vec<u32> is released before returning.
        let values: Vec<u32> = iter.into_iter().collect();
        Self::from_sorted_slice(&values)
    }

    /// Builds a `CompactVec` from a **sorted** (ascending) slice in a
    /// single allocation with no intermediate width upgrades.
    ///
    /// Unlike [`from_sorted_iter`](Self::from_sorted_iter), no temporary
    /// allocation is required because the slice length and maximum value
    /// (`slice.last()`) are known immediately.
    ///
    /// # Panics (debug only)
    ///
    /// In debug builds, panics if the slice is not sorted.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let data = [0u32, 10, 20, 100, 200, 255];
    /// let cv = CompactVec::from_sorted_slice(&data);
    /// assert_eq!(cv.width_bits(), 8);
    /// assert_eq!(cv.len(), 6);
    /// assert_eq!(cv.get(4), Some(200));
    /// ```
    pub fn from_sorted_slice(slice: &[u32]) -> Self {
        if slice.is_empty() {
            return Self::new();
        }
        debug_assert!(
            slice.windows(2).all(|w| w[0] <= w[1]),
            "from_sorted_slice: input must be sorted in ascending order"
        );

        // The last element is the maximum → determines the required width.
        let max_val = slice[slice.len() - 1];
        let width = BitWidth::width_for(max_val);

        let mut cv = Self::new();
        cv.width = width;
        cv.grow_alloc(slice.len());

        for (i, &v) in slice.iter().enumerate() {
            // SAFETY: i < slice.len() == cv.cap; v ≤ max_val, so v fits width.
            unsafe { cv.write_at(i, v) };
        }
        cv.len = slice.len();
        cv
    }

    /// Allocates or reallocates the backing buffer to hold `new_cap` elements
    /// at the current width.
    ///
    /// # Panics / UB prevention
    ///
    /// Calling this with `new_cap == 0` would invoke `alloc::alloc` with a
    /// zero-size layout, which is undefined behaviour. The internal callers
    /// must ensure `new_cap > 0`.
    fn grow_alloc(&mut self, new_cap: usize) {
        let es = self.width.elem_size();
        let new_layout = Layout::array::<u8>(new_cap * es).unwrap();

        let new_ptr = if self.cap == 0 {
            unsafe { alloc::alloc(new_layout) }
        } else {
            let old_layout = Layout::array::<u8>(self.cap * es).unwrap();
            unsafe { alloc::realloc(self.ptr.as_ptr(), old_layout, new_cap * es) }
        };

        self.ptr = NonNull::new(new_ptr).expect("allocation failed");
        self.cap = new_cap;
    }

    /// Allocates a new buffer with `target` width and `new_cap` slots,
    /// re-encodes every existing element, then releases the old buffer.
    ///
    /// This is used for both width upgrades (widening) and width downgrades
    /// triggered by [`shrink_to_fit`](CompactVec::shrink_to_fit).
    fn upgrade_to(&mut self, target: BitWidth, new_cap: usize) {
        let new_es = target.elem_size();
        let new_layout = Layout::array::<u8>(new_cap * new_es).unwrap();
        let new_ptr = NonNull::new(unsafe { alloc::alloc(new_layout) }).expect("allocation failed");

        for i in 0..self.len {
            unsafe {
                let v = self.read_at(i);
                write_val(new_ptr.as_ptr().cast(), i, v, target);
            }
        }

        if self.cap > 0 {
            let old_layout = Layout::array::<u8>(self.cap * self.width.elem_size()).unwrap();
            unsafe {
                alloc::dealloc(self.ptr.as_ptr(), old_layout);
            }
        }

        self.ptr = new_ptr.cast();
        self.cap = new_cap;
        self.width = target;
    }

    /// Returns `true` if `needed` is wider than the current storage width.
    #[inline]
    fn needs_upgrade(&self, needed: BitWidth) -> bool {
        self.width < needed
    }

    /// Re-encodes all elements in the current buffer at `target` width,
    /// keeping capacity unchanged.
    fn upgrade(&mut self, target: BitWidth) {
        self.upgrade_to(target, self.cap);
    }

    /// Reads the element at `idx` from the raw buffer using `self.width`.
    ///
    /// # Safety
    ///
    /// `idx` must be less than `self.len` (and therefore within the
    /// allocation). The buffer must contain validly initialised bytes at
    /// `[idx * elem_size .. (idx + 1) * elem_size)`.
    #[inline(always)]
    unsafe fn read_at(&self, idx: usize) -> u32 {
        unsafe { read_val(self.ptr.as_ptr(), idx, self.width) }
    }

    /// Writes `v` into the raw buffer at `idx` using `self.width`.
    ///
    /// # Safety
    ///
    /// `idx` must be less than `self.cap` (the slot must be allocated), and
    /// `v` must fit within `self.width`.
    #[inline(always)]
    unsafe fn write_at(&mut self, idx: usize, v: u32) {
        unsafe { write_val(self.ptr.as_ptr(), idx, v, self.width) }
    }
}

/// Reads a `u32` from a type-erased byte buffer at element index `idx`.
///
/// All multi-byte reads use `read_unaligned` so alignment of `ptr` is
/// unconstrained. 24-bit values are read as 3 LE bytes into a zero-padded
/// `[u8; 4]`.
///
/// # Safety
///
/// `ptr` must be valid for a read of `(idx + 1) * width.elem_size()` bytes.
#[inline(always)]
const unsafe fn read_val(ptr: *const u8, idx: usize, width: BitWidth) -> u32 {
    match width {
        BitWidth::U8 => unsafe { *ptr.add(idx) as u32 },
        BitWidth::U16 => {
            let p = unsafe { ptr.add(idx * 2) } as *const u16;
            unsafe { p.read_unaligned() as u32 }
        }
        BitWidth::U24 => {
            let p = unsafe { ptr.add(idx * 3) };
            let mut bytes = [0u8; 4];
            unsafe {
                std::ptr::copy_nonoverlapping(p, bytes.as_mut_ptr(), 3);
            }
            u32::from_le_bytes(bytes)
        }
        BitWidth::U32 => {
            let p = unsafe { ptr.add(idx * 4) } as *const u32;
            unsafe { p.read_unaligned() }
        }
    }
}

/// Writes a `u32` into a type-erased byte buffer at element index `idx`.
///
/// 24-bit values are written as the low 3 LE bytes; `v` must be ≤ `0xFF_FFFF`.
///
/// # Safety
///
/// `ptr` must be valid for a write of `(idx + 1) * width.elem_size()` bytes,
/// and `v` must fit within `width`.
#[inline(always)]
const unsafe fn write_val(ptr: *mut u8, idx: usize, v: u32, width: BitWidth) {
    match width {
        BitWidth::U8 => unsafe { *ptr.add(idx) = v as u8 },
        BitWidth::U16 => {
            let p = unsafe { ptr.add(idx * 2) } as *mut u16;
            unsafe { p.write_unaligned(v as u16) };
        }
        BitWidth::U24 => {
            let p = unsafe { ptr.add(idx * 3) };
            let bytes = (v as u32).to_le_bytes();
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, 3);
            }
        }
        BitWidth::U32 => {
            let p = unsafe { ptr.add(idx * 4) } as *mut u32;
            unsafe { p.write_unaligned(v) };
        }
    }
}

impl Drop for CompactVec {
    fn drop(&mut self) {
        if self.cap > 0 {
            let layout = Layout::array::<u8>(self.cap * self.width.elem_size()).unwrap();
            unsafe {
                alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}

/// Borrowing iterator over a [`CompactVec`].
///
/// Created by [`CompactVec::iter`] and by the `IntoIterator` impl for `&CompactVec`.
pub struct Iter<'a> {
    cv: &'a CompactVec,
    start: usize, // was `idx`
    end: usize,   // new: exclusive upper bound
}

impl<'a> Iterator for Iter<'a> {
    type Item = u32;
    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.start >= self.end {
            return None;
        }
        // SAFETY: start < end ≤ original len ≤ cap.
        let v = unsafe { self.cv.read_at(self.start) };
        self.start += 1;
        Some(v)
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.end - self.start;
        (rem, Some(rem))
    }
}

impl ExactSizeIterator for Iter<'_> {}

impl DoubleEndedIterator for Iter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<u32> {
        if self.start >= self.end {
            return None;
        }
        self.end -= 1;
        // SAFETY: new end < old end ≤ original len ≤ cap.
        Some(unsafe { self.cv.read_at(self.end) })
    }
}

/// Consuming iterator over a [`CompactVec`], produced by
/// [`CompactVec::into_iter`].
///
/// The iterator takes ownership of the vector's allocation and
/// releases it when dropped (even if iteration is incomplete).
/// Supports both forward and backward traversal.
pub struct IntoIter {
    cv: CompactVec,
    start: usize, // next index to yield from the front
    end: usize,   // exclusive upper bound (shrinks from the back)
}

impl Iterator for IntoIter {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<u32> {
        if self.start >= self.end {
            return None;
        }
        // SAFETY: start < end ≤ original cv.len ≤ cap.
        let v = unsafe { self.cv.read_at(self.start) };
        self.start += 1;
        Some(v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.end - self.start;
        (rem, Some(rem))
    }
}

impl DoubleEndedIterator for IntoIter {
    #[inline]
    fn next_back(&mut self) -> Option<u32> {
        if self.start >= self.end {
            return None;
        }
        self.end -= 1;
        // SAFETY: new end < old end ≤ original cv.len ≤ cap.
        Some(unsafe { self.cv.read_at(self.end) })
    }
}

impl ExactSizeIterator for IntoIter {}

impl IntoIterator for CompactVec {
    type Item = u32;
    type IntoIter = IntoIter;

    /// Consumes the vector, returning an iterator over its values.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use compact_vec::CompactVec;
    ///
    /// let cv: CompactVec = vec![1u32, 2, 3].into_iter().collect();
    /// let sum: u32 = cv.into_iter().sum();
    /// assert_eq!(sum, 6);
    /// ```
    fn into_iter(self) -> IntoIter {
        let end = self.len;
        IntoIter {
            cv: self,
            start: 0,
            end,
        }
    }
}

impl Default for CompactVec {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CompactVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl FromIterator<u32> for CompactVec {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        let mut cv = Self::new();
        for v in iter {
            cv.push(v);
        }
        cv
    }
}

impl Extend<u32> for CompactVec {
    fn extend<T: IntoIterator<Item = u32>>(&mut self, iter: T) {
        for v in iter {
            self.push(v);
        }
    }
}

impl<'a> IntoIterator for &'a CompactVec {
    type Item = u32;
    type IntoIter = Iter<'a>;
    fn into_iter(self) -> Iter<'a> {
        self.iter()
    }
}

impl From<CompactVec> for Vec<u32> {
    fn from(value: CompactVec) -> Self {
        value.into_iter().collect()
    }
}

impl PartialEq for CompactVec {
    fn eq(&self, other: &Self) -> bool {
        if self.len != other.len {
            return false;
        }
        // Compare element-by-element, decoding from whichever widths are current.
        // This is correct even when widths differ (e.g. [1u8] == [1u32]).
        for i in 0..self.len {
            // SAFETY: i < len ≤ cap for both vecs.
            let a = unsafe { self.read_at(i) };
            let b = unsafe { other.read_at(i) };
            if a != b {
                return false;
            }
        }
        true
    }
}

impl Eq for CompactVec {}

impl PartialEq<Vec<u32>> for CompactVec {
    fn eq(&self, other: &Vec<u32>) -> bool {
        if self.len != other.len() {
            return false;
        }
        for (i, &b) in other.iter().enumerate() {
            // SAFETY: i < len ≤ cap.
            let a = unsafe { self.read_at(i) };
            if a != b {
                return false;
            }
        }
        true
    }
}

impl PartialEq<CompactVec> for Vec<u32> {
    fn eq(&self, other: &CompactVec) -> bool {
        other == self
    }
}

impl From<Vec<u32>> for CompactVec {
    fn from(v: Vec<u32>) -> Self {
        v.into_iter().collect()
    }
}

/// Exposes the live elements as a raw byte slice.
///
/// The slice contains exactly `byte_len()` bytes encoding all
/// `len()` elements in the current width's little-endian format.
/// U24 elements occupy 3 bytes each; all other widths match their
/// standard LE representations.
///
/// This is useful for zero-copy I/O (e.g. writing directly to a
/// file or network buffer) and SIMD processing of the raw payload.
///
/// # Examples
///
/// ```rust
/// use compact_vec::CompactVec;
///
/// let cv: CompactVec = vec![1u32, 2, 3].into_iter().collect(); // U8
/// let bytes: &[u8] = cv.as_ref();
/// assert_eq!(bytes, &[1, 2, 3]);
/// ```
impl AsRef<[u8]> for CompactVec {
    fn as_ref(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: ptr valid for at least len * elem_size bytes; we expose
        // only the live elements (not unused capacity).
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.byte_len()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_u8_and_stores_small_values() {
        let mut cv = CompactVec::new();
        cv.push(0);
        cv.push(255);
        assert_eq!(cv.width_bits(), 8);
        assert_eq!(cv.get(0), Some(0));
        assert_eq!(cv.get(1), Some(255));
        assert_eq!(cv.len(), 2);
    }

    #[test]
    fn upgrades_u8_to_u16() {
        let mut cv = CompactVec::new();
        cv.push(10);
        assert_eq!(cv.width_bits(), 8);
        cv.push(256);
        assert_eq!(cv.width_bits(), 16);
        assert_eq!(cv.get(0), Some(10));
        assert_eq!(cv.get(1), Some(256));
    }

    #[test]
    fn upgrades_u16_to_u24() {
        let mut cv = CompactVec::new();
        cv.push(1000);
        assert_eq!(cv.width_bits(), 16);
        cv.push(u16::MAX as u32 + 1);
        assert_eq!(cv.width_bits(), 24);
        assert_eq!(cv.get(0), Some(1000));
        assert_eq!(cv.get(1), Some(u16::MAX as u32 + 1));
    }

    #[test]
    fn upgrades_u8_directly_to_u24() {
        let mut cv = CompactVec::new();
        cv.push(1);
        cv.push(100_000);
        assert_eq!(cv.width_bits(), 24);
        assert_eq!(cv.get(0), Some(1));
        assert_eq!(cv.get(1), Some(100_000));
    }

    #[test]
    fn push_and_pop() {
        let mut cv = CompactVec::new();
        for i in 0..100u32 {
            cv.push(i);
        }
        assert_eq!(cv.len(), 100);
        for i in (0..100u32).rev() {
            assert_eq!(cv.pop(), Some(i));
        }
        assert!(cv.is_empty());
        assert_eq!(cv.pop(), None);
    }

    #[test]
    fn set_upgrades_if_needed() {
        let mut cv = CompactVec::new();
        cv.push(1u32);
        cv.push(2u32);
        assert_eq!(cv.width_bits(), 8);
        cv.set(0, 70_000);
        assert_eq!(cv.width_bits(), 24);
        assert_eq!(cv.get(0), Some(70_000));
        assert_eq!(cv.get(1), Some(2));
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let cv = CompactVec::new();
        assert_eq!(cv.get(0), None);
        let mut cv2 = CompactVec::new();
        cv2.push(42);
        assert_eq!(cv2.get(1), None);
    }

    #[test]
    fn iter_matches_pushes() {
        let vals: Vec<u32> = vec![0, 1, 255, 256, 65535, 65536, u32::MAX];
        let cv: CompactVec = vals.iter().copied().collect();
        let out: Vec<u32> = cv.iter().collect();
        assert_eq!(vals, out);
        assert_eq!(cv.width_bits(), 32);
    }

    #[test]
    fn with_capacity_no_realloc_for_small_set() {
        let mut cv = CompactVec::with_capacity(64);
        assert!(cv.capacity() >= 64);
        for i in 0..64u32 {
            cv.push(i);
        }
        assert_eq!(cv.len(), 64);
        assert_eq!(cv.width_bits(), 8);
    }

    #[test]
    fn from_iterator() {
        let cv: CompactVec = (0u32..257).collect();
        assert_eq!(cv.len(), 257);
        assert_eq!(cv.width_bits(), 16);
        for i in 0u32..257 {
            assert_eq!(cv.get(i as usize), Some(i));
        }
    }

    #[test]
    fn large_u32_values() {
        let mut cv = CompactVec::new();
        cv.push(u32::MAX);
        assert_eq!(cv.width_bits(), 32);
        assert_eq!(cv.get(0), Some(u32::MAX));
        cv.push(0);
        cv.push(u32::MAX - 1);
        assert_eq!(cv.get(1), Some(0));
        assert_eq!(cv.get(2), Some(u32::MAX - 1));
    }

    #[test]
    fn debug_format() {
        let cv: CompactVec = vec![1u32, 2, 3].into_iter().collect();
        assert_eq!(format!("{cv:?}"), "[1, 2, 3]");
    }

    #[test]
    fn exact_size_iterator() {
        let cv: CompactVec = (0u32..10).collect();
        let mut it = cv.iter();
        assert_eq!(it.len(), 10);
        it.next();
        assert_eq!(it.len(), 9);
    }

    #[test]
    fn get_unchecked_matches_get() {
        let cv: CompactVec = vec![10u32, 20, 300, 70_000].into_iter().collect();
        for i in 0..cv.len() {
            assert_eq!(unsafe { cv.get_unchecked(i) }, cv.get(i).unwrap());
        }
    }

    #[test]
    fn memory_u8_is_quarter_of_u32() {
        let mut cv = CompactVec::new();
        for i in 0u32..=255 {
            cv.push(i);
        }
        assert_eq!(cv.width_bits(), 8);
        let used_bytes = cv.capacity();
        let u32_equivalent = cv.capacity() * 4;
        assert_eq!(used_bytes * 4, u32_equivalent);
    }

    #[test]
    fn u8_storage_is_4x_denser_than_vec_u32() {
        let n = 1024usize;
        let mut cv = CompactVec::with_capacity(n);
        for i in 0..n {
            cv.push(i as u32 % 256);
        }
        assert_eq!(cv.width_bits(), 8);
        let compact_bytes = cv.byte_capacity();
        let naive_bytes = cv.capacity() * 4;
        assert_eq!(compact_bytes * 4, naive_bytes);
    }

    #[test]
    fn u16_storage_is_2x_denser_than_vec_u32() {
        let n = 1024usize;
        let mut cv = CompactVec::with_capacity(n);
        for i in 0..n {
            cv.push(256 + (i as u32 % (65535 - 256)));
        }
        assert_eq!(cv.width_bits(), 16);
        let compact_bytes = cv.byte_capacity();
        let naive_bytes = cv.capacity() * 4;
        assert_eq!(compact_bytes * 2, naive_bytes);
    }

    #[test]
    fn test_u24_boundary_values() {
        let mut cv = CompactVec::new();
        let values = vec![
            0x000000, 0x000001, 0x00FFFF, 0x010000, 0xABCDEF, 0xFFFFFE, 0xFFFFFF,
        ];
        for &v in &values {
            cv.push(v);
        }
        assert_eq!(cv.width_bits(), 24);
        for (i, &expected) in values.iter().enumerate() {
            assert_eq!(cv.get(i), Some(expected), "Failed at index {i}");
        }
    }

    #[test]
    fn test_u24_to_u32_transition() {
        let mut cv = CompactVec::new();
        cv.push(0xFFFFFF);
        assert_eq!(cv.width_bits(), 24);
        cv.push(0x1000000);
        assert_eq!(cv.width_bits(), 32);
        assert_eq!(cv.get(0), Some(0xFFFFFF));
        assert_eq!(cv.get(1), Some(0x1000000));
    }

    #[test]
    fn test_u24_random_access_consistency() {
        let mut cv = CompactVec::new();
        cv.push(0x112233);
        cv.push(0x445566);
        cv.push(0x778899);
        assert_eq!(cv.get(0), Some(0x112233));
        assert_eq!(cv.get(1), Some(0x445566));
        assert_eq!(cv.get(2), Some(0x778899));
    }

    #[test]
    fn test_u24_set_overwrite() {
        let mut cv = CompactVec::new();
        cv.push(0x123456);
        cv.push(0x654321);
        cv.set(0, 0xABCDEF);
        assert_eq!(cv.get(0), Some(0xABCDEF));
        assert_eq!(cv.get(1), Some(0x654321));
    }

    #[test]
    fn test_shrink_capacity_only() {
        let mut cv = CompactVec::new();
        for i in 0..10 {
            cv.push(i);
        }
        let initial_cap = cv.capacity();
        assert!(initial_cap >= 10);
        cv.shrink_to_fit();
        assert_eq!(cv.len(), 10);
        assert_eq!(cv.capacity(), 10);
        assert_eq!(cv.width(), BitWidth::U8);
    }

    #[test]
    fn test_shrink_width_and_capacity() {
        let mut cv = CompactVec::new();
        cv.push(1);
        cv.push(100000);
        assert!(cv.width() > BitWidth::U8);
        cv.pop();
        let high_width = cv.width();
        cv.shrink_to_fit();
        assert_eq!(cv.len(), 1);
        assert_eq!(cv.capacity(), 1);
        assert!(cv.width() < high_width);
        assert_eq!(cv.width(), BitWidth::U8);
        assert_eq!(cv.get(0), Some(1));
    }

    #[test]
    fn test_shrink_empty() {
        let mut cv = CompactVec::with_capacity(100);
        assert_eq!(cv.capacity(), 100);
        cv.shrink_to_fit();
        assert_eq!(cv.len(), 0);
        assert_eq!(cv.capacity(), 0);
        assert_eq!(cv.width(), BitWidth::U8);
    }

    #[test]
    fn test_shrink_no_op() {
        let mut cv = CompactVec::new();
        cv.push(10);
        cv.shrink_to_fit();
        let cap_before = cv.capacity();
        let width_before = cv.width();
        cv.shrink_to_fit();
        assert_eq!(cv.capacity(), cap_before);
        assert_eq!(cv.width(), width_before);
        assert_eq!(cv.get(0), Some(10));
    }

    /// Regression: clone must set `new_cv.width` BEFORE calling `grow_alloc`.
    /// If the order were reversed, grow_alloc would allocate using elem_size=1
    /// (the U8 default), but the subsequent copy would use the original
    /// elem_size (e.g. 4 for U32), writing 8 bytes into a 4-byte buffer.
    #[test]
    fn clone_u32_correct_allocation() {
        let mut cv = CompactVec::new();
        cv.push(u32::MAX); // width = U32, cap = 4
        cv.push(1); // len = 2, cap = 4
        let cloned = cv.clone();
        assert_eq!(cloned.get(0), Some(u32::MAX));
        assert_eq!(cloned.get(1), Some(1));
        assert_eq!(cloned.width(), BitWidth::U32);
    }

    /// Regression: `clear_and_shrink_width` must reset `cap` to 0 so that
    /// the next `grow_alloc` treats it as a fresh allocation (calling
    /// `alloc::alloc`, not `alloc::realloc` with the wrong old layout).
    #[test]
    fn clear_and_shrink_width_then_refill() {
        let mut cv = CompactVec::new();
        cv.push(u32::MAX);
        while cv.len() < cv.capacity() {
            cv.push(0);
        }
        let cap = cv.capacity();
        cv.clear_and_shrink_width();
        assert_eq!(cv.capacity(), 0, "cap must be reset to 0");
        assert_eq!(cv.width(), BitWidth::U8);
        for i in 0..cap as u32 {
            cv.push(i % 256);
        }
        cv.push(42);
        assert_eq!(cv.len(), cap + 1);
    }

    #[test]
    fn partial_eq_same_width() {
        let a: CompactVec = vec![1u32, 2, 3].into_iter().collect();
        let b: CompactVec = vec![1u32, 2, 3].into_iter().collect();
        assert_eq!(a, b);
    }

    #[test]
    fn partial_eq_different_widths() {
        let mut a = CompactVec::new();
        a.push(1);
        let mut b = CompactVec::new();
        b.push(1);
        b.push(u32::MAX);
        b.pop();
        // a is U8, b is U32, both contain [1]
        assert_eq!(a, b);
    }

    #[test]
    fn partial_eq_with_vec() {
        let cv: CompactVec = vec![10u32, 20, 30].into_iter().collect();
        assert_eq!(cv, vec![10u32, 20, 30]);
        assert_ne!(cv, vec![10u32, 20]);
    }

    #[test]
    fn from_vec_u32() {
        let v = vec![0u32, 255, 256, 65536];
        let cv = CompactVec::from(v.clone());
        assert_eq!(cv, v);
    }

    #[test]
    fn truncate_basic() {
        let mut cv: CompactVec = (0u32..10).collect();
        cv.truncate(4);
        assert_eq!(cv.len(), 4);
        assert_eq!(cv.get(3), Some(3));
        assert_eq!(cv.get(4), None);
    }

    #[test]
    fn truncate_noop_when_larger() {
        let mut cv: CompactVec = (0u32..5).collect();
        cv.truncate(100);
        assert_eq!(cv.len(), 5);
    }

    #[test]
    fn truncate_to_zero() {
        let mut cv: CompactVec = (0u32..8).collect();
        cv.truncate(0);
        assert!(cv.is_empty());
        // Width and capacity are unchanged.
        assert!(cv.capacity() >= 8);
    }

    #[test]
    fn truncate_preserves_width() {
        let mut cv: CompactVec = vec![1u32, 70_000].into_iter().collect();
        assert_eq!(cv.width_bits(), 24);
        cv.truncate(1);
        assert_eq!(cv.width_bits(), 24); // width never auto-downgrades
        assert_eq!(cv.get(0), Some(1));
    }

    // --- dedup ---

    #[test]
    fn dedup_removes_consecutive() {
        let mut cv: CompactVec = vec![1u32, 1, 2, 3, 3, 3, 4].into_iter().collect();
        cv.dedup();
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, [1, 2, 3, 4]);
    }

    #[test]
    fn dedup_leaves_non_adjacent() {
        let mut cv: CompactVec = vec![1u32, 2, 1].into_iter().collect();
        cv.dedup();
        assert_eq!(cv.len(), 3);
    }

    #[test]
    fn dedup_all_same() {
        let mut cv: CompactVec = vec![7u32; 100].into_iter().collect();
        cv.dedup();
        assert_eq!(cv.len(), 1);
        assert_eq!(cv.get(0), Some(7));
    }

    #[test]
    fn dedup_empty_and_single() {
        let mut cv = CompactVec::new();
        cv.dedup();
        assert!(cv.is_empty());

        let mut cv2: CompactVec = vec![42u32].into_iter().collect();
        cv2.dedup();
        assert_eq!(cv2.len(), 1);
    }

    #[test]
    fn dedup_on_u32_width() {
        let mut cv: CompactVec = vec![u32::MAX, u32::MAX, 1, 1, 2].into_iter().collect();
        cv.dedup();
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, [u32::MAX, 1, 2]);
        assert_eq!(cv.width_bits(), 32);
    }

    // --- retain ---

    #[test]
    fn retain_evens() {
        let mut cv: CompactVec = (0u32..10).collect();
        cv.retain(|v| v % 2 == 0);
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, [0, 2, 4, 6, 8]);
    }

    #[test]
    fn retain_keep_all() {
        let mut cv: CompactVec = (0u32..5).collect();
        cv.retain(|_| true);
        assert_eq!(cv.len(), 5);
    }

    #[test]
    fn retain_keep_none() {
        let mut cv: CompactVec = (0u32..5).collect();
        cv.retain(|_| false);
        assert!(cv.is_empty());
    }

    #[test]
    fn retain_empty() {
        let mut cv = CompactVec::new();
        cv.retain(|_| true);
        assert!(cv.is_empty());
    }

    #[test]
    fn retain_preserves_order_and_values() {
        let mut cv: CompactVec = vec![10u32, 200, 3000, 40, 500].into_iter().collect();
        cv.retain(|v| v > 100);
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, [200, 3000, 500]);
    }

    // --- extend_from_compact_vec ---

    #[test]
    fn extend_same_width_bulk_copy() {
        let mut a: CompactVec = vec![1u32, 2, 3].into_iter().collect();
        let b: CompactVec = vec![4u32, 5, 6].into_iter().collect();
        assert_eq!(a.width_bits(), 8);
        assert_eq!(b.width_bits(), 8);

        a.extend_from_compact_vec(&b);
        assert_eq!(a.len(), 6);
        let v: Vec<u32> = a.iter().collect();
        assert_eq!(v, [1, 2, 3, 4, 5, 6]);
        assert_eq!(a.width_bits(), 8);
    }

    #[test]
    fn extend_upgrades_self_to_other_width() {
        let mut a: CompactVec = vec![1u32, 2].into_iter().collect(); // U8
        let b: CompactVec = vec![300u32, 400].into_iter().collect(); // U16

        a.extend_from_compact_vec(&b);
        assert_eq!(a.width_bits(), 16);
        assert_eq!(a.len(), 4);
        assert_eq!(a.get(0), Some(1));
        assert_eq!(a.get(2), Some(300));
    }

    #[test]
    fn extend_self_wider_upcasts_other() {
        let mut a: CompactVec = vec![300u32, 400].into_iter().collect(); // U16
        let b: CompactVec = vec![1u32, 2].into_iter().collect(); // U8

        a.extend_from_compact_vec(&b);
        assert_eq!(a.width_bits(), 16); // self stays U16
        assert_eq!(a.len(), 4);
        assert_eq!(a.get(2), Some(1));
        assert_eq!(a.get(3), Some(2));
    }

    #[test]
    fn extend_into_empty_self() {
        let mut a = CompactVec::new();
        let b: CompactVec = vec![10u32, 20, 30].into_iter().collect();
        a.extend_from_compact_vec(&b);
        assert_eq!(a.len(), 3);
        assert_eq!(a.get(2), Some(30));
    }

    #[test]
    fn extend_from_empty_other_noop() {
        let mut a: CompactVec = vec![1u32, 2].into_iter().collect();
        a.extend_from_compact_vec(&CompactVec::new());
        assert_eq!(a.len(), 2);
    }

    // --- binary_search ---

    #[test]
    fn binary_search_found() {
        let cv: CompactVec = vec![10u32, 20, 30, 40, 50].into_iter().collect();
        assert_eq!(cv.binary_search(10), Ok(0));
        assert_eq!(cv.binary_search(30), Ok(2));
        assert_eq!(cv.binary_search(50), Ok(4));
    }

    #[test]
    fn binary_search_not_found() {
        let cv: CompactVec = vec![10u32, 20, 30, 40, 50].into_iter().collect();
        assert_eq!(cv.binary_search(0), Err(0));
        assert_eq!(cv.binary_search(25), Err(2));
        assert_eq!(cv.binary_search(99), Err(5));
    }

    #[test]
    fn binary_search_empty() {
        let cv = CompactVec::new();
        assert_eq!(cv.binary_search(0), Err(0));
    }

    #[test]
    fn binary_search_u32_width() {
        let cv: CompactVec = vec![0u32, u32::MAX / 2, u32::MAX].into_iter().collect();
        assert_eq!(cv.binary_search(u32::MAX), Ok(2));
        assert_eq!(cv.binary_search(1), Err(1));
    }

    #[test]
    fn binary_search_insert_position_is_sorted() {
        let mut cv: CompactVec = vec![2u32, 5, 8, 11].into_iter().collect();
        let Err(pos) = cv.binary_search(7) else {
            panic!()
        };
        // Inserting at pos must maintain sort order.
        cv.push(7);
        cv.sort_unstable();
        assert_eq!(cv.get(pos), Some(7));
    }

    #[test]
    fn sort_unstable_u8() {
        let mut cv: CompactVec = vec![3u32, 1, 4, 1, 5, 9, 2, 6].into_iter().collect();
        assert_eq!(cv.width_bits(), 8);
        cv.sort_unstable();
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, [1, 1, 2, 3, 4, 5, 6, 9]);
    }

    #[test]
    fn sort_unstable_u16() {
        let mut cv: CompactVec = vec![1000u32, 300, 500, 256].into_iter().collect();
        assert_eq!(cv.width_bits(), 16);
        cv.sort_unstable();
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, [256, 300, 500, 1000]);
    }

    #[test]
    fn sort_unstable_u32() {
        let mut cv: CompactVec = vec![u32::MAX, 0u32, u32::MAX / 2].into_iter().collect();
        cv.sort_unstable();
        assert_eq!(cv.get(0), Some(0));
        assert_eq!(cv.get(2), Some(u32::MAX));
    }

    #[test]
    fn sort_stable_preserves_equal() {
        // Can't easily distinguish stable vs unstable with u32 alone,
        // but correctness is verifiable.
        let mut cv: CompactVec = vec![5u32, 2, 2, 1].into_iter().collect();
        cv.sort();
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, [1, 2, 2, 5]);
    }

    #[test]
    fn sort_already_sorted_is_noop() {
        let mut cv: CompactVec = (0u32..10).collect();
        cv.sort_unstable();
        let v: Vec<u32> = cv.iter().collect();
        assert_eq!(v, (0u32..10).collect::<Vec<_>>());
    }

    #[test]
    fn sort_single_element() {
        let mut cv: CompactVec = vec![42u32].into_iter().collect();
        cv.sort_unstable();
        assert_eq!(cv.get(0), Some(42));
    }

    #[test]
    fn into_iter_forward() {
        let cv: CompactVec = vec![10u32, 20, 30].into_iter().collect();
        let v: Vec<u32> = cv.into_iter().collect();
        assert_eq!(v, [10, 20, 30]);
    }

    #[test]
    fn into_iter_reverse() {
        let cv: CompactVec = vec![10u32, 20, 30].into_iter().collect();
        let v: Vec<u32> = cv.into_iter().rev().collect();
        assert_eq!(v, [30, 20, 10]);
    }

    #[test]
    fn into_iter_exact_size() {
        let cv: CompactVec = (0u32..7).collect();
        let mut it = cv.into_iter();
        assert_eq!(it.len(), 7);
        it.next();
        assert_eq!(it.len(), 6);
    }

    #[test]
    fn into_iter_partial_drop_frees_memory() {
        // Drop after consuming only 1 element — no leak or double-free.
        let cv: CompactVec = (0u32..100).collect();
        let mut it = cv.into_iter();
        assert_eq!(it.next(), Some(0));
        drop(it); // should release allocation cleanly
    }

    #[test]
    fn into_iter_sum() {
        let cv: CompactVec = (0u32..=100).collect();
        let sum: u32 = cv.into_iter().sum();
        assert_eq!(sum, 5050);
    }

    #[test]
    fn byte_len_matches_width() {
        let mut cv = CompactVec::new();
        assert_eq!(cv.byte_len(), 0);

        for i in 0u32..4 {
            cv.push(i);
        }
        assert_eq!(cv.width_bits(), 8);
        assert_eq!(cv.byte_len(), 4);

        cv.push(300);
        assert_eq!(cv.width_bits(), 16);
        assert_eq!(cv.byte_len(), 10); // 5 × 2
    }

    #[test]
    fn byte_capacity_equals_cap_times_elem_size() {
        let cv = CompactVec::with_capacity(8);
        assert_eq!(cv.byte_capacity(), cv.capacity() * 1); // U8 by default
    }

    #[test]
    fn as_ref_u8_slice() {
        let cv: CompactVec = vec![1u32, 2, 3].into_iter().collect();
        let bytes: &[u8] = cv.as_ref();
        assert_eq!(bytes, &[1u8, 2, 3]);
    }

    #[test]
    fn as_ref_u16_little_endian() {
        let mut cv = CompactVec::new();
        cv.push(256u32); // 0x0100 in LE → [0x00, 0x01]
        cv.push(1u32); //              → [0x01, 0x00]
        assert_eq!(cv.width_bits(), 16);
        let bytes: &[u8] = cv.as_ref();
        assert_eq!(bytes, &[0x00, 0x01, 0x01, 0x00]);
    }

    #[test]
    fn as_ref_empty_is_empty_slice() {
        let cv = CompactVec::new();
        let bytes: &[u8] = cv.as_ref();
        assert!(bytes.is_empty());
    }

    #[test]
    fn from_sorted_iter_u8() {
        let cv = CompactVec::from_sorted_iter(0u32..256);
        assert_eq!(cv.width_bits(), 8);
        assert_eq!(cv.len(), 256);
        assert_eq!(cv.get(255), Some(255));
    }

    #[test]
    fn from_sorted_iter_u24() {
        let cv = CompactVec::from_sorted_iter([1u32, 1000, 100_000]);
        assert_eq!(cv.width_bits(), 24);
        assert_eq!(cv.len(), 3);
        assert_eq!(cv.get(2), Some(100_000));
    }

    #[test]
    fn from_sorted_iter_u32() {
        let cv = CompactVec::from_sorted_iter([0u32, u32::MAX]);
        assert_eq!(cv.width_bits(), 32);
    }

    #[test]
    fn from_sorted_iter_empty() {
        let cv = CompactVec::from_sorted_iter(core::iter::empty::<u32>());
        assert!(cv.is_empty());
    }

    #[test]
    fn from_sorted_slice_matches_from_sorted_iter() {
        let data: Vec<u32> = (0u32..100).map(|i| i * 3).collect();
        let a = CompactVec::from_sorted_slice(&data);
        let b = CompactVec::from_sorted_iter(data.iter().copied());
        assert_eq!(a, b);
    }

    #[test]
    fn from_sorted_vs_from_iter_same_values() {
        // The two constructors must produce identical logical content.
        let data: Vec<u32> = (0u32..=255).collect();
        let sorted = CompactVec::from_sorted_slice(&data);
        let regular: CompactVec = data.into_iter().collect();
        assert_eq!(sorted, regular);
    }

    #[test]
    fn borrow_iter_rev() {
        let cv: CompactVec = vec![1u32, 2, 3, 4, 5].into_iter().collect();
        let v: Vec<u32> = cv.iter().rev().collect();
        assert_eq!(v, [5, 4, 3, 2, 1]);
    }

    #[test]
    fn borrow_iter_double_ended_mixed() {
        let cv: CompactVec = vec![1u32, 2, 3, 4, 5].into_iter().collect();
        let mut it = cv.iter();
        assert_eq!(it.next(), Some(1));
        assert_eq!(it.next_back(), Some(5));
        assert_eq!(it.next(), Some(2));
        assert_eq!(it.next_back(), Some(4));
        assert_eq!(it.next(), Some(3));
        assert_eq!(it.next(), None);
    }
}
