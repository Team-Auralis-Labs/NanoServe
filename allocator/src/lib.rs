//! Buddy-tier memory allocator, exposed as a C ABI so the C++ inference
//! engine (and eventually anything else) can request/release fixed-power-of-2
//! blocks out of one big pre-reserved arena instead of hammering malloc.
//!
//! Design: classic buddy system.
//!  - Arena size is rounded up to a power of two.
//!  - free_lists[order] holds offsets of free blocks of size 2^order * MIN_BLOCK.
//!  - alloc(): find smallest order >= requested, split down if needed.
//!  - free(): put block back, then repeatedly try to merge with its buddy.
use std::alloc::{alloc, dealloc, Layout};
use std::sync::{Arc, Mutex};

const MIN_BLOCK: usize = 64; // smallest block size in bytes (cache-line sized)

struct Arena {
    base: *mut u8,
    layout: Layout,
    total_size: usize,
    max_order: usize,
    free_lists: Vec<Vec<usize>>, // offsets, indexed by order
}

unsafe impl Send for Arena {}

impl Arena {
    fn new(request_bytes: usize) -> Self {
        let total_size = (request_bytes.max(MIN_BLOCK)).next_power_of_two();
        let max_order = (total_size / MIN_BLOCK).trailing_zeros() as usize;
        let layout = Layout::from_size_align(total_size, 64).unwrap();
        let base = unsafe { alloc(layout) };
        let mut free_lists = vec![Vec::new(); max_order + 1];
        free_lists[max_order].push(0); // whole arena is one free block initially
        Arena { base, layout, total_size, max_order, free_lists }
    }

    fn order_for(&self, size: usize) -> Option<usize> {
        let blocks_needed = (size.max(MIN_BLOCK)).div_ceil(MIN_BLOCK).next_power_of_two();
        let order = blocks_needed.trailing_zeros() as usize;
        if order > self.max_order { None } else { Some(order) }
    }

    fn alloc(&mut self, size: usize) -> Option<usize> {
        let order = self.order_for(size)?;
        let mut o = order;
        while o <= self.max_order && self.free_lists[o].is_empty() {
            o += 1;
        }
        if o > self.max_order {
            return None;
        }
        let mut offset = self.free_lists[o].pop().unwrap();
        while o > order {
            o -= 1;
            let buddy_offset = offset + (MIN_BLOCK << o);
            self.free_lists[o].push(buddy_offset);
        }
        Some(offset)
    }

    fn free(&mut self, mut offset: usize, size: usize) {
        let order = match self.order_for(size) {
            Some(o) => o,
            None => return,
        };
        let mut o = order;
        loop {
            let block_size = MIN_BLOCK << o;
            let buddy_offset = offset ^ block_size;
            let list = &mut self.free_lists[o];
            if let Some(pos) = list.iter().position(|&x| x == buddy_offset) {
                list.remove(pos);
                offset = offset.min(buddy_offset);
                o += 1;
                if o > self.max_order { break; }
            } else {
                self.free_lists[o].push(offset);
                break;
            }
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        unsafe { dealloc(self.base, self.layout) };
    }
}

/// A refcounted handle to a shared arena. Every C/C++/Python caller that
/// wants its own "ownership stake" in the pool should get its own handle via
/// `pool_create` (first owner) or `pool_acquire` (additional owners), and
/// must eventually call `pool_release`/`pool_destroy` on it exactly once.
/// The underlying Arena (and its OS memory) is only freed once the last
/// handle is released — this is Rust's `Arc` doing the same job the borrow
/// checker does inside pure Rust, just carried across the FFI boundary by
/// convention instead of the compiler.
pub struct BuddyPool(Arc<Mutex<Arena>>);

#[no_mangle]
pub extern "C" fn pool_create(size: usize) -> *mut BuddyPool {
    let pool = Box::new(BuddyPool(Arc::new(Mutex::new(Arena::new(size)))));
    Box::into_raw(pool)
}

/// Create a new handle to the SAME underlying arena (increments the Arc
/// refcount). Use this when a second owner — another thread, another
/// long-lived object — needs to keep the pool alive independently.
#[no_mangle]
pub extern "C" fn pool_acquire(pool: *const BuddyPool) -> *mut BuddyPool {
    if pool.is_null() { return std::ptr::null_mut(); }
    let pool = unsafe { &*pool };
    Box::into_raw(Box::new(BuddyPool(Arc::clone(&pool.0))))
}

/// Release this handle. The arena's memory is only actually freed once
/// every handle obtained via pool_create/pool_acquire has been released
/// (i.e. the Arc refcount hits zero) — no manual bookkeeping required.
#[no_mangle]
pub extern "C" fn pool_release(pool: *mut BuddyPool) {
    if pool.is_null() { return; }
    unsafe { drop(Box::from_raw(pool)) };
}

#[no_mangle]
pub extern "C" fn pool_allocate(pool: *mut BuddyPool, req_size: usize) -> *mut u8 {
    let pool = unsafe { &*pool };
    let mut arena = pool.0.lock().unwrap();
    match arena.alloc(req_size) {
        Some(offset) => unsafe { arena.base.add(offset) },
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn pool_free(pool: *mut BuddyPool, ptr: *mut u8, size: usize) {
    let pool = unsafe { &*pool };
    let mut arena = pool.0.lock().unwrap();
    let offset = ptr as usize - arena.base as usize;
    arena.free(offset, size);
}

/// Kept as an alias for `pool_release` (same semantics) so existing callers
/// that expect a "destroy" name keep working.
#[no_mangle]
pub extern "C" fn pool_destroy(pool: *mut BuddyPool) {
    pool_release(pool);
}
