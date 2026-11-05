//! Pins that `RegimeFilter::update` allocates nothing (API_SPEC 1.10).  This
//! binary holds one test, so nothing else allocates while the counter runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use regime::{forward_step, GaussianHmm, Matrix, RegimeFilter};

struct Counting;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every call forwards to the System allocator unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::SeqCst);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn update_does_not_allocate() {
    let r = regime::load_dataset("../data").unwrap().index_returns;
    let mut m = GaussianHmm::with_defaults(3).unwrap();
    m.fit(&Matrix::column(&r[..600]), None).unwrap();
    let p = m.params().unwrap();
    let mut filt = RegimeFilter::new(p, None).unwrap();
    let mut sink = 0.0;
    let before = ALLOCS.load(Ordering::SeqCst);
    for &x in &r {
        sink += filt.update(&[x]).unwrap()[0];
    }
    let after = ALLOCS.load(Ordering::SeqCst);
    assert_eq!(after - before, 0);
    assert!(sink > 0.0);

    // The counter itself works: forward_step allocates.
    let fs_before = ALLOCS.load(Ordering::SeqCst);
    forward_step(p, filt.alpha(), &[0.001]).unwrap();
    assert!(ALLOCS.load(Ordering::SeqCst) > fs_before);
}
