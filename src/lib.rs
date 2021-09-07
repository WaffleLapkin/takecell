//! This crate provides a new cell-like type, `TakeCell`. A `TakeCell` may
//! store arbitrary non-`Copy` types, can be read from at most once and provides
//! direct unique access to the stored contents. The core API looks roughly like
//! this (and there’s much more inside, read on!):
//! ```rust,ignore
//! impl<T> TakeCell<T> {
//!     const fn new() -> OnceCell<T> { ... }
//! }
//! impl<T: ?Sized> TakeCell<T> {
//!     fn get(&self) -> Option<&mut T> { ... }
//! }
//! ```
//! Note that, like with `RefCell` and `Mutex`, the `take` method requires only
//! a shared reference. Because of the single read restriction `take` can
//! return a `&mut T` instead of `RefMut<T>` or `MutexGuard<T>`. In some sense
//! `TakeCell` can be thought as a `Mutex` without unlocking (or rather with
//! unlocking requiring unique access to the `Mutex`, see [`heal`]).
//!
//! [`heal`]: TakeCell::heal
//!
//! This crate is `#![no_std]` and only requires little sychronization via 8-bit
//! atomic.
//!
//! ## Usage examples
//!
//! ### Singletons
//!
//! `TakeCell` is `Sync` (when `T: Sync + Send`) and as such it may be used in
//! `static`s. This can be used to create singletons:
//!
//! ```
//! use takecell::TakeCell;
//!
//! #[non_exhaustive]
//! pub struct Peripherals {
//!     pub something: Something,
//! }
//!
//! pub static PEREPHERALS: TakeCell<Peripherals> = TakeCell::new(Peripherals {
//!     something: Something,
//! });
//! # pub struct Something;
//!
//! let peripherals: &'static mut _ = PEREPHERALS.take().unwrap();
//! ```
//!
//! ### Doing work only once
//!
//! ```
//! use once_cell::sync::OnceCell;
//! use std::sync::{Arc, Condvar, Mutex};
//! use takecell::TakeCell;
//!
//! #[derive(Clone)]
//! struct Job {
//!     // Input can be a type which requires unique access to be used (e.g.: `dyn Read`)
//!     input: Arc<TakeCell<Input>>,
//!     output: Arc<OnceCell<Output>>,
//!     wait: Arc<(Mutex<bool>, Condvar)>,
//! }
//!
//! fn execute(job: Job) -> Output {
//!     match job.input.take() {
//!         Some(input) => {
//!             // Nobody has started executing the job yet, so execute it
//!             let output = input.process();
//!
//!             // Write the output
//!             job.output.set(output);
//!
//!             // Notify other threads that the job is done
//!             let (lock, cvar) = &*job.wait;
//!             let mut done = lock.lock().unwrap();
//!             *done = true;
//!         }
//!         None => {
//!             // Wait for the other thread to do the job
//!             let (lock, cvar) = &*job.wait;
//!             let mut done = lock.lock().unwrap();
//!             // As long as the value inside the `Mutex<bool>` is `false`, we wait
//!             while !*done {
//!                 done = cvar.wait(done).unwrap();
//!             }
//!         }
//!     }
//!
//!     // Read the output
//!     job.output.get().unwrap().clone()
//! }
//!
//! impl Input {
//!     fn process(&mut self) -> Output {
//!         // ...
//! #       Output
//!     }
//! }
//! # struct Input; #[derive(Clone)] struct Output;
//! ```
#![no_std]
use core::{
    cell::UnsafeCell,
    sync::atomic::{AtomicBool, Ordering},
};

/// A cell type which value can be taken only once.
///
/// See [crate-level documentation](mod@self) for more.
#[derive(Default)]
pub struct TakeCell<T: ?Sized> {
    taken: AtomicBool,
    value: UnsafeCell<T>,
}

impl<T> TakeCell<T> {
    /// Creates a new `TakeCell` containing the given value.
    pub const fn new(v: T) -> Self {
        Self {
            taken: AtomicBool::new(false),
            value: UnsafeCell::new(v),
        }
    }

    /// Unwraps the underlying value.
    pub fn into_inner(self) -> T {
        // TODO: make `into_inner` `const` when `UnsafeCell::into_inner` as `const fn`
        // will be stabilized.
        self.value.into_inner()
    }
}

impl<T: ?Sized> TakeCell<T> {
    /// Returns a reference to the underlying value.
    ///
    /// After this function once returns `Some(_)` all consequtive calls before
    /// [`heal`] will return `None` as the reference is already taken.
    ///
    /// [`heal`]: TakeCell::heal
    ///
    /// ## Examples
    ///
    /// ```
    /// # use takecell::TakeCell;
    /// let cell = TakeCell::new(0);
    ///
    /// let uref: &mut _ = cell.take().unwrap();
    /// *uref = 17;
    ///
    /// // Already taken
    /// assert!(cell.take().is_none());
    ///
    /// let value = cell.into_inner();
    /// assert_eq!(value, 17);
    /// ```
    pub fn take(&self) -> Option<&mut T> {
        // ## Safety
        //
        // Aside from `steal` (that is unsafe and it's caller must guarantee that there
        // are no concurent calls to `steal`/`take`) this is the only place where we are
        // changing the value of `self.taken`.
        //
        // This is also the only place (again, aside from `steal`) where we use/provide
        // a reference to the underlying value.
        //
        // Since this `compare_exchange` only changes the value from `false` to `true`,
        // it can only succeed once. This guarantees that the returned reference is
        // unique.
        match self
            .taken
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Some(unsafe { &mut *self.value.get() }),
            Err(_) => None,
        }
    }

    /// Returns `true` if a reference to the underlying value has been already
    /// [`take`]n.
    ///
    /// ie if this function returns `true`, then [`take`] will return `None`.
    /// Note however that the oposite is not true: if this function returned
    /// `false` it doesn't guarantee that [`take`] will return `Some(_)` since
    /// there may have been concurent calls to [`take`].
    ///
    /// [`take`]: TakeCell::take
    pub fn is_taken(&self) -> bool {
        self.taken.load(Ordering::SeqCst)
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// This call borrows `TakeCell` mutably (at compile-time) which guarantees
    /// that we possess the only reference.
    ///
    /// Note that this function is not affected nor affects the [`take`]. ie
    /// this function will return a reference even if [`take`] was already
    /// called
    ///
    /// [`take`]: TakeCell::take
    pub fn get(&mut self) -> &mut T {
        // TODO: make `get` `const` when `UnsafeCell::get_mut` as `const fn`
        // will be stabilized.
        self.value.get_mut()
    }

    /// Heal this cell. After a call to this function next call to [`take`] will
    /// succeed again, even if [`take`] was called before.
    ///
    /// ## Examples
    ///
    /// ```
    /// # use takecell::TakeCell;
    /// let mut cell = TakeCell::new(0);
    ///
    /// assert!(cell.take().is_some());
    /// assert!(cell.is_taken());
    ///
    /// cell.heal();
    ///
    /// assert!(!cell.is_taken());
    /// assert!(cell.take().is_some());
    /// ```
    ///
    /// [`take`]: TakeCell::take
    pub fn heal(&mut self) {
        // Unique reference to self guarantees that the reference retuened from
        // `take`/`steal` (if these function were even called) is dead, thus it's okay
        // to allow a new unique reference to the underlying value to be created.
        self.taken = AtomicBool::new(false);
    }

    /// Unchecked version of [`take`].
    ///
    /// ## Safety
    ///
    /// Call to this function must be the first call to [`steal`] or [`take`]
    /// after cell creation or [`heal`].
    ///
    /// [`take`]: TakeCell::take
    /// [`steal`]: TakeCell::steal
    /// [`heal`]: TakeCell::heal
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn steal(&self) -> &mut T {
        self.taken.store(true, Ordering::SeqCst);

        // ## Safety
        //
        // Guaranteed by the caller
        &mut *self.value.get()
    }
}

impl<T> From<T> for TakeCell<T> {
    fn from(v: T) -> Self {
        Self::new(v)
    }
}

/// ## Safety
///
/// It is possible to pass ownership via `&TakeCell`. As such, `TakeCell<T>` may
/// be `Sync` (`TakeCell<T>: Send`) if and only if `T` is `Send`. Otherwise
/// there may be UB, see [this example], adopted from sslab-gatech rust group.
///
/// [this example]: https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=c5add45a552290fc206fe2e9c768e03f
///
/// `Sync` on the other hand is not required because `TakeCell`'s value is only
/// accesible from one thread at a time.
///
/// This is again similar to a `Mutex`.
unsafe impl<T: ?Sized + Send> Sync for TakeCell<T> {}

// TODO: There may be a slightly different cell, which returns an owned value
// instead of a reference (ie optimized version of `TakeCell<Option<T>>`)

#[cfg(test)]
mod tests {
    use crate::TakeCell;

    #[test]
    fn it_works() {
        let cell = TakeCell::new(0);

        {
            let uref = cell.take().unwrap();
            *uref += 1;
            assert_eq!(*uref, 1);

            assert!(cell.take().is_none());

            *uref += 1;
            assert_eq!(*uref, 2);
        }

        assert!(cell.take().is_none());
        assert_eq!(cell.into_inner(), 2);
    }

    #[test]
    fn unsize() {
        let cell: TakeCell<[i32; 10]> = TakeCell::new([0; 10]);

        let _: &TakeCell<[i32]> = &cell;
        let _: &TakeCell<dyn Send> = &cell;
    }

    #[test]
    fn r#static() {
        static CELL: TakeCell<i32> = TakeCell::new(0);

        {
            let uref: &'static mut i32 = CELL.take().unwrap();
            *uref += 1;
            assert_eq!(*uref, 1);

            assert!(CELL.take().is_none());

            *uref += 1;
            assert_eq!(*uref, 2);
        }

        assert!(CELL.take().is_none());
    }

    #[test]
    fn steal_takes() {
        let cell = TakeCell::new(0);

        // ## Safety
        //
        // There was no calls to take or steal before
        let uref = unsafe { cell.steal() };
        *uref += 1;
        assert_eq!(*uref, 1);

        assert!(cell.is_taken());
        assert!(cell.take().is_none());

        *uref += 1;
        assert_eq!(*uref, 2);
        assert_eq!(cell.into_inner(), 2);
    }
}
