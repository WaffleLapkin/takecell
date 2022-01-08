//! This crate provides two new cell-like types, `TakeCell` and `TakeOwnCell`.
//! Both may store arbitrary non-`Copy` types, can be read from at most once and
//! provide direct unique access to the stored contents. The core API looks
//! roughly like this (and there’s much more inside, read on!):
//! ```rust,ignore
//! impl<T> TakeCell<T> {
//!     const fn new(v: T) -> Self { ... }
//! }
//! impl<T: ?Sized> TakeCell<T> {
//!     fn take(&self) -> Option<&mut T> { ... }
//! }
//!
//! impl<T> TakeOwnCell<T> {
//!     const fn new(v: T) -> Self { ... }
//!     fn take(&self) -> Option<T> { ... }
//! }
//! ```
//! Note that, like with `RefCell` and `Mutex`, the `take` method requires only
//! a shared reference. Because of the single read restriction `take` can
//! return a `&mut T` or `T` instead of `RefMut<T>` or `MutexGuard<T>`. In some
//! sense `TakeCell` can be thought as a `Mutex` without unlocking (or rather
//! with unlocking requiring unique access to the `Mutex`, see [`heal`]).
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
//! `TakeCell` is `Sync` (when `T: Sync`) and as such it may be used in
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
    mem::ManuallyDrop,
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
        // Since this `swap` only changes the value from `false` to `true`, it can only
        // return `false` once. This guarantees that the returned reference is
        // unique.
        //
        // Two threads can't both swap false->true, this is guaranteed by the
        // specification:
        // > All modifications to any particular atomic variable
        // > occur in a total order that is specific to this one atomic variable.
        // > <https://en.cppreference.com/w/cpp/atomic/memory_order>
        //
        // `Relaxed` ordering is ok to use here, because when `TakeCell` is shared we
        // only allow one (1) thread to access the protected memory, so there is no need
        // to synchronize the memory between threads. When `TakeCell` is not shared and
        // can be accessed with `get`, the thread that is holding `&mut TakeCell<_>`
        // must have already synchronized itself with other threads so, again, there is
        // no need for additional synchronization here. See also:
        // <https://discord.com/channels/500028886025895936/628283088555737089/929435782370955344>.
        match self.taken.swap(true, Ordering::Relaxed) {
            // The cell was previously taken
            true => None,
            // The cell wasn't takes before, so we can take it
            false => Some(unsafe { &mut *self.value.get() }),
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
        self.taken.load(Ordering::Relaxed)
    }

    /// Returns a unique reference to the underlying data.
    ///
    /// This call borrows `TakeCell` uniquely (at compile-time) which guarantees
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

    /// Similar to [`is_taken`], but uses unique reference instead of runtime
    /// synchronization.
    ///
    /// [`is_taken`]: TakeCell::is_taken
    pub fn is_taken_unsync(&mut self) -> bool {
        *self.taken.get_mut()
    }

    /// Similar to [`take`], but uses unique reference instead of runtime
    /// synchronization.
    ///
    /// [`take`]: TakeCell::take
    pub fn take_unsync(&mut self) -> Option<&mut T> {
        match self.is_taken_unsync() {
            false => {
                *self.taken.get_mut() = true;
                Some(self.get())
            }
            true => None,
        }
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
        self.taken.store(true, Ordering::Relaxed);

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

/// A cell type which value can be taken only once.
///
/// In difference with [`TakeCell`](crate::TakeCell), this type provides
/// ownership, and not a reference to the inner value. Because of this it can't
/// contain unsized values.
///
/// See [crate-level documentation](mod@self) for more.
#[derive(Default)]
pub struct TakeOwnCell<T>(
    // Invariant: `TakeCell::taken` is true <=> `ManuallyDrop`'s value was taken
    TakeCell<ManuallyDrop<T>>,
);

impl<T> TakeOwnCell<T> {
    /// Creates a new `TakeOwnCell` containing the given value.
    pub const fn new(v: T) -> Self {
        Self(TakeCell::new(ManuallyDrop::new(v)))
    }

    /// Returns the underlying value.
    ///
    /// After this function once returns `Some(_)` all consequtive calls before
    /// [`heal`] will return `None` as the value is already taken.
    ///
    /// [`heal`]: TakeOwnCell::heal
    ///
    /// ## Examples
    ///
    /// ```
    /// # use takecell::TakeOwnCell;
    /// let cell = TakeOwnCell::new(17);
    ///
    /// let value: i32 = cell.take().unwrap();
    /// assert_eq!(value, 17);
    ///
    /// // Already taken
    /// assert!(cell.take().is_none());
    /// assert!(cell.into_inner().is_none());
    /// ```
    pub fn take(&self) -> Option<T> {
        self.0
            .take()
            // ## Safety
            //
            // `TakeCell` guatantees that unique reference to the underlying value is returned only
            // once before `TakeCell::heal`. We ensure a new value is emplaced if it was taken
            // before calling `TakeCell::heal`.
            //
            // In all other places (like `drop` and `get`) we check if the value was taken.
            //
            // This guarantees that the value is not duplicated.
            .map(|value| unsafe { ManuallyDrop::take(value) })
    }

    /// Returns `true` if the underlying value has been already [`take`]n.
    ///
    /// ie if this function returns `true`, then [`take`] will return `None`.
    /// Note however that the oposite is not true: if this function returned
    /// `false` it doesn't guarantee that [`take`] will return `Some(_)` since
    /// there may have been concurent calls to [`take`].
    ///
    /// [`take`]: TakeOwnCell::take
    pub fn is_taken(&self) -> bool {
        self.0.is_taken()
    }

    /// Returns a unique reference to the underlying data.
    ///
    /// This call borrows `TakeOwnCell` uniquely (at compile-time) which
    /// guarantees that we possess the only reference.
    ///
    /// Note that this function does not affect the [`take`]. ie [`take`] may
    /// still return `Some(_)` after a call to this function. The oposite is not
    /// true, after the value is [`take`]n this function returns `None`.
    ///
    /// [`take`]: TakeOwnCell::take
    pub fn get(&mut self) -> Option<&mut T> {
        match self.is_taken() {
            false => {
                // ## Safety
                //
                // While this code doesn't use `unsafe{}` it can be affected by other unsafe
                // blocks (see: `take`).
                //
                // The value may only be accessed if it was not taken before.
                Some(&mut *self.0.get())
            }
            true => None,
        }
    }

    /// Unwraps the underlying value.
    pub fn into_inner(mut self) -> Option<T> {
        self.take_unsync()
    }

    /// Heal this cell. After a call to this function next call to [`take`] will
    /// succeed again, even if [`take`] was called before.
    ///
    /// Returns a reference to the underlying value and `Err(v)` if this cell
    /// was not taken before the call to this function.
    ///
    /// ## Examples
    ///
    /// ```
    /// # use takecell::TakeOwnCell;
    /// let mut cell = TakeOwnCell::new(17);
    ///
    /// let (uref, res) = cell.heal(12);
    /// assert_eq!(res, Err(12));
    /// assert_eq!(*uref, 17);
    /// *uref = 0xAA;
    ///
    /// assert_eq!(cell.take(), Some(0xAA));
    ///
    /// let (uref, res) = cell.heal(12);
    /// assert!(res.is_ok());
    /// assert_eq!(*uref, 12);
    /// *uref = 0xBB;
    ///
    /// assert_eq!(cell.into_inner(), Some(0xBB));
    /// ```
    ///
    /// [`take`]: TakeCell::take
    pub fn heal(&mut self, v: T) -> (&mut T, Result<(), T>) {
        // ## Safety
        //
        // While this code doesn't use `unsafe{}` it can be affected by other unsafe
        // blocks (see: `take`).
        //
        // The value must be emplaced if it was previously taken, before healing the
        // underlying cell.

        let res = match self.0.is_taken() {
            true => {
                *self.0.get() = ManuallyDrop::new(v);
                Ok(())
            }
            false => Err(v),
        };

        self.0.heal();

        let uref = &mut *self.0.get();
        (uref, res)
    }

    /// Similar to [`is_taken`], but uses unique reference instead of runtime
    /// synchronization.
    ///
    /// [`is_taken`]: TakeOwnCell::is_taken
    pub fn is_taken_unsync(&mut self) -> bool {
        self.0.is_taken_unsync()
    }

    /// Similar to [`take`], but uses unique reference instead of runtime
    /// synchronization.
    ///
    /// [`take`]: TakeOwnCell::take
    pub fn take_unsync(&mut self) -> Option<T> {
        self.0
            .take_unsync()
            // ## Safety
            //
            // `TakeCell` guatantees that unique reference to the underlying value is returned only
            // once before `TakeCell::heal`. We ensure a new value is emplaced if it was taken
            // before calling `TakeCell::heal`.
            //
            // In all other places (like `drop` and `get`) we check if the value was taken.
            //
            // This guarantees that the value is not duplicated.
            .map(|value| unsafe { ManuallyDrop::take(value) })
    }

    /// Unchecked version of [`take`].
    ///
    /// ## Safety
    ///
    /// Call to this function must be the first call to [`steal`] or [`take`]
    /// after cell creation or [`heal`].
    ///
    /// [`take`]: TakeOwnCell::take
    /// [`steal`]: TakeOwnCell::steal
    /// [`heal`]: TakeOwnCell::heal
    pub unsafe fn steal(&self) -> T {
        // ## Safety
        //
        // Guaranteed by the caller
        ManuallyDrop::take(self.0.steal())
    }
}

impl<T> From<T> for TakeOwnCell<T> {
    fn from(v: T) -> Self {
        Self::new(v)
    }
}

/// ## Safety
///
/// It is possible to pass ownership via `&TakeOwnCell`. As such,
/// `TakeOwnCell<T>` may be `Sync` (`TakeOwnCell<T>: Send`) if and only if `T`
/// is `Send`. Otherwise there may be UB, see [this example], adopted from
/// sslab-gatech rust group.
///
/// [this example]: https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=c5add45a552290fc206fe2e9c768e03f
///
/// `Sync` on the other hand is not required because `TakeOwnCell`'s value is
/// only accesible from one thread at a time.
///
/// This is again similar to a `Mutex`.
unsafe impl<T: Send> Sync for TakeOwnCell<T> {}

impl<T> Drop for TakeOwnCell<T> {
    fn drop(&mut self) {
        // Drop the underlying value, if the cell still holds it
        let _ = self.take_unsync();
    }
}

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

#[cfg(test)]
mod own_tests {
    use crate::TakeOwnCell;

    #[test]
    fn it_works() {
        let cell = TakeOwnCell::new(17);

        assert_eq!(cell.take(), Some(17));

        assert!(cell.take().is_none());
        assert!(cell.into_inner().is_none());
    }

    #[test]
    fn heal() {
        let mut cell = TakeOwnCell::new(17);

        let (uref, res) = cell.heal(12);
        assert_eq!(res, Err(12));
        assert_eq!(*uref, 17);
        *uref = 0xAA;

        assert_eq!(cell.take(), Some(0xAA));

        let (uref, res) = cell.heal(12);
        assert!(res.is_ok());
        assert_eq!(*uref, 12);
        *uref = 0xBB;

        assert_eq!(cell.into_inner(), Some(0xBB));
    }

    #[test]
    fn r#static() {
        static CELL: TakeOwnCell<i32> = TakeOwnCell::new(42);

        assert!(!CELL.is_taken());

        assert_eq!(CELL.take(), Some(42));

        assert!(CELL.is_taken());
        assert!(CELL.take().is_none());
    }

    #[test]
    fn steal_takes() {
        let cell = TakeOwnCell::new(1);

        assert!(!cell.is_taken());

        // ## Safety
        //
        // There was no calls to take or steal before
        assert_eq!(unsafe { cell.steal() }, 1);

        assert!(cell.is_taken());
        assert!(cell.take().is_none());
        assert!(cell.into_inner().is_none());
    }
}
