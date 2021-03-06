# takecell

[![lib.rs](https://img.shields.io/badge/lib.rs-%20-8844ff)](https://lib.rs/crates/takecell)
[![docs](https://docs.rs/takecell/badge.svg)](https://docs.rs/takecell)

`takecell` provides two new cell-like types, `TakeCell` and `TakeOwnCell`.
 Both may store arbitrary non-`Copy` types, can be read from at most once and
provide direct unique access to the stored contents. The core API looks
_roughly_ like this:

```rust,ignore
impl<T> TakeCell<T> {
    const fn new(v: T) -> Self { ... }
}
impl<T: ?Sized> TakeCell<T> {
    fn take(&self) -> Option<&mut T> { ... }
}

impl<T> TakeOwnCell<T> {
    const fn new(v: T) -> Self { ... }
    fn take(&self) -> Option<T> { ... }
}
```

To use this crate add the following to your `Cargo.toml`:

```toml
[dependencies] 
takecell = "0.1"
```