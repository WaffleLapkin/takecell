# takecell

`takecell` provides two new cell-like types, `TakeCell` and `TakeOwnCell`.
 Both may store arbitrary non-`Copy` types, can be read from at most once and
provide direct unique access to the stored contents. The core API looks
_roughly_ like this:

```rust,ignore
impl<T> TakeCell<T> {
    const fn new(v: T) -> Self { ... }
}
impl<T: ?Sized> TakeCell<T> {
    fn get(&self) -> Option<&mut T> { ... }
}

impl<T> TakeOwnCell<T> {
    const fn new(v: T) -> Self { ... }
    fn get(&self) -> Option<T> { ... }
}
```
