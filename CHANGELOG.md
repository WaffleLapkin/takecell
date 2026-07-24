## Unreleased

## 0.1.2

- Make `TakeOwnCell<T>` covariant in `T` (https://github.com/WaffleLapkin/takecell/pull/7)
- Set MSRV to `1.96`

> [!WARNING]
> This crate is unsound when used with **rust < `1.96`** if all of the following is true:
>
> 1. `TakeOwnCell<T>` is used
> 2. `T` contains a `Box<_>` (e.g. `T = Box<_>`, `T = struct { field: Box<_>}`, etc)
> 3. `TakeOwnCell<T>` is moved after being taken
>
> See 
> - https://github.com/WaffleLapkin/takecell/issues/3
> - https://github.com/WaffleLapkin/takecell/issues/5
> 
> *waffle*: In `1.96.0` I fixed[^1][^2][^3] the issue, so starting from that version `takecell` is sound again :3

[^1]: https://github.com/rust-lang/rust/pull/150447
[^2]: https://github.com/rust-lang/rust/pull/150446
[^3]: https://github.com/rust-lang/rust/pull/155750

## 0.1.1

- Use weaker atomic orderings

## 0.1.0

- Initial release
