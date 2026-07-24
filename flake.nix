# This is a simple nix flake which provides a dev shell for "takecell" development on on NixOS.
# You can either use `nix develop` to activate it manuall, or [`direnv`] to activate it automatically.
#
# [`direnv`]: https://github.com/nix-community/nix-direnv

{
  description = "dev shell for `takecell`";

  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs =
    { self, nixpkgs, rust-overlay }:
    let
      inherit (nixpkgs) lib;
      forEachSystem = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = (nixpkgs.legacyPackages.${system}.extend (import rust-overlay)).pkgs;
        in
        {
          default = pkgs.mkShell {
            name = "takecell-dev-shell";
            packages = [
              (pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml)
            ];
          };
        }
      );
    };
}
