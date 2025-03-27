final: prev:
rec {
  clang-stdenv = prev.llvmPackages_latest.stdenv;
  # rtlil-llvm = final.callPackage ./rtlil-llvm.nix { stdenv = final.clang-stdenv; };
  clang-yosys = (final.pkgs.yosys.override {
    stdenv = final.clang-stdenv;
    enablePython = false;
    gtkwave = null;
  }).overrideAttrs (finalAttrs: previousAttrs: {
    doCheck = false;
    makeFlags = previousAttrs.makeFlags ++ [
      "SMALL=1"
      "ENABLE_ABC=0"
    ];
  });
}