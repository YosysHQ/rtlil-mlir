final: prev:
{
  # Add an alias here can help future migration
  llvmPkgs = final.llvmPackages_17;
  # Use clang instead of gcc to compile, to avoid gcc13 miscompile issue.
  rtlil-llvm = final.callPackage ./rtlil-llvm.nix { stdenv = final.llvmPkgs.stdenv; };
  clang-yosys = (final.pkgs.yosys.override { stdenv = final.llvmPkgs.stdenv; enablePython = false; gtkwave = null; }).overrideAttrs { doCheck = false; };
  # rtlil-mlir = final.callPackage ./rtlil-mlir.nix { stdenv = final.llvmPkgs.stdenv; };
}