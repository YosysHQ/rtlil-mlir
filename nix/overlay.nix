final: prev:
{
  # Add an alias here can help future migration
  llvmPkgs = final.llvmPackages_17;
  # Use clang instead of gcc to compile, to avoid gcc13 miscompile issue.
  rtlil-llvm = final.callPackage ./rtlil-llvm.nix { stdenv = final.llvmPkgs.stdenv; };
  # rtlil-mlir = final.callPackage ./rtlil-mlir.nix { stdenv = final.llvmPkgs.stdenv; };
}