#!/usr/bin/env bash
set -x
mkdir -p build
pushd build
cmake -G Ninja .. -DLLVM_DIR=$(llvm-config --cmakedir) -DMLIR_DIR=$(llvm-config --cmakedir)/../mlir
popd
