mkdir build && cd build
cmake -G Ninja .. -DLLVM_DIR=(llvm-config --cmakedir) -DMLIR_DIR=(llvm-config --cmakedir)/../mlir