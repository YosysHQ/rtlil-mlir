# MLIR RTLIL Dialect

Highly early stage experimentation in representing arbitrary [Yosys](https://github.com/YosysHQ/yosys/) [RTLIL](https://yosyshq.readthedocs.io/projects/yosys/en/latest/yosys_internals/formats/rtlil_rep.html) designs in an [MLIR](https://mlir.llvm.org/) dialect

The ultimate goal is to build a conversion between existing [CIRCT](https://circt.llvm.org/) core dialects and new dialects to build new workflows for formal verification and beyond

## How to build

Linux or MacOS is assumed

Recommended approach:

- Install nix with flakes enabled
- `nix develop` (this will compile LLVM)
- `./steps.sh`
- `ninja -C build`

Manual approach:

- Build Yosys from source
- Build LLVM from source with MLIR enabled
- Make sure yosys, yosys-config and llvm-config are all in PATH
- `./steps.sh`
- `ninja -C build`

## Demo

- `cd examples/cpu`
- `yosys -m ../../build/lib/librtlil-emit.so demo.ys`

Look at `cpu.mlir` and the diff between `before.il` and `after.il`

## Licensing

This code base is dual-licensed for compatibility with both the Yosys and LLVM ecosystems under Apache 2 with LLVM exceptions as well as ISC. See `LICENSE.llvm` and `LICENSE.isc`. Copyright (C) 2025 - 2025 YosysHQ GmbH
