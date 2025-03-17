#ifndef RTLIL_OPS
#define RTLIL_OPS

include "RTLILDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : RTLIL_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "rtlil.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

//  let parser = [{ return ::parseConstantOp(parser, result); }];
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}

def PrintOp : RTLIL_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def WorldOp : RTLIL_Op<"world", [Pure]> {
    let summary = "print RTLIL, World";
    let description = [{
        The "world" operation prints "RTLIL, World", and produces
        no results.
    }];
}

#endif // RTLIL_OPS
