#ifndef RTLIL_OPS
#define RTLIL_OPS

include "RTLILDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

//===----------------------------------------------------------------------===//
// RTLIL dialect definition.
//===----------------------------------------------------------------------===//

def RTLIL_Dialect : Dialect {
    let name = "rtlil";
    let summary = "A rtlil out-of-tree MLIR dialect.";
    let description = [{
        This dialect is minimal example to implement rtlil-world kind of sample code
        for MLIR.
    }];
    let cppNamespace = "::rtlil";
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base rtlil operation definition.
//===----------------------------------------------------------------------===//

class RTLIL_Op<string mnemonic, list<Trait> traits = []> :
        Op<RTLIL_Dialect, mnemonic, traits>;

class RTLIL_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<RTLIL_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

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

// def ArrayRefIntParam : TypeParameter<"::llvm::ArrayRef<int>", "Array of int"> {
//   let allocator = "$_dst = $_allocator.copyInto($_self);";
// }

// def StrAttributeAttr : RTLIL_Attr<"StrAttribute", "str_attr"> {
//   let summary = "a string Attribute";
//   // let mnemonic = "attribute";
//   let parameters = (ins
//     "::mlir::StringAttr":$name,
//     ArrayRefIntParam:$value
//   );
// }

// Ports

def ModportStruct : RTLIL_Attr<"ModportStruct", "mod_port"> {
  let assemblyFormat = "`<` struct(params) `>`";
  let parameters = (ins
                    "StringAttr":$signal);
}

def CellOp : RTLIL_Op<"cell"> {
  let arguments = (ins StrAttr:$instanceName,
                       StrAttr:$moduleName,
                       TypedArrayAttrBase<ModportStruct, "array of modport structs">:$ports
  );
}
                      //  , // TODO back to FlatSymbolRefAttr?
  // let results = (outs Variadic<AnyType>:$results);

class MyDialect_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RTLIL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Here is a simple definition of an "integer" type, with a width parameter.
def My_IntegerType : MyDialect_Type<"Integer", "int"> {
  let summary = "Integer type with arbitrary precision up to a fixed limit";
  let description = [{
    Integer types have a designated bit width.
  }];
  /// Here we defined a single parameter for the type, which is the bitwidth.
  let parameters = (ins "unsigned":$width);

  /// Here we define the textual format of the type declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the type to be output as, for example:
  ///
  ///    !my.int<10> // a 10-bit integer.
  ///
  let assemblyFormat = "`<` $width `>`";

  /// Indicate that our type will add additional verification to the parameters.
  let genVerifyDecl = 1;
}

#endif // RTLIL_OPS
