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
include "mlir/IR/BuiltinTypes.td"

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
    // let useDefaultTypePrinterParser = 1;
    let useDefaultAttributePrinterParser = 1;
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

def ParamAttr : RTLIL_Attr<"Parameter", "param"> {
  let parameters = (ins
                    "::mlir::StringAttr":$name,
                    "::mlir::IntegerAttr":$value);
  let assemblyFormat = "$name $value ";
}

def ParametersAttr
  : TypedArrayAttrBase<ParamAttr, "port to wire connections">;

def CConnAttr : RTLIL_Attr<"CConnection", "conn"> {
  let parameters = (ins
                    "::mlir::StringAttr":$port,
                    "::mlir::StringAttr":$signal);
  let assemblyFormat = " $port $signal ";
}

def CConnectionsAttr
  : TypedArrayAttrBase<CConnAttr, "port to wire connections">;

// TODO WConnections

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

// def ModportStruct : RTLIL_Attr<"ModportStruct", "mod_port"> {
//   let assemblyFormat = "`<` `modport` `:` $signal `>`";
//   let parameters = (ins
//                     "::mlir::StringAttr":$signal);
// }

def CellOp : RTLIL_Op<"cell"> {
  let arguments = (ins StrAttr:$instanceName
                       ,StrAttr:$moduleName
                       ,CConnectionsAttr:$connections
                       ,ParametersAttr:$parameters
  );
}
                      //  , // TODO back to FlatSymbolRefAttr?
  // let results = (outs Variadic<AnyType>:$results);

#endif // RTLIL_OPS
