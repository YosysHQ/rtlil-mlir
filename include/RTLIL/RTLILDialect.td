#ifndef RTLIL_DIALECT
#define RTLIL_DIALECT

include "mlir/IR/OpBase.td"

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

def CellOp : RTLIL_Op<"cell"> {
  let arguments = (ins StrAttr:$instanceName,
                       StrAttr:$moduleName // TODO back to FlatSymbolRefAttr
                    //    Variadic<AnyType>:$inputs,
                    //    StrArrayAttr:$argNames, StrArrayAttr:$resultNames,
                    //    ParamDeclArrayAttr:$parameters,
                    //    OptionalAttr<InnerSymAttr>:$inner_sym
                    );
  let results = (outs Variadic<AnyType>:$results);
}
#endif // RTLIL_DIALECT
