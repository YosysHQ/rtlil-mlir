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
include "mlir/IR/CommonAttrConstraints.td"

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
    // let hasConstantMaterializer = 1;
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

// class RTLIL_Type<string name, string typeMnemonic, list<Trait> traits = []>
//     : DialectType<RTLIL_Dialect, name, traits> {
//   let mnemonic = typeMnemonic;
// }

// Parameters

def ParamAttr : RTLIL_Attr<"Parameter", "param"> {
  let parameters = (ins
                    "::mlir::StringAttr":$name,
                    "::mlir::IntegerAttr":$value);
  let assemblyFormat = "$name $value ";
}

def ParametersAttr
  : TypedArrayAttrBase<ParamAttr, "port to wire connections">;

// Connectivity

class I8EnumAttrCase<string sym, int val, string str = sym>
    : IntEnumAttrCaseBase<I8, sym, str, val>;
def StateS0: I8EnumAttrCase<"S0", 0>;
def StateS1: I8EnumAttrCase<"S1", 1>;
def StateSx: I8EnumAttrCase<"Sx", 2>;
def StateSz: I8EnumAttrCase<"Sz", 3>;
def StateSa: I8EnumAttrCase<"Sa", 4>;
def StateSm: I8EnumAttrCase<"Sm", 5>;

class I8EnumAttr<string name, string summary, list<I8EnumAttrCase> cases> :
    IntEnumAttr<I8, name, summary, cases> {
  let underlyingType = "uint8_t";
}

def StateEnumAttr: I8EnumAttr<"StateEnum", "RTLIL::State model",
    [StateS0, StateS1, StateSx, StateSz, StateSa, StateSm]> {
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
  let cppNamespace = "::rtlil";
}

def RTLILValue : DialectType<RTLIL_Dialect, CPred<"true">, "value"> {}

def ConstAttr
  : TypedArrayAttrBase<StateEnumAttr, "constant multi-valued bitvec">;

def ConstOp : RTLIL_Op<"const"> {
  let arguments = (ins ConstAttr:$value);
  let results = (outs RTLILValue:$result);
}

// TODO Is I32 correct?
def WireOp : RTLIL_Op<"wire"> {
  let arguments = (ins StrAttr:$name
                       ,I32Attr:$width
                       ,I32Attr:$start_offset
                       ,I32Attr:$port_id
                       ,DefaultValuedAttr<BoolAttr, "false">:$port_input
                       ,DefaultValuedAttr<BoolAttr, "false">:$port_output
                       ,DefaultValuedAttr<BoolAttr, "false">:$upto
                       ,BoolAttr:$is_signed
  );
  let results = (outs RTLILValue:$result);
}

def CellOp : RTLIL_Op<"cell"> {
  let arguments = (ins StrAttr:$name
                       ,StrAttr:$type
                       ,Variadic<AnyType>:$connections
                       ,TypedArrayAttrBase<StrAttr, "port names">:$ports
                       ,ParametersAttr:$parameters
  );
}

// def WConnectionOp : RTLIL_Op<"wconnection"> {
//   let arguments = (ins WireOp:$this, WireOp:$that);
// }
                      //  , // TODO back to FlatSymbolRefAttr?
  // let results = (outs Variadic<AnyType>:$results);

#endif // RTLIL_OPS
