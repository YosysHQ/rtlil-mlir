add_mlir_dialect(RTLILOps rtlil)
# add_mlir_doc(RTLILDialect RTLILDialect RTLIL/ -gen-dialect-doc)
# add_mlir_doc(RTLILOps RTLILOps RTLIL/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS RTLILOps.td)
mlir_tablegen(RTLILAttrDefs.h.inc -gen-attrdef-decls
-attrdefs-dialect=rtlil)
mlir_tablegen(RTLILAttrDefs.cpp.inc -gen-attrdef-defs
-attrdefs-dialect=rtlil)
add_public_tablegen_target(MLIRRTLILAttrDefsIncGen)

set(LLVM_TARGET_DEFINITIONS RTLILOps.td)
mlir_tablegen(RTLILEnums.h.inc -gen-enum-decls)
mlir_tablegen(RTLILEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRRTLILEnumDefsIncGen)
