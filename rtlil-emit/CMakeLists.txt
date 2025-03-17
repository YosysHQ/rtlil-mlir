get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRRTLIL
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTargetLLVMIRExport
        MLIRTransforms
        MLIROptLib
        )
# add_llvm_executable(rtlil-opt rtlil-opt.cpp)
add_llvm_executable(rtlil-emit rtlil-emit.cpp)
# TODO what are these
# llvm_update_compile_flags(rtlil-opt)
llvm_update_compile_flags(rtlil-emit)
# target_link_libraries(rtlil-opt PRIVATE ${LIBS})
target_link_libraries(rtlil-emit PRIVATE ${LIBS})

# mlir_check_all_link_libraries(rtlil-opt)
mlir_check_all_link_libraries(rtlil-emit)
