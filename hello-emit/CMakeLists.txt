get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRHello
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
# add_llvm_executable(hello-opt hello-opt.cpp)
add_llvm_executable(hello-emit hello-emit.cpp)
# TODO what are these
# llvm_update_compile_flags(hello-opt)
llvm_update_compile_flags(hello-emit)
# target_link_libraries(hello-opt PRIVATE ${LIBS})
target_link_libraries(hello-emit PRIVATE ${LIBS})

# mlir_check_all_link_libraries(hello-opt)
mlir_check_all_link_libraries(hello-emit)
