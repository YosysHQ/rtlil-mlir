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

# yosys-config --exec --cxx --cxxflags --ldflags \
# -o my_cmd.so -shared my_cmd.cc --ldlibs

execute_process(
    COMMAND yosys-config --ldflags
    OUTPUT_VARIABLE YOSYS_LDFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
    COMMAND yosys-config --cxxflags
    OUTPUT_VARIABLE YOSYS_CXXFLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
separate_arguments(YOSYS_LDFLAGS_LIST UNIX_COMMAND "${YOSYS_LDFLAGS}")
separate_arguments(YOSYS_CXXFLAGS_LIST UNIX_COMMAND "${YOSYS_CXXFLAGS}")
add_compile_options(${YOSYS_CXXFLAGS_LIST})
set(LLVM_REQUIRES_EH ON)
# add_llvm_executable(rtlil-opt rtlil-opt.cpp)
add_llvm_library(rtlil-emit rtlil-emit.cpp SHARED LINK_LIBS ${YOSYS_LDFLAGS_LIST} -Wl,-z,undefs)
# TODO what are these
# llvm_update_compile_flags(rtlil-opt)
llvm_update_compile_flags(rtlil-emit)
# target_link_libraries(rtlil-opt PRIVATE ${LIBS})
target_link_libraries(rtlil-emit PRIVATE ${LIBS})

# -lstdc++ -lm -lrt -lpython3.12 -ldl -L/nix/store/h8wi6kwl1cywxr3f22836vgrrb11fvlb-libxcrypt-4.4.38/lib -lm -lboost_python312 -lboost_system -lboost_filesystem -lreadline -L/nix/store/ixvjg40bxg59mmzjjgs6f2nf2hbal1y4-libffi-3.4.6/lib/../lib64 -lffi -ldl -lz -L/nix/store/krxc78nx57zn9bggwp4slqif88qc8xd1-tcl-8.6.15/lib -ltcl8.6 -ltclstub8.6


# mlir_check_all_link_libraries(rtlil-opt)
mlir_check_all_link_libraries(rtlil-emit)
