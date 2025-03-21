add_mlir_dialect_library(MLIRRTLIL
		RTLILDialect.cpp
		RTLILOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/RTLIL

        DEPENDS
        MLIRRTLILOpsIncGen
        RTLILAttrDefsIncGen
        MLIRRTLILAttrDefsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
