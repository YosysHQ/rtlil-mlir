configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(RTLIL_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        rtlil-emit
        )

add_lit_testsuite(check-rtlil "Running the rtlil regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RTLIL_TEST_DEPENDS}
        )
set_target_properties(check-rtlil PROPERTIES FOLDER "Tests")

add_lit_testsuites(RTLIL ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RTLIL_TEST_DEPENDS})
