// RUN: rtlil-opt %s | FileCheck %s

// CHECK: @rtlil_word_string = internal constant [16 x i8] c"RTLIL, World! \0A\00"
// CHECK: define void @main()
func.func @main() {
    // CHECK: %{{.*}} = call i32 (ptr, ...) @printf(ptr @rtlil_word_string)
    "rtlil.world"() : () -> ()
    return
}
