// RUN: rtlil-opt %s | FileCheck %s

// CHECK: define void @main()
func.func @main() {
    %0 = "rtlil.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    "rtlil.print"(%0) : (tensor<2x3xf64>) -> ()
    return
}
