// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RTLIL/RTLILDialect.h"
#include "RTLIL/RTLILOps.h"

using namespace mlir;
using namespace rtlil;

//===----------------------------------------------------------------------===//
// RTLIL dialect.
//===----------------------------------------------------------------------===//

#include "RTLIL/RTLILOpsDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "RTLIL/RTLILAttrDefs.cpp.inc"
#undef GET_ATTRDEF_CLASSES

#define GET_TYPEDEF_CLASSES
#include "RTLIL/RTLILOpsTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

void RTLILDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RTLIL/RTLILOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "RTLIL/RTLILAttrDefs.cpp.inc"
    >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "RTLIL/RTLILOpsTypes.cpp.inc"
    >();
}

void rtlil::ConstantOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  rtlil::ConstantOp::build(builder, state, dataType, dataAttribute);
}

mlir::Operation *RTLILDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<rtlil::ConstantOp>(
      loc, type, mlir::cast<mlir::DenseElementsAttr>(value));
}