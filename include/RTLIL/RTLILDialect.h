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

#ifndef RTLIL_RTLILDIALECT_H
#define RTLIL_RTLILDIALECT_H

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "RTLIL/RTLILOps.h.inc"
#include "RTLIL/RTLILOpsDialect.h.inc"
#define GET_TYPEDEF_CLASSES
#include "RTLIL/RTLILOpsTypes.h.inc"
#define GET_ATTRDEF_CLASSES
#include "RTLIL/RTLILAttrDefs.h.inc"

#endif // RTLIL_RTLILDIALECT_H
