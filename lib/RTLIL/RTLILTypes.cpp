#include "RTLIL/RTLILTypes.h"
#include "RTLIL/RTLILOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

namespace rtlil {
bool isMValueType(mlir::Type type) {
    if (isa<MValueType>(type))
        return true;
    return false;
}
} // namespace rtlil
