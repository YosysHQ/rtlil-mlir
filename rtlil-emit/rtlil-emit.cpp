#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

// Malarkey
#define GET_OP_CLASSES

#include "RTLIL/RTLILDialect.h"
#include "RTLIL/RTLILPasses.h"

// Malarkey - I think this is just not generally exposed?
// TODO move elsewhere?
namespace mlir {
    class ModuleOp;
    class CellOp;
    class ModuleType;
};

mlir::ModuleOp sayRTLIL(mlir::MLIRContext& context) {
    auto builder = mlir::OpBuilder(&context);
    auto nowhere = builder.getUnknownLoc();
    mlir::ModuleOp moduleOp(mlir::ModuleOp::create(nowhere));
    builder.setInsertionPointToStart(moduleOp.getBody());
    mlir::Value op = builder.create<rtlil::ConstantOp>(nowhere, 1.0);
    mlir::TypeRange newOperands;
    rtlil::ModportStruct port = "foosig";
    mlir::ArrayAttr modports = builder.getArrayAttr({port});
    // rtlil::ModportStructArrayAttr::get();
    // (void)builder.create<rtlil::CellOp>(nowhere, newOperands, "foo", "bar");
    return moduleOp;
}

#include "kernel/yosys.h"
USING_YOSYS_NAMESPACE

void emit_module () {} // TODO

struct MyPass : public Pass {
    MyPass() : Pass("write_mlir", "Write design as MLIR RTLIL dialect") { }
    void execute(std::vector<std::string> args, RTLIL::Design *design) override
    {
        mlir::MLIRContext context;
        context.getOrLoadDialect<rtlil::RTLILDialect>();
        auto mop = sayRTLIL(context);
        mop.print(llvm::outs());
        for (auto mod : design->selected_modules())
            emit_module();
    }
} MyPass;
