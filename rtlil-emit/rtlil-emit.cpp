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

#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"

// Malarkey
#define GET_OP_CLASSES

#include "RTLIL/RTLILDialect.h"
#include "RTLIL/RTLILPasses.h"

// Malarkey - I think this is just not generally exposed?
namespace mlir {
    class ModuleOp;
};

mlir::ModuleOp sayRTLIL(mlir::MLIRContext& context) {
    auto builder = mlir::OpBuilder(&context);
    mlir::ModuleOp moduleOp(mlir::ModuleOp::create(builder.getUnknownLoc()));
    builder.setInsertionPointToStart(moduleOp.getBody());
    mlir::Value op = builder.create<rtlil::ConstantOp>(builder.getUnknownLoc(), 1.0);
    return moduleOp;
}
// int main(int argc, char **argv) {
//     mlir::MLIRContext context;
//     context.getOrLoadDialect<rtlil::RTLILDialect>();
//     auto mop = sayRTLIL(context);
//     mop.print(llvm::outs());

//     return 0;
// }

#include "kernel/yosys.h"
USING_YOSYS_NAMESPACE

struct MyPass : public Pass {
    MyPass() : Pass("my_cmd", "just a simple test") { }
    void execute(std::vector<std::string> args, RTLIL::Design *design) override
    {
        log("Arguments to my_cmd:\n");
        for (auto &arg : args)
            log("  %s\n", arg.c_str());

        log("Modules in current design:\n");
        for (auto mod : design->modules())
            log("  %s (%d wires, %d cells)\n", log_id(mod),
                    GetSize(mod->wires()), GetSize(mod->cells()));
    }
} MyPass;
