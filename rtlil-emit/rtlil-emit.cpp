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

mlir::ModuleOp sayRTLIL(mlir::MLIRContext& ctx) {
    auto builder = mlir::OpBuilder(&ctx);
    auto nowhere = builder.getUnknownLoc();
    mlir::ModuleOp moduleOp(mlir::ModuleOp::create(nowhere));
    builder.setInsertionPointToStart(moduleOp.getBody());
    mlir::TypeRange newOperands;
    auto portname = mlir::StringAttr::get(&ctx, "somesig");
    auto wirename = mlir::StringAttr::get(&ctx, "somewire");
    auto connection = rtlil::CConnectionAttr::get(&ctx, portname, wirename);
    // auto port = rtlil::ModportStructAttr::get(&ctx, portname);
    mlir::ArrayAttr cellconnections = builder.getArrayAttr({connection});

    auto paramname = mlir::StringAttr::get(&ctx, "someparam");
    mlir::Type i64Ty = mlir::IntegerType::get(&ctx, 64);
    auto paramvalue = mlir::IntegerAttr::get(i64Ty, 123);
    auto param = rtlil::ParameterAttr::get(&ctx, paramname, paramvalue);
    // auto port = rtlil::ModportStructAttr::get(&ctx, portname);
    mlir::ArrayAttr params = builder.getArrayAttr({param});
    // rtlil::ModportStructArrayAttr::get();
    (void)builder.create<rtlil::CellOp>(nowhere, newOperands, "foo", "bar", cellconnections, params);
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
        mlir::MLIRContext ctx;
        ctx.getOrLoadDialect<rtlil::RTLILDialect>();
        auto mop = sayRTLIL(ctx);
        mop.print(llvm::outs());
        for (auto mod : design->selected_modules())
            emit_module();
    }
} MyPass;
