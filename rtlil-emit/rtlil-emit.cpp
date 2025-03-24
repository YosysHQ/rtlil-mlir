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
#include "llvm/Support/raw_os_ostream.h"

// Malarkey
// #define GET_OP_CLASSES

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
    (void)builder.create<rtlil::CellOp>(nowhere, "foo", "bar", cellconnections, params);
    // (void)builder.create<rtlil::CellOp>(nowhere, newOperands, "foo", "bar");
    return moduleOp;
}

#include "kernel/yosys.h"
USING_YOSYS_NAMESPACE

// TODO This is awful naming, what's the nomenclature?
// From the MLIR perspective this is an importer, but it's a yosys backend
class MLIRifier {
    mlir::MLIRContext& ctx;
    mlir::OpBuilder b;
    mlir::Location loc;
public:
    MLIRifier(mlir::MLIRContext& context) : ctx(context), b(mlir::OpBuilder(&context)), loc(b.getUnknownLoc()) {}
    rtlil::CellOp convert_cell(RTLIL::Cell* cell) {
        // is this smart?
        std::vector<mlir::Attribute> connections;
        std::vector<mlir::Attribute> parameters;
        for (auto [port, sigspec] : cell->connections()) {
            log_assert(sigspec.is_wire());
            auto portname = mlir::StringAttr::get(&ctx, port.c_str());
            auto wirename = mlir::StringAttr::get(&ctx, sigspec.as_wire()->name.c_str());
            auto connection = rtlil::CConnectionAttr::get(&ctx, portname, wirename);
            connections.push_back(connection);
        }
        for (auto [param, value] : cell->parameters) {
            log_assert(value.is_fully_def());
            auto paramname = mlir::StringAttr::get(&ctx, param.c_str());
            mlir::Type itype = mlir::IntegerType::get(&ctx, value.size());
            auto paramvalue = mlir::IntegerAttr::get(itype, value.as_int());
            auto parameter = rtlil::ParameterAttr::get(&ctx, paramname, paramvalue);
            parameters.push_back(parameter);
        }
        mlir::ArrayAttr cellconnections = b.getArrayAttr(connections);
        mlir::ArrayAttr cellparameters = b.getArrayAttr(parameters);
        mlir::StringAttr cellname = mlir::StringAttr::get(&ctx, cell->name.c_str());
        mlir::StringAttr celltype = mlir::StringAttr::get(&ctx, cell->type.c_str());
        return b.create<rtlil::CellOp>(loc,
            cellname,
            celltype,
            cellconnections,
            cellparameters);
    }

    mlir::ModuleOp convert_module(RTLIL::Module* mod) {
        mlir::ModuleOp moduleOp(mlir::ModuleOp::create(loc));
        b.setInsertionPointToStart(moduleOp.getBody());
        for (auto cell : mod->cells()) {
            (void)convert_cell(cell);
        }
        return moduleOp;
    }
};

struct MlirBackend : public Backend {
    MlirBackend() : Backend("mlir", "Write design as MLIR RTLIL dialect") { }
    // TODO help
    void execute(std::ostream *&f, std::string filename, std::vector<std::string> args, RTLIL::Design *design) override
    {
        log_header(design, "Executing MLIR backend.\n");
		size_t argidx;
		for (argidx = 1; argidx < args.size(); argidx++) {}
        extra_args(f, filename, args, argidx);
        llvm::raw_os_ostream osos(*f);
        mlir::MLIRContext ctx;
        ctx.getOrLoadDialect<rtlil::RTLILDialect>();
        MLIRifier mlirifier(ctx);
        for (auto mod : design->selected_modules())
            mlirifier.convert_module(mod).print(osos);
    }
} MyPass;
