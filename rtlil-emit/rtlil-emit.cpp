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

#include "RTLIL/RTLILDialect.h"
#include "RTLIL/RTLILPasses.h"

// Malarkey - I think this is just not generally exposed?
// TODO move elsewhere?
namespace mlir {
class ModuleOp;
}; // namespace mlir

#include "kernel/yosys.h"
USING_YOSYS_NAMESPACE

// TODO This is awful naming, what's the nomenclature?
// From the MLIR perspective this is an importer, but it's a yosys backend
class MLIRifier {
  mlir::MLIRContext &ctx;
  mlir::OpBuilder b;
  mlir::Location loc;
  // This is really stupid actually
  llvm::DenseMap<RTLIL::Wire *, rtlil::WireOp> wiremap;

public:
  MLIRifier(mlir::MLIRContext &context)
      : ctx(context), b(mlir::OpBuilder(&context)), loc(b.getUnknownLoc()) {}

  rtlil::WireOp convert_wire(RTLIL::Wire *wire) {
    log_assert(!wiremap.contains(wire));
    // TODO custom return type
    return wiremap[wire] = b.create<rtlil::WireOp>(
               loc, mlir::IntegerType::get(&ctx, 32),
               mlir::StringAttr::get(&ctx, wire->name.c_str()),
               mlir::IntegerAttr::get(b.getI32Type(), wire->width),
               mlir::IntegerAttr::get(b.getI32Type(), wire->start_offset),
               mlir::IntegerAttr::get(b.getI32Type(), wire->port_id),
               mlir::BoolAttr::get(&ctx, wire->port_input),
               mlir::BoolAttr::get(&ctx, wire->port_output),
               mlir::BoolAttr::get(&ctx, wire->upto),
               mlir::BoolAttr::get(&ctx, wire->is_signed));
  }

  rtlil::ConstOp convert_const(RTLIL::Const *c) {
    std::vector<mlir::Attribute> const_bits;
    for (auto bit : c->bits())
      const_bits.push_back(
          rtlil::StateEnumAttr::get(&ctx, (rtlil::StateEnum)bit));
    mlir::ArrayAttr aa = b.getArrayAttr(const_bits);
    // TODO custom return type
    return b.create<rtlil::ConstOp>(loc, mlir::IntegerType::get(&ctx, 32),
                                    (mlir::ArrayAttr)aa);
  }

  rtlil::CellOp convert_cell(RTLIL::Cell *cell) {
    // is this smart?
    std::vector<mlir::Value> connections;
    std::vector<mlir::Attribute> parameters;
    std::vector<mlir::Attribute> signature;
    for (auto [port, sigspec] : cell->connections()) {
      auto portname = std::string(port.c_str());
      auto portattr = mlir::StringAttr::get(&ctx, portname);
      if (sigspec.is_fully_const()) {
        std::vector<mlir::Attribute> const_bits;
        RTLIL::Const domain_const = sigspec.as_const();
        rtlil::ConstOp c = convert_const(&domain_const);
        log_assert(c.verify().succeeded());
        signature.push_back(portattr);
        connections.push_back(c.getResult());
      } else if (sigspec.is_wire()) {
        signature.push_back(portattr);
        RTLIL::Wire *wire = sigspec.as_wire();
        log_assert(wiremap.contains(wire));
        connections.push_back(wiremap[wire].getResult());
      } else {
        log_error("Found SigSpec that isn't a constant or full wire "
                  "connection, did you run splice?");
      }
    }
    for (auto [param, value] : cell->parameters) {
      log_assert(value.is_fully_def());
      auto paramname = mlir::StringAttr::get(&ctx, param.c_str());
      mlir::Type itype = mlir::IntegerType::get(&ctx, value.size());
      auto paramvalue = mlir::IntegerAttr::get(itype, value.as_int());
      auto parameter = rtlil::ParameterAttr::get(&ctx, paramname, paramvalue);
      parameters.push_back(parameter);
    }
    mlir::ArrayAttr cellparameters = b.getArrayAttr(parameters);
    mlir::StringAttr cellname = mlir::StringAttr::get(&ctx, cell->name.c_str());
    mlir::StringAttr celltype = mlir::StringAttr::get(&ctx, cell->type.c_str());
    mlir::ArrayAttr cellsignature = b.getArrayAttr(signature);
    return b.create<rtlil::CellOp>(loc, cellname, celltype, connections,
                                   cellsignature, cellparameters);
  }

  rtlil::WConnectionOp convert_connection(RTLIL::SigSig ss) {
    SigSpec sslhs, ssrhs;
    std::tie(sslhs, ssrhs) = ss;
    RTLIL::Wire *lhs, *rhs;
    log_assert(sslhs.is_wire() && ssrhs.is_wire());
    std::tie(lhs, rhs) = std::make_tuple(sslhs.as_wire(), ssrhs.as_wire());
    log_assert(wiremap.contains(lhs) && wiremap.contains(rhs));
    return b.create<rtlil::WConnectionOp>(loc, wiremap[lhs], wiremap[rhs]);
  }

  mlir::ModuleOp convert_module(RTLIL::Module *mod) {
    mlir::ModuleOp moduleOp(mlir::ModuleOp::create(loc, mod->name.c_str()));
    b.setInsertionPointToStart(moduleOp.getBody());
    for (auto wire : mod->wires()) {
      log_assert(convert_wire(wire).verify().succeeded());
    }
    for (auto cell : mod->cells()) {
      log_assert(convert_cell(cell).verify().succeeded());
    }
    for (auto conn : mod->connections()) {
      log_assert(convert_connection(conn).verify().succeeded());
    }
    return moduleOp;
  }
};

struct MlirBackend : public Backend {
  MlirBackend() : Backend("mlir", "Write design as MLIR RTLIL dialect") {}
  // TODO help
  void execute(std::ostream *&f, std::string filename,
               std::vector<std::string> args, RTLIL::Design *design) override {
    log_header(design, "Executing MLIR backend.\n");
    size_t argidx;
    for (argidx = 1; argidx < args.size(); argidx++) {
    }
    extra_args(f, filename, args, argidx);
    llvm::raw_os_ostream osos(*f);
    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<rtlil::RTLILDialect>();
    MLIRifier mlirifier(ctx);
    for (auto mod : design->selected_modules())
      mlirifier.convert_module(mod).print(osos);
  }
} MyPass;
