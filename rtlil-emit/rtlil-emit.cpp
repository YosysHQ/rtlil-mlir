#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/SourceMgr.h"
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
    log_debug("converting wire %s\n", log_id(wire));
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
    log_debug("converting const %s\n", log_const(*c));
    std::vector<mlir::Attribute> const_bits;
    for (auto bit : c->bits())
      const_bits.push_back(
          rtlil::StateEnumAttr::get(&ctx, (rtlil::StateEnum)bit));
    mlir::ArrayAttr aa = b.getArrayAttr(const_bits);
    // TODO custom return type
    return b.create<rtlil::ConstOp>(loc, mlir::IntegerType::get(&ctx, 32),
                                    (mlir::ArrayAttr)aa);
  }

  mlir::Value convert_sigspec(RTLIL::SigSpec sigspec) {
    log_debug("converting sigspec %s\n", log_signal(sigspec));
    if (sigspec.is_fully_const()) {
      std::vector<mlir::Attribute> const_bits;
      RTLIL::Const domain_const = sigspec.as_const();
      rtlil::ConstOp c = convert_const(&domain_const);
      log_assert(mlir::verify(c).succeeded());
      return c.getResult();
    } else if (sigspec.is_wire()) {
      RTLIL::Wire *wire = sigspec.as_wire();
      log_assert(wiremap.contains(wire));
      return wiremap[wire].getResult();
    } else {
      log_error("Found SigSpec that isn't a constant or full wire "
                "connection, did you run splice?\n");
    }
  }

  rtlil::CellOp convert_cell(RTLIL::Cell *cell) {
    // is this smart?
    std::vector<mlir::Value> connections;
    std::vector<mlir::Attribute> parameters;
    std::vector<mlir::Attribute> signature;
    log_debug("converting cell %s\n", log_id(cell));
    for (auto [port, sigspec] : cell->connections()) {
      auto val = convert_sigspec(sigspec);
      connections.push_back(val);
      auto portname = std::string(port.c_str());
      auto portattr = mlir::StringAttr::get(&ctx, portname);
      signature.push_back(portattr);
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
    log_debug("converting connection %s %s\n", log_signal(ss.first),
              log_signal(ss.second));
    return b.create<rtlil::WConnectionOp>(loc, convert_sigspec(ss.first),
                                          convert_sigspec(ss.second));
  }

  mlir::ModuleOp convert_module(RTLIL::Module *mod) {
    log_debug("converting module %s\n", log_id(mod));
    mlir::ModuleOp moduleOp(mlir::ModuleOp::create(loc, mod->name.c_str()));
    b.setInsertionPointToStart(moduleOp.getBody());
    for (auto wire : mod->wires()) {
      log_assert(mlir::verify(convert_wire(wire)).succeeded());
    }
    for (auto cell : mod->cells()) {
      log_assert(mlir::verify(convert_cell(cell)).succeeded());
    }
    for (auto conn : mod->connections()) {
      log_assert(mlir::verify(convert_connection(conn)).succeeded());
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
      break;
    }
    extra_args(f, filename, args, argidx);
    llvm::raw_os_ostream osos(*f);
    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<rtlil::RTLILDialect>();
    MLIRifier convertor(ctx);
    for (auto mod : design->selected_modules())
      convertor.convert_module(mod).print(osos);
  }
} MlirBackend;

class RTLILifier {
  RTLIL::Design *design;

public:
  RTLILifier(RTLIL::Design* d) : design(d) {}
  void convert_wire(RTLIL::Module* mod, rtlil::WireOp op) {
    auto name = op->getAttr("name").cast<mlir::StringAttr>().str();
    // mlir::IntegerType::get(&ctx, 32),
    RTLIL::Wire* w = mod->addWire(name);
    w->width = op->getAttr("width").cast<mlir::IntegerAttr>().getInt();
    w->start_offset = op->getAttr("start_offset").cast<mlir::IntegerAttr>().getInt();
    w->port_id = op->getAttr("port_id").cast<mlir::IntegerAttr>().getInt();
    w->port_input = op->getAttr("port_input").cast<mlir::BoolAttr>().getValue();
    w->port_output = op->getAttr("port_output").cast<mlir::BoolAttr>().getValue();
    w->upto = op->getAttr("upto").cast<mlir::BoolAttr>().getValue();
    w->is_signed = op->getAttr("is_signed").cast<mlir::BoolAttr>().getValue();
  }
  void convert_cell(RTLIL::Module* mod, rtlil::CellOp op) {
  }
  void convert_connection(RTLIL::Module* mod, rtlil::WConnectionOp op) {
  }
  void convert_const(RTLIL::Module* mod, rtlil::ConstOp op) {
  }
  void convert_module(mlir::ModuleOp moduleOp) {
    llvm::StringRef moduleName = moduleOp.getName().value_or("");
    log_assert((moduleName.size() != 0) && "Unnamed module op in RTLIL dialect");
    RTLIL::Module* new_module = design->addModule(moduleName.str());
    for (auto &op : moduleOp.getBody()->getOperations()) {
      if (auto wireOp = mlir::dyn_cast<rtlil::WireOp>(op))
        convert_wire(new_module, wireOp);
      else if (auto cellOp = mlir::dyn_cast<rtlil::CellOp>(op))
        convert_cell(new_module, cellOp);
      else if (auto connOp = mlir::dyn_cast<rtlil::WConnectionOp>(op))
        convert_connection(new_module, connOp);
      else if (auto constOp = mlir::dyn_cast<rtlil::ConstOp>(op))
        convert_const(new_module, constOp);
      else {
        op.dump();
        log_error("Unhandled RTLIL dialect op\n");
      }
    }
  }
};

struct MlirFrontend : public Frontend {
  MlirFrontend() : Frontend("mlir", "Read design from MLIR RTLIL dialect") {}
  // TODO help
  void execute(std::istream *&f, std::string filename,
               std::vector<std::string> args, RTLIL::Design *design) override {
    log_header(design, "Executing MLIR frontend.\n");
    size_t argidx;
    for (argidx = 1; argidx < args.size(); argidx++) {
      break;
    }
    extra_args(f, filename, args, argidx);
    //   llvm::raw_istream osis(*f);
    mlir::MLIRContext ctx;
    ctx.getOrLoadDialect<rtlil::RTLILDialect>();
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
      llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> owningModule =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &ctx);
    if (!owningModule) {
      llvm::errs() << "Error can't load file " << filename << "\n";
    }
    auto moduleOp = std::make_shared<mlir::ModuleOp>(owningModule.release());
    llvm::outs() << "yeah we got some stuff\n";
    // moduleOp->print(llvm::outs());
    // auto opIterator = .begin();
    RTLILifier convertor(design);
    for (auto& operation : moduleOp->getOps()) {
      mlir::ModuleOp op = llvm::dyn_cast<mlir::ModuleOp>(operation);
      if (!op)
        log_assert(false && "Top level MLIR entity isn't a module");
      convertor.convert_module(op);
    }
  }
} MlirFrontend;
