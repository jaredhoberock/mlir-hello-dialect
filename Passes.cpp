#include "Dialect.hpp"
#include "Ops.hpp"
#include "Passes.hpp"
#include <mlir/Conversion/ConvertToLLVM/ToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/Pass/Pass.h>
#include <iostream>

using namespace mlir;

struct PrintOpLowering : public ConversionPattern {
  PrintOpLowering(MLIRContext *ctx) 
    : ConversionPattern(hello::PrintOp::getOperationName(), 1, ctx)
  {}

  LogicalResult matchAndRewrite(Operation *op, 
                                ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const override {
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();
    auto loc = op->getLoc();

    // get a pointer to the string "Hello, world\n"
    StringRef hello("Hello, world\n\0", 14);
    Value helloPtr = getOrCreateGlobalString(loc, rewriter, "hello_msg", hello, moduleOp);

    // Create the printf function declaration if it doesn’t exist
    auto printfRef = getOrInsertPrintf(moduleOp, rewriter);

    // Call printf with the pointer to the string
    rewriter.create<LLVM::CallOp>(loc, getPrintfType(rewriter.getContext()),
                                  printfRef,
                                  ArrayRef<Value>({helloPtr}));

    // Remove the original PrintOp
    rewriter.eraseOp(op);

    return success();
  }

private:
  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                  builder.getIndexAttr(0));
    return builder.create<LLVM::GEPOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext()), global.getType(),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }


  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  static LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) {
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmPtrTy = LLVM::LLVMPointerType::get(context);
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmPtrTy, /*isVarArg=*/true);
    return llvmFnType;
  }

  FlatSymbolRefAttr getOrInsertPrintf(ModuleOp module, 
                                      ConversionPatternRewriter &rewriter) const {

    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf",
                                      getPrintfType(context));
    return SymbolRefAttr::get(context, "printf");
  }
};


// Add patterns to the pattern set
static void populatePrintOpToLLVMPatterns(RewritePatternSet &patterns) {
  patterns.add<PrintOpLowering>(patterns.getContext());
}


namespace hello
{

#define GEN_PASS_DEF_CONVERTHELLOTOLLVMPASS
#include "Passes.hpp.inc"

struct ConvertHelloToLLVMPass 
    : hello::impl::ConvertHelloToLLVMPassBase<ConvertHelloToLLVMPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    populatePrintOpToLLVMPatterns(patterns);

    LLVMConversionTarget target(getContext());
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // end hello
