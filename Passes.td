#ifndef HELLO_PASSES
#define HELLO_PASSES

include "mlir/Pass/PassBase.td"

def ConvertHelloToLLVMPass : Pass<"convert-hello-to-llvm", "ModuleOp"> {
  let summary = "Convert Hello dialect to LLVM dialect";
  let description = [{
    Convert Hello dialect operations to the LLVM dialect.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // HELLO_PASSES
