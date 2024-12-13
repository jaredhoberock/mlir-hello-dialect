#include "Dialect.hpp"
#include "Passes.hpp"
#include "Ops.hpp"
#include <iostream>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

int main() {
  mlir::MLIRContext context;

  // set up the context for using the dialects we want
  // and for translation to LLVM
  {
    // tell MLIR what dialects we want to use
    mlir::DialectRegistry registry;
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<hello::HelloDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    // tell MLIR what translations we want to use
    mlir::registerLLVMDialectTranslation(context);
    mlir::registerBuiltinDialectTranslation(context);
  }
  
  // Create a location
  auto loc = mlir::UnknownLoc::get(&context);
  
  // Create a module
  mlir::OpBuilder builder(&context);
  auto module = mlir::ModuleOp::create(loc);

  // create a module with a single function 'foo' that uses our hello::PrintOp
  {
    // Define a new function 'foo'
    auto funcType = builder.getFunctionType({}, {}); // foo takes no arguments, returns void
    auto fooFunc = builder.create<mlir::func::FuncOp>(loc, "foo", funcType);

    // Create an entry block for the function and set the insertion point
    auto& entryBlock = *fooFunc.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);
    
    // Insert the PrintOp within foo
    builder.create<hello::PrintOp>(loc);

    // Add a return operation to foo
    builder.create<mlir::func::ReturnOp>(loc);

    // Set insertion point back to the module to ensure foo is at module level
    builder.setInsertionPointToEnd(module.getBody());

    module.push_back(fooFunc);
    
    // Verify and print the module
    if (failed(mlir::verify(module))) {
      module.emitError("module verification error");
      return -1;
    }

    std::cout << "Initial module:" << std::endl;
    module.print(llvm::outs());
    std::cout << std::endl;
  }

  // Convert the module to LLVM using Passes
  {
    mlir::PassManager pm(&context);

    // func -> LLVM
    pm.addPass(mlir::createConvertFuncToLLVMPass());

    // Hello -> LLVM
    pm.addPass(hello::createConvertHelloToLLVMPass());

    if (failed(pm.run(module))) {
      module.emitError("Failed to convert to LLVM");
      return -1;
    }

    std::cout << "After conversion to LLVM" << std::endl;
    module.print(llvm::outs());
    std::cout << std::endl;
  }

  // JIT compile and call foo()
  {
    // Initialize LLVM targets (required for JIT)
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    std::cout << "before ExecutionEngine::create" << std::endl;

    // Create the Execution Engine
    auto maybeEngine = mlir::ExecutionEngine::create(module);
    if (!maybeEngine) {
      llvm::errs() << "Failed to create ExecutionEngine\n";
      return -1;
    }
    auto &engine = maybeEngine.get();

    auto maybe_foo = engine->lookup("foo");
    if (!maybe_foo) {
      llvm::errs() << "foo not found in ExecutionEngine\n";
      return -1;
    }

    // Extract the raw void* function pointer
    void *raw_foo = *maybe_foo;

    // Define the function pointer type matching `foo`
    using foo_t = void();

    // Cast to the callable function type
    auto foo = reinterpret_cast<foo_t*>(reinterpret_cast<intptr_t>(raw_foo));

    // Call the function
    foo();
  }

  return 0;
}
