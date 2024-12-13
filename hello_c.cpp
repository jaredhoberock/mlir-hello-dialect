#include "hello_c.h"
#include "Dialect.hpp"
#include "Ops.hpp"
#include "Passes.hpp"
#include <mlir/CAPI/IR.h>
#include <mlir/CAPI/Pass.h>
#include <mlir/IR/Builders.h>

extern "C" {

void helloRegisterDialect(MlirContext context) {
  mlir::MLIRContext *ctx = unwrap(context);
  ctx->loadDialect<hello::HelloDialect>();
}

MlirOperation helloCreatePrintOp(MlirLocation loc) {
  mlir::OpBuilder builder(unwrap(loc)->getContext());
  auto op = builder.create<hello::PrintOp>(unwrap(loc));
  return wrap(op.getOperation());
}

MlirPass helloCreateConvertHelloToLLVMPass() {
  return wrap(hello::createConvertHelloToLLVMPass().release());
}

} // end extern "C"
