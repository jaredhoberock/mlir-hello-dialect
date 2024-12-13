#ifndef HELLO_OPS
#define HELLO_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "Dialect.td"

def PrintOp : Op<Hello_Dialect, "print"> {
  let summary = "print Hello, world";
  let description = [{
    The "print" operation prints "Hello, world" and produces no results."
  }];
}

#endif // HELLO_OPS
