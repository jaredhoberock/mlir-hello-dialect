#ifndef HELLO_DIALECT
#define HELLO_DIALECT

include "mlir/IR/OpBase.td"

def Hello_Dialect : Dialect {
  let name = "hello";
  let summary = "A dialect for hello operations";
  let description = [{
    The hello dialect is a simple example dialect.
  }];
  let cppNamespace = "::hello";
}

#endif // HELLO_DIALECT
