#pragma once

#include <mlir-c/IR.h>
#include <mlir-c/Pass.h>

#ifdef __cplusplus
extern "C" {
#endif

// Register the dialect with an MLIR context
MLIR_CAPI_EXPORTED void helloRegisterDialect(MlirContext context);

// Create a print operation
MLIR_CAPI_EXPORTED MlirOperation helloCreatePrintOp(MlirLocation loc);

// Create the conversion pass
MLIR_CAPI_EXPORTED MlirPass helloCreateConvertHelloToLLVMPass();

#ifdef __cplusplus
}
#endif
