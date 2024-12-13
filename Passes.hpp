#pragma once

#include <memory>
#include <mlir/Pass/Pass.h>

namespace hello {
#define GEN_PASS_DECL
#include "Passes.hpp.inc"
}
