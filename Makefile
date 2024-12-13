LLVM_CONFIG = llvm-config-19
TBLGEN := mlir-tblgen-19

# Compiler flags
CXX := clang++
CXXFLAGS := -fPIC `$(LLVM_CONFIG) --cxxflags`

# LLVM/MLIR libraries
LLVM_LIBS := `$(LLVM_CONFIG) --libs`
LLVM_SYSTEM_LIBS := `$(LLVM_CONFIG) --system-libs`
LLVM_LDFLAGS := `$(LLVM_CONFIG) --ldflags`
MLIR_INCLUDE := `$(LLVM_CONFIG) --includedir`
MLIR_LIBS := -lMLIR

# Dialect library sources (everything except main)
DIALECT_SOURCES := Dialect.cpp Ops.cpp Passes.cpp hello_c.cpp
DIALECT_OBJECTS := $(DIALECT_SOURCES:.cpp=.o)

# Generated files
GENERATED := Dialect.hpp.inc Dialect.cpp.inc Ops.hpp.inc Ops.cpp.inc Passes.hpp.inc

.PHONY: all clean

all: hello

# TableGen rules
Dialect.hpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-decls -I $(MLIR_INCLUDE) $< -o $@

Dialect.cpp.inc: Dialect.td
	$(TBLGEN) --gen-dialect-defs -I $(MLIR_INCLUDE) $< -o $@

Ops.hpp.inc: Ops.td
	$(TBLGEN) --gen-op-decls -I $(MLIR_INCLUDE) $< -o $@

Ops.cpp.inc: Ops.td
	$(TBLGEN) --gen-op-defs -I $(MLIR_INCLUDE) $< -o $@

Passes.hpp.inc: Passes.td
	$(TBLGEN) --gen-pass-decls -I $(MLIR_INCLUDE) $< -o $@

# TableGen rules for C API
Dialect_c.h: Dialect.td Ops.td
	$(TBLGEN) --gen-dialect-c-interface-wrapper -I $(MLIR_INCLUDE) Ops.td -o $@

Dialect_c.cpp: Dialect.td Ops.td
	$(TBLGEN) --gen-dialect-c-interface-impl -I $(MLIR_INCLUDE) Ops.td -o $@

# Object file rules
%.o: %.cpp $(GENERATED)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Build the dialect static library
libhello_dialect.a: $(DIALECT_OBJECTS)
	ar rcs $@ $(DIALECT_OBJECTS)

# Build the main executable, linking against the static library
hello: main.o libhello_dialect.a
	$(CXX) $^ -L$(LLVM_BUILD)/lib $(MLIR_LIBS) $(LLVM_LIBS) $(LLVM_SYSTEM_LIBS) $(LLVM_LDFLAGS) -o $@

clean:
	rm -f *.o *.inc *.a hello
