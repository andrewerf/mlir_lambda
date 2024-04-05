//
// Created by Andrey Aralov on 3/30/24.
//
#pragma once

#include <mlir/IR/Dialect.h>


namespace mlir::lambda
{

class LambdaDialect : public Dialect
{
public:
    explicit LambdaDialect( MLIRContext* ctx );

    /// Return the name of the dialect.
    static llvm::StringRef getDialectNamespace();
};

}
