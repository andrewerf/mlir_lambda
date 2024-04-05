//
// Created by Andrey Aralov on 3/30/24.
//
#include "Lambda.h"
#include "LambdaOps.h"

namespace mlir::lambda
{

LambdaDialect::LambdaDialect( mlir::MLIRContext *ctx ):
    Dialect( getDialectNamespace(), ctx, TypeID::get<LambdaDialect>() )
{
    addOperations
        < LambdaOp
        , ReturnOp
        >();
}

llvm::StringRef LambdaDialect::getDialectNamespace()
{
    return "lambda";
}


}
