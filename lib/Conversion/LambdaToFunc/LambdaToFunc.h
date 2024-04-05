//
// Created by Andrey Aralov on 3/31/24.
//
#pragma once
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>

#include "lib/Dialect/Lambda/IR/LambdaOps.h"


namespace mlir
{
std::unique_ptr<Pass> createConvertLambdaToFuncPass();
}

