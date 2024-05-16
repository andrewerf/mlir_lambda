//
// Created by Andrey Aralov on 4/6/24.
//
#pragma once
#include <memory>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Pass/Pass.h>


namespace mlir
{
std::unique_ptr<Pass> createConvertLambdaToLLVMPass();
}