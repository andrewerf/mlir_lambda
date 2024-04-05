project(MLIRLambda)


add_mlir_dialect_library(MLIRLambda
    Dialect/Lambda/IR/LambdaOps.cpp
    Dialect/Lambda/IR/Lambda.cpp
    Conversion/LambdaToFunc/LambdaToFunc.cpp
    LINK_LIBS PUBLIC
)
