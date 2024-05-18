//
// Created by Andrey Aralov on 3/31/24.
//
#include <ranges>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/IR/IRMapping.h>

#include "LambdaToFunc.h"
#include "lib/Dialect/Lambda/IR/Lambda.h"


namespace
{

using namespace mlir;

class LambdaToFunc : public ConversionPattern
{
public:
    LambdaToFunc( MLIRContext *context );

    LogicalResult match( Operation *op ) const override;
    void rewrite( Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter ) const override;
};

class ReturnToFunc : public ConversionPattern
{
public:
    explicit ReturnToFunc( MLIRContext* ctx );

    LogicalResult match( Operation *op ) const override;
    void rewrite( Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter ) const override;
};


class LambdaLoweringPass : public PassWrapper<LambdaLoweringPass, OperationPass<ModuleOp>>
{
public:
void runOnOperation() override;
};

ReturnToFunc::ReturnToFunc( MLIRContext *ctx ):
        ConversionPattern( lambda::ReturnOp::getOperationName(), 1, ctx )
{}

LogicalResult ReturnToFunc::match( mlir::Operation *op ) const
{
    return success( llvm::isa<lambda::ReturnOp>( op ) );
}

void ReturnToFunc::rewrite( mlir::Operation *op, ArrayRef<mlir::Value>, mlir::ConversionPatternRewriter& rewriter ) const
{
    rewriter.replaceOp( op,
                        rewriter.create( op->getLoc(), rewriter.getStringAttr( "func.return" ), op->getOperands(), op->getResultTypes() ) );
}


LambdaToFunc::LambdaToFunc( MLIRContext *context ):
        ConversionPattern( lambda::LambdaOp::getOperationName(), 1, context )
{}

LogicalResult LambdaToFunc::match( Operation *op ) const
{
    return success( llvm::isa<lambda::LambdaOp>( op ) );
}


/// Given a lambda, returns corresponding function type (with capture added)
FunctionType getLambdaFunctionType( lambda::LambdaOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter )
{
    auto lambdaFuncType = op.getProperties().functionType;
    llvm::SmallVector<Type> funcInputs;
    std::ranges::copy( operands | std::views::transform( &Value::getType ), std::back_inserter( funcInputs ) );
    std::ranges::copy( lambdaFuncType.getInputs(), std::back_inserter( funcInputs ) );
    return rewriter.getFunctionType( funcInputs, lambdaFuncType.getResults() );
}

std::string getLambdaName()
{
    static int x = 0;
    return "__lambda" + std::to_string( x++ );
}

/// Makes a function in the current module, that corresponds to the lambda
func::FuncOp makeLambdaFunction( lambda::LambdaOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter )
{
    Operation* rootOp = op->getParentOfType<ModuleOp>();
    auto& baseBlock = *rootOp->getRegion( 0 ).getBlocks().begin();

    rewriter.setInsertionPointToStart( &baseBlock );
    auto funcOp = rewriter.create<::mlir::func::FuncOp>( baseBlock.getParent()->getLoc(), getLambdaName(), getLambdaFunctionType( op, operands, rewriter ) );
    rewriter.inlineRegionBefore( op.getRegion(), funcOp.getBody(), funcOp.end() );
    rewriter.setInsertionPoint( op );
    return funcOp;
}


LLVM::LLVMStructType makeLLVMStructType( lambda::LambdaOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter )
{
    auto funcType = getLambdaFunctionType( op, operands, rewriter );

    llvm::SmallVector<Type> captureTypes;
    captureTypes.push_back( funcType );
    std::ranges::copy( operands | std::views::transform( &Value::getType ), std::back_inserter( captureTypes ) );

    return LLVM::LLVMStructType::getLiteral( rewriter.getContext(), captureTypes );
}


void LambdaToFunc::rewrite( Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter ) const
{
    auto lambdaOp = llvm::dyn_cast<lambda::LambdaOp>( op );
    assert( lambdaOp && "Operation should be lambda" );
    auto loc = op->getLoc();

    // function operation
    auto funcOp = makeLambdaFunction( lambdaOp, operands, rewriter );

    // initial op is replaced with a function constant
    auto makeFuncOp = rewriter.create<::mlir::func::ConstantOp>( loc, funcOp.getFunctionType(), FlatSymbolRefAttr::get( funcOp ) );
    rewriter.replaceOp( op, makeFuncOp );

    // result of the lambda creation (the SSA value)
    Value value = makeFuncOp->getResult( 0 );

    auto makeLambdaOp = rewriter.create<lambda::MakeLambdaOp>( loc, value, operands );

    rewriter.replaceAllUsesWith( op->getResults(), makeLambdaOp->getResults() );

//    // create a struct type and construct it as undef
//    auto structType = makeLLVMStructType( lambdaOp, operands, rewriter );
//    auto undefOp = rewriter.create<LLVM::UndefOp>( loc, structType );
//
//    Value undefOpValue = undefOp.getRes();
//    undefOpValue.print( llvm::outs() );
//
//    // store function ptr
//    rewriter.create<LLVM::InsertValueOp>( loc, structType, undefOpValue, value, rewriter.getDenseI64ArrayAttr( { 0 } ) );
//
//    // store operands (which represent the capture list) in the structure
//    for ( auto val : operands )
//    {
////        rewriter.create<LLVM::InsertValueOp>( loc,  )
//    }

}


void LambdaLoweringPass::runOnOperation()
{
    mlir::ConversionTarget target( getContext() );
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalOp<lambda::MakeLambdaOp>();
    target.addLegalOp<lambda::CallOp>();
    target.addIllegalDialect<lambda::LambdaDialect>();

    mlir::RewritePatternSet patterns( &getContext() );
    patterns.add<ReturnToFunc>( &getContext() );
    patterns.add<LambdaToFunc>( &getContext() );
    if ( mlir::failed( mlir::applyPartialConversion( getOperation(), target, std::move( patterns ) ) ) )
        signalPassFailure();
}

}


namespace mlir
{

std::unique_ptr<Pass> createConvertLambdaToFuncPass()
{
    return std::make_unique<LambdaLoweringPass>( LambdaLoweringPass() );
}

}