//
// Created by Andrey Aralov on 3/31/24.
//
#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

void LambdaToFunc::rewrite( Operation *op, ArrayRef<Value>, ConversionPatternRewriter &rewriter ) const
{
    auto lambdaOp = llvm::dyn_cast<lambda::LambdaOp>( op );
    assert( lambdaOp && "Operation should be lambda" );
    auto lambdaFuncType = lambdaOp.getProperties().functionType;

    Operation* rootOp = op->getParentOfType<ModuleOp>();

    assert( rootOp != nullptr && "Root op must exist" );
    assert( rootOp != op && "Root op must be different from op" );
    assert( rootOp->getRegions().size() == 1 && "Root must have one region" );
    assert( rootOp->getRegion( 0 ).getBlocks().size() == 1 && "Root's region must have one block" );

    auto& baseBlock = *rootOp->getRegion( 0 ).getBlocks().begin();

    rewriter.setInsertionPointToStart( &baseBlock );
    auto funcOp = rewriter.create<::mlir::func::FuncOp>( baseBlock.getParent()->getLoc(), "fff", lambdaFuncType );
    rewriter.inlineRegionBefore( lambdaOp->getRegion( 0 ), funcOp.getBody(), funcOp.end() );

    rewriter.setInsertionPoint( op );
    auto newOp = rewriter.create<::mlir::func::ConstantOp>( op->getLoc(), op->getResultTypes(), FlatSymbolRefAttr::get( funcOp ) );
    rewriter.replaceOp( op, newOp );
}


void LambdaLoweringPass::runOnOperation()
{
    mlir::ConversionTarget target( getContext() );
    target.addLegalDialect<func::FuncDialect>();
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