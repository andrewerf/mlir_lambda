//
// Created by Andrey Aralov on 4/6/24.
//
#include <ranges>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/IRMapping.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>


#include "LambdaToLLVM.h"
#include "lib/Dialect/Lambda/IR/LambdaOps.h"
#include "lib/Dialect/Lambda/IR/Lambda.h"


namespace
{

using namespace mlir;

class LambdaLoweringPass : public PassWrapper<LambdaLoweringPass, OperationPass<ModuleOp>>
{
public:
    void runOnOperation() override;
};

class MakeLambdaToLLVM : public ConversionPattern
{
public:
    MakeLambdaToLLVM( LLVMTypeConverter& conv, MLIRContext *context );

    LogicalResult match( Operation *op ) const override;
    void rewrite( Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter ) const override;
};

class CallToLLVM : public ConversionPattern
{
public:
    CallToLLVM( LLVMTypeConverter& conv, MLIRContext *context );

    LogicalResult match( Operation *op ) const override;
    void rewrite( Operation *op, ArrayRef<Value> operands, ConversionPatternRewriter &rewriter ) const override;
};




MakeLambdaToLLVM::MakeLambdaToLLVM( LLVMTypeConverter& conv, mlir::MLIRContext *context ):
    ConversionPattern( conv, lambda::MakeLambdaOp::getOperationName(), 1, context )
{}

LogicalResult MakeLambdaToLLVM::match( mlir::Operation *op ) const
{
    return success( llvm::isa<lambda::MakeLambdaOp>( op ) );
}

LLVM::LLVMStructType makeLLVMStructType( TypeConverter* conv, lambda::MakeLambdaOp op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter )
{
    auto lambdaType = llvm::dyn_cast<lambda::LambdaType>( op->getResult( 0 ).getType() );
    assert( lambdaType );

    SmallVector<Type> convTypes;
    for ( auto type : lambdaType.getElementTypes() )
    {
        convTypes.push_back( conv->convertType( type ) );
    }

    return LLVM::LLVMStructType::getLiteral( rewriter.getContext(), convTypes );
}

void MakeLambdaToLLVM::rewrite( mlir::Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter ) const
{
    auto makeLambdaOp = llvm::dyn_cast<lambda::MakeLambdaOp>( op );
    assert( makeLambdaOp && "Operation should be lambda" );
    auto loc = op->getLoc();

    if ( operands.size() == 1 )
    {
        rewriter.replaceOp( op, operands[0] );
    }
    else
    {
        // create a struct type and construct it as undef
        auto structType = makeLLVMStructType( getTypeConverter(), makeLambdaOp, operands, rewriter );
        auto undefOp = rewriter.create<LLVM::UndefOp>( loc, structType );

        Value undefOpValue = undefOp.getRes();

        Operation *updatedOp;

        // store operands (which represent the callee and the capture list) in the structure
        for ( auto [i, val]: std::views::enumerate( operands ) )
        {
            auto top = rewriter.create<LLVM::InsertValueOp>( loc, structType, undefOpValue, val, rewriter.getDenseI64ArrayAttr( {i} ) );
            undefOpValue = top.getRes();
            updatedOp = top;
        }
        rewriter.replaceOp( op, updatedOp );
    }
}


CallToLLVM::CallToLLVM( mlir::LLVMTypeConverter& conv, mlir::MLIRContext *context ):
    ConversionPattern( lambda::CallOp::getOperationName(), 1, context )
{}

LogicalResult CallToLLVM::match( mlir::Operation *op ) const
{
    return success( llvm::isa<lambda::CallOp>( op ) );
}

void CallToLLVM::rewrite( mlir::Operation *op, ArrayRef<mlir::Value> operands, mlir::ConversionPatternRewriter& rewriter ) const
{
    auto callee = operands[0];
    auto loc = op->getLoc();

    if ( auto calleeType = llvm::dyn_cast<LLVM::LLVMStructType>( callee.getType() ) )
    {
//    auto calleeFuncType = llvm::dyn_cast<FunctionType>( calleeType.getBody()[0] );
//    assert( calleeFuncType );

        // extract capture to SSA values. First one is the function
        auto loadFuncOp = rewriter.create<LLVM::ExtractValueOp>( loc, callee, rewriter.getDenseI64ArrayAttr( {0} ) );
        auto funcVal = loadFuncOp.getRes();

        SmallVector<Value> args;
        for ( int64_t i = 1; i < calleeType.getBody().size(); ++i )
        {
            auto loadArgOp = rewriter.create<LLVM::ExtractValueOp>( loc, callee, rewriter.getDenseI64ArrayAttr( {i} ) );
            auto argVal = loadArgOp.getRes();
            args.push_back( argVal );
        }

        for ( int64_t i = 1; i < operands.size(); ++i )
        {
            args.push_back( operands[i] );
        }

        auto newOp = rewriter.create<func::CallIndirectOp>( loc, rewriter.getType<IntegerType>( 32 ), funcVal, args );
        rewriter.replaceOp( op, newOp );
    }
    else
    {
        SmallVector<Value> args;
        for ( int64_t i = 1; i < operands.size(); ++i )
        {
            args.push_back( operands[i] );
        }

        auto callOp = rewriter.create<func::CallIndirectOp>( loc, rewriter.getType<IntegerType>( 32 ), callee, args );
        rewriter.replaceOp( op, callOp );
    }
}



void LambdaLoweringPass::runOnOperation()
{
    mlir::ConversionTarget target( getContext() );
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<lambda::LambdaDialect>();

    LLVMTypeConverter typeConverter( &getContext() );

    mlir::RewritePatternSet patterns( &getContext() );
    patterns.add<MakeLambdaToLLVM>( typeConverter, &getContext() );
    patterns.add<CallToLLVM>( typeConverter, &getContext() );
    populateFuncToLLVMConversionPatterns( typeConverter, patterns );
    if ( mlir::failed( mlir::applyPartialConversion( getOperation(), target, std::move( patterns ) ) ) )
        signalPassFailure();
}


}


namespace mlir
{

std::unique_ptr<Pass> createConvertLambdaToLLVMPass()
{
    return std::make_unique<LambdaLoweringPass>();
}

}