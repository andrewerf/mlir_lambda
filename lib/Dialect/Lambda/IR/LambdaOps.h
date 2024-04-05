//
// Created by Andrey Aralov on 3/31/24.
//
#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/Dialect/Func/IR/FuncOps.h.inc>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/IR/OpImplementation.h>


namespace mlir::lambda
{


class LambdaOp : public Op
            < LambdaOp
            , OpTrait::OneRegion
            , OpTrait::OneResult
            , OpTrait::ZeroOperands
            , OpTrait::IsIsolatedFromAbove
            , CallableOpInterface::Trait
            >
{
public:
    struct Properties
    {
        FunctionType functionType;
    };

    using Op::Op;
    using Op::print;

    static llvm::StringRef getOperationName();
    static llvm::StringRef getAttributeNames();

    // Properties and attributes
    static llvm::hash_code computePropertiesHash( const Properties& props );
    static Attribute getPropertiesAsAttr( MLIRContext* ctx, const Properties& props );
    static LogicalResult setPropertiesFromAttr( Properties& props, Attribute attr, InFlightDiagnostic *diag );
    static std::optional<Attribute> getInherentAttr( MLIRContext* ctx, const Properties &props, llvm::StringRef name );
    static void setInherentAttr( Properties& props, llvm::StringRef name, Attribute attr );
    static void populateInherentAttrs( MLIRContext* ctx, const Properties &props, NamedAttrList &attrs);
    static LogicalResult verifyInherentAttrs( OperationName opName, NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag );
    //


    // CallableOpInterface
    Region * getCallableRegion();
    ArrayRef<Type> getCallableResults();
    ArrayAttr getCallableArgAttrs();
    ArrayAttr getCallableResAttrs();
    //


    static mlir::ParseResult parse( OpAsmParser &parser,
                                    OperationState &result );

};



class ReturnOp : public Op
            < ReturnOp
            , OpTrait::ZeroResults
            , OpTrait::OneOperand
            , OpTrait::HasParent<LambdaOp>::Impl
            , OpTrait::IsTerminator
            >
{
public:
    using Op::Op;
    using Op::print;

    static llvm::StringRef getOperationName();
    static llvm::StringRef getAttributeNames();

    LogicalResult verify();

    static mlir::ParseResult parse( OpAsmParser &parser,
                                    OperationState &result );
};


}