//
// Created by Andrey Aralov on 3/31/24.
//
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionImplementation.h>
#include "LambdaOps.h"

namespace mlir::lambda
{


llvm::StringRef LambdaOp::getOperationName()
{ return "lambda.lambda"; }

llvm::StringRef LambdaOp::getAttributeNames()
{ return {}; }



llvm::hash_code LambdaOp::computePropertiesHash( const LambdaOp::Properties& props )
{
    return llvm::hash_value( props.functionType.getAsOpaquePointer() );
}

Attribute LambdaOp::getPropertiesAsAttr( mlir::MLIRContext *ctx,
                                              const mlir::lambda::LambdaOp::Properties& props )
{
    Builder builder{ ctx };
    return builder.getDictionaryAttr( { builder.getNamedAttr( "function_type", TypeAttr::get( props.functionType ) ) } );
}

LogicalResult
LambdaOp::setPropertiesFromAttr( mlir::lambda::LambdaOp::Properties& props,
                                 mlir::Attribute attr,
                                 mlir::InFlightDiagnostic *diag )
{
    const auto msgToDiag = [diag] ( std::string_view msg )
    {
        if ( diag )
            *diag << msg;
    };

    auto dict = ::llvm::dyn_cast<DictionaryAttr>( attr );
    if ( !dict )
    {
        msgToDiag( "expected DictionaryAttr to set properties" );
        return failure();
    }

    const auto functionType = dict.get( "function_type" );
    if ( !functionType )
    {
        msgToDiag( "expected entry for function_type in DictionaryAttr to set properties" );
        return failure();
    }

    const auto convertedFunctionType = llvm::dyn_cast<TypeAttr>( functionType );
    if ( !convertedFunctionType )
    {
        msgToDiag( "expected entry for function_type to have type FunctionType" );
        return failure();
    }

    props.functionType = llvm::dyn_cast<FunctionType>( convertedFunctionType.getValue() );

    return success();
}

std::optional<Attribute> LambdaOp::getInherentAttr( mlir::MLIRContext *ctx, const mlir::lambda::LambdaOp::Properties& props, llvm::StringRef name )
{
    if ( name == "function_type" )
        return TypeAttr::get( props.functionType );
    return std::nullopt;
}

void LambdaOp::setInherentAttr( Properties& props, llvm::StringRef name, mlir::Attribute attr )
{
    if ( name == "function_type" )
        props.functionType = dyn_cast<FunctionType>( llvm::dyn_cast<TypeAttr>( attr ).getValue() );
}

void LambdaOp::populateInherentAttrs( mlir::MLIRContext *ctx, const mlir::lambda::LambdaOp::Properties& props, mlir::NamedAttrList& attrs )
{
    attrs.append( "function_type", TypeAttr::get( props.functionType ) );
}

LogicalResult
LambdaOp::verifyInherentAttrs( mlir::OperationName opName, mlir::NamedAttrList& attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag )
{
    return failure();
}

Region *LambdaOp::getCallableRegion()
{
    return &getRegion();
}

ArrayRef<Type> LambdaOp::getCallableResults()
{
    return getProperties().functionType.getResults();
}

ArrayAttr LambdaOp::getCallableArgAttrs()
{
    return {};
}

ArrayAttr LambdaOp::getCallableResAttrs()
{
    return {};
}


ParseResult LambdaOp::parse( OpAsmParser& parser,
                             OperationState& result )
{
//    llvm::SmallVector<OpAsmParser::UnresolvedOperand> capture;
//    if ( parser.parseLess() || parser.parseOperandList( capture ) || parser.parseGreater() )
//        return mlir::failure();

    llvm::SmallVector<OpAsmParser::Argument> args;
    llvm::SmallVector<Type> resultTypes;
    llvm::SmallVector<DictionaryAttr> resultAttrs;
    bool isVariadic;
    if ( function_interface_impl::parseFunctionSignature( parser, false, args, isVariadic, resultTypes, resultAttrs ) )
        return mlir::failure();

    llvm::SmallVector<Type> argTypes;
    for ( const auto& arg : args )
        argTypes.push_back( arg.type );

    auto& functionType = result.getOrAddProperties<Properties>().functionType;
    functionType = parser.getBuilder().getFunctionType( argTypes, resultTypes );

    auto body = result.addRegion();
    if ( parser.parseRegion( *body, args ) )
        return mlir::failure();

    result.addTypes( functionType );

    return mlir::success();
}


StringRef ReturnOp::getOperationName()
{ return "lambda.return"; }

StringRef ReturnOp::getAttributeNames()
{ return {}; }

LogicalResult ReturnOp::verify()
{
    if ( OpState::verify().failed() )
        return failure();
    auto parent = getParentOp();

    const auto propStorage = parent->getPropertiesStorage();
    assert( propStorage && "Expected non-empty property storage" );

    const auto typedPropStorage = propStorage.as<LambdaOp::Properties*>();
    assert( typedPropStorage && "Expected property storage to have type detail::Properties" );

    if ( typedPropStorage->functionType.getResult( 0 ) != getOperand().getType() )
        return failure();

    return success();
}

mlir::ParseResult ReturnOp::parse( OpAsmParser& parser,
                                   OperationState& result )
{
    OpAsmParser::UnresolvedOperand operand;
    Type type;
    if (   parser.parseOperand( operand )
        || parser.parseColonType( type )
        || parser.resolveOperand( operand, type, result.operands ) )
        return mlir::failure();

    return mlir::success();
}




}

