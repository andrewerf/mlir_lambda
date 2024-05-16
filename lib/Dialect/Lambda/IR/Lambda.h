//
// Created by Andrey Aralov on 3/30/24.
//
#pragma once

#include <mlir/IR/Dialect.h>


namespace mlir::lambda
{

namespace detail
{
class LambdaTypeStorage;
}

class LambdaType :  public Type::TypeBase<LambdaType, Type, detail::LambdaTypeStorage>
{
public:
    using Base::Base;

    static LambdaType get( llvm::ArrayRef<mlir::Type> elementTypes );

    /// Returns the element types of this struct type.
    llvm::ArrayRef<mlir::Type> getElementTypes();

    /// Returns the number of element type held by this struct.
    size_t getNumElementTypes();
};

class LambdaDialect : public Dialect
{
public:
    explicit LambdaDialect( MLIRContext* ctx );

    /// Return the name of the dialect.
    static llvm::StringRef getDialectNamespace();

    Type parseType( DialectAsmParser &parser ) const override;
    void printType( Type type, DialectAsmPrinter &printer ) const override;
};

}
