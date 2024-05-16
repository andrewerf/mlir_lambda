//
// Created by Andrey Aralov on 3/30/24.
//
#include "Lambda.h"
#include "LambdaOps.h"

#include <mlir/IR/DialectImplementation.h>


namespace mlir::lambda::detail
{

class LambdaTypeStorage : public TypeStorage
{
public:
    using KeyTy = llvm::ArrayRef<Type>;

    LambdaTypeStorage( llvm::ArrayRef<mlir::Type> elementTypes_ ):
        elementTypes( elementTypes_ )
    {}


    bool operator==( const KeyTy &key ) const
    {
        return key == elementTypes;
    }

    static llvm::hash_code hashKey( const KeyTy &key ) {
        return llvm::hash_value(key);
    }
    static KeyTy getKey( llvm::ArrayRef<mlir::Type> elementTypes ) {
        return KeyTy(elementTypes);
    }

    static LambdaTypeStorage *construct(
            mlir::TypeStorageAllocator &allocator,
            const KeyTy &key )
    {
        // Copy the elements from the provided `KeyTy` into the allocator.
        llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

        // Allocate the storage instance and construct it.
        return new ( allocator.allocate<LambdaTypeStorage>() ) LambdaTypeStorage( elementTypes );
    }

    ArrayRef<Type> elementTypes;
};

}

namespace mlir::lambda
{

LambdaType LambdaType::get( llvm::ArrayRef<Type> elementTypes )
{
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first parameter is the context to unique in. The
    // parameters after are forwarded to the storage instance.
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get( ctx, elementTypes );
}

llvm::ArrayRef<mlir::Type> LambdaType::getElementTypes()
{
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
}

size_t LambdaType::getNumElementTypes()
{ return getElementTypes().size(); }


LambdaDialect::LambdaDialect( mlir::MLIRContext *ctx ):
    Dialect( getDialectNamespace(), ctx, TypeID::get<LambdaDialect>() )
{
    addOperations
        < LambdaOp
        , ReturnOp
        , CallOp
        , MakeLambdaOp
        >();
    addTypes
        < LambdaType
        >();
}

llvm::StringRef LambdaDialect::getDialectNamespace()
{
    return "lambda";
}

Type LambdaDialect::parseType( mlir::DialectAsmParser& parser ) const
{
    if ( parser.parseKeyword( "lambda_type" ) || parser.parseLess() )
        return {};

    SmallVector<mlir::Type> elementTypes;
    do {
        // Parse the current element type.
        SMLoc typeLoc = parser.getCurrentLocation();
        mlir::Type elementType;
        if ( parser.parseType( elementType ) )
            return nullptr;

        elementTypes.push_back( elementType );

        // Parse the optional: `,`
    } while ( succeeded( parser.parseOptionalComma() ) );

    if ( parser.parseGreater() )
        return Type();

    return LambdaType::get( elementTypes );
}

void LambdaDialect::printType( Type type, DialectAsmPrinter& printer ) const
{
    auto structType = type.cast<LambdaType>();

    // Print the struct type according to the parser format.
    printer << "lambda_type<";
    llvm::interleaveComma( structType.getElementTypes(), printer );
    printer << '>';
}

}
