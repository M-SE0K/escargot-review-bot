// Test file for refactor pass detection
// This file contains patterns that the refactor pass should detect:
// 1. Nested conditionals that can be converted to guard clauses
// 2. Repeated patterns that can be extracted to local helpers
// 3. Try-catch with multiple return paths (RAII/ScopeGuard opportunity)
// 4. Redundant temporaries/copies

#include "Value.h"
#include "Object.h"
#include "ErrorObject.h"

namespace Escargot {

// Pattern 1: Nested conditionals - should suggest guard-clause refactor
Value processValue(ExecutionState& state, Value input) {
    if (input.isObject()) {
        Object* obj = input.asObject();
        if (obj != nullptr) {
            if (obj->hasProperty(state, "value")) {
                Value prop = obj->get(state, "value");
                if (prop.isString()) {
                    return prop;
                } else {
                    ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Expected string");
                }
            } else {
                ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Property not found");
            }
        } else {
            ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Invalid object");
        }
    } else {
        ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Not an object");
    }
    return Value();
}

// Pattern 2: Repeated error handling pattern - should suggest local helper
Value processIterator(ExecutionState& state, Object* iterator) {
    Value nextMethod = Object::getMethod(state, iterator, state.context()->staticStrings().next);
    if (nextMethod.isUndefined()) {
        ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Iterator.next is not callable");
        return Value();
    }
    
    Value result = Object::call(state, nextMethod, iterator, 0, nullptr);
    if (result.isAbrupt()) {
        IteratorObject::iteratorClose(state, iterator, result);
        ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Iterator.next failed");
        return Value();
    }
    
    Value done = Object::get(state, result, state.context()->staticStrings().done);
    if (done.isAbrupt()) {
        IteratorObject::iteratorClose(state, iterator, result);
        ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Iterator.done check failed");
        return Value();
    }
    
    return result;
}

// Pattern 3: Try-catch with multiple return paths - should suggest RAII/ScopeGuard
Value allocateAndProcess(ExecutionState& state, size_t size) {
    void* buffer = GC_MALLOC(size);
    if (buffer == nullptr) {
        ErrorObject::throwBuiltinError(state, ErrorCode::RangeError, "Allocation failed");
        return Value();
    }
    
    try {
        // Process buffer
        if (someCondition(state)) {
            GC_FREE(buffer);
            return Value();
        }
        
        if (anotherCondition(state)) {
            GC_FREE(buffer);
            ErrorObject::throwBuiltinError(state, ErrorCode::TypeError, "Processing failed");
            return Value();
        }
        
        Value result = processBuffer(state, buffer);
        GC_FREE(buffer);
        return result;
    } catch (...) {
        GC_FREE(buffer);
        throw;
    }
}

// Pattern 4: Redundant temporaries in loop - should suggest hoisting
Value processArray(ExecutionState& state, ArrayObject* array) {
    Value result = ArrayObject::createArray(state);
    for (size_t i = 0; i < array->length(); i++) {
        Value element = array->get(state, i);
        ObjectPropertyName propName(state, state.context()->staticStrings().value);
        Value converted = element.toNumber(state);
        Object::set(state, result, propName, converted);
        
        ObjectPropertyName propName2(state, state.context()->staticStrings().value);
        Value converted2 = element.toNumber(state);
        Object::set(state, result, propName2, converted2);
    }
    return result;
}

} // namespace Escargot
