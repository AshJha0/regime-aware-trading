#ifndef REGIME_TESTS_MINI_JSON_HPP
#define REGIME_TESTS_MINI_JSON_HPP

/// \file mini_json.hpp
/// \brief Tiny self-contained JSON reader for the flat golden.json schema.
///
/// Supports objects, arrays, strings, numbers, booleans and null — enough
/// for {"cases": [{"name", "inputs": {}, "expect": {...}, "tol"}]}.  Not a
/// general-purpose parser (no \u escapes, no streaming); test-tree only.

#include <cctype>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mini_json {

/// A parsed JSON value (tagged union over the supported types).
struct Value {
    enum class Type { Null, Bool, Number, String, Array, Object };
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;
    std::vector<Value> arr;
    std::map<std::string, Value> obj;

    const Value& at(const std::string& key) const {
        auto it = obj.find(key);
        if (it == obj.end()) throw std::runtime_error("mini_json: missing key '" + key + "'");
        return it->second;
    }
    double num() const {
        if (type != Type::Number) throw std::runtime_error("mini_json: not a number");
        return number;
    }
    const std::string& string() const {
        if (type != Type::String) throw std::runtime_error("mini_json: not a string");
        return str;
    }
};

namespace detail {

struct Parser {
    const std::string& text;
    std::size_t pos = 0;

    explicit Parser(const std::string& t) : text(t) {}

    void skip_ws() {
        while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) ++pos;
    }
    char peek() {
        skip_ws();
        if (pos >= text.size()) throw std::runtime_error("mini_json: unexpected end of input");
        return text[pos];
    }
    void expect(char c) {
        if (peek() != c)
            throw std::runtime_error(std::string("mini_json: expected '") + c + "' at " +
                                     std::to_string(pos));
        ++pos;
    }

    Value parse_value() {
        const char c = peek();
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') {
            Value v;
            v.type = Value::Type::String;
            v.str = parse_string();
            return v;
        }
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') {
            pos += 4;  // null
            return Value{};
        }
        return parse_number();
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos < text.size() && text[pos] != '"') {
            char c = text[pos++];
            if (c == '\\' && pos < text.size()) {
                const char e = text[pos++];
                switch (e) {
                    case 'n': out.push_back('\n'); break;
                    case 't': out.push_back('\t'); break;
                    default: out.push_back(e); break;
                }
            } else {
                out.push_back(c);
            }
        }
        if (pos >= text.size()) throw std::runtime_error("mini_json: unterminated string");
        ++pos;  // closing quote
        return out;
    }

    Value parse_number() {
        skip_ws();
        const char* start = text.c_str() + pos;
        char* end = nullptr;
        Value v;
        v.type = Value::Type::Number;
        v.number = std::strtod(start, &end);
        if (end == start) throw std::runtime_error("mini_json: bad number at " + std::to_string(pos));
        pos += static_cast<std::size_t>(end - start);
        return v;
    }

    Value parse_bool() {
        Value v;
        v.type = Value::Type::Bool;
        if (text.compare(pos, 4, "true") == 0) {
            v.boolean = true;
            pos += 4;
        } else if (text.compare(pos, 5, "false") == 0) {
            v.boolean = false;
            pos += 5;
        } else {
            throw std::runtime_error("mini_json: bad literal at " + std::to_string(pos));
        }
        return v;
    }

    Value parse_array() {
        expect('[');
        Value v;
        v.type = Value::Type::Array;
        if (peek() == ']') {
            ++pos;
            return v;
        }
        while (true) {
            v.arr.push_back(parse_value());
            const char c = peek();
            if (c == ',') {
                ++pos;
            } else if (c == ']') {
                ++pos;
                return v;
            } else {
                throw std::runtime_error("mini_json: bad array at " + std::to_string(pos));
            }
        }
    }

    Value parse_object() {
        expect('{');
        Value v;
        v.type = Value::Type::Object;
        if (peek() == '}') {
            ++pos;
            return v;
        }
        while (true) {
            const std::string key = parse_string();
            expect(':');
            v.obj[key] = parse_value();
            const char c = peek();
            if (c == ',') {
                ++pos;
            } else if (c == '}') {
                ++pos;
                return v;
            } else {
                throw std::runtime_error("mini_json: bad object at " + std::to_string(pos));
            }
        }
    }
};

}  // namespace detail

/// Parse a complete JSON document.
inline Value parse(const std::string& text) {
    detail::Parser p(text);
    Value v = p.parse_value();
    p.skip_ws();
    return v;
}

}  // namespace mini_json

#endif  // REGIME_TESTS_MINI_JSON_HPP
