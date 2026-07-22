// Minimal JSON parser for the ORT session profile file.
//
// Extracted into a header so it can be unit-tested independently of the
// ORT C API (the test program does not need to link against ORT).
//
// The profile is a JSON array of objects. Each object may have:
//   "cat": "Node"            — a node kernel execution event
//   "args": { "provider": "CPUExecutionProvider", ... }
//
// count_cpu_nodes() returns the number of top-level objects with
// cat == "Node" whose args.provider is "CPUExecutionProvider" — i.e. the
// number of nodes that ran on the CPU EP (the fallback provider).
//
// We do not use a full JSON parser library (the harness must stay a single
// translation unit with no external deps). Instead we scan the file with a
// small state machine that tracks the current top-level object's cat and
// args.provider. This is sufficient for the ORT profile format, which is a
// flat array of flat objects.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace ort_smoke {

// A very small, allocation-light JSON tokenizer sufficient for parsing the
// ORT profile. It walks the input and reports tokens; the caller assembles
// them into the structure it needs.
struct JsonTok {
    enum class Type { String, Number, LBrace, RBrace, LBracket, RBracket, Comma, Colon, True, False, Null, End };
    Type type;
    std::string text;   // for String (unescaped) and Number
};

class JsonLexer {
public:
    explicit JsonLexer(const std::string& s) : s_(s), pos_(0) {}

    JsonTok next() {
        skip_ws();
        if (pos_ >= s_.size()) return {JsonTok::Type::End, ""};
        char c = s_[pos_];
        switch (c) {
            case '{': ++pos_; return {JsonTok::Type::LBrace, ""};
            case '}': ++pos_; return {JsonTok::Type::RBrace, ""};
            case '[': ++pos_; return {JsonTok::Type::LBracket, ""};
            case ']': ++pos_; return {JsonTok::Type::RBracket, ""};
            case ',': ++pos_; return {JsonTok::Type::Comma, ""};
            case ':': ++pos_; return {JsonTok::Type::Colon, ""};
            case '"': return read_string();
            case 't': return read_keyword("true", JsonTok::Type::True);
            case 'f': return read_keyword("false", JsonTok::Type::False);
            case 'n': return read_keyword("null", JsonTok::Type::Null);
            default:
                if (c == '-' || (c >= '0' && c <= '9')) return read_number();
                // Unknown character — skip it (the profile is well-formed so
                // this should not happen, but we stay robust).
                ++pos_;
                return next();
        }
    }

private:
    void skip_ws() {
        while (pos_ < s_.size()) {
            char c = s_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    JsonTok read_string() {
        // Assumes s_[pos_] == '"'.
        ++pos_;
        std::string out;
        while (pos_ < s_.size()) {
            char c = s_[pos_++];
            if (c == '"') {
                return {JsonTok::Type::String, out};
            }
            if (c == '\\') {
                if (pos_ >= s_.size()) break;
                char e = s_[pos_++];
                switch (e) {
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case '/': out += '/'; break;
                    case 'n': out += '\n'; break;
                    case 'r': out += '\r'; break;
                    case 't': out += '\t'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case 'u': {
                        // Skip 4 hex digits; emit '?' for non-ASCII (we only
                        // care about ASCII provider names).
                        pos_ += 4;
                        out += '?';
                        break;
                    }
                    default: out += e; break;
                }
            } else {
                out += c;
            }
        }
        return {JsonTok::Type::String, out};
    }

    JsonTok read_number() {
        size_t start = pos_;
        if (pos_ < s_.size() && s_[pos_] == '-') ++pos_;
        while (pos_ < s_.size() && ((s_[pos_] >= '0' && s_[pos_] <= '9') ||
                                    s_[pos_] == '.' || s_[pos_] == 'e' ||
                                    s_[pos_] == 'E' || s_[pos_] == '+' ||
                                    s_[pos_] == '-')) {
            ++pos_;
        }
        return {JsonTok::Type::Number, s_.substr(start, pos_ - start)};
    }

    JsonTok read_keyword(const char* kw, JsonTok::Type t) {
        size_t len = std::strlen(kw);
        if (pos_ + len <= s_.size() && s_.compare(pos_, len, kw) == 0) {
            pos_ += len;
            return {t, kw};
        }
        // Mismatch — skip one char and continue.
        ++pos_;
        return next();
    }

    const std::string& s_;
    size_t pos_;
};

// Skip a single JSON value (string, number, object, array, true, false,
// null) starting from the given token (which has already been read).
inline void skip_value(JsonLexer& lex, const JsonTok& tok) {
    if (tok.type == JsonTok::Type::String || tok.type == JsonTok::Type::Number ||
        tok.type == JsonTok::Type::True || tok.type == JsonTok::Type::False ||
        tok.type == JsonTok::Type::Null) {
        return;
    }
    if (tok.type == JsonTok::Type::LBrace) {
        int depth = 1;
        while (depth > 0) {
            JsonTok t = lex.next();
            if (t.type == JsonTok::Type::End) return;
            if (t.type == JsonTok::Type::LBrace) ++depth;
            else if (t.type == JsonTok::Type::RBrace) --depth;
        }
        return;
    }
    if (tok.type == JsonTok::Type::LBracket) {
        int depth = 1;
        while (depth > 0) {
            JsonTok t = lex.next();
            if (t.type == JsonTok::Type::End) return;
            if (t.type == JsonTok::Type::LBracket) ++depth;
            else if (t.type == JsonTok::Type::RBracket) --depth;
        }
        return;
    }
}

// Parse the ORT profile JSON and count node events whose provider is
// CPUExecutionProvider. Returns the count, or -1 on parse error.
//
// The profile is a JSON array of objects. We walk the array and, for each
// object, track whether cat == "Node" and what args.provider is. We do not
// need to fully deserialize nested structures — when we descend into a
// nested object or array we just skip it (the provider field is a direct
// child of args, which is a direct child of the event object).
inline int count_cpu_nodes(const std::string& profile_json) {
    JsonLexer lex(profile_json);
    int cpu_node_count = 0;
    JsonTok tok = lex.next();
    if (tok.type != JsonTok::Type::LBracket) return -1;  // expect top-level array

    while (true) {
        tok = lex.next();
        if (tok.type == JsonTok::Type::End) return -1;
        if (tok.type == JsonTok::Type::RBracket) break;  // end of array
        if (tok.type == JsonTok::Type::Comma) continue;  // separator between elements
        if (tok.type != JsonTok::Type::LBrace) return -1;  // expect object

        // Parse one event object.
        bool is_node = false;
        std::string provider;
        bool have_provider = false;
        while (true) {
            tok = lex.next();
            if (tok.type == JsonTok::Type::End) return -1;
            if (tok.type == JsonTok::Type::RBrace) break;  // end of object
            if (tok.type == JsonTok::Type::Comma) continue;
            if (tok.type != JsonTok::Type::String) return -1;  // expect key
            std::string key = tok.text;
            tok = lex.next();
            if (tok.type != JsonTok::Type::Colon) return -1;
            tok = lex.next();
            if (key == "cat") {
                if (tok.type == JsonTok::Type::String) {
                    is_node = (tok.text == "Node");
                }
            } else if (key == "args") {
                // args is an object. Parse it looking for "provider".
                if (tok.type != JsonTok::Type::LBrace) {
                    // Skip whatever value this is.
                    skip_value(lex, tok);
                    continue;
                }
                while (true) {
                    tok = lex.next();
                    if (tok.type == JsonTok::Type::End) return -1;
                    if (tok.type == JsonTok::Type::RBrace) break;
                    if (tok.type == JsonTok::Type::Comma) continue;
                    if (tok.type != JsonTok::Type::String) return -1;
                    std::string akey = tok.text;
                    tok = lex.next();
                    if (tok.type != JsonTok::Type::Colon) return -1;
                    tok = lex.next();
                    if (akey == "provider") {
                        if (tok.type == JsonTok::Type::String) {
                            provider = tok.text;
                            have_provider = true;
                        }
                    } else {
                        // Skip the value (could be string, number, object,
                        // array, bool, null).
                        skip_value(lex, tok);
                    }
                }
            } else {
                // Skip the value for any other key.
                skip_value(lex, tok);
            }
        }
        if (is_node && have_provider && provider == "CPUExecutionProvider") {
            ++cpu_node_count;
        }
    }
    return cpu_node_count;
}

}  // namespace ort_smoke
