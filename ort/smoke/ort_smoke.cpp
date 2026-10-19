// Native ORT runtime smoke harness.
//
// Loads the ORT shared library built by the current job via the platform
// dynamic loader (dlopen / LoadLibrary) and exercises the ORT C API directly.
// It never uses the onnxruntime Python wheel as proof that the new library
// works.
//
// It runs one full HTDemucs inference window. One zero-filled float32 tensor
// is built per model input, using each input's declared static shape from the
// session, so the harness carries no model interface of its own. The shipped
// spectral interface is:
//   inputs  spectral=[1,2,2,2048,336], mix=[1,2,343980]
//   outputs spectral_out=[1,4,2,2,2048,336], time_out=[1,4,2,343980]
// It rejects any output containing NaN or Inf. The expected output shape is
// required from the caller via --expect-output-shape (the semicolon-joined,
// space-free per-output shape string) so the shape gate stays interface-aware
// without this harness hard-coding an interface.
//
// The requested execution provider is mandatory. CPU is tested in a separate
// invocation; a non-CPU provider failure must never be hidden by a replacement
// CPU session.
//
// To measure the fallback node count, the harness enables the ORT session
// profiler before creating the session. After inference it calls
// SessionEndProfiling to obtain the profile file path, parses the profile
// JSON, and counts the number of node events whose "args.provider" is
// "CPUExecutionProvider" — i.e. the number of nodes that ran on the CPU
// (the fallback EP) rather than the requested provider. When the requested
// provider is CPU itself, this count is the total node count (every node
// ran on CPU) and used_fallback is false.
//
// It writes a JSON object to stdout with the inference results. A Python
// wrapper (scripts/run_native_smoke.py) adds the runtime/model digests and
// target and writes the final report file.
//
// Usage:
//   ort_smoke --lib <path-to-libonnxruntime> --model <path-to.onnx> \
//             --provider <cpu|coreml|xnnpack|directml> [--target <triple>] \
//             [--expect-output-shape "<shape-string>"]
//
// Build:
//   See ort/smoke/CMakeLists.txt. The ORT C API header is included from the
//   pinned ORT source checkout so the API struct layout matches the built
//   library's ORT_API_VERSION.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <stdexcept>
#include <climits>
#if defined(_WIN32)
#  include <windows.h>
#  include <psapi.h>
#else
#  include <sys/resource.h>
#endif

// Platform dynamic loading.
#if !defined(_WIN32)
#  include <dlfcn.h>
#endif

// ORT C API header (from the pinned source checkout, passed via -I).
#include <onnxruntime/core/session/onnxruntime_c_api.h>

// Profile JSON parser (shared with the unit test).
#include "profile_parser.hpp"

namespace {

// Minimal JSON string escaper.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

// RAII wrapper for the loaded dynamic library.
class DynLib {
public:
#if defined(_WIN32)
    using Handle = HMODULE;
#else
    using Handle = void*;
#endif

    explicit DynLib(const std::string& path) : handle_(nullptr), error_("") {
#if defined(_WIN32)
        handle_ = LoadLibraryExA(
            path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!handle_) {
            DWORD code = GetLastError();
            error_ = "LoadLibraryA failed (code " + std::to_string(code) + ") for " + path;
        }
#else
        handle_ = dlopen(path.c_str(), RTLD_NOW);
        if (!handle_) {
            const char* err = dlerror();
            error_ = std::string("dlopen failed: ") + (err ? err : "unknown") + " for " + path;
        }
#endif
    }
    ~DynLib() {
#if defined(_WIN32)
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
    }
    bool ok() const { return handle_ != nullptr; }
    const std::string& error() const { return error_; }

    template <typename T>
    T sym(const char* name, std::string& err) {
#if defined(_WIN32)
        auto p = GetProcAddress(handle_, name);
#else
        auto p = dlsym(handle_, name);
#endif
        if (!p) {
#if defined(_WIN32)
            DWORD code = GetLastError();
            err = std::string("GetProcAddress failed for ") + name + " (code " + std::to_string(code) + ")";
#else
            const char* e = dlerror();
            err = std::string("dlsym failed for ") + name + ": " + (e ? e : "unknown");
#endif
            return nullptr;
        }
        return reinterpret_cast<T>(p);
    }

private:
    Handle handle_;
    std::string error_;
};

// Read a file into a buffer.
std::vector<uint8_t> read_file(const std::string& path, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        err = "cannot open file: " + path;
        return {};
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    if (sz < 0) {
        err = "ftell failed: " + path;
        fclose(f);
        return {};
    }
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (sz > 0 && fread(buf.data(), 1, static_cast<size_t>(sz), f) != static_cast<size_t>(sz)) {
        err = "short read: " + path;
        fclose(f);
        return {};
    }
    fclose(f);
    return buf;
}

// Check that a float buffer is entirely finite (no NaN, no Inf).
bool all_finite(const float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(data[i])) return false;
    }
    return true;
}

// Custom ORT logger that writes to stderr (NOT stdout).
// The harness prints its JSON report to stdout; if ORT's default logger
// also writes to stdout, log lines pollute the JSON and break parsing.
// Routing logs to stderr keeps stdout clean for the JSON output.
void stderr_logger(void* /*param*/, OrtLoggingLevel severity,
                   const char* category, const char* logid,
                   const char* /*code_location*/, const char* message) {
    const char* sev = "?";
    switch (severity) {
        case ORT_LOGGING_LEVEL_VERBOSE: sev = "V"; break;
        case ORT_LOGGING_LEVEL_INFO:    sev = "I"; break;
        case ORT_LOGGING_LEVEL_WARNING: sev = "W"; break;
        case ORT_LOGGING_LEVEL_ERROR:   sev = "E"; break;
        case ORT_LOGGING_LEVEL_FATAL:   sev = "F"; break;
    }
    fprintf(stderr, "[ort][%s][%s][%s] %s\n", sev, category, logid, message);
}

long long native_peak_rss_kb() {
    long long peak_rss_kb = 0;
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS memory{};
        if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &memory, sizeof(memory))) throw std::runtime_error("cannot measure native peak RSS");
        peak_rss_kb = static_cast<long long>(memory.PeakWorkingSetSize / 1024);
#else
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("cannot measure native peak RSS");
#  if defined(__APPLE__)
        peak_rss_kb = usage.ru_maxrss / 1024;
#  else
        peak_rss_kb = usage.ru_maxrss;
#  endif
#endif
    return peak_rss_kb;
}

void print_json_field(const char* key, const std::string& value, bool last) {
    printf("  \"%s\": \"%s\"%s\n", key, json_escape(value).c_str(), last ? "" : ",");
}

void print_json_field_bool(const char* key, bool value, bool last) {
    printf("  \"%s\": %s%s\n", key, value ? "true" : "false", last ? "" : ",");
}

// Print the trailing fields that every JSON output (success or failure)
// must include so the schema is consistent across all paths.
void print_json_trailer(int cpu_node_count, bool used_fallback,
                        int total_node_count = -1,
                        int provider_node_count = -1) {
    if (total_node_count >= 0) {
        printf("  \"total_node_count\": %d,\n", total_node_count);
    } else {
        printf("  \"total_node_count\": null,\n");
    }
    if (cpu_node_count >= 0) {
        printf("  \"cpu_node_count\": %d,\n", cpu_node_count);
        printf("  \"fallback_node_count\": %d,\n", cpu_node_count);
    } else {
        printf("  \"cpu_node_count\": null,\n");
        printf("  \"fallback_node_count\": null,\n");
    }
    if (provider_node_count >= 0) {
        printf("  \"provider_node_count\": %d,\n", provider_node_count);
    } else {
        printf("  \"provider_node_count\": null,\n");
    }
    print_json_field_bool("used_fallback", used_fallback, true);
    printf("}\n");
}

}  // namespace

int main(int argc, char** argv) {
    std::string lib_path;
    std::string model_path;
    std::string provider = "cpu";
    std::string target = "unknown";
    // Expected semicolon-joined per-output shape string, supplied by the
    // caller. There is deliberately no default: the only shape worth
    // defaulting to belonged to the retired waveform interface, so a caller
    // that forgot the flag would have silently gated on a shape no current
    // model produces.
    std::string expect_output_shape;
    std::string input_dir, output_dir;
    int warmup = 0, iters = 1;
    auto parse_count = [](const char* value) {
        char* end = nullptr;
        long n = std::strtol(value, &end, 10);
        if (!*value || *end || n < 0 || n > INT_MAX) throw std::invalid_argument("invalid iteration count");
        return static_cast<int>(n);
    };

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--lib" && i + 1 < argc) lib_path = argv[++i];
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--provider" && i + 1 < argc) provider = argv[++i];
        else if (a == "--target" && i + 1 < argc) target = argv[++i];
        else if (a == "--expect-output-shape" && i + 1 < argc) expect_output_shape = argv[++i];
        else if (a == "--input-dir" && i + 1 < argc) input_dir = argv[++i];
        else if (a == "--output-dir" && i + 1 < argc) output_dir = argv[++i];
        else if (a == "--warmup" && i + 1 < argc) {
            try { warmup = parse_count(argv[++i]); } catch (...) { return 2; }
        }
        else if (a == "--iters" && i + 1 < argc) {
            try { iters = parse_count(argv[++i]); } catch (...) { return 2; }
        }
        else if (a == "--help") {
            fprintf(stderr, "usage: ort_smoke --lib <lib> --model <onnx> "
                            "--expect-output-shape <str> "
                            "--provider <name> [--target <triple>]\n");
            return 0;
        }
    }

    if (lib_path.empty() || model_path.empty() || expect_output_shape.empty() || (iters == 0 && warmup != 0) || input_dir.empty() != output_dir.empty()) {
        fprintf(stderr, "ERROR: --lib, --model, and --expect-output-shape are required\n");
        return 2;
    }

    // Begin JSON output on stdout.
    printf("{\n");
    print_json_field("target", target, false);
    print_json_field("requested_provider", provider, false);

    // Load the runtime shared library.
    std::string load_err;
    DynLib lib(lib_path);
    if (!lib.ok()) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error", lib.error(), false);
        print_json_field("inference", "not_attempted", false);
        print_json_field("available_providers", "", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", "", false);
        print_json_trailer(-1, false);
        return 1;
    }

    // Resolve OrtGetApiBase.
    std::string sym_err;
    auto get_api_base = lib.sym<const OrtApiBase* (*)()>("OrtGetApiBase", sym_err);
    if (!get_api_base) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error", sym_err, false);
        print_json_field("inference", "not_attempted", false);
        print_json_field("available_providers", "", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", "", false);
        print_json_trailer(-1, false);
        return 1;
    }

    const OrtApiBase* api_base = get_api_base();
    if (!api_base) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error", "OrtGetApiBase returned null", false);
        print_json_field("inference", "not_attempted", false);
        print_json_field("available_providers", "", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", "", false);
        print_json_trailer(-1, false);
        return 1;
    }

    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    if (!api) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error",
                         "GetApi(ORT_API_VERSION) returned null; header/library mismatch",
                         false);
        print_json_field("inference", "not_attempted", false);
        print_json_field("available_providers", "", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", "", false);
        print_json_trailer(-1, false);
        return 1;
    }

    // Record available providers.
    char** providers = nullptr;
    int n_providers = 0;
    std::string available;
    if (api->GetAvailableProviders(&providers, &n_providers) == nullptr) {
        for (int i = 0; i < n_providers; ++i) {
            if (i) available += ",";
            available += providers[i];
        }
        api->ReleaseAvailableProviders(providers, n_providers);
    }
    print_json_field("available_providers", available, false);

    // Create environment with a custom logger that writes to stderr,
    // keeping stdout clean for the JSON report.
    OrtEnv* env = nullptr;
    {
        OrtStatus* st = api->CreateEnvWithCustomLogger(
            stderr_logger, nullptr, ORT_LOGGING_LEVEL_WARNING,
            "ort_smoke", &env);
        if (st) {
            print_json_field("session_creation", "failed", false);
            print_json_field("session_creation_error", api->GetErrorMessage(st), false);
            api->ReleaseStatus(st);
            print_json_field("inference", "not_attempted", false);
            print_json_field("output_shape", "", false);
            print_json_field_bool("finite_output", false, false);
            print_json_field("provider_assignment", "", false);
            print_json_trailer(-1, false);
            return 1;
        }
    }

    // Read model bytes.
    std::string model_err;
    std::vector<uint8_t> model_bytes = read_file(model_path, model_err);
    if (model_bytes.empty()) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error", model_err, false);
        print_json_field("inference", "not_attempted", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", "", false);
        print_json_trailer(-1, false);
        api->ReleaseEnv(env);
        return 1;
    }

    // Append the requested execution provider. Try the requested one first;
    // on failure fall back to CPU.
    std::string provider_assigned;
    bool used_fallback = false;
    std::string last_session_error;  // captured for diagnostics

    auto try_create_session = [&](const std::string& prov) -> OrtSession* {
        // Reset options for each attempt.
        OrtSessionOptions* o = nullptr;
        OrtStatus* st = api->CreateSessionOptions(&o);
        if (st) {
            last_session_error = std::string("CreateSessionOptions: ") +
                                 api->GetErrorMessage(st);
            api->ReleaseStatus(st);
            return nullptr;
        }
        // CPU EP is the default and does not need to be appended explicitly.
        if (prov != "cpu" && prov != "CPUExecutionProvider") {
            if (prov == "directml" || prov == "DmlExecutionProvider") {
                // DirectML requires sequential execution and disabled memory
                // patterns. The source-built Windows runtime exports the DML
                // provider factory beside OrtGetApiBase.
                st = api->DisableMemPattern(o);
                if (!st) st = api->SetSessionExecutionMode(o, ORT_SEQUENTIAL);
                if (st) {
                    last_session_error = std::string("configure DirectML session: ") +
                                         api->GetErrorMessage(st);
                    api->ReleaseStatus(st);
                    api->ReleaseSessionOptions(o);
                    return nullptr;
                }

                using AppendDmlFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
                std::string dml_symbol_error;
                auto append_dml = lib.sym<AppendDmlFn>(
                    "OrtSessionOptionsAppendExecutionProvider_DML", dml_symbol_error);
                if (!append_dml) {
                    last_session_error = dml_symbol_error;
                    api->ReleaseSessionOptions(o);
                    return nullptr;
                }
                st = append_dml(o, 0);
                if (st) {
                    last_session_error = std::string("AppendExecutionProvider(DML): ") +
                                         api->GetErrorMessage(st);
                    api->ReleaseStatus(st);
                    api->ReleaseSessionOptions(o);
                    return nullptr;
                }
            } else {
                std::string ep_name;
                if (prov == "coreml" || prov == "CoreMLExecutionProvider") {
                    ep_name = "CoreMLExecutionProvider";
                } else if (prov == "xnnpack" || prov == "XnnpackExecutionProvider") {
                    ep_name = "XnnpackExecutionProvider";
                } else {
                    last_session_error = "unknown provider: " + prov;
                    api->ReleaseSessionOptions(o);
                    return nullptr;
                }
                st = api->SessionOptionsAppendExecutionProvider(
                    o, ep_name.c_str(), nullptr, nullptr, 0);
                if (st) {
                    last_session_error = std::string("AppendExecutionProvider(") +
                                         ep_name + "): " + api->GetErrorMessage(st);
                    api->ReleaseStatus(st);
                    api->ReleaseSessionOptions(o);
                    return nullptr;
                }
            }
        }
        // Enable profiling so we can measure the fallback node count after
        // inference. The profile file prefix is a path without extension;
        // ORT appends a timestamp and ".json". Use a unique prefix per
        // attempt so the profile of a failed attempt does not overwrite the
        // successful one.
        static int profile_counter = 0;
        std::string profile_prefix = "ort_smoke_profile_";
        profile_prefix += std::to_string(profile_counter++);
#if defined(_WIN32)
        // ORTCHAR_T is wchar_t on Windows.
        std::wstring wprefix(profile_prefix.begin(), profile_prefix.end());
        st = api->EnableProfiling(o, wprefix.c_str());
#else
        st = api->EnableProfiling(o, profile_prefix.c_str());
#endif
        if (st) {
            // Profiling is optional for the smoke test; if it fails to enable
            // we still proceed (fallback_node_count will be -1 / null).
            api->ReleaseStatus(st);
        }
        st = api->SetIntraOpNumThreads(o, 2);
        if (!st) st = api->SetInterOpNumThreads(o, 1);
        if (st) {
            last_session_error = std::string("Set thread count: ") + api->GetErrorMessage(st);
            api->ReleaseStatus(st);
            api->ReleaseSessionOptions(o);
            return nullptr;
        }
        OrtSession* sess = nullptr;
        st = api->CreateSessionFromArray(
            env, model_bytes.data(), model_bytes.size(), o, &sess);
        api->ReleaseSessionOptions(o);
        if (st) {
            last_session_error = std::string("CreateSessionFromArray: ") +
                                 api->GetErrorMessage(st);
            api->ReleaseStatus(st);
            return nullptr;
        }
        return sess;
    };

    using Clock = std::chrono::steady_clock;
    auto rss_before = native_peak_rss_kb();
    auto session_start = Clock::now();
    OrtSession* session = try_create_session(provider);
    double cold_load_s = std::chrono::duration<double>(Clock::now() - session_start).count();
    if (session) {
        provider_assigned = provider;
    }

    if (!session) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error",
                         last_session_error.empty()
                             ? "session creation failed for requested provider"
                             : last_session_error,
                         false);
        print_json_field("inference", "not_attempted", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", "", false);
        print_json_trailer(-1, used_fallback);
        api->ReleaseEnv(env);
        return 1;
    }

    print_json_field("session_creation", "ok", false);

    // Inspect inputs/outputs.
    size_t n_inputs = 0, n_outputs = 0;
    api->SessionGetInputCount(session, &n_inputs);
    api->SessionGetOutputCount(session, &n_outputs);

    OrtAllocator* allocator = nullptr;
    if (api->GetAllocatorWithDefaultOptions(&allocator) != nullptr) {
        print_json_field("inference", "failed", false);
        print_json_field("inference_error", "GetAllocatorWithDefaultOptions failed", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", provider_assigned, false);
        print_json_trailer(-1, used_fallback);
        api->ReleaseSession(session);
        api->ReleaseEnv(env);
        return 1;
    }

    OrtMemoryInfo* mem_info = nullptr;
    api->CreateCpuMemoryInfo(OrtAllocatorType::OrtDeviceAllocator,
                              OrtMemType::OrtMemTypeDefault, &mem_info);

    // Build one zero-filled float32 tensor per model input, using each
    // input's declared static shape from the session. This adapts to the
    // single-input waveform interface and the two-input spectral-core
    // interface without hard-coding either shape.
    std::vector<char*> input_names_owned(n_inputs, nullptr);
    std::vector<const char*> input_names_c(n_inputs, nullptr);
    std::vector<OrtValue*> input_tensors(n_inputs, nullptr);
    std::vector<std::vector<float>> input_buffers(n_inputs);
    std::string input_build_error;

    for (size_t i = 0; i < n_inputs; ++i) {
        api->SessionGetInputName(session, i, allocator, &input_names_owned[i]);
        input_names_c[i] = input_names_owned[i];

        OrtTypeInfo* in_type_info = nullptr;
        OrtStatus* ti_st = api->SessionGetInputTypeInfo(session, i, &in_type_info);
        if (ti_st) {
            input_build_error = std::string("SessionGetInputTypeInfo[") +
                std::to_string(i) + "]: " + api->GetErrorMessage(ti_st);
            api->ReleaseStatus(ti_st);
            break;
        }
        const OrtTensorTypeAndShapeInfo* in_tensor_info = nullptr;
        api->CastTypeInfoToTensorInfo(in_type_info, &in_tensor_info);
        if (!in_tensor_info) {
            input_build_error = "input " + std::to_string(i) + " is not a tensor";
            api->ReleaseTypeInfo(in_type_info);
            break;
        }
        ONNXTensorElementDataType input_type;
        ti_st = api->GetTensorElementType(in_tensor_info, &input_type);
        if (ti_st || input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            input_build_error = "native harness requires float32 input tensors";
            if (ti_st) api->ReleaseStatus(ti_st);
            api->ReleaseTypeInfo(in_type_info);
            break;
        }
        size_t ndims = 0;
        api->GetDimensionsCount(in_tensor_info, &ndims);
        std::vector<int64_t> dims(ndims, 0);
        api->GetDimensions(in_tensor_info, dims.data(), ndims);
        api->ReleaseTypeInfo(in_type_info);

        int64_t count = 1;
        bool dynamic = (ndims == 0);
        for (size_t d = 0; d < ndims; ++d) {
            if (dims[d] <= 0) { dynamic = true; break; }
            count *= dims[d];
        }
        if (dynamic) {
            input_build_error = std::string("input '") +
                (input_names_owned[i] ? input_names_owned[i] : "?") +
                "' has a non-static shape; the smoke harness requires static input dims";
            break;
        }
        input_buffers[i].assign(static_cast<size_t>(count), 0.0f);
        if (!input_dir.empty()) {
            std::string name(input_names_c[i]);
            if (name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos) {
                input_build_error = "unsafe input tensor name";
                break;
            }
            std::string error;
            auto bytes = read_file(input_dir + "/" + name + ".f32", error);
            if (!error.empty() || bytes.size() != static_cast<size_t>(count) * sizeof(float)) {
                input_build_error = "input tensor byte size mismatch: " + name + "; " + error;
                break;
            }
            std::memcpy(input_buffers[i].data(), bytes.data(), bytes.size());
            if (!all_finite(input_buffers[i].data(), static_cast<size_t>(count))) {
                input_build_error = "nonfinite input tensor: " + name;
                break;
            }
        }
        OrtStatus* ct_st = api->CreateTensorWithDataAsOrtValue(
            mem_info, input_buffers[i].data(),
            static_cast<size_t>(count) * sizeof(float),
            dims.data(), ndims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &input_tensors[i]);
        if (ct_st) {
            input_build_error = std::string("CreateTensorWithDataAsOrtValue[") +
                std::to_string(i) + "]: " + api->GetErrorMessage(ct_st);
            api->ReleaseStatus(ct_st);
            break;
        }
    }

    auto free_inputs = [&]() {
        for (size_t i = 0; i < n_inputs; ++i) {
            if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
            if (input_names_owned[i]) api->AllocatorFree(allocator, input_names_owned[i]);
        }
    };

    if (!input_build_error.empty()) {
        print_json_field("inference", "failed", false);
        print_json_field("inference_error", input_build_error, false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", provider_assigned, false);
        print_json_trailer(-1, used_fallback);
        free_inputs();
        api->ReleaseMemoryInfo(mem_info);
        api->ReleaseSession(session);
        api->ReleaseEnv(env);
        return 1;
    }

    // Get output names.
    std::vector<char*> output_names(n_outputs, nullptr);
    for (size_t i = 0; i < n_outputs; ++i) {
        api->SessionGetOutputName(session, i, allocator, &output_names[i]);
    }
    // Run() takes const char* const* / const OrtValue* const*; build
    // const-correct arrays from the allocator-returned char* names.
    std::vector<const char*> output_names_c(n_outputs, nullptr);
    for (size_t i = 0; i < n_outputs; ++i) output_names_c[i] = output_names[i];

    std::vector<OrtValue*> outputs(n_outputs, nullptr);
    auto outputs_finite = [&]() {
        for (auto output : outputs) {
            OrtTensorTypeAndShapeInfo* info = nullptr;
            OrtStatus* error = api->GetTensorTypeAndShape(output, &info);
            size_t count = 0;
            ONNXTensorElementDataType type;
            if (!error) error = api->GetTensorElementType(info, &type);
            if (!error) error = api->GetTensorShapeElementCount(info, &count);
            if (info) api->ReleaseTensorTypeAndShapeInfo(info);
            float* data = nullptr;
            if (!error) error = api->GetTensorMutableData(output, reinterpret_cast<void**>(&data));
            if (error) { api->ReleaseStatus(error); return false; }
            if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || !data || !count || !all_finite(data, count)) return false;
        }
        return true;
    };
    auto first_start = Clock::now();
    OrtStatus* st = api->Run(session, nullptr,
                             input_names_c.data(), input_tensors.data(), n_inputs,
                             output_names_c.data(), n_outputs, outputs.data());

    double first_window_s = std::chrono::duration<double>(Clock::now() - first_start).count();
    bool finite_runs = !st && outputs_finite();
    std::vector<double> latencies;
    if (!st && finite_runs && !input_dir.empty()) {
        for (int64_t iteration = 0; iteration < static_cast<int64_t>(warmup) + iters; ++iteration) {
            for (auto& output : outputs) { api->ReleaseValue(output); output = nullptr; }
            auto start = Clock::now();
            st = api->Run(session, nullptr, input_names_c.data(), input_tensors.data(), n_inputs,
                          output_names_c.data(), n_outputs, outputs.data());
            double seconds = std::chrono::duration<double>(Clock::now() - start).count();
            if (st) break;
            finite_runs = outputs_finite();
            if (!finite_runs) break;
            if (iteration >= warmup) latencies.push_back(seconds);
        }
    }
    if (st) {
        print_json_field("inference", "failed", false);
        print_json_field("inference_error", api->GetErrorMessage(st), false);
        api->ReleaseStatus(st);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", provider_assigned, false);
        print_json_trailer(-1, used_fallback);
        for (size_t i = 0; i < n_outputs; ++i) if (output_names[i]) api->AllocatorFree(allocator, output_names[i]);
        free_inputs();
        api->ReleaseMemoryInfo(mem_info);
        api->ReleaseSession(session);
        api->ReleaseEnv(env);
        return 1;
    }

    print_json_field("inference", "ok", false);

    // Inspect output shape and finiteness.
    std::string shape_str;
    bool finite = finite_runs;
    for (size_t i = 0; i < n_outputs; ++i) {
        OrtTypeInfo* type_info = nullptr;
        api->GetTypeInfo(outputs[i], &type_info);
        const OrtTensorTypeAndShapeInfo* tensor_info = nullptr;
        if (type_info) {
            api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
        }
        if (tensor_info) {
            size_t ndims = 0;
            api->GetDimensionsCount(tensor_info, &ndims);
            std::vector<int64_t> dims(ndims, 0);
            api->GetDimensions(tensor_info, dims.data(), ndims);
            if (i) shape_str += ";";
            shape_str += "[";
            for (size_t d = 0; d < ndims; ++d) {
                if (d) shape_str += ",";
                shape_str += std::to_string(dims[d]);
            }
            shape_str += "]";
            // Read the data and check finiteness.
            float* out_data = nullptr;
            api->GetTensorMutableData(outputs[i], reinterpret_cast<void**>(&out_data));
            size_t elem_count = 1;
            for (size_t d = 0; d < ndims; ++d) elem_count *= static_cast<size_t>(dims[d]);
            if (out_data && elem_count > 0) {
                if (!all_finite(out_data, elem_count)) finite = false;
                if (!output_dir.empty()) {
                    std::string path = output_dir + "/output_" + std::to_string(i) + ".f32";
                    FILE* file = fopen(path.c_str(), "wb");
                    bool written = file != nullptr;
                    if (file) {
                        written = fwrite(out_data, sizeof(float), elem_count, file) == elem_count;
                        if (fclose(file) != 0) written = false;
                    }
                    if (!written) finite = false;
                }
            }
            // NOTE: Do NOT call ReleaseTensorTypeAndShapeInfo on tensor_info.
            // CastTypeInfoToTensorInfo returns a non-owning cast pointer
            // ("Do not free this value, it will be valid until type_info is
            // freed"). Freeing it causes a double-free / heap corruption.
            // It is released when type_info is released below.
        }
        if (type_info) api->ReleaseTypeInfo(type_info);
    }
    print_json_field("output_shape", shape_str, false);
    const bool shape_valid = shape_str == expect_output_shape;
    print_json_field_bool("finite_output", finite, false);
    print_json_field("provider_assignment", provider_assigned, false);

    // End profiling and count total, CPU, and exact requested-provider nodes.
    int cpu_node_count = -1;
    int total_node_count = -1;
    std::string profile_json;
    char* profile_path_c = nullptr;
    OrtStatus* prof_st = api->SessionEndProfiling(session, allocator, &profile_path_c);
    if (prof_st) {
        // Profiling was not enabled or failed; record null.
        api->ReleaseStatus(prof_st);
    } else if (profile_path_c) {
        std::string profile_path(profile_path_c);
        api->AllocatorFree(allocator, profile_path_c);
        std::string prof_err;
        std::vector<uint8_t> prof_bytes = read_file(profile_path, prof_err);
        if (!prof_bytes.empty()) {
            profile_json.assign(prof_bytes.begin(), prof_bytes.end());
            cpu_node_count = ort_smoke::count_cpu_nodes(profile_json);
            total_node_count = ort_smoke::count_total_nodes(profile_json);
        }
        // Best-effort cleanup of the profile file (it is also gitignored).
        remove(profile_path.c_str());
    }

    const char* requested_profile_provider = nullptr;
    if (provider == "cpu") {
        requested_profile_provider = "CPUExecutionProvider";
    } else if (provider == "coreml") {
        requested_profile_provider = "CoreMLExecutionProvider";
    } else if (provider == "xnnpack") {
        requested_profile_provider = "XnnpackExecutionProvider";
    } else if (provider == "directml") {
        requested_profile_provider = "DmlExecutionProvider";
    }

    int provider_node_count = -1;
    if (!profile_json.empty() && requested_profile_provider != nullptr) {
        provider_node_count = ort_smoke::count_nodes(
            profile_json, requested_profile_provider);
    }
    if (!input_dir.empty()) {
        long long peak_rss_kb = native_peak_rss_kb();
        printf("  \"cold_load_s\": %.9f,\n", cold_load_s);
        printf("  \"first_window_s\": %.9f,\n", first_window_s);
        printf("  \"peak_rss_kb\": %lld,\n", peak_rss_kb);
        printf("  \"rss_delta_kb\": %lld,\n", peak_rss_kb - rss_before);
        printf("  \"warm_latencies_s\": [");
        for (size_t i = 0; i < latencies.size(); ++i) printf("%s%.9f", i ? "," : "", latencies[i]);
        printf("],\n");
    }
    print_json_trailer(
        cpu_node_count, used_fallback, total_node_count, provider_node_count);

    const bool provider_used =
        provider_assigned == provider && provider_node_count > 0;
    const bool strict_success = finite && shape_valid && provider_used && !used_fallback;

    // Cleanup.
    for (size_t i = 0; i < n_outputs; ++i) {
        if (output_names[i]) api->AllocatorFree(allocator, output_names[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    free_inputs();
    api->ReleaseMemoryInfo(mem_info);
    api->ReleaseSession(session);
    api->ReleaseEnv(env);
    return strict_success ? 0 : 3;
}
