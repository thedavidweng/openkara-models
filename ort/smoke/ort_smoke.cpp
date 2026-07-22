// Native ORT runtime smoke harness.
//
// Loads the ORT shared library built by the current job via the platform
// dynamic loader (dlopen / LoadLibrary) and exercises the ORT C API directly.
// It never uses the onnxruntime Python wheel as proof that the new library
// works.
//
// It runs one full HTDemucs inference window:
//   input  [1, 2, 343980]  (float32 zeros)
//   output [1, 4, 2, 343980] (drums/bass/other/vocals)
// and rejects any output containing NaN or Inf.
//
// The requested execution provider is tried first; if session creation with
// that provider fails, the harness falls back to CPU and records the fallback.
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
//             --provider <cpu|coreml|xnnpack|directml> [--target <triple>]
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

// Platform dynamic loading.
#if defined(_WIN32)
#  include <windows.h>
#else
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
        handle_ = LoadLibraryA(path.c_str());
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

void print_json_field(const char* key, const std::string& value, bool last) {
    printf("  \"%s\": \"%s\"%s\n", key, json_escape(value).c_str(), last ? "" : ",");
}

void print_json_field_bool(const char* key, bool value, bool last) {
    printf("  \"%s\": %s%s\n", key, value ? "true" : "false", last ? "" : ",");
}

// Print the trailing fields that every JSON output (success or failure)
// must include so the schema is consistent across all paths. The caller
// has already printed session_creation / inference / output_shape /
// finite_output / provider_assignment as appropriate; this prints the
// final fallback_node_count and used_fallback and closes the object.
void print_json_trailer(int fallback_node_count, bool used_fallback) {
    if (fallback_node_count >= 0) {
        printf("  \"fallback_node_count\": %d,\n", fallback_node_count);
    } else {
        printf("  \"fallback_node_count\": null,\n");
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

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--lib" && i + 1 < argc) lib_path = argv[++i];
        else if (a == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (a == "--provider" && i + 1 < argc) provider = argv[++i];
        else if (a == "--target" && i + 1 < argc) target = argv[++i];
        else if (a == "--help") {
            fprintf(stderr, "usage: ort_smoke --lib <lib> --model <onnx> "
                            "--provider <name> [--target <triple>]\n");
            return 0;
        }
    }

    if (lib_path.empty() || model_path.empty()) {
        fprintf(stderr, "ERROR: --lib and --model are required\n");
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

    // Session options.
    OrtSessionOptions* opts = nullptr;
    if (api->CreateSessionOptions(&opts) != nullptr) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error", "CreateSessionOptions failed", false);
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
        // ORT v1.27.1 exposes a single generic
        // SessionOptionsAppendExecutionProvider(options, provider_name,
        //   provider_options_keys, provider_options_values, num_keys).
        // The per-provider functions (CPU/CoreML/Xnnpack/DML) do not exist in
        // the base OrtApi struct; they are registered by name here.
        // CPU EP is the default and does not need to be appended explicitly;
        // appending it can return an error, so skip it for the cpu provider.
        if (prov != "cpu" && prov != "CPUExecutionProvider") {
            std::string ep_name;
            if (prov == "coreml" || prov == "CoreMLExecutionProvider") {
                ep_name = "CoreMLExecutionProvider";
            } else if (prov == "xnnpack" || prov == "XnnpackExecutionProvider") {
                ep_name = "XnnpackExecutionProvider";
            } else if (prov == "directml" || prov == "DmlExecutionProvider") {
                ep_name = "DmlExecutionProvider";
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

    OrtSession* session = try_create_session(provider);
    if (session) {
        provider_assigned = provider;
    } else {
        // Fallback to CPU.
        session = try_create_session("cpu");
        if (session) {
            provider_assigned = "cpu";
            used_fallback = true;
        }
    }

    if (!session) {
        print_json_field("session_creation", "failed", false);
        print_json_field("session_creation_error",
                         last_session_error.empty()
                             ? "session creation failed for requested provider and CPU fallback"
                             : last_session_error,
                         false);
        print_json_field("inference", "not_attempted", false);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", "", false);
        print_json_trailer(-1, used_fallback);
        api->ReleaseSessionOptions(opts);
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
        api->ReleaseSessionOptions(opts);
        api->ReleaseEnv(env);
        return 1;
    }

    // Get input name.
    char* input_name_c = nullptr;
    if (n_inputs > 0) {
        api->SessionGetInputName(session, 0, allocator, &input_name_c);
    }

    // Build input tensor [1, 2, 343980] float32 zeros.
    const int64_t batch = 1;
    const int64_t channels = 2;
    const int64_t frames = 343980;
    std::vector<int64_t> input_shape = {batch, channels, frames};
    size_t input_count = static_cast<size_t>(batch * channels * frames);
    std::vector<float> input_data(input_count, 0.0f);

    OrtMemoryInfo* mem_info = nullptr;
    api->CreateCpuMemoryInfo(OrtAllocatorType::OrtDeviceAllocator,
                              OrtMemType::OrtMemTypeDefault, &mem_info);

    OrtValue* input_tensor = nullptr;
    OrtStatus* st = api->CreateTensorWithDataAsOrtValue(
        mem_info, input_data.data(), input_count * sizeof(float),
        input_shape.data(), input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);
    if (st) {
        print_json_field("inference", "failed", false);
        print_json_field("inference_error", api->GetErrorMessage(st), false);
        api->ReleaseStatus(st);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", provider_assigned, false);
        print_json_trailer(-1, used_fallback);
        if (input_name_c) api->AllocatorFree(allocator, input_name_c);
        api->ReleaseMemoryInfo(mem_info);
        api->ReleaseSession(session);
        api->ReleaseSessionOptions(opts);
        api->ReleaseEnv(env);
        return 1;
    }

    // Run inference.
    const char* input_names[] = {input_name_c};
    std::vector<char*> output_names;
    output_names.resize(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        api->SessionGetOutputName(session, i, allocator, &output_names[i]);
    }
    // Run() takes const char* const* / const OrtValue* const*; build
    // const-correct arrays from the allocator-returned char* names.
    std::vector<const char*> output_names_c(n_outputs, nullptr);
    for (size_t i = 0; i < n_outputs; ++i) output_names_c[i] = output_names[i];
    const OrtValue* input_arr[] = {input_tensor};

    std::vector<OrtValue*> outputs(n_outputs, nullptr);
    st = api->Run(session, nullptr, input_names, input_arr, 1,
                  output_names_c.data(), n_outputs, outputs.data());

    if (st) {
        print_json_field("inference", "failed", false);
        print_json_field("inference_error", api->GetErrorMessage(st), false);
        api->ReleaseStatus(st);
        print_json_field("output_shape", "", false);
        print_json_field_bool("finite_output", false, false);
        print_json_field("provider_assignment", provider_assigned, false);
        print_json_trailer(-1, used_fallback);
        for (size_t i = 0; i < n_outputs; ++i) if (output_names[i]) api->AllocatorFree(allocator, output_names[i]);
        if (input_name_c) api->AllocatorFree(allocator, input_name_c);
        api->ReleaseValue(input_tensor);
        api->ReleaseMemoryInfo(mem_info);
        api->ReleaseSession(session);
        api->ReleaseSessionOptions(opts);
        api->ReleaseEnv(env);
        return 1;
    }

    print_json_field("inference", "ok", false);

    // Inspect output shape and finiteness.
    std::string shape_str;
    bool finite = true;
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
    print_json_field_bool("finite_output", finite, false);
    print_json_field("provider_assignment", provider_assigned, false);

    // End profiling and parse the profile file to count the nodes that ran
    // on the CPU EP (the fallback provider). SessionEndProfiling returns
    // the path to the profile JSON file via the allocator.
    int fallback_node_count = -1;
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
            std::string prof_json(prof_bytes.begin(), prof_bytes.end());
            fallback_node_count = ort_smoke::count_cpu_nodes(prof_json);
        }
        // Best-effort cleanup of the profile file (it is also gitignored).
        remove(profile_path.c_str());
    }

    print_json_trailer(fallback_node_count, used_fallback);

    // Cleanup.
    for (size_t i = 0; i < n_outputs; ++i) {
        if (output_names[i]) api->AllocatorFree(allocator, output_names[i]);
        if (outputs[i]) api->ReleaseValue(outputs[i]);
    }
    if (input_name_c) api->AllocatorFree(allocator, input_name_c);
    api->ReleaseValue(input_tensor);
    api->ReleaseMemoryInfo(mem_info);
    api->ReleaseSession(session);
    api->ReleaseSessionOptions(opts);
    api->ReleaseEnv(env);
    return finite ? 0 : 3;
}
