// Unit tests for the ORT profile JSON parser (ort/smoke/profile_parser.hpp).
//
// This is a standalone C++ program that does not link against ORT. It is
// compiled and run by tests/test_profile_parser.py.
//
// Each test prints "PASS" or "FAIL: <reason>" to stdout and returns 0 on
// success, non-zero on failure.

#include <cstdio>
#include <string>
#include "profile_parser.hpp"

using ort_smoke::count_cpu_nodes;

static int failures = 0;

void check_eq(int actual, int expected, const char* name) {
    if (actual != expected) {
        printf("FAIL: %s: expected %d, got %d\n", name, expected, actual);
        ++failures;
    } else {
        printf("PASS: %s\n", name);
    }
}

int main() {
    // Empty array.
    check_eq(count_cpu_nodes("[]"), 0, "empty_array");

    // No node events.
    check_eq(count_cpu_nodes(
        R"([{"cat":"Session","name":"model_loading","args":{}}])"),
        0, "no_node_events");

    // One CPU node.
    check_eq(count_cpu_nodes(
        R"([{"cat":"Node","name":"n0","args":{"provider":"CPUExecutionProvider","op_name":"Conv"}}])"),
        1, "one_cpu_node");

    // One non-CPU node (CoreML).
    check_eq(count_cpu_nodes(
        R"([{"cat":"Node","name":"n0","args":{"provider":"CoreMLExecutionProvider"}}])"),
        0, "one_coreml_node");

    // Mixed: 2 CPU + 1 CoreML.
    check_eq(count_cpu_nodes(
        R"([
            {"cat":"Node","name":"n0","args":{"provider":"CoreMLExecutionProvider"}},
            {"cat":"Node","name":"n1","args":{"provider":"CPUExecutionProvider"}},
            {"cat":"Session","name":"session_initialization","args":{}},
            {"cat":"Node","name":"n2","args":{"provider":"CPUExecutionProvider"}}
        ])"),
        2, "mixed_cpu_and_coreml");

    // Node with no provider field — should not count.
    check_eq(count_cpu_nodes(
        R"([{"cat":"Node","name":"n0","args":{"op_name":"Conv"}}])"),
        0, "node_no_provider");

    // Node with empty args — should not count.
    check_eq(count_cpu_nodes(
        R"([{"cat":"Node","name":"n0","args":{}}])"),
        0, "node_empty_args");

    // Nested args values (arrays/objects) should be skipped correctly.
    check_eq(count_cpu_nodes(
        R"([{"cat":"Node","name":"n0","args":{"provider":"CPUExecutionProvider","input_type_shape":[{"float":[1,3,224,224]}],"thread_scheduling_stats":{"main_thread":{"thread_pool_name":"pool-1"}}}}])"),
        1, "nested_args_values");

    // Realistic ORT profile excerpt.
    check_eq(count_cpu_nodes(
        R"([
            {"cat":"Session","name":"model_loading_array","dur":49,"args":{}},
            {"cat":"Session","name":"session_initialization","dur":183,"args":{}},
            {"cat":"Node","name":"n0_kernel_time","dur":1058,"args":{"provider":"CPUExecutionProvider","op_name":"Conv","node_index":0}},
            {"cat":"Node","name":"n1_kernel_time","dur":1048,"args":{"provider":"CPUExecutionProvider","op_name":"PRelu","node_index":1}},
            {"cat":"Node","name":"n3_kernel_time","dur":428,"args":{"provider":"CPUExecutionProvider","op_name":"ReduceMax","node_index":3}}
        ])"),
        3, "realistic_profile");

    // Invalid JSON (not an array) — should return -1.
    check_eq(count_cpu_nodes("{}"), -1, "not_array");

    // Empty string — should return -1.
    check_eq(count_cpu_nodes(""), -1, "empty_string");

    // Truncated JSON — should return -1.
    check_eq(count_cpu_nodes("[{\"cat\":\"Node\""), -1, "truncated");

    if (failures == 0) {
        printf("ALL PASS\n");
        return 0;
    }
    printf("%d FAILURE(S)\n", failures);
    return 1;
}
