py_library(
    name = "logger",
    srcs = ["logger.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "communication_bench",
    srcs = ["communication_benchmark.py"],
    visibility = ["//visibility:public"],
)


py_binary(
    name = "communication_benchmark",
    srcs = ["communication_benchmark.py"],
    deps = [":logger",
            ":communication_bench",
    ],
)
