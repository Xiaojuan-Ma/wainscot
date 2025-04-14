py_library(
    name = "transformer",
    srcs = ["transformer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":beam_search_v1",
        ":model_utils",
        ":attention_layer",
        ":embedding_layer",
        ":ffn_layer",
    ],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":model_helpers",
    ],
)

py_library(
    name = "embedding_layer",
    srcs = ["embedding_layer.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":tpu",
    ],
)

py_library(
    name = "attention_layer",
    srcs = ["attention_layer.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "beam_search_v1",
    srcs = ["beam_search_v1.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "ffn_layer",
    srcs = ["ffn_layer.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "model_helpers",
    srcs = ["model_helpers.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "model_params",
    srcs = ["model_params.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "schedule",
    srcs = ["schedule.py"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "tpu",
    srcs = ["tpu.py"],
    visibility = ["//visibility:public"],
)






