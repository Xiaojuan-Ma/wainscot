py_library(
    name = "nets_factory",
    srcs = ["nets_factory.py"],
    visibility = ["//visibility:public"],
    deps = [
        ":cifarnet",
        ":inception_v3",
        ":nasnet",
        ":pnasnet",
    ],
)

py_library(
    name = "cifarnet",
    srcs = ["cifarnet.py"],
    deps = [],
)

py_library(
    name = "inception_utils",
    srcs = ["inception_utils.py"],
)

py_library(
    name = "nasnet_utils",
    srcs = ["nasnet_utils.py"],
)

py_library(
    name = "inception_v3",
    srcs = ["inception_v3.py"],
    deps = [
        ":inception_utils",
    ],
)

py_library(
    name = "nasnet",
    srcs = ["nasnet.py"],
    deps = [
        ":nasnet_utils",
    ],
)


py_library(
    name = "pnasnet",
    srcs = ["pnasnet.py"],
    deps = [
        ":nasnet_utils",
    ],
)