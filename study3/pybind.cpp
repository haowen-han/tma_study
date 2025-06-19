#include "ops.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    //my custom ops
    pybind11::module ops = m.def_submodule("ops", "my custom operators");
    ops.def(
        "double_buffer_add",
        &double_buffer_add,
        "");
    ops.def(
        "origin_add",
        &origin_add,
        "");
}