#pragma once
#include <torch/extension.h>
void double_buffer_add(torch::Tensor &input);

void origin_add(torch::Tensor &input);