#!/bin/sh
# 在内存较充足的电脑上把官方 BGE 模型导出为适合服务器 CPU 的 AVX2 INT8 ONNX。
set -eu

output_dir=${1:-models/bge-small-zh-v1.5-int8}
if [ -e "$output_dir/model_quantized.onnx" ]; then
    echo "目标模型已存在：$output_dir/model_quantized.onnx" >&2
    exit 1
fi

work_dir=$(mktemp -d)
trap 'rm -rf "$work_dir"' EXIT HUP INT TERM
fp32_dir="$work_dir/fp32"
int8_dir="$work_dir/int8"

optimum-cli export onnx \
    --model BAAI/bge-small-zh-v1.5 \
    --task feature-extraction \
    --framework pt \
    --optimize O2 \
    --output_dir "$fp32_dir"

optimum-cli onnxruntime quantize \
    --onnx_model "$fp32_dir" \
    --avx2 \
    -o "$int8_dir"

mkdir -p "$output_dir"
quantized_model=$(find "$int8_dir" -maxdepth 1 -type f -name '*quantized.onnx' | head -n 1)
if [ -z "$quantized_model" ]; then
    echo "量化命令没有生成 ONNX 文件。" >&2
    exit 1
fi
cp "$quantized_model" "$output_dir/model_quantized.onnx"

for file_name in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt; do
    if [ -f "$int8_dir/$file_name" ]; then
        cp "$int8_dir/$file_name" "$output_dir/$file_name"
    elif [ -f "$fp32_dir/$file_name" ]; then
        cp "$fp32_dir/$file_name" "$output_dir/$file_name"
    fi
done

if [ ! -f "$output_dir/tokenizer.json" ]; then
    echo "Tokenizer 文件不完整，导出失败。" >&2
    exit 1
fi

echo "已生成：$output_dir"
