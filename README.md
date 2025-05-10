# Encoder converter
:rocket: Easy way to convert huggingface encoder model to other formats.
## Description
Encoder converter is a package that allows you to convert the huggingface encoder model to other formats (e.g. onnx).
## Features
Unfinished features will be implemented in future versions.
- [x] Convert encoder model to onnx.
- [x] Convert encoder model to openvino.
- [ ] Convert model with custom wrapper.
## Installation
```bash
pip install encoder-converter
```
## Usage
### Run
```bash
convertencoder --model-name project/huggingface_repo --format onnx --output-dir /my/output/dir --cache-dir /cache/dir --model-output-name t5_encoder
```
### Parameters
| Parameter             | Description                                               | Default   |
|-----------------------|-----------------------------------------------------------|-----------|
| `--model-name`        | Huggingface model name                                    |           |
| `--format`            | Compiled model format. Available: `onnx`, `openvino`      |           |
| `--output-dir`        | Path to save compiled model and tokenizer artifacts.      |           |
| `--cache-dir`         | Path to a directory in which a downloaded pretrained model configuration should be cached while compiling.                                                             |  `/tmp`   |
| `--model-output-name` | If not specified, the default output model name will be parsed depends on the `model_name` parameter.                                                             |           |
If `--model-output-name` is not specified, you can find the complied model at `output_dir`/`huggingface_repo`.`extension`:
1. output_dir - `--ouput-dir` parameter.
2. huggingface_repo - extracts from the `--model-name` parameter.
3. extension - depends on selected `format` parameter:
    * `onnx` - `onnx`
    * `openvino` - `xml`
