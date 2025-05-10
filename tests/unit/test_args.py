import argparse
from unittest.mock import patch

import pytest

from encoder_converter.args import huggingface_model_name, parse_args


def test_successful_huggingface_model_name():
    EXPECTED_MODEL_NAME = "test-project/test-repo"
    model_name = huggingface_model_name(EXPECTED_MODEL_NAME)
    assert model_name == EXPECTED_MODEL_NAME


def test_faulure_huggingface_model_name():
    with pytest.raises(argparse.ArgumentTypeError):
        _ = huggingface_model_name("test")


@pytest.mark.parametrize(
    "model_name,format,output_dir,cache_dir",
    [
        ("test-project/test-repo", "onnx", "test-output", "test-cache"),
        ("test-project/test-repo", "openvino", "test-output", "test-cache"),
    ],
)
def test_successful_parse_args(model_name, format, output_dir, cache_dir):
    PROG_NAME = "encoder-converter"
    with patch(
        "sys.argv",
        [PROG_NAME, "-m", model_name, "-f", format, "-o", output_dir, "-c", cache_dir],
    ):
        args = parse_args()
        assert args.model_name == model_name
        assert args.format == format
        assert args.output_dir == output_dir
        assert args.cache_dir == cache_dir


@pytest.mark.parametrize(
    "model_name,format,output_dir,cache_dir",
    [
        ("test-project", "onnx", "test-output", "test-cache"),
        ("test-project/test-repo", "test", "test-output", "test-cache"),
    ],
)
def test_failure_parse_args(model_name, format, output_dir, cache_dir):
    PROG_NAME = "encoder-converter"
    with patch(
        "sys.argv",
        [PROG_NAME, "-m", model_name, "-f", format, "-o", output_dir, "-c", cache_dir],
    ):
        with pytest.raises(SystemExit):
            _ = parse_args()
