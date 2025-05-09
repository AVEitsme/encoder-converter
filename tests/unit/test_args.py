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


def test_successful_parse_args():
    EXPECTED_PROG_NAME = "encoder-converter"
    EXPECTED_MODEL = "test-project/test-repo"
    EXPECTED_OUTPUT_DIR = "test-output"
    EXPECTED_CACHE_DIR = "test-cache"
    with patch(
        "sys.argv",
        [
            EXPECTED_PROG_NAME,
            "-m",
            EXPECTED_MODEL,
            "-o",
            EXPECTED_OUTPUT_DIR,
            "-c",
            EXPECTED_CACHE_DIR,
        ],
    ):
        args = parse_args()
        assert args.model == EXPECTED_MODEL
        assert args.output_dir == EXPECTED_OUTPUT_DIR
        assert args.cache_dir == EXPECTED_CACHE_DIR


def test_failure_parse_args():
    EXPECTED_PROG_NAME = "encoder-converter"
    EXPECTED_MODEL = "test-model"
    EXPECTED_CACHE_DIR = "test-cache"
    with patch(
        "sys.argv",
        [
            EXPECTED_PROG_NAME,
            "-m",
            EXPECTED_MODEL,
            "-c",
            EXPECTED_CACHE_DIR,
        ],
    ):
        with pytest.raises(SystemExit):
            _ = parse_args()
