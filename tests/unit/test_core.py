from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from encoder_converter.core import (
    ModelFormatNotSupportedError,
    convert_to_onnx,
    convert_to_openvino,
    generate_model_path,
)


@pytest.mark.parametrize(
    "base_dir,model_name,extension,expected_path",
    [
        ("/base/path", "encoder", "onnx", "/base/path/encoder.onnx"),
        ("/base/path", "encoder", "bin", "/base/path/encoder.bin"),
        ("/base/path", "encoder", "xml", "/base/path/encoder.xml"),
    ],
)
def test_generate_model_path(base_dir, model_name, extension, expected_path):
    actual_path = generate_model_path(
        base_dir=Path(base_dir), model_name=model_name, extension=extension
    )
    assert actual_path == expected_path


def test_model_format_not_supported_error():
    EXPECTED_STRING = "Unexpected compiled model format `test`. Supported formats are 'onnx' and 'openvino'."
    exc = ModelFormatNotSupportedError(format="test")
    assert str(exc) == EXPECTED_STRING


def test_convert_to_onnx():
    EXPECTED_GENERATED_MODEL_PATH = "/base/path/encoder.test"
    mock_model = MagicMock()
    mock_dummy_input = MagicMock()
    mock_output_dir = MagicMock()
    mock_output_dir.exists.return_value = False
    mock_output_dir.mkdir.return_value = None
    with (
        patch("torch.onnx.export", return_value=None) as mock_export,
        patch(
            "encoder_converter.core.generate_model_path",
            return_value=EXPECTED_GENERATED_MODEL_PATH,
        ) as mock_generate_model_path,
    ):
        actual_path = convert_to_onnx(
            model=mock_model, dummy_input=mock_dummy_input, output_dir=mock_output_dir
        )
        mock_output_dir.exists.assert_called_once()
        mock_output_dir.mkdir.assert_called_once()
        mock_export.assert_called_once_with(
            mock_model,
            mock_dummy_input,
            mock_generate_model_path.return_value,
            input_names=["input_ids"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {1: "sequence_length"},
            },
        )
        assert actual_path == EXPECTED_GENERATED_MODEL_PATH


def test_convert_to_openvino():
    INPUT_MODEL_PATH = "input-test-path"
    OUTPUT_MODEL_PATH = Path("test-out")
    MODEL_NAME = "encoder"
    EXTENSION = "xml"
    EXPECTED_GENERATED_MODEL_PATH = "/base/path/encoder.test"
    mock_model = MagicMock()

    with (
        patch("openvino.convert_model", return_value=mock_model) as mock_convert,
        patch("openvino.save_model", return_value=None) as mock_save,
        patch(
            "encoder_converter.core.generate_model_path",
            return_value=EXPECTED_GENERATED_MODEL_PATH,
        ) as mock_generate_model_path,
    ):
        actual_path = convert_to_openvino(
            INPUT_MODEL_PATH,
            OUTPUT_MODEL_PATH,
            MODEL_NAME,
        )
        mock_convert.assert_called_once_with(INPUT_MODEL_PATH)
        mock_save.assert_called_once_with(mock_model, EXPECTED_GENERATED_MODEL_PATH)
        mock_generate_model_path.assert_called_once_with(
            base_dir=OUTPUT_MODEL_PATH, model_name=MODEL_NAME, extension=EXTENSION
        )
        assert actual_path == EXPECTED_GENERATED_MODEL_PATH
