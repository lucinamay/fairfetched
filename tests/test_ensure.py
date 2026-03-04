"""Tests for _ensure module, specifically ensure_url function."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fairfetched.get._ensure import ensure_url


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_ensure_url_downloads_file(temp_dir):
    """Test that ensure_url successfully downloads a file."""
    target_path = temp_dir / "subdir" / "testfile.txt"
    test_content = b"Hello, World!"

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[test_content])
        mock_get.return_value = mock_response

        result = ensure_url("http://example.com/testfile.txt", target_path)

    # Verify the file was created
    assert result.exists()
    assert result == target_path
    assert target_path.parent.exists()
    assert target_path.read_bytes() == test_content


def test_ensure_url_creates_parent_directories(temp_dir):
    """Test that ensure_url creates necessary parent directories."""
    target_path = temp_dir / "deep" / "nested" / "directory" / "file.txt"
    test_content = b"Test content"

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[test_content])
        mock_get.return_value = mock_response

        result = ensure_url("http://example.com/file.txt", target_path)

    # Verify all parent directories were created
    assert target_path.parent.exists()
    assert result.exists()


def test_ensure_url_skips_existing_file(temp_dir):
    """Test that ensure_url skips download if file already exists."""
    target_path = temp_dir / "existing.txt"
    original_content = b"Original content"
    target_path.write_bytes(original_content)

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        result = ensure_url("http://example.com/file.txt", target_path)

    # Verify file was not downloaded (mock_get not called)
    mock_get.assert_not_called()
    assert result == target_path
    assert target_path.read_bytes() == original_content


def test_ensure_url_force_redownload(temp_dir):
    """Test that ensure_url can force redownload with force=True."""
    target_path = temp_dir / "file.txt"
    target_path.write_bytes(b"Old content")
    new_content = b"New content"

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[new_content])
        mock_get.return_value = mock_response

        result = ensure_url("http://example.com/file.txt", target_path, force=True)

    # Verify the file was re-downloaded
    mock_get.assert_called_once()
    assert target_path.read_bytes() == new_content


def test_ensure_url_handles_chunked_response(temp_dir):
    """Test that ensure_url correctly handles chunked response."""
    target_path = temp_dir / "chunked.txt"
    chunks = [b"Hello", b", ", b"World", b"!"]

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=chunks)
        mock_get.return_value = mock_response

        result = ensure_url("http://example.com/chunked.txt", target_path)

    # Verify all chunks were written
    assert target_path.read_bytes() == b"Hello, World!"


def test_ensure_url_raises_on_http_error(temp_dir):
    """Test that ensure_url raises on HTTP errors."""
    target_path = temp_dir / "file.txt"

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="HTTP 404"):
            ensure_url("http://example.com/notfound.txt", target_path)


def test_ensure_url_with_string_path(temp_dir):
    """Test that ensure_url works with string paths."""
    target_path_str = str(temp_dir / "file.txt")
    test_content = b"Test"

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[test_content])
        mock_get.return_value = mock_response

        result = ensure_url("http://example.com/file.txt", target_path_str)

    # Verify it works with string paths
    assert Path(result).exists()
    assert Path(result).read_bytes() == test_content


def test_ensure_url_binary_file(temp_dir):
    """Test that ensure_url correctly handles binary files."""
    target_path = temp_dir / "binary.bin"
    binary_content = bytes(range(256))

    with patch("fairfetched.get._ensure.requests.get") as mock_get:
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[binary_content])
        mock_get.return_value = mock_response

        result = ensure_url("http://example.com/binary.bin", target_path)

    # Verify binary content is preserved
    assert target_path.read_bytes() == binary_content
