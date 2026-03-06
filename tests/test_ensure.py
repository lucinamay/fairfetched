"""Tests for _ensure module, specifically ensure_url function."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from fairfetched.utils.ensure import ensure_url


@pytest.fixture
def temp_dir():
    """Create a temporary directory that gets cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_ensure_url_downloads_file(temp_dir):
    """Test that ensure_url successfully downloads a file."""
    target_path = temp_dir / "subdir" / "testfile.txt"
    test_content = b"Hello, World!"

    mock_resp = Mock()
    mock_resp.read = Mock(side_effect=[test_content, b""])
    mock_resp.getheader = Mock(return_value=str(len(test_content)))
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
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

    mock_resp = Mock()
    mock_resp.read = Mock(side_effect=[test_content, b""])
    mock_resp.getheader = Mock(return_value=str(len(test_content)))
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = ensure_url("http://example.com/file.txt", target_path)

    # Verify all parent directories were created
    assert target_path.parent.exists()
    assert result.exists()


def test_ensure_url_skips_existing_file(temp_dir):
    """Test that ensure_url skips download if file already exists."""
    target_path = temp_dir / "existing.txt"
    original_content = b"Original content"
    target_path.write_bytes(original_content)

    with patch("urllib.request.urlopen") as mock_urlopen:
        result = ensure_url("http://example.com/file.txt", target_path)

    # Verify file was not downloaded (mock_urlopen not called)
    mock_urlopen.assert_not_called()
    assert result == target_path
    assert target_path.read_bytes() == original_content


def test_ensure_url_force_redownload(temp_dir):
    """Test that ensure_url can force redownload with force=True."""
    target_path = temp_dir / "file.txt"
    target_path.write_bytes(b"Old content")
    new_content = b"New content"

    mock_resp = Mock()
    mock_resp.read = Mock(side_effect=[new_content, b""])
    mock_resp.getheader = Mock(return_value=str(len(new_content)))
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        _ = ensure_url("http://example.com/file.txt", target_path, force=True)

    # Verify the file was re-downloaded
    mock_urlopen.assert_called_once()
    assert target_path.read_bytes() == new_content


def test_ensure_url_handles_chunked_response(temp_dir):
    """Test that ensure_url correctly handles chunked response."""
    target_path = temp_dir / "chunked.txt"
    chunks = [b"Hello", b", ", b"World", b"!"]

    mock_resp = Mock()
    mock_resp.read = Mock(side_effect=chunks + [b""])
    mock_resp.getheader = Mock(return_value=str(sum(len(c) for c in chunks)))
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        _ = ensure_url("http://example.com/chunked.txt", target_path)

    # Verify all chunks were written
    assert target_path.read_bytes() == b"Hello, World!"


def test_ensure_url_raises_on_http_error(temp_dir):
    """Test that ensure_url raises on HTTP errors."""
    target_path = temp_dir / "file.txt"

    mock_resp = Mock()
    mock_resp.__enter__ = Mock(side_effect=Exception("HTTP 404"))
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(Exception, match="HTTP 404"):
            ensure_url("http://example.com/notfound.txt", target_path)


def test_ensure_url_with_string_path(temp_dir):
    """Test that ensure_url works with string paths."""
    target_path_str = str(temp_dir / "file.txt")
    test_content = b"Test"

    mock_resp = Mock()
    mock_resp.read = Mock(side_effect=[test_content, b""])
    mock_resp.getheader = Mock(return_value=str(len(test_content)))
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        result = ensure_url("http://example.com/file.txt", target_path_str)

    # Verify it works with string paths
    assert Path(result).exists()
    assert Path(result).read_bytes() == test_content


def test_ensure_url_binary_file(temp_dir):
    """Test that ensure_url correctly handles binary files."""
    target_path = temp_dir / "binary.bin"
    binary_content = bytes(range(256))

    mock_resp = Mock()
    mock_resp.read = Mock(side_effect=[binary_content, b""])
    mock_resp.getheader = Mock(return_value=str(len(binary_content)))
    mock_resp.__enter__ = Mock(return_value=mock_resp)
    mock_resp.__exit__ = Mock(return_value=None)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        _ = ensure_url("http://example.com/binary.bin", target_path)

    # Verify binary content is preserved
    assert target_path.read_bytes() == binary_content
