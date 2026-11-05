import socket

import huggingface_hub
import pytest
from pytest_socket import SocketBlockedError, SocketConnectBlockedError

from conftest import HF_HOSTS


def test_timeout_is_60_seconds(request):
    assert int(request.config.getini("timeout")) == 60


def test_data_tests_are_deselected_by_default(request):
    assert "not data" in request.config.getoption("markexpr")


def test_network_is_blocked_by_default():
    with pytest.raises(SocketBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_hub_download_is_blocked_by_default():
    with pytest.raises(RuntimeError, match="blocked"):
        huggingface_hub.hf_hub_download("PolyAI/banking77", "data/train-00000-of-00001.parquet")


@pytest.mark.data
def test_data_marker_allows_only_hugging_face_hosts(request):
    marker = request.node.get_closest_marker("allow_hosts")
    assert marker is not None and list(marker.args[0]) == HF_HOSTS
    assert all(h == "hf.co" or h.endswith((".huggingface.co", ".hf.co")) or h == "huggingface.co" for h in HF_HOSTS)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(SocketConnectBlockedError):
        sock.connect(("93.184.216.34", 80))  # not a Hugging Face address; refused before any packet is sent
