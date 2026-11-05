"""Test harness guards: no network by default, no real downloads, `data` tests reach Hugging Face only.

pyproject.toml sets the 60 s timeout, `--disable-socket` and `-m 'not data'`.
"""
import huggingface_hub
import pytest

# The only hosts a `data` test may connect to. pytest-socket resolves names to IPs when the
# test starts; if a CDN rotates to an unlisted address the connection fails closed.
HF_HOSTS = [
    "huggingface.co",
    "hf.co",
    "cdn-lfs.huggingface.co",
    "cdn-lfs-us-1.huggingface.co",
    "cdn-lfs.hf.co",
    "cas-bridge.xethub.hf.co",
    "cas-server.xethub.hf.co",
    "transfer.xethub.hf.co",
]


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("data"):
            item.add_marker(pytest.mark.allow_hosts(HF_HOSTS))


@pytest.fixture(autouse=True)
def _no_real_downloads(request, monkeypatch):
    """A default test can never fetch from the Hub, even if a socket guard were lifted."""
    if request.node.get_closest_marker("data"):
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("hf_hub_download is blocked in tests; inject a fake downloader")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", blocked)
