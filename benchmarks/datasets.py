"""Small, chunked dataset helpers shared by heavyweight benchmarks."""
from pathlib import Path
from urllib.request import Request, urlopen


SIFT1M_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"


def download_file(url, destination, chunk_bytes=8 * 1024 * 1024):
    """Download atomically so an interrupted transfer is never accepted."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "VectorDB-benchmark/1.0"})
    with urlopen(request) as response, partial.open("wb") as output:
        while True:
            chunk = response.read(chunk_bytes)
            if not chunk:
                break
            output.write(chunk)
    partial.replace(destination)
    return destination


def ensure_sift1m(path, download=False):
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"{path} does not exist; rerun with --download")
        download_file(SIFT1M_URL, path)
    return path


def validate_sift_file(handle):
    required = {"train", "test", "neighbors"}
    missing = required.difference(handle.keys())
    if missing:
        raise ValueError(f"SIFT HDF5 is missing datasets: {sorted(missing)}")
    train, test, neighbors = (
        handle["train"], handle["test"], handle["neighbors"])
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("train and test must be 2D with matching dimensions")
    if neighbors.ndim != 2 or len(neighbors) != len(test):
        raise ValueError("neighbors must contain one row per test query")
    return train, test, neighbors


def iter_rows(dataset, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(dataset), batch_size):
        yield start, dataset[start:start + batch_size]
