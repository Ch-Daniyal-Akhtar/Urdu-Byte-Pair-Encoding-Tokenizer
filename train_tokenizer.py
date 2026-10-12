"""Verify the reference trainer, then train the exact 50,000-token model."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import time
from pathlib import Path

from bpe import BPETokenizer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_utf8_prefix(path: Path, byte_count: int) -> str:
    with path.open("rb") as stream:
        payload = stream.read(byte_count)
    return payload.decode("utf-8", errors="ignore")


def verify_implementations(path: Path, sample_mb: float, merge_count: int) -> dict[str, object]:
    sample = read_utf8_prefix(path, int(sample_mb * 1024 * 1024))
    target = 257 + merge_count
    reference = BPETokenizer(vocab_size=target)
    fast = BPETokenizer(vocab_size=target)

    started = time.perf_counter()
    reference_merges = reference.train_reference(sample, vocab_size=target)
    reference_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fast_merges = fast.train_fast_text(sample, vocab_size=target)
    fast_seconds = time.perf_counter() - started
    if reference_merges != fast_merges:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(reference_merges, fast_merges), start=1)
                if left != right
            ),
            min(len(reference_merges), len(fast_merges)) + 1,
        )
        raise RuntimeError(f"reference/fast merge mismatch at merge {mismatch}")
    return {
        "sample_bytes": len(sample.encode("utf-8")),
        "merges_compared": len(reference_merges),
        "identical": True,
        "reference_seconds": reference_seconds,
        "fast_seconds": fast_seconds,
    }


def total_ram_bytes() -> int | None:
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_physical)
        return None
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, ValueError, OSError):
        return None


def hardware() -> dict[str, object]:
    ram = total_ram_bytes()
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "ram_gib": ram / 1024**3 if ram else None,
        "python": platform.python_version(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the assignment Urdu byte-level BPE tokenizer")
    parser.add_argument("--input", type=Path, default=Path("corpus/train.txt"))
    parser.add_argument("--output", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--vocab-size", type=int, default=50_000)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--verify-sample-mb", type=float, default=1.0)
    parser.add_argument("--verify-merges", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=1_000)
    arguments = parser.parse_args()

    verification = verify_implementations(arguments.input, arguments.verify_sample_mb, arguments.verify_merges)
    print("verification=" + json.dumps(verification, ensure_ascii=False), flush=True)

    tokenizer = BPETokenizer(vocab_size=arguments.vocab_size)
    started = time.perf_counter()

    def report(step: int, total: int, pair: tuple[int, int], count: int) -> None:
        if step == 1 or step % arguments.log_every == 0 or step == total:
            elapsed = time.perf_counter() - started
            print(
                f"merge={step:,}/{total:,} pair={pair} frequency={count:,} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    tokenizer.train_fast_file(
        arguments.input,
        vocab_size=arguments.vocab_size,
        min_frequency=arguments.min_frequency,
        progress=report,
    )
    elapsed = time.perf_counter() - started
    if tokenizer.vocab_size != arguments.vocab_size:
        raise RuntimeError(f"trained vocabulary is {tokenizer.vocab_size}, expected {arguments.vocab_size}")

    metadata = {
        "training_file": str(arguments.input).replace("\\", "/"),
        "training_file_sha256": file_sha256(arguments.input),
        "training_file_bytes": arguments.input.stat().st_size,
        "wall_clock_seconds": elapsed,
        "merges_per_second": len(tokenizer.merges) / elapsed,
        "reference_fast_verification": verification,
        "hardware": hardware(),
    }
    tokenizer.save(arguments.output, metadata=metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
