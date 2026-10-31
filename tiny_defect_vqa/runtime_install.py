"""Install selected official TensorRT DLLs with per-member integrity checks."""

import io
import urllib.request
import argparse
import base64
import csv
import hashlib
import json
import zipfile
import struct
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VERSION = "10.14.1.48"
WHEEL = f"https://pypi.nvidia.com/tensorrt-cu13-libs/tensorrt_cu13_libs-{VERSION}-py2.py3-none-win_amd64.whl"


def verify_original_record(path, expected):
    """Verify RECORD, including NVIDIA's demonstrated post-RECORD PE signing.

    This release signs DLLs after writing RECORD. Restore only the PE checksum
    and certificate directory and remove the appended WIN_CERTIFICATE: that
    exact unsigned payload must match RECORD, and Windows must validate the
    current NVIDIA Authenticode signature. Neither check may be skipped.
    """

    def encode(value):
        return "sha256=" + base64.urlsafe_b64encode(value.digest()).decode().rstrip("=")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    if encode(digest) == expected:
        return "exact original wheel RECORD"
    with path.open("rb") as handle:
        header = bytearray(handle.read(4096))
        if header[:2] != b"MZ":
            raise IOError(f"RECORD mismatch for a non-PE file: {path.name}")
        pe = struct.unpack_from("<I", header, 0x3C)[0]
        if header[pe : pe + 4] != b"PE\0\0":
            raise IOError("Invalid PE header")
        optional = pe + 24
        magic = struct.unpack_from("<H", header, optional)[0]
        if magic not in {0x10B, 0x20B}:
            raise IOError("Unsupported PE optional header")
        security = optional + (112 if magic == 0x20B else 96) + 8 * 4
        offset, size = struct.unpack_from("<II", header, security)
        if offset < len(header) or size == 0 or offset + size != path.stat().st_size:
            raise IOError("RECORD mismatch is not the known appended-signature case")
        header[optional + 64 : optional + 68] = b"\0" * 4
        header[security : security + 8] = b"\0" * 8
        unsigned = hashlib.sha256(header)
        remaining = offset - len(header)
        while remaining:
            block = handle.read(min(8 << 20, remaining))
            if not block:
                raise IOError("Truncated signed PE")
            unsigned.update(block)
            remaining -= len(block)
    if encode(unsigned) != expected:
        raise IOError(
            f"Unsigned payload does not match original wheel RECORD: {path.name}"
        )
    literal = str(path.resolve()).replace("'", "''")
    shell = shutil.which("pwsh") or shutil.which("powershell.exe")
    if shell is None:
        raise RuntimeError(
            "Windows PowerShell is required for the signed-SDK integrity check"
        )
    command = f"[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); $ErrorActionPreference='Stop'; $s=Get-AuthenticodeSignature -LiteralPath '{literal}'; [pscustomobject]@{{status=[string]$s.Status;signer=$s.SignerCertificate.Subject}} | ConvertTo-Json -Compress"
    result = subprocess.run(
        [shell, "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )
    signature = json.loads(result.stdout)
    if signature["status"] != "Valid" or "O=NVIDIA Corporation" not in signature.get(
        "signer", ""
    ):
        raise IOError(f"NVIDIA Authenticode verification failed: {signature}")
    return "exact unsigned wheel RECORD plus valid NVIDIA Authenticode"


class HTTPRangeFile(io.RawIOBase):
    """Seekable HTTP byte ranges for reading a few members of the 19-GB ZIP."""

    def __init__(self, url, block_size=1 << 20):
        super().__init__()
        self.url, self.block_size, self.position = url, block_size, 0
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as response:
            self.length = int(response.headers["Content-Length"])
        self.cache_start, self.cache = 0, b""
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = (
            offset
            if whence == io.SEEK_SET
            else (
                self.position + offset
                if whence == io.SEEK_CUR
                else self.length + offset
            )
        )
        if not 0 <= position <= self.length:
            raise ValueError("ZIP seek is outside the official archive")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        size = min(size, self.length - self.position)
        if size == 0:
            return b""
        if size > 128 << 20:
            raise ValueError(
                "A single requested ZIP range exceeds the bounded subset budget"
            )
        if (
            self.cache_start <= self.position
            and self.position + size <= self.cache_start + len(self.cache)
        ):
            start = self.position - self.cache_start
            result = self.cache[start : start + size]
        else:
            start = self.position
            stop = min(self.length, start + max(size, self.block_size))
            self.cache = self.fetch_range(start, stop)
            self.transferred += len(self.cache)
            self.cache_start = start
            result = self.cache[:size]
        self.position += len(result)
        return result

    def fetch_range(self, start, stop):
        url = self.url + ("&" if "?" in self.url else "?") + f"segment={start}"
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{stop - 1}"}
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            expected = f"bytes {start}-{stop - 1}/{self.length}"
            if (
                response.status != 206
                or response.headers.get("Content-Range") != expected
            ):
                raise RuntimeError(
                    "The server ignored the exact byte range; refusing an unbounded transfer"
                )
            data = response.read()
        if len(data) != stop - start:
            raise IOError("The official archive returned an incomplete byte range")
        return data


def install_selected(output, sm=89):
    """Extract only official GPU-specific SDK payloads, not a partial pip wheel.

    Each DLL is streamed through ZIP CRC and its original wheel RECORD SHA256.
    Full runtime wheels on this network can silently truncate. Independent
    byte ranges avoid disabling pip's integrity checks or installing that file.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with HTTPRangeFile(WHEEL) as remote, zipfile.ZipFile(remote) as archive:
        names = archive.namelist()
        record_name = next(name for name in names if name.endswith(".dist-info/RECORD"))
        records = {
            row[0]: row[1]
            for row in csv.reader(
                io.StringIO(archive.read(record_name).decode("utf-8"))
            )
        }
        license_name = next(
            name for name in names if name.endswith(".dist-info/LICENSE.txt")
        )
        (output / "LICENSE.txt").write_bytes(archive.read(license_name))
    dlls = [
        "nvinfer_10.dll",
        "nvonnxparser_10.dll",
        "nvinfer_plugin_10.dll",
        f"nvinfer_builder_resource_sm{sm}_10.dll",
    ]

    def extract(name):
        member = "tensorrt_libs/" + name
        if member not in records:
            raise ValueError(f"The official runtime does not contain {name}")
        expected = records[member]
        if not expected.startswith("sha256="):
            raise ValueError(f"Official RECORD has no SHA256 for {name}")
        target = output / name
        if target.exists():
            proof = verify_original_record(target, expected)
            return dict(
                file=name,
                bytes=target.stat().st_size,
                record_integrity_verified=True,
                proof=proof,
                cache=True,
            )
        temporary = target.with_suffix(".part")
        if temporary.exists():
            proof = verify_original_record(temporary, expected)
            temporary.replace(target)
            return dict(
                file=name,
                bytes=target.stat().st_size,
                record_integrity_verified=True,
                proof=proof,
                cache=True,
            )
        with (
            HTTPRangeFile(WHEEL, block_size=8 << 20) as source,
            zipfile.ZipFile(source) as archive,
        ):
            temporary = target.with_suffix(".part")
            value = hashlib.sha256()
            with archive.open(member) as incoming, temporary.open("wb") as outgoing:
                while block := incoming.read(8 << 20):
                    outgoing.write(block)
                    value.update(block)
            observed = "sha256=" + base64.urlsafe_b64encode(
                value.digest()
            ).decode().rstrip("=")
            proof = (
                "exact original wheel RECORD"
                if observed == expected
                else verify_original_record(temporary, expected)
            )
            temporary.replace(target)
        print(f"Verified official {name}: {target.stat().st_size} bytes", flush=True)
        return dict(
            file=name,
            bytes=target.stat().st_size,
            record_integrity_verified=True,
            proof=proof,
            cache=False,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(extract, dlls))
    report = dict(
        version=VERSION,
        source=WHEEL,
        selected_compute_capability=sm,
        files=files,
        packaging="selected original SDK DLLs; not a complete installed tensorrt-cu13-libs wheel",
        activation="add this directory and CUDA13/bin to the process PATH before importing the separately installed TensorRT Python bindings",
    )
    (output / "runtime_manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sm", type=int, default=89)
    args = parser.parse_args()
    print(json.dumps(install_selected(args.output, args.sm), indent=2))


if __name__ == "__main__":
    main()
