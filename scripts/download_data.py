"""Download the UCI "Predict Students' Dropout and Academic Success" dataset.

Source: Realinho, V., Vieira Martins, M., Machado, J. & Baptista, L. (2021).
UCI Machine Learning Repository, https://doi.org/10.24432/C5MC89 (CC BY 4.0).

The file is saved unchanged as data/students.csv and checked against the
SHA-256 of the version used for the results in the README. Column renaming
and encoding happen in src/preprocessing.py.

Usage:
    python scripts/download_data.py
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/697/"
    "predict+students+dropout+and+academic+success.zip"
)
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "students.csv"
EXPECTED_SHA256 = "3ef126de5cefff26eb11fbb4237f1a1401cb64b488e2f1d598c23cedeb4c45ae"


def main() -> None:
    with urllib.request.urlopen(UCI_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    content = archive.read("data.csv")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(content)

    digest = hashlib.sha256(content).hexdigest()
    status = "matches" if digest == EXPECTED_SHA256 else "DOES NOT match"
    print(f"Wrote {OUTPUT.name} ({len(content):,} bytes)")
    print(f"SHA-256 {digest} {status} the version used in the README")


if __name__ == "__main__":
    main()
