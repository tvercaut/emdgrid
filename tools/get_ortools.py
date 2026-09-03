#!/usr/bin/env python3

import argparse
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import traceback
import zipfile
from pathlib import Path

RELEASE_TAG = "v9.15"
API_URL = f"https://api.github.com/repos/google/or-tools/releases/tags/{RELEASE_TAG}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and extract OR-Tools binaries."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected asset without downloading it.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("third_party"),
        help="Directory where the archive will be extracted.",
    )

    return parser.parse_args()


def get_linux_distro_info():
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return None, None

    data = {}
    for line in os_release.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip().strip('"')

    distro_id = data.get("ID")
    version_id = data.get("VERSION_ID")
    return distro_id, version_id


def get_platform_patterns():
    system = platform.system()
    machine = platform.machine().lower()

    if machine in ("amd64", "x86_64"):
        arch_patterns = ["amd64", "x86_64", "x64"]
    elif machine in ("arm64", "aarch64"):
        arch_patterns = ["arm64", "aarch64"]
    else:
        raise RuntimeError(f"Unsupported architecture: {machine}")

    distro_id, version_id = None, None

    if system == "Windows":
        os_patterns = ["win", "windows"]
        ext_patterns = [".zip"]
    elif system == "Linux":
        os_patterns = ["linux"]
        ext_patterns = [".tar.gz"]
        distro_id, version_id = get_linux_distro_info()
    elif system == "Darwin":
        os_patterns = ["osx", "mac", "darwin"]
        ext_patterns = [".tar.gz", ".zip"]
    else:
        raise RuntimeError(f"Unsupported operating system: {system}")

    return os_patterns, arch_patterns, ext_patterns, distro_id, version_id


def score_asset(name, os_patterns, arch_patterns, ext_patterns, distro_id, version_id):
    name_lower = name.lower()

    score = 0

    # Look for C++ binaries
    if "cpp" in name_lower:
        score += 100
    else:
        score -= 100

    if any(p in name_lower for p in os_patterns):
        score += 10

    if any(p in name_lower for p in arch_patterns):
        score += 10

    if distro_id and distro_id.lower() in name_lower:
        score += 50
        if version_id and version_id.lower() in name_lower:
            score += 20

    if any(name_lower.endswith(ext) for ext in ext_patterns):
        score += 5

    return score


def fetch_release_metadata():
    with tempfile.NamedTemporaryFile(
        suffix=".json",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "-o",
                str(tmp_path),
                API_URL,
            ],
            check=True,
        )

        return json.loads(tmp_path.read_text(encoding="utf-8"))

    finally:
        tmp_path.unlink(missing_ok=True)


def download_file(url: str, destination: Path):
    subprocess.run(
        [
            "curl",
            "-fL",
            "-o",
            str(destination),
            url,
        ],
        check=True,
    )


def extract_archive(archive: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    if archive.name.endswith(".tar.gz"):
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(output_dir)

    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(output_dir)

    else:
        raise RuntimeError(f"Unsupported archive format: {archive}")


def main():
    args = parse_args()

    print(f"Fetching release metadata for {RELEASE_TAG}...")
    release = fetch_release_metadata()

    assets = release.get("assets", [])
    if not assets:
        raise RuntimeError("No assets found")

    os_patterns, arch_patterns, ext_patterns, distro_id, version_id = (
        get_platform_patterns()
    )

    print(
        f"Platform: system={platform.system()}, machine={platform.machine()}, distro={distro_id}, version={version_id}"
    )

    best_asset = max(
        assets,
        key=lambda a: score_asset(
            a["name"],
            os_patterns,
            arch_patterns,
            ext_patterns,
            distro_id,
            version_id,
        ),
    )

    url = best_asset["browser_download_url"]
    filename = best_asset["name"]

    print(f"Selected asset: {filename}")
    print(f"URL: {url}")
    print(f"Output directory: {args.output_path}")

    if args.dry_run:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        archive = Path(tmpdir) / filename

        print("Downloading archive...")
        download_file(url, archive)

        print("Extracting archive...")
        extract_archive(archive, args.output_path)

    extracted_dirs = list(args.output_path.glob("or-tools*"))
    if not extracted_dirs:
        raise RuntimeError("Extracted directory not found")

    extracted_output_path = extracted_dirs[0].resolve()

    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"ortools_dir={extracted_output_path}\n")

    print(f"Extracted to: {extracted_output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
