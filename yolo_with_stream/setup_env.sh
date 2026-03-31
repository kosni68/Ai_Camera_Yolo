#!/usr/bin/env bash
set -euo pipefail

write_step() {
  printf '\n==> %s\n' "$1"
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

run_with_privileges() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
    return
  fi

  if command_exists sudo; then
    sudo "$@"
    return
  fi

  fail "This script needs sudo or root privileges to install Ubuntu packages."
}

ensure_apt_packages() {
  local missing=()
  local package

  if ! command_exists apt-get; then
    fail "This script is intended for Ubuntu or Debian systems with apt-get."
  fi

  for package in "$@"; do
    if ! dpkg -s "$package" >/dev/null 2>&1; then
      missing+=("$package")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    return 0
  fi

  write_step "Installing Ubuntu packages: ${missing[*]}"
  if ! run_with_privileges apt-get update; then
    fail "Failed to update the apt package index."
  fi

  if ! run_with_privileges env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"; then
    fail "Failed to install Ubuntu packages: ${missing[*]}"
  fi
}

resolve_python312() {
  if command_exists python3.12; then
    python3.12 -c 'import sys; print(sys.executable)'
    return 0
  fi

  if command_exists python3 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
    python3 -c 'import sys; print(sys.executable)'
    return 0
  fi

  return 1
}

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv_dir="$project_dir/.venv"
venv_python="$venv_dir/bin/python"
requirements_file="$project_dir/requirements.txt"

if [[ "$(uname -s)" != "Linux" ]]; then
  fail "This script is intended to run on Linux."
fi

if [[ ! -f "$requirements_file" ]]; then
  fail "The requirements file was not found at '$requirements_file'."
fi

write_step "Checking Python 3.12"
python312_path="$(resolve_python312 || true)"
if [[ -z "$python312_path" ]]; then
  ensure_apt_packages python3.12 python3.12-venv
  python312_path="$(resolve_python312 || true)"
fi

if [[ -z "$python312_path" ]]; then
  fail "Python 3.12 is required. Install 'python3.12' and 'python3.12-venv', then rerun this script."
fi

echo "Using Python 3.12 at $python312_path"

if ! "$python312_path" -m venv --help >/dev/null 2>&1; then
  ensure_apt_packages python3.12-venv
fi

write_step "Checking Tesseract OCR and OpenCV system libraries"
ensure_apt_packages tesseract-ocr libgl1 libglib2.0-0 libgomp1 libgtk-3-0

if [[ -d "$venv_dir" ]]; then
  write_step "Reusing existing virtual environment"

  if [[ ! -x "$venv_python" ]]; then
    fail "The existing virtual environment at '$venv_dir' is incomplete. Remove it manually and rerun this script."
  fi

  venv_version="$("$venv_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  if [[ -z "$venv_version" ]]; then
    fail "Could not inspect the existing virtual environment at '$venv_dir'."
  fi

  if [[ "$venv_version" != "3.12" ]]; then
    fail "The existing virtual environment at '$venv_dir' uses Python $venv_version. Remove it manually and rerun this script to recreate it with Python 3.12."
  fi
else
  write_step "Creating virtual environment"
  if ! "$python312_path" -m venv "$venv_dir"; then
    fail "Failed to create the virtual environment at '$venv_dir'."
  fi

  if [[ ! -x "$venv_python" ]]; then
    fail "Failed to create the virtual environment at '$venv_dir'."
  fi
fi

write_step "Upgrading pip tooling"
if ! "$venv_python" -m pip install --upgrade pip setuptools wheel; then
  fail "Failed to upgrade pip tooling inside '$venv_dir'."
fi

write_step "Installing Python dependencies"
if ! "$venv_python" -m pip install -r "$requirements_file"; then
  fail "Failed to install Python dependencies from '$requirements_file'."
fi

write_step "Validating Python imports"
if ! "$venv_python" - <<'PY'
import cv2
import pytesseract
import torch
import torchvision
import ultralytics

print(f"cv2 {cv2.__version__}")
print(f"torch {torch.__version__}")
print(f"torchvision {torchvision.__version__}")
print(f"ultralytics {ultralytics.__version__}")
print(f"pytesseract {getattr(pytesseract, '__version__', 'unknown')}")
PY
then
  fail "Python package validation failed."
fi

write_step "Validating Tesseract"
if ! tesseract --version; then
  fail "Tesseract validation failed."
fi

printf '\n'
printf 'Setup complete.\n'
printf 'Next commands from repo root:\n'
printf 'source ./yolo_with_stream/.venv/bin/activate\n'
printf 'python ./yolo_with_stream/plate_recognition_tesseract.py\n'
printf '\n'
printf 'If you are already in yolo_with_stream:\n'
printf 'source ./.venv/bin/activate\n'
printf 'python ./plate_recognition_tesseract.py\n'
