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
  run_with_privileges apt-get update
  run_with_privileges env DEBIAN_FRONTEND=noninteractive apt-get install -y "${missing[@]}"
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

python_has_ensurepip() {
  local python_bin="$1"
  "$python_bin" -c 'import ensurepip' >/dev/null 2>&1
}

python_has_pip() {
  local python_bin="$1"
  "$python_bin" -m pip --version >/dev/null 2>&1
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

if ! python_has_ensurepip "$python312_path"; then
  ensure_apt_packages python3.12-venv
fi

if ! python_has_ensurepip "$python312_path"; then
  fail "Python 3.12 was found, but ensurepip is missing. Install 'python3.12-venv', then rerun this script."
fi

write_step "Checking OpenCV runtime libraries"
ensure_apt_packages libgl1 libglib2.0-0 libgomp1 libgtk-3-0

create_venv=true
if [[ -d "$venv_dir" ]]; then
  write_step "Reusing existing virtual environment"

  if [[ ! -x "$venv_python" ]]; then
    write_step "Removing incomplete virtual environment"
    rm -rf "$venv_dir"
  else
    venv_version="$("$venv_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    if [[ -z "$venv_version" ]]; then
      fail "Could not inspect the existing virtual environment at '$venv_dir'."
    fi

    if [[ "$venv_version" != "3.12" ]]; then
      fail "The existing virtual environment at '$venv_dir' uses Python $venv_version. Remove it manually and rerun this script to recreate it with Python 3.12."
    fi

    if ! python_has_pip "$venv_python"; then
      write_step "Removing incomplete virtual environment without pip"
      rm -rf "$venv_dir"
    else
      create_venv=false
    fi
  fi
fi

if [[ "$create_venv" == true ]]; then
  write_step "Creating virtual environment"
  "$python312_path" -m venv "$venv_dir"
fi

if [[ ! -x "$venv_python" ]]; then
  fail "Failed to create the virtual environment at '$venv_dir'."
fi

if ! python_has_pip "$venv_python"; then
  write_step "Bootstrapping pip inside the virtual environment"
  "$venv_python" -m ensurepip --upgrade
fi

write_step "Upgrading pip tooling"
"$venv_python" -m pip install --upgrade pip setuptools wheel

write_step "Installing Python dependencies"
"$venv_python" -m pip install -r "$requirements_file"

write_step "Validating Python imports"
"$venv_python" - <<'PY'
import cv2
import easyocr
import numpy
import torch
import ultralytics

print(f"cv2 {cv2.__version__}")
print(f"easyocr {easyocr.__version__}")
print(f"numpy {numpy.__version__}")
print(f"torch {torch.__version__}")
print(f"ultralytics {ultralytics.__version__}")
print(f"cuda_available {torch.cuda.is_available()}")
PY

printf '\n'
printf 'Setup complete.\n'
printf 'Next commands from repo root:\n'
printf 'source ./number-plate-recognition-easyocr-rtsp/.venv/bin/activate\n'
printf 'python ./number-plate-recognition-easyocr-rtsp/number-plate-recognition.py\n'
printf '\n'
printf 'If you are already in number-plate-recognition-easyocr-rtsp:\n'
printf 'source ./.venv/bin/activate\n'
printf 'python ./number-plate-recognition.py\n'
