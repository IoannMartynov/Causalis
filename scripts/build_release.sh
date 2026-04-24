#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./scripts/build_release.sh X.Y.Z [--allow-dirty]

Build local causalis release artifacts by overriding setuptools_scm with an
explicit version. Existing causalis artifacts in dist/ are removed first so
local smoke checks only inspect the requested build.

Production releases are published by pushing an annotated vX.Y.Z tag to GitHub,
not by uploading local artifacts manually.

The script uses the current environment's python interpreter by default. Set
PYTHON_BIN=/path/to/python if you want to force a specific interpreter.
EOF
}

error() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

allow_dirty=0
version=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --allow-dirty)
      allow_dirty=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -n "$version" ]]; then
        error "Expected exactly one version argument."
      fi
      version="$1"
      ;;
  esac
  shift
done

if [[ -z "$version" ]]; then
  usage >&2
  exit 1
fi

if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  error "Version must be plain semver like 0.3.1."
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

python_bin="${PYTHON_BIN:-}"

if [[ -z "$python_bin" ]]; then
  if command -v python >/dev/null 2>&1; then
    python_bin=python
  elif command -v python3 >/dev/null 2>&1; then
    python_bin=python3
  else
    error "Could not find python or python3 on PATH."
  fi
fi

if ! "$python_bin" -m build --version >/dev/null 2>&1; then
  error "The selected interpreter does not have the build package installed."
fi

cd "$repo_root"

if [[ "$allow_dirty" -ne 1 ]] && [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  error "Working tree has tracked changes. Commit or stash them first, or rerun with --allow-dirty."
fi

mkdir -p dist
find dist -maxdepth 1 -type f \( -name 'causalis-*.whl' -o -name 'causalis-*.tar.gz' -o -name 'causalis-*.zip' \) -delete

printf 'Building causalis %s with %s\n' "$version" "$python_bin"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CAUSALIS="$version" "$python_bin" -m build

printf '\nBuilt release artifacts in %s/dist\n' "$repo_root"
printf 'Run twine check --strict dist/* for a local metadata smoke check.\n'
printf 'Publish production releases by pushing an annotated vX.Y.Z tag.\n'
