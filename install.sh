#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="telescope"
PREFIX=""
SOLVER=""
RECREATE=0
SKIP_ENV=0
UPDATE_ENV=0
JOBS=""

usage() {
  cat <<'USAGE'
Usage: ./install.sh [OPTIONS]

One-command installer for Telescope:
  1. Creates or reuses/updates the conda/mamba environment from environment.yml
  2. Builds and installs the `telescope` binary (`cargo install --locked`)
  3. Installs the bundled R engine scripts into `$PREFIX/share/telescope/workflow/scripts/`
  4. Verifies the installed binary and R/Bioconductor dependencies

Options:
  -n, --name <NAME>       Target conda/mamba environment name [default: telescope]
  -p, --prefix <PATH>     Explicit environment prefix path (overrides --name)
  -s, --solver <TOOL>     Package manager to use: mamba, micromamba, or conda
      --update-env        Force environment solver update even when dependencies already verify
      --recreate          Remove and recreate the target environment if it exists
      --skip-env          Skip environment creation/update (use existing target prefix)
  -j, --jobs <N>          Number of parallel Cargo build jobs
  -h, --help              Show this help message and exit
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
      ENV_NAME="${2:?missing value for $1}"
      shift 2
      ;;
    -p|--prefix)
      PREFIX="${2:?missing value for $1}"
      shift 2
      ;;
    -s|--solver)
      SOLVER="${2:?missing value for $1}"
      shift 2
      ;;
    --update-env)
      UPDATE_ENV=1
      shift
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --skip-env)
      SKIP_ENV=1
      shift
      ;;
    -j|--jobs)
      JOBS="${2:?missing value for $1}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

pick_solver() {
  if [[ -n "$SOLVER" ]]; then
    if ! command -v "$SOLVER" >/dev/null 2>&1 && [[ ! -x "$SOLVER" ]]; then
      echo "ERROR: requested solver '$SOLVER' not found" >&2
      exit 1
    fi
    echo "$SOLVER"
    return
  fi
  for candidate in mamba micromamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      echo "$candidate"
      return
    fi
  done
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    local base_guess
    base_guess="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    for rel in condabin/mamba bin/mamba; do
      if [[ -x "$base_guess/$rel" ]]; then
        echo "$base_guess/$rel"
        return
      fi
    done
    echo "$CONDA_EXE"
    return
  fi
  echo "ERROR: neither mamba, micromamba, nor conda was found on PATH" >&2
  exit 1
}

resolve_base_dir() {
  local tool="$1"
  local base=""
  if command -v conda >/dev/null 2>&1; then
    base="$(conda info --base 2>/dev/null || true)"
  elif [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    base="$("$CONDA_EXE" info --base 2>/dev/null || true)"
  fi
  if [[ -z "$base" ]]; then
    local tool_path
    tool_path="$(command -v "$tool" 2>/dev/null || echo "$tool")"
    base="$(cd "$(dirname "$tool_path")/.." && pwd)"
  fi
  echo "$base"
}

verify_r_packages() {
  local quiet="${1:-0}"
  "$PREFIX/bin/Rscript" --vanilla - "$quiet" <<'RS'
args <- commandArgs(trailingOnly = TRUE)
quiet <- identical(args[1], "1")
pkgs <- c(
  "optparse", "arrow", "dplyr", "tidyr",
  "DEXSeq", "DESeq2", "limma", "S4Vectors",
  "GenomicAlignments", "GenomicFeatures", "data.table"
)
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  if (!quiet) {
    stop("Missing R packages: ", paste(missing, collapse = ", "))
  }
  quit(status = 1)
}
if (!quiet) {
  cat("Verified R/Bioconductor packages:", length(pkgs), "loaded\n")
}
RS
}

env_ready() {
  [[ -x "$PREFIX/bin/cargo" && -x "$PREFIX/bin/Rscript" ]] && verify_r_packages 1 >/dev/null 2>&1
}

if [[ "$SKIP_ENV" -eq 0 ]]; then
  SOLVER_BIN="$(pick_solver)"
  if [[ -z "$PREFIX" ]]; then
    BASE_DIR="$(resolve_base_dir "$SOLVER_BIN")"
    PREFIX="$BASE_DIR/envs/$ENV_NAME"
  fi
else
  if [[ -z "$PREFIX" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      PREFIX="$CONDA_PREFIX"
    else
      BASE_DIR="$(resolve_base_dir "${SOLVER:-conda}")"
      PREFIX="$BASE_DIR/envs/$ENV_NAME"
    fi
  fi
fi

echo "==> Telescope installer"
echo "    repo   : $REPO_DIR"
echo "    prefix : $PREFIX"

if [[ "$SKIP_ENV" -eq 0 ]]; then
  echo "    solver : $SOLVER_BIN"
  SOLVER_BASE="$(basename "$SOLVER_BIN")"
  if [[ -d "$PREFIX" && "$RECREATE" -eq 1 ]]; then
    echo "==> Removing existing environment at $PREFIX"
    rm -rf "$PREFIX"
  fi

  if [[ -d "$PREFIX" ]]; then
    if [[ "$UPDATE_ENV" -eq 1 ]] || ! env_ready; then
      echo "==> Updating existing environment from environment.yml"
      if [[ "$SOLVER_BASE" == "micromamba" ]]; then
        "$SOLVER_BIN" install -y -p "$PREFIX" -f "$REPO_DIR/environment.yml"
      else
        "$SOLVER_BIN" env update -p "$PREFIX" -f "$REPO_DIR/environment.yml"
      fi
    else
      echo "==> Environment already satisfies dependencies (skipping solver update; pass --update-env to force)"
    fi
  else
    echo "==> Creating environment from environment.yml"
    if [[ "$SOLVER_BASE" == "micromamba" ]]; then
      "$SOLVER_BIN" create -y -p "$PREFIX" -f "$REPO_DIR/environment.yml"
    else
      "$SOLVER_BIN" env create -y -p "$PREFIX" -f "$REPO_DIR/environment.yml"
    fi
  fi
fi

if [[ ! -x "$PREFIX/bin/cargo" ]]; then
  echo "ERROR: cargo not found at $PREFIX/bin/cargo" >&2
  exit 1
fi
if [[ ! -x "$PREFIX/bin/Rscript" ]]; then
  echo "ERROR: Rscript not found at $PREFIX/bin/Rscript" >&2
  exit 1
fi

export CONDA_PREFIX="$PREFIX"
export PATH="$PREFIX/bin:$PATH"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$REPO_DIR/target}"
if [[ -x "$PREFIX/bin/x86_64-conda-linux-gnu-cc" ]]; then
  export CC="${CC:-$PREFIX/bin/x86_64-conda-linux-gnu-cc}"
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="${CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER:-$PREFIX/bin/x86_64-conda-linux-gnu-cc}"
fi

echo "==> Building and installing telescope binary into $PREFIX/bin"
CARGO_ARGS=(install --path "$REPO_DIR/crates/telescope-cli" --locked --root "$PREFIX" --force)
if [[ -n "$JOBS" ]]; then
  CARGO_ARGS+=(--jobs "$JOBS")
fi
"$PREFIX/bin/cargo" "${CARGO_ARGS[@]}"

SCRIPTS_DEST="$PREFIX/share/telescope/workflow/scripts"
echo "==> Installing R engine scripts into $SCRIPTS_DEST"
mkdir -p "$SCRIPTS_DEST"
cp -f "$REPO_DIR"/workflow/scripts/*.R "$SCRIPTS_DEST/"

echo "==> Verifying installation"
"$PREFIX/bin/telescope" --version
"$PREFIX/bin/telescope" quant -h >/dev/null
"$PREFIX/bin/telescope" diff -h >/dev/null
verify_r_packages 0

echo "==> Installation complete!"
if [[ "$PREFIX" == */envs/* ]]; then
  echo "    Activate with: conda activate $(basename "$PREFIX")"
else
  echo "    Add to PATH  : export PATH=\"$PREFIX/bin:\$PATH\""
fi
