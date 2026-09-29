#!/usr/bin/env bash
# Builds cuda/libadcfft.so from cuda/adcfft.cu.
#
# There was no build script until this repo was set up -- the .so already in
# cuda/ was built by hand (see shell history: `nvcc --version` checks, no
# recorded full build command). This reconstructs it from what the shared
# object actually links against (`ldd`: libcufft.so.12 from
# /usr/local/cuda/targets/sbsa-linux/lib) and the toolchain installed on this
# Jetson (CUDA 13.2, /usr/local/cuda-13.2/bin/nvcc). Re-run this any time
# adcfft.cu changes.
set -euo pipefail
cd "$(dirname "$0")"

# Resolve through PATH first, then the usual CUDA install. `[ -x "$NVCC" ]`
# was checking a bare name as if it were a path, so whenever nvcc WAS on PATH
# the script tested ./nvcc, found nothing, and refused to build.
NVCC="${NVCC:-}"
if [ -z "$NVCC" ]; then
    NVCC="$(command -v nvcc 2>/dev/null || true)"
fi
[ -n "$NVCC" ] || NVCC=/usr/local/cuda/bin/nvcc
command -v "$NVCC" >/dev/null 2>&1 || {
    echo "nvcc not found -- set NVCC=/path/to/nvcc" >&2; exit 1; }
echo "using $NVCC ($("$NVCC" --version | tail -1))"

"$NVCC" -O3 -arch=sm_87 -shared -Xcompiler -fPIC -o libadcfft.so adcfft.cu -lcufft
echo "built cuda/libadcfft.so"
