#!/usr/bin/env bash
# Open the MATLAB prototype and capacity model from a Linux terminal.
set -euo pipefail

if [[ ${1:-} == -h || ${1:-} == --help ]]; then
    printf 'Usage: %s [fundus-image]\nOpens MATLAB, runs the quality-gated demo and displays the SimEvents model.\n' "$0"
    exit 0
fi
if (( $# > 1 )); then
    printf 'Expected at most one image path. Use --help for usage.\n' >&2
    exit 2
fi
if ! command -v matlab >/dev/null 2>&1; then
    printf 'MATLAB is not on PATH. Add its bin directory to PATH and retry.\n' >&2
    exit 1
fi

NETRAI_PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
NETRAI_DEMO_IMAGE=''
if (( $# == 1 )); then
    if [[ ! -f $1 ]]; then
        printf 'Image file not found: %s\n' "$1" >&2
        exit 2
    fi
    NETRAI_DEMO_IMAGE=$(realpath -- "$1")
fi
export NETRAI_PROJECT_DIR NETRAI_DEMO_IMAGE

# Environment variables keep user paths out of executable MATLAB command text.
exec matlab -desktop -r "cd(fullfile(getenv('NETRAI_PROJECT_DIR'),'matlab')); try; result = demo_netrai(string(getenv('NETRAI_DEMO_IMAGE'))); open_system('models/NetrAI_Telemedicine_Model.slx'); catch exception; disp(getReport(exception,'extended','hyperlinks','off')); end"
