#!/usr/bin/env bash
#
# Download the corpora that tokenization/trainer/train_bpe.py expects into data/.
#
#   ./download_data.sh               # TinyStories and OpenWebText (about 7 GB down, 14 GB on disk)
#   ./download_data.sh tinystories   # TinyStories only (2.3 GB)
#   ./download_data.sh owt           # OpenWebText only (4.7 GB down, 12 GB unpacked)
#
# Files already in data/ are kept. An interrupted download resumes where it
# stopped, so the script is safe to rerun. The data folder is in .gitignore
# because these files are far too large for GitHub.
#
# Sources:
#   https://huggingface.co/datasets/roneneldan/TinyStories
#   https://huggingface.co/datasets/stanford-cs336/owt-sample

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${DATA_DIR:-$HERE/data}"

TINYSTORIES=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
OWT=https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main

# unpack ARCHIVE UNPACKED: gunzip ARCHIVE into UNPACKED and remove the archive.
# The output is written under a temporary name first, so UNPACKED only ever
# appears complete. On failure a corrupt archive is removed, so the next run
# downloads it again; an intact one (say the disk filled up) is kept.
unpack() {
    local archive=$1 unpacked=$2
    echo "unzip $(basename "$archive")"
    if ! gunzip -c "$archive" > "$unpacked.partial"; then
        rm -f "$unpacked.partial"
        if gzip -t "$archive" 2>/dev/null; then
            echo "unpacking failed but the archive is intact; kept $(basename "$archive")" >&2
        else
            rm -f "$archive"
            echo "corrupt archive; removed $(basename "$archive"), rerun to download it again" >&2
        fi
        exit 1
    fi
    mv "$unpacked.partial" "$unpacked"
    rm -f "$archive"
}

# fetch URL NAME: download URL to data/NAME unless the final file already exists.
# A .gz NAME is unpacked afterwards.
fetch() {
    local url=$1 name=$2
    local target="$DATA_DIR/$name" final="$DATA_DIR/${name%.gz}"
    if [[ -s "$final" ]]; then
        echo "have  $final"
        return
    fi
    if [[ "$name" == *.gz && -f "$target" ]]; then
        # A finished download whose unpacking was interrupted: unpack it again.
        unpack "$target" "$final"
        echo "done  $final"
        return
    fi
    echo "get   $name"
    # -C - resumes a partial file; the .partial name keeps an interrupted
    # download from being mistaken for a finished one.
    curl --location --fail --retry 5 --retry-all-errors --continue-at - \
        --progress-bar --output "$target.partial" "$url/$name"
    mv "$target.partial" "$target"
    if [[ "$name" == *.gz ]]; then
        unpack "$target" "$final"
    fi
    echo "done  $final"
}

main() {
    local which=${1:-all}
    case "$which" in
        all|tinystories|owt) ;;
        *) echo "usage: $0 [all|tinystories|owt]" >&2; exit 2 ;;
    esac
    mkdir -p "$DATA_DIR"
    if [[ "$which" == all || "$which" == tinystories ]]; then
        fetch "$TINYSTORIES" TinyStoriesV2-GPT4-train.txt
        fetch "$TINYSTORIES" TinyStoriesV2-GPT4-valid.txt
    fi
    if [[ "$which" == all || "$which" == owt ]]; then
        fetch "$OWT" owt_train.txt.gz
        fetch "$OWT" owt_valid.txt.gz
    fi
    echo
    ls -lh "$DATA_DIR"/*.txt
}

main "$@"
