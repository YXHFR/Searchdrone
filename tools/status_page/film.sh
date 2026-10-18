#!/usr/bin/env bash
# The status page's demo film from the panel's browser recordings.
#
#   bash tools/status_page/film.sh <out dir> <recording.webm>...
#
# Joins the recordings in the order given, speeds them up to fit in MAX_S seconds (never
# slows them down) and writes <out>/demo.mp4 (H.264, plays on phones) and <out>/demo.webm
# (VP9, plays in Chromium builds without H.264). No sound.
set -euo pipefail

MAX_S="${MAX_S:-85}"
WIDTH=960

out="${1:?ange var filmen ska ligga}"
shift
[ "$#" -ge 1 ] || { echo "film.sh: inga inspelningar"; exit 2; }
mkdir -p "$out"

total=0
inputs=()
for clip in "$@"; do
    len="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$clip")"
    total="$(awk -v a="$total" -v b="$len" 'BEGIN { print a + b }')"
    inputs+=(-i "$clip")
done
factor="$(awk -v t="$total" -v m="$MAX_S" 'BEGIN { f = t / m; if (f < 1) f = 1; printf "%.4f", f }')"
echo "film.sh: $# inspelningar, ${total} s, ${factor} gånger snabbare"

joined=""
for i in $(seq 0 $(($# - 1))); do joined+="[$i:v]"; done
graph="${joined}concat=n=$#:v=1:a=0,setpts=PTS/${factor},fps=25,scale=${WIDTH}:-2[v]"

ffmpeg -v error -y "${inputs[@]}" -filter_complex "$graph" -map "[v]" -an \
    -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart "$out/demo.mp4"
ffmpeg -v error -y "${inputs[@]}" -filter_complex "$graph" -map "[v]" -an \
    -c:v libvpx-vp9 -b:v 0 -crf 38 -row-mt 1 -deadline good -cpu-used 4 "$out/demo.webm"

for f in "$out/demo.mp4" "$out/demo.webm"; do
    len="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")"
    echo "film.sh: $f ${len} s, $(stat -c %s "$f") byte"
    awk -v l="$len" -v m="$MAX_S" 'BEGIN { exit !(l > 0 && l <= m + 5) }'
done
