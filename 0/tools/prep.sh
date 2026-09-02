#!/usr/bin/env bash
# Media prep for CS180 Project 0.
#
#   ./tools/prep.sh convert   HEIC -> JPG, downscale everything for the web
#   ./tools/prep.sh gif       build media/part3/dolly-zoom.gif from frame-*.jpg
#   ./tools/prep.sh all       both of the above
#   ./tools/prep.sh rotate media/part2/x.jpg 270   fix a sideways photo by hand
#   ./tools/prep.sh frames media/part3/clip.MOV 8    pull stills out of a video take
#
# Run from the project-0 directory (the one containing index.html).

set -euo pipefail
cd "$(dirname "$0")/.."

MAXDIM=1800          # longest edge for the stills
GIF_MAXDIM=${GIF_MAXDIM:-620}       # longest edge of the dolly-zoom gif (portrait clips stay light)
GIF_FPS=5            # frames per second
PINGPONG=1           # 1 = play forward then backward, for a seamless loop

convert_media() {
  shopt -s nullglob nocaseglob

  # HEIC -> JPG (sips ships with macOS; no extra install needed)
  for f in media/*/*.heic; do
    out="${f%.*}.jpg"
    [ -f "$out" ] && continue
    echo "  heic -> jpg   $f"
    sips -s format jpeg -s formatOptions 95 "$f" --out "$out" >/dev/null
  done

  # Bake EXIF orientation into the pixels, strip the tag, downscale.
  # sips rotates the buffer but LEAVES the orientation tag, so the browser
  # rotates a second time and the photo lands on its side. normalize.py
  # settles it once so disk and browser agree.
  local jpgs=(media/*/*.jpg media/*/*.jpeg)
  [ ${#jpgs[@]} -gt 0 ] && python3 tools/normalize.py "${jpgs[@]}"

  shopt -u nullglob nocaseglob
  echo "convert: done"
}

# Pull N evenly-spaced stills out of a video clip. Phones shoot the dolly zoom
# as a continuous take; the assignment wants stills, so this samples them.
extract_frames() {
  local src="$1" count="${2:-8}" head="${3:-0.04}" tail="${4:-0.96}"
  [ -f "$src" ] || { echo "extract: no such file: $src" >&2; return 1; }
  local dur; dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src")
  echo "  $src  (${dur}s) -> $count stills"
  rm -f media/part3/frame-*.jpg
  for i in $(seq 1 "$count"); do
    local t; t=$(python3 -c "print(round($dur*($head+($tail-$head)*($i-1)/($count-1)),3))")
    printf -v out 'media/part3/frame-%02d.jpg' "$i"
    ffmpeg -hide_banner -loglevel error -ss "$t" -i "$src" -frames:v 1 \
      -vf "scale=-2:${MAXDIM}" -q:v 3 "$out" -y
    echo "    t=${t}s  ->  $out"
  done
}

# Build a dolly-zoom GIF straight from a video take.
#   gifclip <video> <out-name> [frames] [head] [tail]
# Samples evenly between head/tail (as fractions of duration), ping-pongs the
# result, and caps the longest edge so portrait takes stay light.
gifclip() {
  local src="$1" name="$2" n="${3:-14}" head="${4:-0.02}" tail="${5:-0.97}"
  [ -f "$src" ] || { echo "gifclip: no such file: $src" >&2; return 1; }
  local dur; dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$src")
  local tmp="media/part3/.tmp_$name"; rm -rf "$tmp"; mkdir -p "$tmp"
  for i in $(seq 1 "$n"); do
    local t; t=$(python3 -c "print(round($dur*($head+($tail-$head)*($i-1)/($n-1)),3))")
    ffmpeg -hide_banner -loglevel error -ss "$t" -i "$src" -frames:v 1 \
      -vf "scale=w=${GIF_MAXDIM}:h=${GIF_MAXDIM}:force_original_aspect_ratio=decrease:flags=lanczos" \
      -q:v 3 "$(printf '%s/f%03d.jpg' "$tmp" "$i")" -y
  done
  ffmpeg -hide_banner -loglevel error -y -framerate "$GIF_FPS" \
    -pattern_type glob -i "$tmp/f*.jpg" \
    -filter_complex "split[f][r];[r]reverse[rv];[f][rv]concat=n=2:v=1[all];[all]split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5" \
    -loop 0 "media/part3/${name}.gif"
  rm -rf "$tmp"
  echo "  ${name}.gif  ($n frames from ${dur}s)  $(du -h "media/part3/${name}.gif" | cut -f1)"
}

build_gif() {
  local frames=(media/part3/frame-*.jpg)
  if [ ${#frames[@]} -eq 0 ]; then
    echo "gif: no media/part3/frame-*.jpg found — nothing to do." >&2
    return 1
  fi
  echo "  ${#frames[@]} frames -> media/part3/dolly-zoom.gif"

  # Cap the LONGEST edge, so a portrait take doesn't balloon into a huge file.
  local chain="scale=w=${GIF_MAXDIM}:h=${GIF_MAXDIM}:force_original_aspect_ratio=decrease:flags=lanczos"
  if [ "$PINGPONG" = "1" ]; then
    chain="${chain},split[f][r];[r]reverse[rv];[f][rv]concat=n=2:v=1[all];[all]"
  else
    chain="${chain},"
  fi
  chain="${chain}split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5"

  ffmpeg -hide_banner -loglevel error -y \
    -framerate "$GIF_FPS" -pattern_type glob -i 'media/part3/frame-*.jpg' \
    -filter_complex "$chain" -loop 0 \
    media/part3/dolly-zoom.gif

  echo "gif: done — $(du -h media/part3/dolly-zoom.gif | cut -f1)"
}

case "${1:-all}" in
  convert) convert_media ;;
  gif)     build_gif ;;
  all)     convert_media; build_gif ;;
  rotate)
    [ $# -eq 3 ] || { echo "usage: $0 rotate <file.jpg> <90|180|270>" >&2; exit 1; }
    sips -r "$3" "$2" >/dev/null && echo "rotated $2 by $3 (clockwise)" ;;
  frames)
    [ $# -ge 2 ] || { echo "usage: $0 frames <video.mov> [count]" >&2; exit 1; }
    extract_frames "$2" "${3:-8}" ;;
  gifclip)
    [ $# -ge 3 ] || { echo "usage: $0 gifclip <video> <out-name> [frames] [head] [tail]" >&2; exit 1; }
    shift; gifclip "$@" ;;
  *)       sed -n '2,12p' "$0"; exit 1 ;;
esac
