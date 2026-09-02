#!/usr/bin/env python3
"""
Bake EXIF orientation into the pixels, strip the tag, and downscale for the web.

sips rotates the pixel buffer but leaves the orientation tag in place, so the
browser applies the rotation a SECOND time and the photo lands on its side.
Normalizing here -- once -- means what you see on disk is what the browser draws.
"""
import sys, pathlib
from PIL import Image, ImageOps

MAXDIM = 1800

for arg in sys.argv[1:]:
    p = pathlib.Path(arg)
    im = Image.open(p)
    before = im.size
    im = ImageOps.exif_transpose(im)          # apply orientation to pixels
    if max(im.size) > MAXDIM:
        im.thumbnail((MAXDIM, MAXDIM), Image.LANCZOS)
    clean = Image.new(im.mode, im.size)        # new image => no EXIF carried over
    clean.putdata(list(im.getdata()))
    clean.save(p, "JPEG", quality=86, optimize=True)
    print(f"  {p}  {before[0]}x{before[1]} -> {im.size[0]}x{im.size[1]}  (orientation baked, EXIF stripped)")
