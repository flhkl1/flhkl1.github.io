#!/usr/bin/env python3
"""
Generates assets/berkeley-night.svg -- a Starry-Night-over-Berkeley backdrop.

Every mark is a stroked path pushed through an feTurbulence displacement filter,
which is what gives the sky its impasto wobble instead of clean vector arcs.
Re-run after changing any constant below; the output is deterministic.
"""
import math, pathlib, random

W, H = 1600, 900
random.seed(11)

# --- palette ------------------------------------------------------------
SKY_HI, SKY_MID, SKY_LO = "#080f2a", "#0f2456", "#1b3f86"
BLUES  = ["#1d4a95", "#2a63b8", "#3f83cf", "#5b9bd8", "#84badf"]
CREAM  = "#d3e5f6"
GOLD   = "#ffd23f"
GOLD_D = "#f2a52b"
HILL_F, HILL_B = "#0a1c2e", "#102a3c"
CYPRESS = "#07161f"
TOWER   = "#0a1526"

out = []
add = out.append

def wave(y0, amp, freq, phase, x0=-80, x1=W + 80, step=26):
    """A long sinusoidal brushstroke sweeping across the sky."""
    pts = []
    x = x0
    while x <= x1:
        y = y0 + amp * math.sin(freq * x + phase) + 14 * math.sin(2.7 * freq * x + phase * 1.7)
        pts.append((x, y))
        x += step
    d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"
    for i in range(1, len(pts) - 1):
        xc = (pts[i][0] + pts[i + 1][0]) / 2
        yc = (pts[i][1] + pts[i + 1][1]) / 2
        d += f" Q {pts[i][0]:.1f} {pts[i][1]:.1f} {xc:.1f} {yc:.1f}"
    return d

def spiral(cx, cy, r0, r1, turns, step=0.22):
    """Archimedean vortex -- the signature Starry Night eddy."""
    pts, t, tmax = [], 0.0, turns * 2 * math.pi
    while t <= tmax:
        r = r0 + (r1 - r0) * (t / tmax)
        pts.append((cx + r * math.cos(t), cy + r * 0.62 * math.sin(t)))
        t += step
    d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"
    for i in range(1, len(pts) - 1):
        xc = (pts[i][0] + pts[i + 1][0]) / 2
        yc = (pts[i][1] + pts[i + 1][1]) / 2
        d += f" Q {pts[i][0]:.1f} {pts[i][1]:.1f} {xc:.1f} {yc:.1f}"
    return d

# --- defs ---------------------------------------------------------------
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="{W}" height="{H}" preserveAspectRatio="xMidYMax slice" role="img" '
    f'aria-label="A Starry Night styled painting of the Berkeley hills and the Campanile">')
add('<defs>')
add(f'''<linearGradient id="sky" x1="0" y1="0" x2="0.15" y2="1">
  <stop offset="0" stop-color="{SKY_HI}"/><stop offset="0.55" stop-color="{SKY_MID}"/>
  <stop offset="1" stop-color="{SKY_LO}"/></linearGradient>''')
# The impasto filter: fractal noise displacing every stroke.
add('''<filter id="impasto" x="-15%" y="-15%" width="130%" height="130%">
  <feTurbulence type="fractalNoise" baseFrequency="0.011 0.017" numOctaves="3" seed="7" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="24" xChannelSelector="R" yChannelSelector="G"/>
</filter>''')
add('''<filter id="impasto-soft" x="-15%" y="-15%" width="130%" height="130%">
  <feTurbulence type="fractalNoise" baseFrequency="0.02 0.03" numOctaves="2" seed="3" result="n"/>
  <feDisplacementMap in="SourceGraphic" in2="n" scale="11" xChannelSelector="R" yChannelSelector="G"/>
</filter>''')
add(f'''<radialGradient id="halo"><stop offset="0" stop-color="{GOLD}" stop-opacity="0.95"/>
  <stop offset="0.35" stop-color="{GOLD}" stop-opacity="0.30"/>
  <stop offset="1" stop-color="{GOLD}" stop-opacity="0"/></radialGradient>''')
add('</defs>')

add(f'<rect width="{W}" height="{H}" fill="url(#sky)"/>')

# --- sky brushwork ------------------------------------------------------
add('<g filter="url(#impasto)" fill="none" stroke-linecap="round">')
for i in range(26):
    y0 = 20 + i * 24 + random.uniform(-9, 9)
    amp = random.uniform(16, 46)
    freq = random.uniform(0.0042, 0.0088)
    col = random.choice(BLUES + [CREAM if random.random() < 0.16 else BLUES[2]])
    add(f'<path d="{wave(y0, amp, freq, random.uniform(0,6.3))}" stroke="{col}" '
        f'stroke-width="{random.uniform(4,13):.1f}" opacity="{random.uniform(0.30,0.72):.2f}"/>')

# two vortices, the way the sky curls in the original
for cx, cy, r0, r1, turns, n in ((430, 250, 12, 165, 2.7, 7), (1130, 175, 9, 120, 2.4, 6)):
    for k in range(n):
        col = CREAM if k % 3 == 0 else random.choice(BLUES[1:])
        add(f'<path d="{spiral(cx + random.uniform(-9,9), cy + random.uniform(-7,7), r0 + k*4, r1 - k*13, turns)}" '
            f'stroke="{col}" stroke-width="{random.uniform(3,8):.1f}" opacity="{random.uniform(0.35,0.8):.2f}"/>')
add('</g>')

# --- stars & moon -------------------------------------------------------
stars = [(180,120,34),(700,95,26),(905,300,22),(1290,330,24),(560,410,18),
         (1010,120,20),(300,330,20),(1420,150,28),(120,430,16),(790,235,15)]
for sx, sy, sr in stars:
    add(f'<circle cx="{sx}" cy="{sy}" r="{sr*2.5:.0f}" fill="url(#halo)"/>')
    add(f'<circle cx="{sx}" cy="{sy}" r="{sr*0.42:.0f}" fill="{GOLD}" opacity="0.95"/>')
# crescent moon, upper right
add(f'<g filter="url(#impasto-soft)"><circle cx="1455" cy="128" r="140" fill="url(#halo)"/>'
    f'<path d="M 1492 66 A 66 66 0 1 0 1492 190 A 52 52 0 1 1 1492 66 Z" fill="{GOLD}" opacity="0.92"/></g>')

# --- Berkeley hills -----------------------------------------------------
add(f'<g filter="url(#impasto-soft)">')
add(f'<path d="M 0 596 C 150 548 260 572 380 540 C 520 502 620 548 760 520 '
    f'C 900 492 1010 534 1160 508 C 1320 480 1450 512 {W} 486 L {W} {H} L 0 {H} Z" fill="{HILL_B}"/>')
add(f'<path d="M 0 668 C 180 630 300 660 440 632 C 590 602 700 646 850 620 '
    f'C 1010 592 1140 632 1300 606 C 1420 586 1520 606 {W} 592 L {W} {H} L 0 {H} Z" fill="{HILL_F}"/>')
add('</g>')

# --- Campanile (Sather Tower) ------------------------------------------
# Tall shaft, stepped cornice, pyramidal cap, lit clock face.
tx, base, top = 1002, 700, 300
add('<g filter="url(#impasto-soft)">')
add(f'<path d="M {tx-30} {base} L {tx-24} {top+64} L {tx+24} {top+64} L {tx+30} {base} Z" fill="{TOWER}"/>')
add(f'<rect x="{tx-34}" y="{top+52}" width="68" height="16" fill="{TOWER}"/>')
add(f'<path d="M {tx-34} {top+52} L {tx} {top-30} L {tx+34} {top+52} Z" fill="{TOWER}"/>')
add(f'<circle cx="{tx}" cy="{top+150}" r="13" fill="{GOLD}" opacity="0.9"/>')
for r in range(3):  # belfry openings, lit
    add(f'<rect x="{tx-17+r*15}" y="{top+74}" width="8" height="26" rx="4" fill="{GOLD_D}" opacity="0.75"/>')
add('</g>')

# --- campus rooftops ----------------------------------------------------
add('<g filter="url(#impasto-soft)">')
roofs = [(60,700,150,58),(230,712,120,46),(380,694,180,64),(600,716,140,42),
         (770,702,110,56),(1080,708,170,50),(1290,696,140,62),(1450,714,150,44)]
for rx, ry, rw, rh in roofs:
    add(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh+120}" fill="{HILL_F}"/>')
    add(f'<path d="M {rx-8} {ry} L {rx+rw/2} {ry-26} L {rx+rw+8} {ry} Z" fill="{HILL_F}"/>')
    for w in range(int(rw // 34)):  # a few lit windows
        if random.random() < 0.55:
            add(f'<rect x="{rx+12+w*34}" y="{ry+18}" width="9" height="12" '
                f'fill="{GOLD_D}" opacity="{random.uniform(0.45,0.85):.2f}"/>')
add('</g>')

# --- cypress, foreground left ------------------------------------------
add(f'<g filter="url(#impasto)"><path d="M 168 {H+20} C 96 720 150 560 128 430 '
    f'C 118 344 168 292 196 250 C 214 320 248 352 250 432 C 252 556 300 700 236 {H+20} Z" '
    f'fill="{CYPRESS}"/></g>')

# --- foreground ground band --------------------------------------------
add(f'<path d="M 0 806 C 260 786 520 812 800 796 C 1080 780 1340 806 {W} 790 '
    f'L {W} {H} L 0 {H} Z" fill="#061019"/>')

add('</svg>')

dest = pathlib.Path(__file__).resolve().parent.parent / "assets" / "berkeley-night.svg"
dest.write_text("\n".join(out))
print(f"wrote {dest}  ({dest.stat().st_size/1024:.1f} KB)")
