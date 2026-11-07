---
title: Image Warping & Mosaicing
subtitle: CS 180 Project 4
date: November 2026
cover: assets/background.jpg
---

<!-- This file is the whole page. Cheat sheet: ../README-posts.md
     The cover is the cylindrical panorama of my loft from Part B. -->

<details><summary>About this project</summary>

This is the fourth project of [CS180 (Computer Vision)](https://cal-cs180.github.io/fa26/hw/proj3/partA.html) with Alexei Efros and Ren Ng. The stated goal is to take two or more photos, register them, warp them, and composite them into one mosaic. Part A does this with correspondences I picked by hand. Part B does it automatically, following [Brown et al. (MOPS)](https://cal-cs180.github.io/fa26/hw/proj3/Papers/MOPS.pdf).

Definitions:
A homography is a projective transformation of the plane. It maps straight lines to straight lines, but not parallel lines to parallel lines.
A mosaic (or panorama) is several photos warped into one shared frame and blended together.

</details>

## The problem

When you take a panorama on your phone, you're really taking a bunch of photos while turning in place, and the phone glues them together. This project is basically building that glue from scratch.

The idea is simple: if the camera only rotates (the center of projection stays put), then any two photos of the scene are related by a single 3 × 3 matrix, a homography. Find that matrix, warp one photo into the other's frame, blend the overlap, and you have a mosaic.

## Meta (Why?)

Problem: Image Mosaicing (also called image stitching)

Formally: Given images $I_1, \dots, I_n$ taken from the same center of projection, and a reference image $I_r$, find homographies $H_i$ such that

$$
I_i(\mathbf{x}) \approx I_r(H_i \mathbf{x}) \quad \text{for every } \mathbf{x} \text{ in the overlap}
$$

then warp every $I_i$ by $H_i$ and blend them on one canvas.

This is the same image registration problem as Project 1, but with a much bigger transformation space. In Project 1, $\mathcal{T}$ was just translations (2 degrees of freedom), small enough to brute force. Here it's homographies (8 degrees of freedom), so brute force is out and we have to solve for $H$ directly from matching points.

Commercial applications: Phone panorama modes, Google Street View, drone and satellite orthomosaics, digital pathology slide scanners, 360° VR capture

Subproblems:

- Solving for $H$ from point matches (least squares)

- Resampling an image under a warp without holes or aliasing (inverse warping + interpolation)

- Hiding the seams where photos overlap (blending)

- Finding the point matches without a human (features, matching, RANSAC)

## A.1 Shooting the pictures

I took three sets on an iPhone, turning the phone in place with about 35–55% overlap between neighbours: a view over downtown Berkeley from a rooftop, a sweep of my loft, and a rooftop sunset over the Bay.

![city 1](media/a/city_1.jpg "city 1") ![city 2](media/a/city_2.jpg "city 2")

![interior 1](media/a/interior_1.jpg "interior 1") ![interior 2](media/a/interior_2.jpg "interior 2") ![interior 3](media/a/interior_3.jpg "interior 3")

![sunset 1](media/a/sunset_1.jpg "sunset 1") ![sunset 2](media/a/sunset_2.jpg "sunset 2") ![sunset 3](media/a/sunset_3.jpg "sunset 3")

## A.2 Recovering homographies

### Deriving the linear system

A homography maps a point $\mathbf{p} = (x, y, 1)$ to $\mathbf{p}' \sim H\mathbf{p}$, where $\sim$ means equal up to scale. Since scale doesn't matter, we can fix $h_{33} = 1$, which leaves 8 unknowns:

$$
\begin{bmatrix} w x' \\ w y' \\ w \end{bmatrix} = \begin{bmatrix} h_1 & h_2 & h_3 \\ h_4 & h_5 & h_6 \\ h_7 & h_8 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
$$

**Start: divide out the scale.**

$$
x' = \frac{h_1 x + h_2 y + h_3}{h_7 x + h_8 y + 1}, \qquad y' = \frac{h_4 x + h_5 y + h_6}{h_7 x + h_8 y + 1}
$$

This isn't linear in $h$, because the unknowns are in the denominator.

**Fix 1: multiply through by the denominator.**

$$
h_1 x + h_2 y + h_3 - h_7 x x' - h_8 y x' = x'
$$

$$
h_4 x + h_5 y + h_6 - h_7 x y' - h_8 y y' = y'
$$

Now it's linear. Each correspondence gives two rows of $A\mathbf{h} = \mathbf{b}$:

$$
\begin{bmatrix} x & y & 1 & 0 & 0 & 0 & -x x' & -y x' \\ 0 & 0 & 0 & x & y & 1 & -x y' & -y y' \end{bmatrix} \mathbf{h} = \begin{bmatrix} x' \\ y' \end{bmatrix}
$$

**Fix 2: four points is too few.**

4 points give exactly 8 equations, so we could solve it directly. But my clicks are off by a pixel or two, and with exactly 4 points all of that error goes straight into $H$. So I use 9–15 points per pair and solve the overdetermined $2n \times 8$ system by least squares, $\min_\mathbf{h} \lVert A\mathbf{h} - \mathbf{b} \rVert^2$.

**Fix 3: the columns are on wildly different scales.**

The $x x'$ columns are around $10^5$ while the $1$ columns are just 1, which makes the system badly conditioned. So before solving, I normalize each point set (Hartley normalization): shift it to zero mean and scale it so the average distance from the origin is $\sqrt{2}$. Then I undo it after: $H = T_2^{-1} \hat{H} T_1$.

<details><summary>Code</summary>

```python
def computeH(im1_pts, im2_pts):
    T1, T2 = norm_T(im1_pts), norm_T(im2_pts)
    q1, q2 = apply(T1, im1_pts), apply(T2, im2_pts)
    A = np.zeros((2*n, 8)); b = np.zeros(2*n)
    x, y, xp, yp = q1[:,0], q1[:,1], q2[:,0], q2[:,1]
    A[0::2, :3] = np.c_[x, y, ones];  A[0::2, 6:] = np.c_[-x*xp, -y*xp]
    A[1::2, 3:6] = np.c_[x, y, ones]; A[1::2, 6:] = np.c_[-x*yp, -y*yp]
    b[0::2], b[1::2] = xp, yp
    h = np.linalg.lstsq(A, b, rcond=None)[0]
    H = np.linalg.inv(T2) @ np.append(h, 1).reshape(3, 3) @ T1
    return H / H[2, 2]
```

</details>

### Correspondences

I picked points by eye on zoomed-in views of each overlap (window corners, lamps, roof edges, the tree on the skyline), then snapped each one to the best local match with NCC in a ±10 px window, which just cleans up clicks that are off by a pixel or two. Points on repeated stuff like stair treads that didn't agree with the rest got dropped.

![city points](media/a/A2_points_city_1_city_2.jpg "city 1 ↔ city 2 · 12 points")

![interior points](media/a/A2_points_interior_1_interior_2.jpg "interior 1 ↔ interior 2 · 9 points")

![interior points](media/a/A2_points_interior_2_interior_3.jpg "interior 2 ↔ interior 3 · 11 points")

![sunset points](media/a/A2_points_sunset_1_sunset_2.jpg "sunset 1 ↔ sunset 2 · 14 points")

![sunset points](media/a/A2_points_sunset_2_sunset_3.jpg "sunset 2 ↔ sunset 3 · 15 points")

| pair | $H$ (image 1 → image 2) | points | mean reprojection error |
|---|---|---|---|
| city 1 → 2 | $\begin{bmatrix} 2.346 & -0.1297 & -1260 \\ 0.9191 & 2.12 & -718.8 \\ 0.001448 & 4.583\times10^{-5} & 1 \end{bmatrix}$ | 12 | 0.87 px |
| interior 1 → 2 | $\begin{bmatrix} 3.375 & -0.09479 & -2017 \\ 1.643 & 3.092 & -1338 \\ 0.002624 & 0.000121 & 1 \end{bmatrix}$ | 9 | 1.18 px |
| interior 2 → 3 | $\begin{bmatrix} 2.995 & -0.01285 & -1838 \\ 1.384 & 2.761 & -1091 \\ 0.002257 & 5.496\times10^{-5} & 1 \end{bmatrix}$ | 11 | 1.16 px |
| sunset 1 → 2 | $\begin{bmatrix} 1.932 & -0.2096 & -809.9 \\ 0.7134 & 1.74 & -548.4 \\ 0.00101 & 3.871\times10^{-5} & 1 \end{bmatrix}$ | 14 | 1.15 px |
| sunset 2 → 3 | $\begin{bmatrix} 1.949 & -0.2266 & -901.9 \\ 0.8801 & 1.541 & -486.9 \\ 0.001274 & -0.0002339 & 1 \end{bmatrix}$ | 15 | 0.92 px |

The bottom rows ($h_7, h_8$) are what make this a homography and not just an affine warp. They're around $10^{-3}$, which sounds tiny, but multiplied by $x \approx 900$ px they change the denominator by a factor of 2–3 across the image. That's the "keystone" stretch you'll see in the mosaics below.

## A.3 Warping the images

### Inverse warping

The naive way to warp is to take every source pixel, push it through $H$, and paint it wherever it lands. This leaves holes, because a warp that magnifies spreads the source pixels apart.

So we go the other way. First, push the 4 corners through $H$ to find the bounding box of the output. Then for every output pixel, ask where it came from by sending it back through $H^{-1}$, and sample the source there. Every output pixel gets exactly one value, so there are no holes. An alpha channel marks which output pixels actually landed inside the source.

The catch is that $H^{-1}\mathbf{p}'$ almost never lands exactly on a pixel center, so we need to interpolate.

- **Nearest neighbour** rounds to the closest pixel center.
- **Bilinear** takes the four surrounding pixels and blends them by how close they are. If $(a, b)$ is the fractional offset from the top-left neighbour:

$$
I(x + a, y + b) \approx (1-a)(1-b)\, I_{00} + a(1-b)\, I_{10} + (1-a)b\, I_{01} + ab\, I_{11}
$$

<details><summary>Code</summary>

```python
def warpImageBilinear(im, H):
    xmin, ymin, xmax, ymax = warp_bbox(im.shape, H)      # corners through H
    xs, ys = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))
    src = np.linalg.inv(H) @ np.stack([xs.ravel(), ys.ravel(), ones])
    sx, sy = src[0]/src[2], src[1]/src[2]                # where it came from
    x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
    a, b = sx - x0, sy - y0
    out = ((1-a)*(1-b)*im[y0, x0]   + a*(1-b)*im[y0, x0+1] +
           (1-a)*b    *im[y0+1, x0] + a*b    *im[y0+1, x0+1])
    alpha = (sx >= 0) & (sx <= w-1) & (sy >= 0) & (sy <= h-1)
    return out, alpha
```

</details>

### Rectification

As a sanity check, I took two things I know are rectangles but were photographed at an angle, clicked their four corners, and mapped them to a real rectangle: the corkboard (4:3) and a book cover (24.5 × 30 cm).

![corkboard input](media/a/A3_rect_corkboard_src.jpg "input + clicked corners") ![corkboard NN](media/a/A3_rect_corkboard_nn.jpg "nearest neighbour · 31 ms") ![corkboard bilinear](media/a/A3_rect_corkboard_bilinear.jpg "bilinear · 62 ms")

![book input](media/a/A3_rect_book_src.jpg "input + clicked corners") ![book NN](media/a/A3_rect_book_nn.jpg "nearest neighbour · 18 ms") ![book bilinear](media/a/A3_rect_book_bilinear.jpg "bilinear · 43 ms")

The corkboard comes out square on the wall, and you can read the book title like you're looking straight down at it.

<div class="compare" data-left="media/a/A3_rect_corkboard_nn.jpg" data-right="media/a/A3_rect_corkboard_bilinear.jpg" data-left-label="nearest neighbour" data-right-label="bilinear" data-caption="drag · look at the photo grid and the edges of the board"></div>

![zoom](media/a/A3_rect_corkboard_zoom.jpg "4× zoom · nearest neighbour (left) vs. bilinear (right)")

### Speed vs. quality

Nearest neighbour is about 2× faster (31 vs. 62 ms on the corkboard, 16 vs. 21 s for the whole interior mosaic including blending). This is because it does one lookup per pixel instead of four plus some arithmetic.

The cost shows up wherever the warp magnifies. In the zoom, nearest neighbour turns every edge into a staircase, since neighbouring output pixels keep rounding to the same source pixel. Bilinear is a bit softer but clean. Every warp in this project magnifies somewhere, so I use bilinear for all the final results.

## A.4 Blending into a mosaic

### Process

1. Pick a reference photo (the middle one for the three-photo sets) and leave it unwarped.
2. Chain the pairwise homographies so every photo has an $H$ into the reference frame. For example, interior 3 uses $H_{2 \to 3}^{-1}$.
3. Push every photo's corners through its $H$ to size one shared canvas.
4. Inverse-warp each photo onto the canvas, with its alpha mask.
5. Blend.

![city 1 warped](media/a/A4_warped_city_city_1.jpg "city 1 in the mosaic frame") ![city 2 warped](media/a/A4_warped_city_city_2.jpg "city 2 warped into it")

### Blending

The phone re-metered between shots, so the same wall can be noticeably brighter in one photo than another. We want some way to hide that.

First I just pasted each photo on top of the last, and you could see a hard seam wherever the exposure changed. Averaging the overlap got rid of that seam, but then you could see the outline of every photo instead, since the overlap is a 50/50 mix and the rest isn't.

Feathering is better. Each photo gets weighted by its distance transform, which is 1 at the photo's center and falls off linearly to 0 at its edge, so every photo fades out before it hits its own border. The problem is that anywhere the photos are slightly misaligned (parallax, mostly), fine detail shows up twice. This is called ghosting.

What I ended up using is the Laplacian pyramid idea from Project 2, with 2 bands. Split each photo into a low band (a $\sigma = 2$ Gaussian blur) and a high band (the photo minus the blur). The low band (color, exposure) gets the smooth feathered weights. The high band (edges, texture) only comes from whichever photo is closer to its own center, so it can't ghost.

<div class="slider" data-src="media/a/seam_{v}.jpg" data-values="overwrite,average,feather,laplacian" data-labels="overwrite,average,feather,2-band Laplacian" data-start="3" data-caption="the same corner of the interior mosaic under each blend"></div>

### Mosaics

![city mosaic](media/a/A4_mosaic_city.jpg "city · 2 photos · reference = city 1")

![interior mosaic](media/a/A4_mosaic_interior.jpg "interior · 3 photos · reference = interior 2")

![sunset mosaic](media/a/A4_mosaic_sunset.jpg "sunset · 3 photos · reference = sunset 2")

They stitch cleanly, but they look like bowties. This is the keystone stretch from earlier: the further a photo is rotated away from the reference, the more a flat projection has to stretch it. This motivates the bells & whistles below.

## A.5 Bells & whistles: cylindrical mosaics

### Estimating the focal length

To project onto a cylinder, we need the focal length $f$ in pixels. It turns out we can get it from the homographies we already have.

For a camera that only rotates, the homography between two photos is

$$
H = K R K^{-1}, \qquad K = \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

where $R$ is the rotation between the two shots. $K^{-1}$ turns a pixel into a ray, $R$ rotates the ray, and $K$ turns it back into a pixel.

So if we guess the right $f$, then $K^{-1} H K$ should be a rotation matrix, which means $R^\top R = I$. If we guess wrong, it won't be. So I just search over $f$ and keep the value where all five $K^{-1} H K$ are closest to rotations:

$$
f^* = \arg\min_f \sum_{\text{pairs}} \big\lVert R_f^\top R_f - I \big\rVert, \qquad R_f = K_f^{-1} H K_f
$$

![focal](media/a/A5_focal.jpg "rotation-consistency cost vs. f · minimum at 798 px")

The minimum is at $f \approx 798$ px, which is a horizontal field of view of about 59° for these 900 px wide photos.

### Projecting onto a cylinder

Instead of a flat plane, imagine wrapping the mosaic around a cylinder centered on the camera. A point on the cylinder is an angle $\theta$ around it and a height $h$ up it, which looks like the ray

$$
\mathbf{r}(\theta, h) = (\sin\theta,\ h,\ \cos\theta)
$$

For every output pixel $(u, v)$, I set $\theta = u/f$ and $h = v/f$, project the ray into the reference image ($K\mathbf{r}$), then into photo $i$ ($H_i^{-1} K \mathbf{r}$), and sample it bilinearly. Each pixel is sampled straight from the original photo, so nothing gets warped twice and no resolution is lost. Equal angles now take equal width, so the edge stretch goes away, and vertical lines stay vertical.

![city cylinder](media/a/A5_cyl_city.jpg "city · cylindrical")

![interior cylinder](media/a/A5_cyl_interior.jpg "interior · cylindrical")

![sunset cylinder](media/a/A5_cyl_sunset.jpg "sunset · cylindrical")

## Part B: Autostitching

Clicking points is slow, and my clicks are only good to a pixel or two. Part B finds the matches automatically. The pipeline follows MOPS at a single scale:

```
Algorithm 1: AutoStitch(I₁, I₂)
Input:  two overlapping photos
Output: homography H mapping I₁ into I₂

 1:  for k ∈ {1, 2} do
 2:      C_k ← HarrisCorners(I_k)                 ▷ B.1
 3:      C_k ← ANMS(C_k, 1000)                    ▷ B.1
 4:      D_k ← Descriptors(I_k, C_k)              ▷ B.2
 5:  end for
 6:  M ← RatioTestMatch(D₁, D₂, 0.7)              ▷ B.3
 7:  H ← RANSAC(M, 5000 iterations, 3 px)         ▷ B.4
 8:  return H
```

## B.1 Harris corners and ANMS

### Harris

A corner is a spot where the image changes in every direction. Edges only change in one direction, and flat areas don't change at all. Harris measures this with the structure tensor of the gradients in a small window, and gives a high score when both of its eigenvalues are large. I take every local maximum of that score at least 20 px from the border (so the 40 × 40 descriptor window fits), which gives 5,000–11,000 corners per photo.

### ANMS

The problem is that the strongest corners all bunch up. In my loft they pile onto the rug and the kitchen backsplash, since that's where the contrast is. But we only care about the overlap, and a patch of the overlap with no corners can't be matched.

The idea behind adaptive non-maximal suppression is to keep corners that are the strongest *in their neighbourhood*, not the strongest overall. For each corner, find the distance to the nearest corner that is clearly stronger than it:

$$
r_i = \min_j \lVert \mathbf{x}_i - \mathbf{x}_j \rVert \quad \text{s.t.} \quad f(\mathbf{x}_i) < 0.9\, f(\mathbf{x}_j)
$$

A big $r_i$ means nothing nearby beats it. Keep the 1000 corners with the largest $r_i$, and they spread out evenly over the whole photo.

![harris anms](media/b/B1_harris_anms_interior_2.jpg "all Harris corners · strongest 1000 · 1000 after ANMS")

![city](media/b/B1_harris_anms_city_1.jpg "city 1") ![sunset](media/b/B1_harris_anms_sunset_2.jpg "sunset 2")

Small aside about computational feasibility.

ANMS compares every corner with every other corner, so it's $O(n^2)$. With 10,000 corners that's $10^8$ distances, which is 800 MB as a float64 matrix. So I only consider the 8,000 strongest candidates and compute the distances in chunks of 1,024 rows, which keeps memory to about 100 MB.

## B.2 Feature descriptors

Around each corner I take a 40 × 40 window, blur it ($\sigma = 2.5$), and sample an 8 × 8 grid every 5 px. Each of the 64 values then summarizes a 5 × 5 patch. In plain English, the descriptor is a tiny, blurry thumbnail of the corner's neighbourhood. The blur matters a lot. If the corner is off by a pixel or two, a sharp patch changes completely, but a blurry one barely changes.

Then, just like NCC in Project 1, we normalize for brightness and contrast: subtract the mean and divide by the standard deviation. This is called bias/gain normalization, and it means the same spot in a brighter or darker exposure gives the same descriptor.

![descriptors](media/b/B2_descriptors.jpg "six features from interior 2 · the 40×40 window and its normalized 8×8 descriptor")

## B.3 Matching

For every descriptor in photo 1, find its nearest neighbour in photo 2. The question is when to trust it.

**Start: threshold the distance.** Keep a match if the distance is small. This doesn't work well. In a photo of stairs, every stair tread has a close match: the right tread, and also the one above and below it. A small distance doesn't tell you the match is unique.

**Fix: compare to the runner-up.** Lowe's ratio test keeps a match only if the best neighbour is clearly better than the second best:

$$
\frac{e_{1\text{-NN}}}{e_{2\text{-NN}}} < 0.7
$$

A correct match should be a lot better than the next best option. A feature with no real partner has two neighbours that are about equally bad, so its ratio is close to 1. I also require the two features to be each other's nearest neighbour.

Here's the same plot as MOPS Fig. 6b, but on my photos:

![ratio](media/b/B3_ratio_hist.jpg "1-NN/2-NN ratio for correct vs. incorrect matches · dashed line = 0.7")

![matches](media/b/B3_matches_city_1_city_2.jpg "city · 181 matches")

![matches](media/b/B3_matches_interior_2_interior_3.jpg "interior 2 ↔ 3 · 83 matches")

![matches](media/b/B3_matches_sunset_2_sunset_3.jpg "sunset 2 ↔ 3 · 148 matches")

Most matches are right, but a few wrong ones still get through, mostly on stuff that repeats: stair treads, solar panels, window panes. Even one wrong match can wreck a least-squares fit, which is why we need RANSAC.

## B.4 RANSAC

If we pick 4 matches at random and they all happen to be correct, the $H$ they give will agree with every other correct match. Wrong matches mostly won't agree with each other.

```
Algorithm 2: RANSAC(matches, N, ε)
Input:  point matches (pᵢ, p'ᵢ), iterations N, inlier threshold ε
Output: homography H and its inlier set

 1:  best ← ∅
 2:  for k ← 1 to N do
 3:      S ← 4 random matches
 4:      H ← computeH(S)                          ▷ exact fit, 4 points
 5:      inliers ← { i : ‖H pᵢ − p'ᵢ‖ < ε }
 6:      if |inliers| > |best| then best ← inliers
 7:  end for
 8:  H ← computeH(best)                           ▷ least squares on all inliers
 9:  return H, best
```

I use $N = 5000$ and $\epsilon = 3$ px. Green lines are inliers and red ones were thrown out.

![ransac](media/b/B4_ransac_city_1_city_2.jpg "city · 175 inliers / 6 outliers")

![ransac](media/b/B4_ransac_interior_1_interior_2.jpg "interior 1 ↔ 2 · 29 inliers / 41 outliers")

![ransac](media/b/B4_ransac_interior_2_interior_3.jpg "interior 2 ↔ 3 · 65 inliers / 18 outliers")

![ransac](media/b/B4_ransac_sunset_1_sunset_2.jpg "sunset 1 ↔ 2 · 165 inliers / 13 outliers")

![ransac](media/b/B4_ransac_sunset_2_sunset_3.jpg "sunset 2 ↔ 3 · 119 inliers / 29 outliers")

Small aside about how many iterations we actually need.

If a fraction $w$ of the matches are correct, a random sample of 4 is all-correct with probability $w^4$. To have a 99% chance of at least one clean sample in $N$ tries:

$$
N = \frac{\log(1 - 0.99)}{\log(1 - w^4)}
$$

The worst pair here is interior 1 ↔ 2, with $w = 29/70 \approx 0.41$. That gives $w^4 \approx 0.029$ and $N \approx 157$. The city pair ($w \approx 0.97$) needs about 2. So 5000 is overkill by a lot, but each iteration is microseconds, so the whole thing still takes under a second (0.7–0.8 s per pair).

| pair | ratio-test matches | RANSAC inliers | automatic $H$ vs. my clicked points |
|---|---|---|---|
| city 1 → 2 | 181 | 175 | 1.04 px |
| interior 1 → 2 | 70 | 29 | 4.33 px |
| interior 2 → 3 | 83 | 65 | 2.02 px |
| sunset 1 → 2 | 178 | 165 | 1.81 px |
| sunset 2 → 3 | 148 | 119 | 1.53 px |

### Manual vs. automatic

Same reference photo, same warping, same blending. The only difference is where $H$ came from.

![manual](media/b/B4_mosaic_city_manual.jpg "city · manual") ![auto](media/b/B4_mosaic_city_auto.jpg "city · automatic")

![manual](media/b/B4_cyl_interior_manual.jpg "interior · manual") ![auto](media/b/B4_cyl_interior_auto.jpg "interior · automatic")

![manual](media/b/B4_cyl_sunset_manual.jpg "sunset · manual") ![auto](media/b/B4_cyl_sunset_auto.jpg "sunset · automatic")

They're basically identical. The auto ones are a bit sharper in the city and sunset overlaps, which I think is just because 29–175 inliers spread over the whole overlap beats my ~12 clicks.

![interior auto](media/b/B4_mosaic_interior_auto.jpg "interior · automatic · flat")

![sunset auto](media/b/B4_mosaic_sunset_auto.jpg "sunset · automatic · flat")

## Failures

**Parallax in the loft.** Interior 1 → 2 is the one pair where manual and automatic disagree (by 4.3 px on my clicked points). It's also the pair with the lowest inlier ratio (29 of 70). This is because I was standing a few feet from the stairs and didn't turn the phone exactly about its lens. A homography is only exact when the camera purely rotates (or the scene is flat). With some translation, near things (the stair rail) shift more than far things (the kitchen), and no single $H$ fits both. RANSAC locks onto whichever depth has the most matches, and my clicks were mostly on the stairs, so each version is sharp in a slightly different place. Neither one is really right.

**Wide angle.** The flat mosaics stretch a lot. The sunset mosaic in particular is mostly stretched-out sky. The fix is the cylinder in A.5.

**Corners at infinity.** When I chained homographies for the three-photo sets, some corners got sent toward infinity: as the denominator $h_7 x + h_8 y + 1$ approaches 0, the point flies off. My first interior canvas was about 5,000 × 3,600 px and ran out of memory. So I clip the canvas to a window around the reference, and I throw out any pixel whose ray points behind the camera.

## Bells & whistles (Part B)

### Rotation-invariant descriptors

The descriptors from B.2 are axis-aligned, so if the camera rolls, the same corner gives a completely different thumbnail.

Following MOPS §2, I give each corner an orientation: the direction of the blurred image gradient at that point ($\sigma = 4.5$). Then I sample the 8 × 8 grid in a 40 × 40 window *rotated* to that direction. So every patch gets described in its own coordinate frame, and it doesn't matter how the camera was turned.

To test it, I rotated sunset 2 by 25° and matched it against sunset 1.

![axis aligned](media/b/B5_rot_axis_aligned.jpg "axis-aligned · 4 matches, 4 inliers") ![rotation invariant](media/b/B5_rot_rotation_invariant.jpg "rotation-invariant · 135 matches, 121 inliers")

Axis-aligned descriptors find 4 matches, which is just the minimum RANSAC needs. Oriented descriptors find 135, and 121 of them are inliers.

![orientations](media/b/B5_orientations.jpg "oriented windows on the rotated photo · red = dominant gradient direction")

### Panorama recognition

The last thing I tried was throwing all eight photos in together and seeing if it could sort them into panoramas by itself. I shuffled them and ran the matcher on every pair (28 pairs). A pair counts as overlapping if it passes the test from [Brown & Lowe](https://doi.org/10.1007/s11263-006-0002-3): the number of RANSAC inliers $n_i$ must satisfy

$$
n_i > 5.9 + 0.22\, n_f
$$

where $n_f$ is the number of ratio-test matches. The overlapping pairs form a graph, and each connected piece of the graph is one panorama. Each one is stitched from a tree of verified homographies, starting from the photo with the most inliers. It isn't told the grouping or the order.

![shuffled](media/b/B5_pano_input.jpg "input · all eight photos in random order")

![matrix](media/b/B5_pano_matrix.jpg "verified matches between every pair")

Real pairs have 29–175 inliers, and every wrong pair has exactly 4, which is just the 4 points RANSAC fit to. It recovers all three panoramas:

![pano 1](media/b/B5_pano_group1.jpg "recognized · city 1, city 2")

![pano 2](media/b/B5_pano_group2.jpg "recognized · sunset 1, 2, 3")

![pano 3](media/b/B5_pano_group3.jpg "recognized · interior 1, 2, 3")

## Beyond Bells & Whistles: The Modern Research Frontier

In my opinion, this project, like Project 1, is a pretty friendly version of the problem. A few observations:

- **A: The camera only rotates.** That's what makes one $H$ per photo enough. The loft shows what happens when it isn't quite true.
- **B: Corners are distinctive.** Harris + a blurry thumbnail works because each corner's neighbourhood looks different from most others. Repeated structure (stairs, solar panels) and textureless areas (sky, white walls) break this.
- **C: The scene holds still.** Nothing in these photos really moves between shots.

This sparked my curiosity about what happens when these don't hold, and the loft already showed me one of them. Relaxing each one leads to a different area people are working on:

| Name | Explanation | Groups Working on It |
|---|---|---|
| **Learned features** | A network detects and describes keypoints | [SuperPoint](https://arxiv.org/abs/1712.07629) (Magic Leap) |
| **Learned matching** | Match using both images' features at once | [SuperGlue](https://arxiv.org/abs/1911.11763), [LightGlue](https://arxiv.org/abs/2306.13643) |
| **Detector-free matching** | Match densely, even on blank walls | [LoFTR](https://arxiv.org/abs/2104.00680) |
| **Pose-free 3D** | Unposed photos straight to 3D + cameras | [DUSt3R](https://arxiv.org/abs/2312.14132) (Naver Labs) |

We can break **A** by letting the camera move, which is what happened (a little) in my loft. Then each photo's relationship isn't a homography but a full 3D rotation and translation, plus a depth for every pixel, which is structure from motion. DUSt3R goes as far as skipping the matching step entirely and regressing the 3D geometry straight from the image pair.

We can break **B** by matching on things a hand-designed descriptor can't see. SuperGlue and LightGlue use attention between both images' features, so a corner on one stair tread can be matched by looking at where the rest of the staircase is, not just at its own 40 × 40 patch. LoFTR matches every region densely, which helps on the white walls in my loft where Harris finds nothing.

We can break **C** by stitching scenes that move, like a crowd or water. Then there's no single correct mosaic, and the problem becomes deciding which photo each moving object should come from, which is closer to the seam-finding and deghosting work in commercial panorama software.

## Sources

- [CS180 Project 4A description](https://cal-cs180.github.io/fa26/hw/proj3/partA.html) and [Project 4B description](https://cal-cs180.github.io/fa26/hw/proj3/partB.html)
- Brown, Szeliski and Winder, [Multi-Image Matching using Multi-Scale Oriented Patches](https://cal-cs180.github.io/fa26/hw/proj3/Papers/MOPS.pdf), CVPR 2005
- Brown and Lowe, [Automatic Panoramic Image Stitching using Invariant Features](https://doi.org/10.1007/s11263-006-0002-3), IJCV 2007
- Fischler and Bolles, [Random Sample Consensus](https://doi.org/10.1145/358669.358692), CACM 1981
- DeTone, Malisiewicz and Rabinovich, [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)
- Sarlin, DeTone, Malisiewicz and Rabinovich, [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763), CVPR 2020
- Lindenberger, Sarlin and Pollefeys, [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643), ICCV 2023
- Sun, Shen, Wang, Bao and Zhou, [LoFTR: Detector-Free Local Feature Matching with Transformers](https://arxiv.org/abs/2104.00680), CVPR 2021
- Wang, Leroy, Cabon, Chidlovskii and Revaud, [DUSt3R: Geometric 3D Vision Made Easy](https://arxiv.org/abs/2312.14132), CVPR 2024

## Thanks

Alexei Efros

Ren Ng
