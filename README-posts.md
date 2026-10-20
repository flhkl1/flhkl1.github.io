# Writing the project pages

Each project is one Markdown file:

```
0/post.md   1/post.md   2/post.md   3/post.md
```

`N/index.html` is a tiny shell that never needs editing. The look lives in `assets/post.css`
and the rendering in `assets/post.js`. Edit `post.md`, save, and refresh.

**Preview locally.** The page fetches `post.md`, so opening the HTML file directly won't work.
Serve the repo root instead:

```
cd ~/Desktop/Classes/project0/hello/flhkl1.github.io
python3 -m http.server 8000      # then open http://localhost:8000/3/
```

(`0/tools/serve.sh` works too.)

---

## The top of the file

```
---
title: Flow Matching
subtitle: CS 180 Project 3
date: October 2026
---
```

Optional extra keys: `cover: assets/background.jpg` adds a full-width banner image (project 0).
`effect: flow` makes the title animate out of noise (project 3).

## Text

Normal Markdown: `## Section`, `### Subsection`, `**bold**`, `*italic*`, `` `code` ``,
`[link](https://…)`, `- bullets`, `1. numbered`, `> quote`, tables, and fenced code blocks.

## Math

`$x_t = (1-t)\epsilon + t x_1$` inline. Put display math on its own lines:

```
$$
\hat{x}_1 = x_t + (1 - t)\, v_\theta(x_t, t, c)
$$
```

## Images

Put images on their own line (or a few on one line). The text in quotes is the caption. Images on the
same line, or on lines with no blank line between them, become one row of figures.

```
![alt text](media/a/cond_0.jpg "seed 180")

![a](media/x.jpg "left") ![b](media/y.jpg "middle") ![c](media/z.jpg "right")
```

Rows of 3 or more get a bit wider than the text column. Click any figure to see it large.

## Placeholders

A paragraph starting with `TODO` shows as a dashed box and is hidden in print/PDF. Anything before
the ` — ` (space, em dash, space) becomes its label:

```
TODO A·1.2 — The blur strengths you tried, and how the best result compares at each t.
```

Delete the whole line and write your paragraph in its place. `NEEDED …` works the same way but
shows an orange "result still missing" box. It is *not* hidden in print, so you don't miss one.

## Folds ("› Code")

A blank line after `<summary>` and before `</details>` is required:

~~~
<details><summary>Code</summary>

```python
def f(x): ...
```

</details>
~~~

## Widgets

Each widget is one line of HTML. The `data-caption` text shows below it.

**Slider**: one image, many versions. `{v}` in `data-src` is replaced by each value.

```
<div class="slider" data-src="media/part2/taj_sharpened_with_alpha_{v}.jpg"
     data-values="1,2,3,4,5,8,10" data-labels="α = 1,α = 2,…" data-caption="…"></div>
```

- `data-sets="Taj=taj|Golik=golik"` adds buttons that swap `{set}` in the path.
- `data-panels="apple_gaussian,apple_laplacian"` with `{p}` in the path shows several images side by side
  under one slider. `data-panel-labels` gives them labels.
- `data-start="2"` picks the starting index.

**Compare**: drag to reveal the left image over the right one.

```
<div class="compare" data-left="before.jpg" data-right="after.jpg"
     data-left-label="before" data-right-label="after" data-caption="…"></div>
```

`data-sets` works here too, with `{set}` in both paths.

**Distance**: a size slider for hybrid images.

```
<div class="distance" data-images="Name=path.jpg|Other=path2.jpg" data-caption="…"></div>
```

**Flip**: a visual anagram with a rotate button.

```
<div class="flip" data-src="anagram.jpg" data-up="upright prompt" data-down="rotated prompt"></div>
```

## Printing to PDF for Gradescope

Print the page from the browser (Save as PDF). TODO boxes, buttons and sliders are hidden in print.
Sliders print their current frame, and project 0's GIFs print as stills.
