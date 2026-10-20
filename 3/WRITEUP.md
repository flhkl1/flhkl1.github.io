# Project 3 write-up checklist

Every paragraph on the Project 3 page is yours to write. The whole page is one Markdown file,
`3/post.md`. Each spot you need to write is a line starting with `TODO`, which shows on the page as a
dashed box with its label and what the spec asks for. TODO boxes are hidden when you print to PDF.

**To fill a slot:** search `post.md` for the label (e.g. `TODO A·1.2`), delete that line, and write
your paragraphs in its place as plain text. Formatting cheat sheet (images, math, widgets):
`README-posts.md` at the repo root.

Tick each box below when it's done.

---

## Part A: The Power of Flow Models

| ✓ | search for | Section | What the spec asks for | Facts already on the page |
|---|----|---------|------------------------|---------------------------|
| [ ] | `TODO intro` | Top of page | *(optional)* Overview of the whole project | — |
| [ ] | `TODO A·intro` | Part A banner | *(optional)* Intro to Part A | — |
| [ ] | `TODO A·1.1` | 1.1 Noisy inputs | Original + three noisy inputs, labeled by time | One ε reused at t = 0.25 / 0.50 / 0.75 |
| [ ] | `TODO A·1.2` | 1.2 Classical denoising | Blur strengths tried, best result beside each noisy input | Best σ: 8 (t=0.25), 4 (t=0.50), 2 (t=0.75), picked by lowest MSE from {1,2,4,8,16} |
| [ ] | `TODO A·1.3` | 1.3 One-step denoising | Original, input, estimate at each t | One velocity call, "a high quality photo" |
| [ ] | `TODO A·1.4` | 1.4 Velocity | Predicted (1−t)v̂ vs ground-truth (1−t)v_gt, error heatmap | **See issue #1 below before writing this** |
| [ ] | `TODO A·1.5` | 1.5 Euler | Compare blur, one-step, iterative on the t = 0.50 input | MSE: blur 0.0241, one-step 0.0084, Euler 0.0099 |
| [ ] | `TODO A·1.6` | 1.6 From noise | Five samples, seeds recorded | Seeds 180–184, 50 steps |
| [ ] | `TODO A·2.1` | 2.1 CFG | Compare w = 7 to 1.6 with the same noise | w = 7, empty-string null prompt |
| [ ] | `TODO A·3.1` | 3.1 Editing | Campanile + two of your images, seven start times | t₀ ∈ {0.02, 0.04, 0.06, 0.10, 0.14, 0.25, 0.50} |
| [ ] | `TODO A·3.1b` | 3.1 Drawings / web | One web image + two of your drawings | — |
| [ ] | `TODO A·3.2` | 3.2 Text-guided editing | Your prompt per image and the edits | Prompts: "A rocket on a launch pad", "Donald Trump", "Zombie Sunflower" |
| [ ] | `TODO A·3.3` | 3.3 Visual anagrams | Two illusions, shown upright and rotated, with both prompts | old man ↔ people around a campfire; snowy mountain village ↔ horse |
| [ ] | `TODO A·3.4` | 3.4 Hybrids | Two hybrids at full and reduced size, both prompts | skull (far) × waterfall (close); cat (far) × forest (close); σ = 16 |

## Part B: Flow Matching from Scratch

| ✓ | search for | Section | What the spec asks for | Facts already on the page |
|---|----|---------|------------------------|---------------------------|
| [ ] | `TODO B·intro` | Part B banner | *(optional)* Intro to Part B | — |
| [ ] | `TODO B·1.1` | 1.1 UNet | *(optional)* Architecture notes | forward() is shown in a code fold |
| [ ] | `TODO B·1.2` | 1.2 Noising process | Visualization of the noising process | Top row of `ood.png` serves as this (σ = 0 to 1.0) |
| [ ] | `TODO B·1.2.1` | 1.2.1 Training | Loss curve, test results after epochs 1 and 5 | D=128, batch 256, Adam 1e-4, 5 epochs, 1175 steps |
| [ ] | `TODO B·1.2.2` | 1.2.2 Out of distribution | Results with σ it wasn't trained on | Same test 7, σ ∈ {0, .2, .4, .5, .6, .8, 1.0} |
| [ ] | `TODO B·1.2.3` | 1.2.3 Pure noise | **Required:** patterns in the outputs, relation to the digits, why | Hint in spec: MSE → centroid |
| [ ] | `TODO B·2.1` | 2.1 Time conditioning | *(optional)* How t is injected | FCBlocks multiply unflatten and up1 |
| [ ] | `TODO B·2.2` | 2.2 Training | Loss curve | D=64, batch 64, Adam 1e-2, γ = 0.1^(1/10), 10 epochs |
| [ ] | `TODO B·2.3` | 2.3 Sampling | Results at 1, 5, 10 epochs | 40 samples, seeds 0–39, 50 steps |
| [ ] | `TODO B·2.4` | 2.4 Class conditioning | *(optional)* One-hot c, extra FCBlocks, dropout | p_uncond = 0.1 |
| [ ] | `TODO B·2.5` | 2.5 Training | Loss curve | same hyperparameters as 2.2 |
| [ ] | `TODO B·2.6` | 2.6 Sampling | Results at 1, 5, 10 epochs, 4 of each digit | γ = 5, 300 steps, seed 0 |
| [ ] | `TODO B·2.6 no scheduler` | 2.6 No scheduler | **Required:** what you did to compensate, and the comparison | — |
| [ ] | `TODO end` | Takeaway | *(optional)* | — |

---

## Results that were missing (all done)

| ✓ | Where | Result | File |
|---|-------|--------|------|
| [x] | B · 1.2.1 | Test-set denoising at σ = 0.5 after epochs 1 and 5 | `media/b/denoise_epoch1.png`, `denoise_epoch5.png` |
| [x] | B · 1.2.3 | Pure-noise training loss | `media/b/loss_pure.png` |
| [x] | B · 2.6 | No-scheduler samples (lr 1e-3 constant + EMA 0.999) at epochs 1, 5, 10, loss curve, and a drag-compare against the scheduled run | `media/b/cc_nosched_epoch*.png`, `loss_cc_nosched.png` |

All Part B figures now come from the same fresh run (`~/Downloads/partb_outputs (1)`). To rebuild them:
`python3 3/tools/prep3.py ~/Desktop/results . ~/Downloads/"partb_outputs (1)"`

---

## Problems I found in the notebooks (worth fixing before you submit)

1. **1.4 `velocity_diagnostics` returns the wrong quantities.**
   `predicted_update` is set to `one_step_denoise(...)`, which is the clean-image *estimate* x̂₁,
   not the update (1−t)v̂. `ground_truth_update` is set to `interpolate(...)`, which is xₜ,
   not x₁ − xₜ. That's why the "predicted" panel looks like a clean Campanile.
   It should be:
   ```python
   v_hat = predict_velocity(x_t, t, condition)
   predicted_update    = (1 - t) * v_hat
   ground_truth_update = clean_image - x_t          # = (1 - t)(x1 - eps)
   error_map = ((predicted_update - ground_truth_update) ** 2).mean(1).sqrt()
   ```
   After a re-run, replace `velocity_*.png` and run `python3 3/tools/prep3.py ~/Desktop/results .` again.

2. **Sections 1.1–1.5 use the Campanile as "your own image".** The spec says to use one image of
   your own, *not* the Campanile.

3. **3.1 "two images of your own" are duplicates.** `my_image_1` is the Campanile again and
   `my_image_2` is the same file as the web image (`bird.png`). Both PNGs are byte-identical to the
   Campanile and web-image runs, so the page shows only Campanile / web image / drawing 1 / drawing 2.
   The 3.2 text edits reuse those same inputs.

4. **Bells & Whistles** (Part A and Part B) aren't in either notebook. They're required only for CS 280A.

---

## Regenerating the media

`3/tools/prep3.py` crops the notebook's matplotlib panel grids into single tiles and trims the
Part B plots. If you re-run anything, run it again from the repo root:

```
python3 3/tools/prep3.py ~/Desktop/results .
```

The panel crop boxes are hard-coded for the current figure sizes (1760×880 grids, etc.).
`3/tools/_chk.jpg` is a leftover preview image. You can delete it.
