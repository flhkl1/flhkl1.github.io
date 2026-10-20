# Builds 3/media from the raw notebook outputs.
#   python3 3/tools/prep3.py ~/Desktop/results . [PARTB_DIR]
# Crops the matplotlib panel grids into single tiles so the page can lay them
# out itself, and trims the white margins off the Part B plots.
import sys, os
from PIL import Image, ImageChops
R, S = sys.argv[1], sys.argv[2]
A = os.path.join(R, 'parta_updated/20261018-035100-7846d2')
# Part B can come from its own unzipped folder: pass it as a third argument.
B = sys.argv[3] if len(sys.argv) > 3 else os.path.join(R, 'partb_outputs')
oa = os.path.join(S, '3/media/a'); ob = os.path.join(S, '3/media/b')
os.makedirs(oa, exist_ok=True); os.makedirs(ob, exist_ok=True)

# panel boxes, measured once off the saved figures
ROW1 = [(25,435),(465,875)]            # plain panel grids
COL4 = [(15,426),(455,866),(895,1306),(1335,1746)]
TROWS = [(52,449),(478,875)]           # grids with a suptitle
TCOL4 = [(21,419),(461,859),(901,1299),(1341,1739)]
LROW = [(39,435)]                      # small-title single rows
LCOL = [(22,419),(462,859),(902,1299),(1342,1739)]

def jpg(im, path):
    im.convert('RGB').save(path, 'JPEG', quality=86, optimize=True, progressive=True)

def tiles(src, rows, cols):
    im = Image.open(src).convert('RGB')
    return [im.crop((c[0], r[0], c[1], r[1])) for r in rows for c in cols]

def trim(im, pad=12):
    bg = Image.new(im.mode, im.size, (255,255,255))
    bb = ImageChops.difference(im, bg).convert('L').point(lambda p: 255 if p > 12 else 0).getbbox()
    if not bb: return im
    l,t,r,b = bb
    return im.crop((max(0,l-pad), max(0,t-pad), min(im.width,r+pad), min(im.height,b+pad)))

T = ['02','04','06','10','14','25','50']

for name, t in zip(['orig','25','50','75'], tiles(f'{A}/training_inputs.png', [ROW1[0]], COL4)):
    jpg(t, f'{oa}/noisy_{name}.jpg')
for t in ['0.25','0.5','0.75']:
    k = t.split('.')[1].ljust(2,'0')
    jpg(tiles(f'{A}/blur_{t}.png', [ROW1[0]], COL4[:2])[1], f'{oa}/blur_{k}.jpg')
    jpg(tiles(f'{A}/one_step_{t}.png', LROW, LCOL[:3])[2], f'{oa}/onestep_{k}.jpg')
for t in ['0.25','0.50','0.75']:
    jpg(trim(Image.open(f'{A}/velocity_{t}.png').convert('RGB')), f'{oa}/velocity_{t[2:]}.jpg')
for name, t in zip(['50','60','70','80','90','100'], tiles(f'{A}/trajectory.png', ROW1, COL4[:3])):
    jpg(t, f'{oa}/traj_{name}.jpg')
for name, t in zip(['orig','blur','onestep','euler'], tiles(f'{A}/comparison.png', LROW, LCOL)):
    jpg(t, f'{oa}/cmp_{name}.jpg')
for i in range(5):
    jpg(Image.open(f'{A}/conditional_{i}.png'), f'{oa}/cond_{i}.jpg')
    jpg(Image.open(f'{A}/cfg_{i}.png'), f'{oa}/cfg_{i}.jpg')
for src, name in [('campanile','campanile'),('web_image','web'),('drawing_1','drawing1'),('drawing_2','drawing2')]:
    for k, t in zip(['orig']+T, tiles(f'{A}/edit_{src}.png', ROW1, COL4)):
        jpg(t, f'{oa}/edit_{name}_{k}.jpg')
for src, name in [('campanile','campanile'),('my_image_1','tower2'),('my_image_2','chick')]:
    for k, t in zip(['orig']+T, tiles(f'{A}/text_edit_{src}.png', TROWS, TCOL4)):
        jpg(t, f'{oa}/text_{name}_{k}.jpg')
for i in range(2):
    im = Image.open(f'{A}/anagram_{i}.png').convert('RGB')
    jpg(im, f'{oa}/anagram_{i}.jpg'); jpg(im.rotate(180), f'{oa}/anagram_{i}_rot.jpg')
    jpg(Image.open(f'{A}/hybrid_{i}.png'), f'{oa}/hybrid_{i}.jpg')

P = os.path.join(B, 'results/partb/all')
for src, dst in [(f'{B}/ood_1_2_2.png','ood.png'),(f'{B}/pure_noise_epoch1.png','pure_e1.png'),
                 (f'{B}/pure_noise_epoch5.png','pure_e5.png'),(f'{B}/train_loss_1_2_1.png','loss_denoise.png'),
                 (f'{B}/tc_loss.png','loss_tc.png'),(f'{P}/cc_sched_loss.png','loss_cc.png'),
                 (f'{B}/denoise_epoch1.png','denoise_epoch1.png'),(f'{B}/denoise_epoch5.png','denoise_epoch5.png'),
                 (f'{B}/train_loss_1_2_3.png','loss_pure.png'),(f'{P}/cc_nosched_loss.png','loss_cc_nosched.png')]:
    if not os.path.exists(src): print('skip (not in this run):', os.path.basename(src)); continue
    im = Image.open(src).convert('RGB')
    if dst.startswith('pure'): im = im.crop((0, 60, im.width, im.height))   # drop the suptitle
    trim(im).save(f'{ob}/{dst}', optimize=True)
for k in ['tc_epoch1','tc_epoch5','tc_epoch10','cc_epoch1','cc_epoch5','cc_epoch10',
          'cc_nosched_epoch1','cc_nosched_epoch5','cc_nosched_epoch10']:
    if not os.path.exists(f'{P}/{k}.png'): print('skip (not in this run):', k); continue
    im = Image.open(f'{P}/{k}.png').convert('RGB')
    im = im.crop((0, 40, im.width, im.height))      # drop the title
    trim(im, pad=0).save(f'{ob}/{k}.png', optimize=True)
print('done')
