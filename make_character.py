#!/usr/bin/env python3
"""
make_character.py - build a character JSON for talking_character.py from an image.

Finds the eyes, mouth and skin in a flat-colored cartoon portrait (open mouth,
eyes open, plain background), derives the blink/mouth shapes, placement and
colors from them, then asks which Kokoro voice the character should have,
playing a sample of each one you try.

Usage:
    python3 make_character.py IMAGE.png                    # writes IMAGE.json
    python3 make_character.py IMAGE.png -o imp.json --voice af_bella --speed 1.1
    python3 make_character.py IMAGE.png --eyes 380,620 630,600 --mouth 515,710

If detection picks the wrong features, pass rough points with --eyes / --mouth
(source-image pixels); the shapes are still traced automatically. Check the
result in the preview_*.png it writes (detected shapes | half blink, mid mouth |
closed eyes and mouth).
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, wave
from types import SimpleNamespace
import numpy as np
import cv2
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import talking_character as tc

LANGS = {'a': ('en-us', 'American English'), 'b': ('en-gb', 'British English'), 'e': ('es', 'Spanish'),
         'f': ('fr-fr', 'French'), 'h': ('hi', 'Hindi'), 'i': ('it', 'Italian'), 'j': ('ja', 'Japanese'),
         'p': ('pt-br', 'Portuguese'), 'z': ('cmn', 'Mandarin')}

# ======================================================================
# Image analysis
# ======================================================================
def quantize(a, k=16):
    # k-means in Lab: flat cartoon colors collapse to a small palette
    lab = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    px = lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, cen = cv2.kmeans(px[::13].copy(), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    lbl = np.concatenate([((c[:, None] - cen[None]) ** 2).sum(-1).argmin(1) for c in np.array_split(px, 16)])
    return lab, lbl.reshape(a.shape[:2]), cen

def blobs(mask, amin, amax):
    n, cc, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if amin <= area <= amax:
            out.append(dict(mask=cc == i, box=(x, y, w, h), area=int(area), c=cen[i], fill=area / (w * h)))
    return out

def mask_box(m):
    ys, xs = np.nonzero(m)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

class Analyzer:
    def __init__(self, a):
        self.a = a
        self.h, self.w = a.shape[:2]
        self.lab, self.lbl, self.cen = quantize(a)
        self.dark = self.lab[..., 0] < 45
        # per pixel: extent of its same-color component (small = a local detail like an eye part)
        self.ext = np.zeros(self.lbl.shape, np.int32)
        for k in range(len(self.cen)):
            m = self.lbl == k
            n, cc, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
            self.ext[m] = np.maximum(st[:, 2], st[:, 3])[cc[m]]

    def skin_near(self, x0, y0, x1, y1):
        # most common light-ish color in a box, plus clusters close to it in color
        x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(self.w, int(x1)), min(self.h, int(y1))
        lb = self.lbl[y0:y1, x0:x1].ravel()
        lb = lb[self.cen[lb, 0] > 60]
        if not len(lb):
            return None
        k = np.bincount(lb, minlength=len(self.cen)).argmax()
        near = [j for j in range(len(self.cen)) if np.linalg.norm(self.cen[j] - self.cen[k]) < 12]
        return np.isin(self.lbl, near)

    def mouth_hole(self, skin, eyes, hint=None):
        # an open mouth is a hole in the skin, below and between the eyes
        (ex1, ey1), (ex2, ey2) = eyes
        dx = abs(ex2 - ex1); mx, my = (ex1 + ex2) / 2, (ey1 + ey2) / 2
        cs, hier = cv2.findContours(skin.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hier is None:
            return None, 0
        skin_lab = np.median(self.lab[skin], 0)
        best, best_s = None, 0
        for c, hh in zip(cs, hier[0]):
            if hh[3] < 0:
                continue  # outer boundary, not a hole
            area = cv2.contourArea(c)
            if area < 0.003 * dx * dx or area > 0.6 * dx * dx:
                continue
            m = np.zeros(skin.shape, np.uint8); cv2.drawContours(m, [c], -1, 1, -1)
            m = m > 0
            hy, hx = np.argwhere(m).mean(0)
            if hint is not None:
                if not m[hint[1], hint[0]]:
                    continue
            elif not (my + 0.15 * dx < hy < my + 1.3 * dx and abs(hx - mx) < 0.45 * dx):
                continue
            contrast = min(1.0, np.linalg.norm(self.lab[m].astype(np.float32) - skin_lab, axis=1).mean() / 30)
            s = area * contrast * (1 - abs(hx - mx) / dx)
            if s > best_s:
                best, best_s = m, s
        return best, best_s

    def eye_pairs(self):
        A = self.h * self.w
        bl = blobs(self.dark, 0.0003 * A, 0.03 * A)
        pairs = []
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                b1, b2 = sorted((bl[i], bl[j]), key=lambda b: b['c'][0])
                dx = b2['c'][0] - b1['c'][0]; dy = abs(b2['c'][1] - b1['c'][1])
                if dx <= 0.5 * (b1['box'][2] + b2['box'][2]) or dx > 0.6 * self.w or dy > 0.4 * dx:
                    continue
                ratio = min(b1['area'], b2['area']) / max(b1['area'], b2['area'])
                if ratio < 0.35:
                    continue
                s = ratio * (1 - dy / dx) * min(b1['area'], b2['area']) ** 0.5 * min(b1['fill'], b2['fill']) ** 0.5
                pairs.append((s, b1, b2))
        return sorted(pairs, key=lambda p: -p[0])

    def blob_at(self, pt, r):
        # the dark blob nearest a hinted point (or the color patch under it)
        x, y = pt
        n, cc = cv2.connectedComponents(self.dark.astype(np.uint8), connectivity=8)
        win = cc[max(0, y - r):y + r, max(0, x - r):x + r]
        ids, cnt = np.unique(win[win > 0], return_counts=True)
        if len(ids):
            m = cc == ids[cnt.argmax()]
        else:
            m = self.lbl == self.lbl[y, x]
            _, cc = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
            m = cc == cc[y, x]
        ys, xs = np.nonzero(m)
        return dict(mask=m, c=np.array([xs.mean(), ys.mean()]))

    def find(self, eye_hint=None, mouth_hint=None):
        if eye_hint:
            r = max(4, abs(eye_hint[1][0] - eye_hint[0][0]) // 6)
            cands = [(1, *sorted((self.blob_at(p, r) for p in eye_hint), key=lambda b: b['c'][0]))]
        else:
            cands = self.eye_pairs()[:15]
        best = None
        for s, b1, b2 in cands:
            eyes = (b1['c'], b2['c'])
            dx = b2['c'][0] - b1['c'][0]
            bottom = max(mask_box(b1['mask'])[3], mask_box(b2['mask'])[3])
            skin = self.skin_near(b1['c'][0], bottom, b2['c'][0], bottom + 0.3 * dx)
            if skin is None:
                continue
            mouth, ms = self.mouth_hole(skin, eyes, mouth_hint)
            if mouth is not None and (best is None or s * ms > best[0]):
                best = (s * ms, b1, b2, skin, mouth)
        if best is None:
            sys.exit('Could not find the eyes and an open mouth; pass rough points with --eyes X,Y X,Y --mouth X,Y')
        return best[1:]

    def eye_region(self, blob, skin, dx):
        # grow from the dark blob through nearby non-skin details (lashes, iris, white),
        # but not through big shapes like hair or glasses frames
        local = (~skin & (self.ext < 0.6 * dx)) | blob['mask']
        _, cc = cv2.connectedComponents(local.astype(np.uint8), connectivity=8)
        m = np.isin(cc, np.unique(cc[blob['mask']]))
        yy, xx = np.mgrid[:self.h, :self.w]
        m &= (xx - blob['c'][0]) ** 2 + (yy - blob['c'][1]) ** 2 < (0.55 * dx) ** 2
        return fill(m)

def fill(m, close=5):
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((close, close), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(m)
    cv2.drawContours(out, [max(cs, key=cv2.contourArea)], -1, 1, -1)
    return out > 0

def polygon(m, pad):
    m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)))
    c = max(cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(c, True)
    while True:
        p = cv2.approxPolyDP(c, eps, True)
        if len(p) <= 24: break
        eps *= 1.3
    return [[int(x), int(y)] for x, y in p[:, 0]]

def curve(x0, y0, x1, y1, sag, n):
    # n points from (x0,y0) to (x1,y1), bowing down by sag in the middle
    return [[int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t + 4 * sag * t * (1 - t)))]
            for t in np.linspace(0, 1, n)]

def edge_y(m, left):
    # mean y of the outermost fifth of a region
    x0, _, x1, _ = mask_box(m)
    band = m[:, x0:x0 + max(1, (x1 - x0) // 5)] if left else m[:, x1 - max(1, (x1 - x0) // 5):x1]
    return np.nonzero(band)[0].mean()

def outline(an, hole):
    # mouth outline: the darkest of the bands just inside the hole, and how many bands match it
    m = hole.astype(np.uint8); k = np.ones((3, 3), np.uint8)
    bands = []
    for _ in range(12):
        e = cv2.erode(m, k); b = (m > 0) & (e == 0); m = e
        if not b.any(): break
        bands.append((np.median(an.a[b], 0), np.median(an.lab[b], 0).astype(np.float32)))
    if len(bands) < 3:
        return [int(v) for v in bands[0][0]], 3
    ref = min(bands[1:4], key=lambda b: b[1][0])
    width = 0
    for _, lab in bands[1:]:
        if np.linalg.norm(lab - ref[1]) < 20: width += 1
        elif width: break
    return [int(v) for v in ref[0]], width + 1  # +1: the antialiased outermost band

def darken_to(rgb, lmax):
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = min(lab[0, 0, 0], lmax)
    return [int(v) for v in cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]]

def lighten_to(rgb, lmin):
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] = max(lab[0, 0, 0], lmin)
    return [int(v) for v in cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)[0, 0]]

def palette(an, bg, skin):
    # ink: darkest sizeable color; accent: most vivid; code text: vivid, different hue
    A = an.h * an.w
    bg_lab = cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    sk = np.bincount(an.lbl[skin]).argmax()
    cols = []
    for k in range(len(an.cen)):
        m = an.lbl == k
        area = m.sum() / A
        if area < 0.005 or k == sk or np.linalg.norm(an.cen[k] - bg_lab) < 15:
            continue
        rgb = [int(v) for v in np.median(an.a[m], 0)]
        hsv = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2HSV)[0, 0].astype(float)
        cols.append(dict(rgb=rgb, area=area, L=an.cen[k][0], hue=hsv[0] * 2, chroma=np.hypot(an.cen[k][1] - 128, an.cen[k][2] - 128)))
    ink = darken_to(min(cols, key=lambda c: c['L'])['rgb'], 60) if cols else [40, 30, 45]
    vivid = sorted([c for c in cols if 40 < c['L'] < 235], key=lambda c: -c['chroma'] * c['area'] ** 0.5)
    accent = darken_to(vivid[0]['rgb'], 150) if vivid else [226, 72, 24]
    fg = [150, 240, 200]
    for c in vivid[1:]:
        dh = abs(c['hue'] - vivid[0]['hue']); dh = min(dh, 360 - dh)
        if dh > 60 and c['chroma'] > 25:
            fg = lighten_to(c['rgb'], 190); break
    return dict(ink=ink, accent=accent, code_bg=ink, code_fg=fg)

def analyze(path, eye_hint=None, mouth_hint=None):
    src = Image.open(path).convert('RGB')
    a = np.asarray(src)
    h, w = a.shape[:2]
    bg = [int(v) for v in np.median(np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3), a[-8:, :8].reshape(-1, 3)]), 0)]
    an = Analyzer(a)
    b1, b2, skin, mouth = an.find(eye_hint, mouth_hint)
    dx = b2['c'][0] - b1['c'][0]
    skin_rgb = [int(v) for v in np.median(a[skin], 0)]

    # mouth
    mline, mwidth = outline(an, mouth)
    mx0, my0, mx1, my1 = mask_box(mouth)
    mh = my1 - my0
    ys, xs = np.nonzero(mouth)
    lc = (xs.min(), ys[xs == xs.min()].mean()); rc = (xs.max(), ys[xs == xs.max()].mean())
    mouth_cfg = dict(skin_color=skin_rgb, line_color=mline, line_width=max(3, mwidth), mid_ratio=0.55,
                     hinge_y=int(my0 - 3), cover_polygon=polygon(mouth, max(2, mwidth // 2)),
                     closed_line=curve(lc[0] + 4, min(lc[1], my0 + 0.3 * mh), rc[0] - 4, min(rc[1], my0 + 0.3 * mh), 0.15 * mh, 7))

    # eyes
    shapes, dark_px = [], []
    for b in (b1, b2):
        m = an.eye_region(b, skin, dx)
        x0, y0, x1, y1 = mask_box(m); ew, eh = x1 - x0, y1 - y0
        ly, ry = edge_y(m, True) + 0.1 * eh, edge_y(m, False) + 0.1 * eh
        shapes.append(dict(cover_polygon=polygon(m, max(2, mwidth // 2)),
                           closed_line=curve(x0 + 0.04 * ew, ly, x1 - 0.04 * ew, ry, 0.2 * eh, 5)))
        px = a[m]; L = an.lab[m][:, 0]
        dark_px.append(px[L <= np.percentile(L, 15)])
    eye_w = max(max(p[0] for p in s['cover_polygon']) - min(p[0] for p in s['cover_polygon']) for s in shapes)
    eyes_cfg = dict(line_color=[int(v) for v in np.median(np.concatenate(dark_px), 0)],
                    line_width=int(max(mwidth, round(0.035 * eye_w), 3)),
                    half_ratio=0.45, interval=[2.5, 5.5], duration=0.16, double_chance=0.15, shapes=shapes)

    # placement: sway around a point below the mouth; bubble tail aims at the face's left edge
    mcx, mcy = xs.mean(), ys.mean()
    band = skin[int(mcy - 0.1 * dx):int(mcy + 0.1 * dx), :int(mcx)]
    cols = np.nonzero(band.any(0))[0]
    face_left = cols.min() if len(cols) else mx0 - 0.5 * dx
    tail = [int(face_left + 0.1 * dx), int(mcy)]
    height_frac, overhang = 1.06, 40
    sc = 1080 * height_frac / h
    cx0 = 1920 - w * sc + overhang
    bubble_right = int(np.clip(cx0 + face_left * sc - 30, 600, 930))

    cfg = dict(image=None, height_frac=height_frac, right_overhang=overhang, top_offset=-30,
               pivot=[int(mcx), int(0.9 * h)], tail_target=tail, bubble_right=bubble_right,
               motion=dict(sway=26, bob=9, tilt_deg=1.6), voice=None,
               mouth=mouth_cfg, eyes=eyes_cfg, colors=palette(an, bg, skin))
    return cfg

# ======================================================================
# Voice
# ======================================================================
def play(audio, sr):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix='.wav'); os.close(fd)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())
    for cmd in (['paplay'], ['pw-play'], ['aplay', '-q'], ['afplay'], ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet']):
        if shutil.which(cmd[0]):
            subprocess.run(cmd + [path]); os.remove(path); return
    print(f'  (no audio player found; sample saved to {path})')

def pick_voice(args):
    tts = tc.Speaker({}, args.kokoro_dir)
    voices = sorted(tts.model().get_voices())
    name, speed = args.voice, args.speed
    if name and name not in voices:
        sys.exit(f"Unknown voice {name!r}; choose from: {', '.join(voices)}")
    if name and speed:
        return name, speed
    print('\nKokoro voices (prefix: language, then f = female / m = male):')
    for p, (_, lang) in LANGS.items():
        for g, gname in (('f', 'female'), ('m', 'male')):
            vs = [v[3:] for v in voices if v.startswith(p + g + '_')]
            if vs:
                print(f'  {p}{g}_  {lang + " " + gname + ":":28s} {" ".join(vs)}')

    def sample(n, sp):
        tts.name, tts.speed, tts.lang = n, sp, LANGS.get(n[0], ('en-us',))[0]
        play(tts.say(args.sample_text), tts.SR)

    name = name or 'af_heart'
    speed = speed or 1.0
    print('\nType a voice name (or just the part after the prefix) to hear it; press Enter to keep the current one.')
    if not args.voice:
        while True:
            ans = input(f'voice [{name}]: ').strip()
            if not ans:
                break
            match = [v for v in voices if v == ans or v[3:] == ans]
            if not match:
                print(f'  no voice called {ans!r}'); continue
            name = match[0]; print(f'  {name}...'); sample(name, speed)
    while True:
        ans = input(f'speed [{speed:g}] (0.5-2, Enter to keep): ').strip()
        if not ans:
            break
        try:
            speed = float(ans)
        except ValueError:
            print('  not a number'); continue
        if not 0.5 <= speed <= 2:
            print('  speed must be between 0.5 and 2'); speed = 1.0; continue
        sample(name, speed)
    return name, speed

# ======================================================================
# Output
# ======================================================================
def dumps(v, ind=0):
    # indented like character.json, with point lists kept on one line
    pad = '  ' * (ind + 1)
    if isinstance(v, dict):
        flat = all(not isinstance(x, (dict, list)) or (isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x)) for x in v.values())
        if flat and len(v) <= 4 and ind > 0:
            return '{' + ', '.join(f'{json.dumps(k)}: {json.dumps(x)}' for k, x in v.items()) + '}'
        return '{\n' + ',\n'.join(f'{pad}{json.dumps(k)}: {dumps(x, ind + 1)}' for k, x in v.items()) + '\n' + '  ' * ind + '}'
    if isinstance(v, list) and any(isinstance(x, dict) for x in v):
        return '[\n' + ',\n'.join(pad + dumps(x, ind + 1) for x in v) + '\n' + '  ' * ind + ']'
    if isinstance(v, list):
        return json.dumps(v, separators=(',', ':')) if v and isinstance(v[0], list) else json.dumps(v)
    return json.dumps(v)

def write_preview(cfg_path, cfg, out):
    # render the real sprites via talking_character.setup(): shapes | half blink, mid mouth | closed
    fd, script = tempfile.mkstemp(suffix='.txt'); os.close(fd)
    with open(script, 'w') as f:
        f.write('Hi.\n')
    args = SimpleNamespace(character=cfg_path, script=script, size=1080, fps=30, cps=32, read_cps=38, hold_base=0.9,
                           voice=None, speed=None, no_voice=True, kokoro_dir='~/.cache/talking-character',
                           no_audio=True, no_blink=False)
    try:
        tc.setup(args)
    finally:
        os.remove(script)
    src = Image.open(os.path.join(os.path.dirname(os.path.abspath(cfg_path)), cfg['image'])).convert('RGB')
    def sprite(e, m):
        f = Image.fromarray(tc.G['frames'][e][m], 'RGBA').resize(src.size, Image.LANCZOS)
        bg = Image.new('RGBA', src.size, tuple(tc.G['BG']) + (255,)); bg.alpha_composite(f)
        return bg.convert('RGB')
    shapes = src.copy(); d = ImageDraw.Draw(shapes)
    lw = max(2, src.height // 300)
    for e in cfg['eyes']['shapes']:
        d.polygon([tuple(p) for p in e['cover_polygon']], outline=(0, 220, 255), width=lw)
        d.line([tuple(p) for p in e['closed_line']], fill=(255, 0, 200), width=lw)
    m = cfg['mouth']
    d.polygon([tuple(p) for p in m['cover_polygon']], outline=(0, 220, 255), width=lw)
    d.line([tuple(p) for p in m['closed_line']], fill=(255, 0, 200), width=lw)
    xs = [p[0] for p in m['cover_polygon']]
    d.line([(min(xs) - 20, m['hinge_y']), (max(xs) + 20, m['hinge_y'])], fill=(255, 200, 0), width=lw)
    for (x, y), col in ((cfg['pivot'], (255, 200, 0)), (cfg['tail_target'], (0, 200, 80))):
        r = 4 * lw; d.ellipse((x - r, y - r, x + r, y + r), fill=col)
    panels = [shapes, sprite(1, 1), sprite(2, 0)]
    sheet = Image.new('RGB', (src.width * 3, src.height), 'white')
    for i, p in enumerate(panels):
        sheet.paste(p, (i * src.width, 0))
    sheet.thumbnail((2400, 800))
    sheet.save(out)

# ======================================================================
def point(s):
    try:
        x, y = (int(round(float(v))) for v in s.split(','))
    except ValueError:
        raise argparse.ArgumentTypeError(f'expected X,Y, got {s!r}')
    return (x, y)

def main():
    ap = argparse.ArgumentParser(description='Create a character JSON for talking_character.py from an image.')
    ap.add_argument('image', help='character image: flat-colored portrait, eyes open, mouth open, plain background')
    ap.add_argument('-o', '--output', default=None, help='JSON to write (default: image name + .json)')
    ap.add_argument('--eyes', type=point, nargs=2, metavar='X,Y', help='rough centers of the two eyes, if detection misses them')
    ap.add_argument('--mouth', type=point, metavar='X,Y', help='a point inside the open mouth, if detection misses it')
    ap.add_argument('--voice', default=None, help='Kokoro voice, e.g. af_bella (skips the voice prompt)')
    ap.add_argument('--speed', type=float, default=None, help='speech speed multiplier (skips the speed prompt)')
    ap.add_argument('--sample-text', default="Hi there! Let me walk you through how this works, one step at a time.",
                    help='line spoken when trying voices')
    ap.add_argument('--kokoro-dir', default='~/.cache/talking-character', help='where Kokoro model files and cached speech go')
    ap.add_argument('--no-preview', action='store_true', help="don't write the preview_*.png check image")
    ap.add_argument('-f', '--force', action='store_true', help='overwrite an existing JSON')
    args = ap.parse_args()

    out = args.output or os.path.splitext(args.image)[0] + '.json'
    if os.path.exists(out) and not args.force:
        sys.exit(f'{out} exists; pass -f to overwrite or -o to pick another name')
    if args.speed is not None and not 0.5 <= args.speed <= 2:
        sys.exit('--speed must be between 0.5 and 2')
    if not (args.voice and args.speed) and not sys.stdin.isatty():
        sys.exit('Choosing a voice needs a terminal; pass --voice NAME --speed N instead.')

    print(f'analyzing {args.image} ...', file=sys.stderr)
    cfg = analyze(args.image, args.eyes, args.mouth)
    cfg['image'] = os.path.relpath(os.path.abspath(args.image), os.path.dirname(os.path.abspath(out)))
    m, e = cfg['mouth'], cfg['eyes']['shapes']
    print(f"  eyes near {[s['closed_line'][2] for s in e]}, mouth near {m['closed_line'][3]}, skin {m['skin_color']}", file=sys.stderr)

    name, speed = pick_voice(args)
    cfg['voice'] = dict(name=name, speed=speed, lang=LANGS.get(name[0], ('en-us',))[0])

    with open(out, 'w') as f:
        f.write(dumps(cfg) + '\n')
    print(f'wrote {out}', file=sys.stderr)
    if not args.no_preview:
        prev = os.path.join(os.path.dirname(os.path.abspath(out)), 'preview_' + os.path.splitext(os.path.basename(out))[0] + '.png')
        write_preview(out, cfg, prev)
        print(f'wrote {prev}  (detected shapes | half blink | eyes and mouth closed)', file=sys.stderr)
    print(f'try it: python3 talking_character.py SCRIPT.txt -c {out} --size 480 --preview 1.9', file=sys.stderr)

if __name__ == '__main__':
    main()
