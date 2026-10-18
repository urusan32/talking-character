#!/usr/bin/env python3
"""
talking_character.py - turn a plain-text script into a talking-character video.

The character (from a single still image) speaks each bubble with a Kokoro
text-to-speech voice, its mouth following the loudness of the speech while the
bubble types out; it blinks every few seconds and drifts/bobs gently. With
--no-voice it plays soft "voice blips" and flaps its mouth instead.

Usage:
    python3 talking_character.py SCRIPT.txt -o out.mp4
    python3 talking_character.py SCRIPT.txt --size 720 --preview 0 150 900   # quick stills

Script format (plain text, UTF-8):
    # title: The GIL, explained!        optional title card (first line(s) only)
    # subtitle: + free-threaded Python   optional, shown under the title
    == PART 1 · WHAT IS THE GIL?         section tag shown top-left (until next ==)
    Hi! This line is one bubble.

    A blank line starts a new bubble.
    A new line inside a bubble starts a new paragraph.
    > $ lines starting with "> " are code (monospace, dark box, not spoken)
    [pause 1.5]                          on its own: hold with no bubble for N seconds
    // comment lines are ignored

Character settings (image, mouth and eye shapes, placement) live in a JSON file;
see character.json next to this script.

Requires: Python 3.10+, Pillow, NumPy, OpenCV (opencv-python-headless), ffmpeg,
and kokoro-onnx for the voice (model files are downloaded on first use).
"""
import argparse, hashlib, json, math, multiprocessing as mp, os, random, re, subprocess, sys, time, urllib.request, wave
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))

# ======================================================================
# Script parsing
# ======================================================================
def parse_script(text):
    meta = {'title': None, 'subtitle': None}
    items, tag, cur = [], None, []

    def flush():
        nonlocal cur
        if cur:
            items.append({'kind': 'bubble', 'tag': tag, 'lines': cur})
        cur = []

    for raw in text.splitlines():
        line = raw.rstrip()
        s = line.strip()
        m = re.match(r'#\s*(title|subtitle)\s*:\s*(.*)$', s, re.I)
        if m:
            meta[m.group(1).lower()] = m.group(2).strip(); continue
        if s.startswith('//') or (s.startswith('#') and not items and not cur):
            continue
        if s.startswith('=='):
            flush(); tag = s.lstrip('=').strip() or None; continue
        m = re.match(r'\[pause\s+([\d.]+)\]$', s, re.I)
        if m:
            flush(); items.append({'kind': 'pause', 'secs': float(m.group(1))}); continue
        if not s:
            flush(); continue
        if line.lstrip().startswith('>'):
            code = line.lstrip()[1:]
            cur.append(('code', code[1:] if code.startswith(' ') else code))
        else:
            if cur and cur[-1][0] == 'body' and raw.startswith((' ', '\t')):
                cur[-1] = ('body', cur[-1][1] + ' ' + s)   # indented = continuation
            else:
                cur.append(('body', s))
    flush()
    if not any(i['kind'] == 'bubble' for i in items):
        sys.exit('Script has no dialogue.')
    return meta, items

# ======================================================================
# Globals filled by setup() (inherited by forked workers)
# ======================================================================
G = {}

def find_font(names):
    dirs = ['/usr/share/fonts', '/usr/local/share/fonts', os.path.expanduser('~/.fonts'),
            os.path.expanduser('~/.local/share/fonts'), HERE]
    for n in names:
        if os.path.isabs(n) and os.path.exists(n):
            return n
        for d in dirs:
            for root, _, files in os.walk(d):
                if n in files:
                    return os.path.join(root, n)
    return None

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default(size)

def setup(args):
    cfg = json.load(open(args.character))
    cdir = os.path.dirname(os.path.abspath(args.character))
    H = int(args.size); W = int(round(H * 16 / 9)) // 2 * 2
    S = H / 1080.0
    G.update(W=W, H=H, S=S, FPS=args.fps, CPS=args.cps)

    # ---------- character image + mouth variants ----------
    src = Image.open(os.path.join(cdir, cfg['image'])).convert('RGB')
    crop = cfg.get('crop') or [0, 0, *src.size]   # source-pixel box kept in the video; all other coords stay in full-image pixels
    srcw, srch = crop[2] - crop[0], crop[3] - crop[1]
    def smooth(pts, n=8):
        # Catmull-Rom spline through pts, so few control points give a clean curve
        p = [pts[0]] + list(pts) + [pts[-1]]; out = []
        for i in range(1, len(p) - 2):
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = p[i - 1:i + 3]
            for k in range(n):
                t = k / n; t2, t3 = t * t, t * t * t
                out.append(tuple(0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (3 * b - a - 3 * c + d) * t3)
                                 for a, b, c, d in ((x0, x1, x2, x3), (y0, y1, y2, y3))))
        return out + [tuple(pts[-1])]

    for fx in cfg.get('fixups', []):   # repaint part of the source: fill a polygon with the background, optionally outline an edge
        fd = ImageDraw.Draw(src)
        fd.polygon([tuple(q) for q in fx['fill']], fill=tuple(cfg['background']))
        if fx.get('line'):
            fd.line(smooth(fx['line']), fill=tuple(fx.get('color', [17, 18, 20])), width=fx.get('width', 9), joint='curve')
    a = np.asarray(src)
    bg = tuple(cfg.get('background') or np.median(
        np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3), a[-8:, :8].reshape(-1, 3)]), 0).astype(int).tolist())
    G['BG'] = bg
    def cover(base, poly, skin):
        # fill poly with skin, feathering the edge by a pixel so no seam shows
        msk = Image.new('L', base.size, 0); ImageDraw.Draw(msk).polygon(poly, fill=255)
        return Image.composite(Image.new('RGB', base.size, skin), base, msk.filter(ImageFilter.GaussianBlur(1.2)))

    def squash(base, poly, skin, ratio, anchor_y=None, bottom=False):
        # cover poly with skin, then paste the original region squashed vertically
        # toward anchor_y (top hinge) or, with bottom=True, toward the region's bottom
        out = cover(base, poly, skin)
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        box = (min(xs), min(ys) if anchor_y is None else anchor_y, max(xs) + 1, max(ys) + 1)
        reg = base.crop(box)
        msk = Image.new('L', base.size, 0); ImageDraw.Draw(msk).polygon(poly, fill=255)
        nh = max(1, int(reg.height * ratio))
        at = (box[0], box[3] - nh) if bottom else box[:2]
        out.paste(reg.resize((reg.width, nh), Image.LANCZOS), at,
                  msk.crop(box).resize((reg.width, nh), Image.LANCZOS))
        return out

    # eyes: open / half / closed versions of the base image
    eyes = None if args.no_blink else cfg.get('eyes')
    if eyes:
        eskin = tuple(eyes.get('skin_color') or (cfg.get('mouth') or {}).get('skin_color', [255, 219, 207]))
        ecol = tuple(eyes.get('line_color', [10, 12, 16]))
        half, closed = src, src.copy()
        for e in eyes['shapes']:
            poly = [tuple(p) for p in e['cover_polygon']]
            half = squash(half, poly, eskin, eyes.get('half_ratio', 0.5), bottom=True)
            closed = cover(closed, poly, eskin)
            ImageDraw.Draw(closed).line(smooth(e['closed_line']), fill=ecol,
                                        width=eyes.get('line_width', 7), joint='curve')
        eye_bases = [src, half, closed]
    else:
        eye_bases = [src]

    # mouth: closed / mid / open, applied on top of each eye state
    mouth = cfg.get('mouth')
    def mouth_variants(base):
        if not mouth:
            return [base, base, base]  # no mouth config: just bob
        poly = [tuple(p) for p in mouth['cover_polygon']]
        lip = [tuple(p) for p in mouth['closed_line']]
        skin = tuple(mouth['skin_color']); line_col = tuple(mouth.get('line_color', [118, 20, 22]))
        closed = base.copy(); ImageDraw.Draw(closed).polygon(poly, fill=skin)
        ImageDraw.Draw(closed).line(lip, fill=line_col, width=mouth.get('line_width', 4), joint='curve')
        top = mouth.get('hinge_y', min(p[1] for p in poly) - 6)
        mid = squash(base, poly, skin, mouth.get('mid_ratio', 0.55), anchor_y=top)
        return [closed, mid, base]

    def keyed(img):
        f = np.asarray(img.crop(tuple(crop))).astype(np.float32)
        d = np.sqrt(((f - np.array(bg, np.float32)) ** 2).sum(-1))
        alpha = np.clip((d - 14) / 26, 0, 1) * 255
        return np.dstack([f, alpha]).astype(np.uint8)

    ch_h = int(H * cfg.get('height_frac', 1.06))
    sc = ch_h / srch
    ch_w = int(srcw * sc)
    # G['frames'][eye_state][mouth_state]
    G['frames'] = [[cv2.resize(keyed(v), (ch_w, ch_h), interpolation=cv2.INTER_AREA) for v in mouth_variants(eb)]
                   for eb in eye_bases]
    G['CH'] = (ch_w, ch_h)
    G['STREAK'] = None
    st = (cfg.get('cry') or {}).get('streaks')
    if st:   # mask of the pale tear streaks, so setup_cry's flow can be drawn only on them
        f = a[crop[1]:crop[3], crop[0]:crop[2]].astype(np.float32)
        m = np.clip(1 - (np.sqrt(((f - np.array(st['color'], np.float32)) ** 2).sum(-1)) - st.get('tol', 28)) / 14, 0, 1)
        m = cv2.resize(m, (ch_w, ch_h), interpolation=cv2.INTER_AREA)
        ys, xs = np.where(m > 0.02)
        box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        G['STREAK'] = (m[box[1]:box[3], box[0]:box[2], None], box, st)
    G['CX0'] = W - ch_w + int(cfg.get('right_overhang', 40) * S)
    G['CY0'] = int(cfg.get('top_offset', -30) * S)
    px, py = cfg.get('pivot', [crop[0] + srcw * 0.55, crop[1] + srch * 0.88])
    G['PIVOT'] = ((px - crop[0]) * sc, (py - crop[1]) * sc)
    tx, ty = cfg.get('tail_target', [crop[0] + srcw * 0.3, crop[1] + srch * 0.68])
    G['TAILT'] = ((tx - crop[0]) * sc, (ty - crop[1]) * sc)
    G['SRC2CH'] = (crop[0], crop[1], sc)
    mv = cfg.get('motion', {})
    G['MOTION'] = (mv.get('sway', 26) * S, mv.get('bob', 9) * S, mv.get('tilt_deg', 1.6))
    G['CRYCFG'] = cfg.get('cry')

    # ---------- fonts ----------
    bold = find_font(cfg.get('fonts', {}).get('body', []) + ['DejaVuSans-Bold.ttf', 'NotoSans-Bold.ttf', 'LiberationSans-Bold.ttf', 'FreeSansBold.ttf'])
    mono = find_font(cfg.get('fonts', {}).get('code', []) + ['DejaVuSansMono-Bold.ttf', 'NotoSansMono-Bold.ttf', 'LiberationMono-Bold.ttf', 'FreeMonoBold.ttf'])
    G['FONT_BOLD'], G['FONT_MONO'] = bold, mono
    G['F_TAG'] = load_font(bold, int(30 * S))
    G['F_TITLE'] = load_font(bold, int(72 * S))
    col = cfg.get('colors', {})
    G['INK'] = tuple(col.get('ink', [40, 30, 45]))
    G['ACCENT'] = tuple(col.get('accent', [226, 72, 24]))
    G['CODEBG'] = tuple(col.get('code_bg', [34, 40, 54]))
    G['CODEFG'] = tuple(col.get('code_fg', [150, 240, 200]))
    G['BX0'], G['BX1'] = int(70 * S), int(cfg.get('bubble_right', 930) * S)

    # ---------- script & timeline ----------
    meta, items = parse_script(open(args.script, encoding='utf-8').read())
    G['META'] = meta
    title_s = 2.6 if meta['title'] else 0.0
    G['TITLE_S'] = title_s
    vcfg = dict(cfg.get('voice', {}))
    if args.voice: vcfg['name'] = args.voice
    if args.speed: vcfg['speed'] = args.speed
    tts = None if args.no_voice else Speaker(vcfg, args.kokoro_dir)
    bubbles, clips, t = [], [], title_s
    for it in items:
        if it['kind'] == 'pause':
            t += it['secs']; continue
        lay = layout_bubble(it['lines'])
        n = lay['nchars']
        text = ' '.join(x for k, x in it['lines'] if k == 'body')
        clip = tts.say(text) if tts and text.strip() else None
        if clip is not None:
            # type the bubble out over the speech, then hold briefly
            rev = max(len(clip) / tts.SR, n / (args.cps * 3))
            hold = args.hold_base
            clips.append((t + 0.15, clip))
        else:
            rev = n / args.cps
            hold = args.hold_base + n / args.read_cps
        bubbles.append(dict(tag=it['tag'], lay=lay, s=t, r=t + rev, e=t + rev + hold, cps=n / rev))
        t += rev + hold + 0.25
    G['BUBBLES'] = bubbles
    G['TOTAL'] = t + 1.2
    G['NF'] = int(G['TOTAL'] * args.fps)

    # speech track: every clip mixed onto one timeline; per-frame loudness drives the mouth
    G['SPEECH'] = None
    if tts:
        sr = tts.SR; buf = np.zeros(int(G['TOTAL'] * sr), np.float32)
        for at, clip in clips:
            i0 = int(at * sr); buf[i0:i0 + len(clip)] += clip[:len(buf) - i0]
        G['SPEECH'], G['SR'] = buf, sr
        G['MOUTH'] = mouth_track(buf, sr, args.fps, G['NF'])

    # deterministic mouth sequence (changes every 3 frames, never repeats)
    r = random.Random(3); seq, prev = [], 0
    for _ in range(G['NF'] // 3 + 2):
        prev = r.choice([c for c in (0, 1, 2, 1, 2) if c != prev]); seq.append(prev)
    G['MSEQ'] = seq

    # deterministic blink schedule: (start, duration) every few seconds, sometimes a double blink
    blinks = []
    if eyes:
        lo, hi = eyes.get('interval', [2.5, 5.5]); dur = eyes.get('duration', 0.16)
        r = random.Random(5); t = r.uniform(0.8, lo)
        while t < G['TOTAL']:
            blinks.append((t, dur))
            if r.random() < eyes.get('double_chance', 0.15):
                blinks.append((t + dur + 0.12, dur))
            t += r.uniform(lo, hi)
    G['BLINKS'] = blinks

    # background: pre-render dot grid once
    step = int(60 * S)
    dots = Image.new('RGBA', (W + 2 * step, H + 2 * step), (0, 0, 0, 0))
    dd = ImageDraw.Draw(dots); rr = max(2, int(3 * S))
    for gx in range(0, W + 2 * step, step):
        for gy in range(0, H + 2 * step, step):
            dd.ellipse((gx - rr, gy - rr, gx + rr, gy + rr), fill=(255, 255, 255, 40))
    G['DOTS'], G['STEP'] = dots, step
    br = random.Random(7)
    G['BLOBS'] = [(br.uniform(0, W), br.uniform(0, H), br.uniform(80, 230) * S, br.uniform(0, 6.28), br.uniform(0.05, 0.12)) for _ in range(9)]
    G['AUDIO'] = not args.no_audio
    if G['CRYCFG']:
        setup_cry(G['CRYCFG'])

# ======================================================================
# Crying effects (character.json "cry"): tear spray, sobbing, rising water
# ======================================================================
SOB_BEATS = ((0.15, 1.0), (0.42, 0.7))   # (seconds into the sob period, strength) of each jolt

def setup_cry(c):
    S = G['S']
    sob = c.get('sob')
    G['SOB'] = (sob['period'], sob.get('amp', 14) * S, sob.get('tilt_deg', 1.2)) if sob else None
    G['WATER'] = c.get('water')
    G['RAIN'] = c.get('rain', 0)
    x0, y0, sc = G['SRC2CH']
    ev = []   # one row per tear: start time and position, velocity, size
    r = random.Random(21)
    for e in c.get('emitters', []):
        t = -2.5
        while t < G['TOTAL']:
            t += r.expovariate(e['rate'])
            dx, dy, _ = char_motion(t)
            sx, sy = e.get('spread', [0, 0])
            ev.append((t, G['CX0'] + dx + (e['pos'][0] - x0) * sc, G['CY0'] + dy + (e['pos'][1] - y0) * sc,
                       (e['vel'][0] + r.gauss(0, sx)) * S, (e['vel'][1] + r.gauss(0, sy)) * S,
                       e.get('size', 8) * S * r.uniform(0.7, 1.25)))
    G['TEARS'] = np.array(ev, np.float32).reshape(-1, 6)
    G['RAINDROPS'] = [(r.uniform(0, G['W']), r.uniform(0, G['H']), r.uniform(220, 420) * S, r.uniform(4, 8) * S) for _ in range(G['RAIN'])]

def sob_pulse(tt):
    # jolt strength at time tt, plus a broader envelope for the trembling in between
    if not G.get('SOB'): return 0.0, 0.0
    k = tt % G['SOB'][0]
    jolt = sum(a * math.exp(-((k - c) / 0.06) ** 2) for c, a in SOB_BEATS)
    return jolt, math.exp(-((k - 0.3) / 0.35) ** 2)

def char_motion(tt):
    # character offset (dx, dy) in pixels and rotation in degrees
    sway, bob, tilt = G['MOTION']
    dx = sway * math.sin(tt * 0.45) + sway * 0.3 * math.sin(tt * 1.1)
    dy = bob * math.sin(tt * 1.6) + bob * 0.55 * math.sin(tt * 0.7)
    ang = tilt * math.sin(tt * 0.6 + 0.5)
    if G.get('SOB'):
        jolt, trem = sob_pulse(tt)
        dy += G['SOB'][1] * jolt
        ang -= G['SOB'][2] * jolt
        dx += 2.5 * G['S'] * trem * math.sin(tt * 41)
    return dx, dy, ang

def flow_streaks(ch, tt):
    # bands of light and shade sliding down the tear streaks
    m, (x0, y0, x1, y1), st = G['STREAK']
    S = G['S']
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    ph = ys / (st.get('wavelength', 46) * S) * 2 * np.pi - tt * st.get('speed', 7) + 0.9 * np.sin(xs / (30 * S) + tt)
    v = (0.5 + 0.5 * np.sin(ph))[..., None]
    lit = np.array(st.get('light', [238, 253, 250]), np.float32); dark = np.array(st.get('dark', [150, 214, 214]), np.float32)
    col = dark * (1 - v) + lit * v
    out = ch.copy(); reg = out[y0:y1, x0:x1]
    reg[..., :3] = (reg[..., :3] * (1 - m * 0.6) + col * m * 0.6).astype(np.uint8)
    return out

def water_level(tt):
    # surface height in pixels above the bottom edge: rises over the whole video
    w = G['WATER']
    return G['H'] * (w['start'] + (w['end'] - w['start']) * min(1, tt / G['TOTAL']))

def water_surface(tt):
    xs = np.arange(G['W'], dtype=np.float32); S = G['S']
    return G['H'] - water_level(tt) + S * (7 * np.sin(xs / (90 * S) + tt * 1.7) + 4 * np.sin(xs / (37 * S) - tt * 2.6))

def draw_crying(img, tt):
    """Tear drops flying and falling, then the rising water over the character."""
    arr = np.array(img)
    W, H, S = G['W'], G['H'], G['S']
    surf = water_surface(tt) if G['WATER'] else np.full(W, H + 100, np.float32)
    g = 1700 * S
    T = G['TEARS']
    dt = tt - T[:, 0]
    live = (dt >= 0) & (dt < 3.0)
    for t0, x0, y0, vx, vy, r in T[live]:
        d = tt - t0
        x = x0 + vx * d; y = y0 + vy * d + 0.5 * g * d * d
        if not (0 <= x < W) or y > H or y > surf[int(x)] - r:
            continue
        ang = math.degrees(math.atan2(vy + g * d, vx))
        c = (int(x), int(y)); ax = (max(2, int(r * 1.5)), max(2, int(r)))
        cv2.ellipse(arr, c, (ax[0] + 2, ax[1] + 2), ang, 0, 360, (36, 160, 170, 255), -1, cv2.LINE_AA)
        cv2.ellipse(arr, c, ax, ang, 0, 360, (190, 238, 236, 255), -1, cv2.LINE_AA)
        cv2.circle(arr, (int(x - r * 0.3), int(y - r * 0.3)), max(1, int(r * 0.28)), (255, 255, 255, 255), -1, cv2.LINE_AA)
    if G['WATER']:
        rows = np.arange(H, dtype=np.float32)[:, None]
        depth = np.clip((rows - surf[None, :]) / (60 * S), 0, 1)            # 0 at the surface, 1 when deep
        a = (rows >= surf[None, :]).astype(np.float32)
        near = np.clip(1 - np.abs(rows - surf[None, :]) / (3.5 * S), 0, 1)  # bright line along the surface
        col = np.array(G['WATER']['color'], np.float32); deep = col * 0.78
        tint = col[None, None, :] * (1 - depth[..., None]) + deep[None, None, :] * depth[..., None]
        al = (a * G['WATER'].get('alpha', 0.62))[..., None]
        out = arr[..., :3].astype(np.float32) * (1 - al) + tint * al
        out = out * (1 - near[..., None] * 0.8) + np.array([205, 245, 245], np.float32) * near[..., None] * 0.8
        arr[..., :3] = out.astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')

def make_sfx(sr):
    """Quiet water plinks (more as the water rises) and a sniffle on every sob."""
    buf = np.zeros(int(G['TOTAL'] * sr), np.float32); r = random.Random(33)
    t = np.arange(int(0.16 * sr)) / sr
    k = 0.4
    while k < G['TOTAL']:
        f = r.uniform(700, 1500)
        env = np.exp(-t * 38) * np.minimum(1, t * 1500)
        plink = np.sin(2 * np.pi * (f * t + 90 * (1 - np.exp(-t * 30)))) * env * r.uniform(0.03, 0.06)
        i0 = int(k * sr); buf[i0:i0 + len(plink)] += plink[:len(buf) - i0]
        k += r.expovariate(2.0 + 5.0 * k / G['TOTAL'])
    if G.get('SOB'):
        n = int(0.22 * sr); u = np.arange(n) / sr
        for s0 in np.arange(0, G['TOTAL'], G['SOB'][0]):
            for c, a in SOB_BEATS:
                noise = np.random.RandomState(int(s0 * 10 + c * 100)).randn(n).astype(np.float32)
                lp = np.convolve(noise, np.ones(6) / 6, 'same'); sn = (lp - np.convolve(lp, np.ones(60) / 60, 'same'))
                sn *= np.sin(np.pi * np.clip(u / 0.22, 0, 1)) ** 2 * 0.22 * a
                i0 = int((s0 + c + 0.05) * sr)
                if i0 + n <= len(buf): buf[i0:i0 + n] += sn
    return buf

# ======================================================================
# Bubble layout
# ======================================================================
def wrap(text, font, maxw):
    out, cur = [], ''
    for w in text.split(' '):
        test = (cur + ' ' + w).strip()
        if font.getlength(test) <= maxw or not cur:
            cur = test
        else:
            out.append(cur); cur = w
    if cur: out.append(cur)
    return out

def layout_bubble(lines):
    S, H = G['S'], G['H']
    pad = int(46 * S)
    avail = G['BX1'] - G['BX0'] - 2 * pad
    max_h = H - int(200 * S)
    for size in range(44, 21, -2):
        fb = load_font(G['FONT_BOLD'], int(size * S))
        msize = int(size * 0.86 * S)
        fm = load_font(G['FONT_MONO'], msize)
        codes = [t for k, t in lines if k == 'code']
        while codes and msize > 12 and max(fm.getlength(c) for c in codes) > avail - 20 * S:
            msize -= 1; fm = load_font(G['FONT_MONO'], msize)
        rh_b, rh_c = int(size * 1.32 * S), int(msize * 1.37)
        rows, prev = [], None
        for k, t in lines:
            if k == 'code':
                rows.append(['code', t, int((10 if prev == 'code' else 22) * S), rh_c])
            else:
                for j, part in enumerate(wrap(t, fb, avail)):
                    rows.append(['body', part, int(18 * S) if (j == 0 and rows) else 0, rh_b])
            prev = k
        hgt = sum(g + h for _, _, g, h in rows) + 2 * pad - int(6 * S)
        if hgt <= max_h:
            break
    nchars = sum(len(t) + (0 if k == 'code' else 1) for k, t, _, _ in rows)
    return dict(rows=rows, h=hgt, fb=fb, fm=fm, pad=pad, nchars=nchars)

# ======================================================================
# Drawing
# ======================================================================
def ease_out_back(x):
    return 1 + 2.5 * (x - 1) ** 3 + 1.5 * (x - 1) ** 2

def background(tt):
    W, H, S = G['W'], G['H'], G['S']
    img = Image.new('RGBA', (W, H), G['BG'] + (255,))
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for x, y, r, ph, sp in G['BLOBS']:
        cx = x + 40 * S * math.sin(tt * sp + ph); cy = y + 30 * S * math.cos(tt * sp * 1.3 + ph)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255, 34))
    for x, y, v, r in G.get('RAINDROPS', []):
        cy = (y + v * tt) % (H + 4 * r) - 2 * r
        d.ellipse((x - r * 0.6, cy - r * 1.4, x + r * 0.6, cy + r * 1.4), fill=(255, 255, 255, 46))
    img.alpha_composite(layer)
    step = G['STEP']; off = (tt * 12 * G['S']) % step
    ox, oy = int(step - off), int(step - off * 0.5)
    img.alpha_composite(G['DOTS'].crop((ox, oy, ox + W, oy + H)))
    return img

def draw_bubble(img, b, tt, tip):
    s, r, e = b['s'], b['r'], b['e']
    if tt < s or tt > e + 0.25:
        return
    S, H, INK = G['S'], G['H'], G['INK']
    lay = b['lay']; pad = lay['pad']
    BX0, BX1 = G['BX0'], G['BX1']
    hgt = lay['h']
    by0 = int(min(max(470 * S - hgt / 2, 130 * S), H - 30 * S - hgt)); by1 = by0 + hgt
    # local canvas bounding bubble + tail + shadow
    sh = int(10 * S)
    lx0, ly0 = BX0 - 4, min(by0, tip[1]) - 4
    lx1, ly1 = max(BX1, tip[0]) + sh + 8, max(by1, tip[1]) + sh + 8
    lw, lh = lx1 - lx0, ly1 - ly0
    def L(x, y): return (x - lx0, y - ly0)

    ttop = max(by0 + int(30 * S), by1 - int(130 * S))
    tail = [L(BX1 - int(60 * S), ttop), L(BX1 - int(60 * S), by1 - int(30 * S)), L(*tip)]
    m = Image.new('L', (lw, lh), 0); md = ImageDraw.Draw(m)
    md.rounded_rectangle((*L(BX0, by0), *L(BX1, by1)), int(44 * S), fill=255)
    md.polygon(tail, fill=255)
    marr = np.asarray(m)
    k = max(3, int(13 * S) | 1)
    inner = cv2.erode(marr, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = np.zeros((lh, lw, 4), np.uint8)
    shadow = np.zeros_like(marr); shadow[sh:, sh:] = marr[:-sh, :-sh] if sh else marr
    sa = (shadow.astype(np.uint16) * 70 // 255).astype(np.uint8)
    out[..., :3] = (20, 60, 70); out[..., 3] = sa
    on = marr > 0
    out[on] = INK + (255,)
    oi = inner > 0
    out[oi] = (255, 255, 255, 255)
    # soften edges using the mask's own antialiasing
    out[..., 3] = np.maximum(sa, marr)
    layer = Image.fromarray(out, 'RGBA')
    d = ImageDraw.Draw(layer)

    shown = int((tt - s) * b['cps'])
    rows = lay['rows']
    # code box (appears once its first character is reached)
    y = by0 + pad - int(4 * S); cb = None; before = 0; counted = 0
    for kind, txt, gap, rh in rows:
        y += gap
        if kind == 'code':
            if cb is None: cb = [y - int(8 * S)]; before = counted
            cb.append(y + rh - int(4 * S))
        counted += len(txt) + (0 if kind == 'code' else 1)
        y += rh
    if cb and shown > before:
        d.rounded_rectangle((*L(BX0 + pad - int(14 * S), cb[0]), *L(BX1 - pad + int(14 * S), cb[-1])), int(16 * S), fill=G['CODEBG'])
    y = by0 + pad - int(4 * S); left = shown
    for kind, txt, gap, rh in rows:
        y += gap
        part = txt[:max(0, left)]
        left -= len(txt) + (0 if kind == 'code' else 1)
        if part:
            if kind == 'code':
                d.text(L(BX0 + pad + int(6 * S), y + int(4 * S)), part, font=lay['fm'], fill=G['CODEFG'])
            else:
                d.text(L(BX0 + pad, y), part, font=lay['fb'], fill=INK)
        y += rh

    pin = min(1, (tt - s) / 0.28); pout = max(0, min(1, (tt - e) / 0.25))
    alpha = (1 - pout) * min(1, pin * 2.5)
    px, py = lx0, ly0
    if pin < 1:
        scale = max(0.05, 0.75 + 0.25 * ease_out_back(pin))
        ax, ay = L(BX1 - int(40 * S), by1 - int(80 * S))
        layer = layer.resize((max(1, int(lw * scale)), max(1, int(lh * scale))), Image.BILINEAR)
        px, py = int(lx0 + ax - ax * scale), int(ly0 + ay - ay * scale)
    if alpha < 1:
        arr = np.asarray(layer).copy(); arr[..., 3] = (arr[..., 3] * alpha).astype(np.uint8)
        layer = Image.fromarray(arr, 'RGBA')
    img.alpha_composite(layer, (max(0, px), max(0, py)),
                        (max(0, -px), max(0, -py), layer.width, layer.height))

def draw_tag(img, tt):
    cur = None
    for b in G['BUBBLES']:
        if tt >= b['s'] - 0.2: cur = b['tag']
    if not cur: return
    S = G['S']; d = ImageDraw.Draw(img); f = G['F_TAG']
    w = f.getlength(cur)
    d.rounded_rectangle((int(60 * S), int(46 * S), int(60 * S + w + 48 * S), int(106 * S)), int(30 * S), fill=G['ACCENT'])
    d.text((int(84 * S), int(58 * S)), cur, font=f, fill='white')

def draw_title(img, tt):
    T = G['TITLE_S']; meta = G['META']
    if not meta['title'] or tt > T + 0.3: return
    S, INK = G['S'], G['INK']
    a = 1.0 if tt < T - 0.4 else max(0, (T + 0.3 - tt) / 0.7)
    p = min(1, tt / 0.5)
    words = meta['title'].split(' ')
    # split title over two lines, second in accent color
    commas = [i for i, w in enumerate(words[:-1]) if w.endswith((',', ':', '—'))]
    if commas:
        half = commas[0] + 1
    else:  # balance by character count
        tot = len(meta['title']); acc = 0; half = 1
        for i, w in enumerate(words[:-1]):
            acc += len(w) + 1
            if acc >= tot / 2: half = i + 1; break
    l1, l2 = ' '.join(words[:half]), ' '.join(words[half:])
    f = G['F_TITLE']
    while max(f.getlength(l1), f.getlength(l2)) > G['BX1'] - G['BX0'] - 90 * S and f.size > 30:
        f = load_font(G['FONT_BOLD'], f.size - 4)
    layer = Image.new('RGBA', img.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
    y = int(380 * S + 40 * S * (1 - ease_out_back(p)))
    x0 = G['BX0']
    d.rounded_rectangle((x0, y, G['BX1'] - int(30 * S), y + int(300 * S)), int(40 * S), fill=(255, 255, 255, 235), outline=INK, width=max(2, int(6 * S)))
    d.text((x0 + int(45 * S), y + int(45 * S)), l1, font=f, fill=INK)
    d.text((x0 + int(45 * S), y + int(130 * S)), l2, font=f, fill=G['ACCENT'])
    if meta['subtitle']:
        d.text((x0 + int(48 * S), y + int(228 * S)), meta['subtitle'], font=G['F_TAG'], fill=(90, 90, 110))
    if a < 1:
        arr = np.asarray(layer).copy(); arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
        layer = Image.fromarray(arr, 'RGBA')
    img.alpha_composite(layer)

def mouth_state(fi, tt):
    if G['SPEECH'] is not None:
        return G['MOUTH'][fi] if fi < len(G['MOUTH']) else 0
    if any(b['s'] <= tt < b['r'] + 0.08 for b in G['BUBBLES']):
        return G['MSEQ'][fi // 3]
    return 0

def eye_state(tt):
    # 0 open, 1 half, 2 closed: half -> closed -> half over each blink
    if len(G['frames']) == 1:
        return 0
    for s, d in G['BLINKS']:
        if s <= tt < s + d:
            x = (tt - s) / d
            return 2 if 0.25 <= x < 0.7 else 1
        if s > tt:
            break
    return 0

def render_frame(fi):
    tt = fi / G['FPS']
    img = background(tt)
    dx, dy, ang = char_motion(tt)
    ch = G['frames'][eye_state(tt)][mouth_state(fi, tt)]
    if G['STREAK']:
        ch = flow_streaks(ch, tt)
    cw, chh = G['CH']
    M = cv2.getRotationMatrix2D(G['PIVOT'], ang, 1.0)
    rot = cv2.warpAffine(ch, M, (cw, chh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    ox, oy = int(G['CX0'] + dx), int(G['CY0'] + dy)
    cimg = Image.fromarray(rot, 'RGBA')
    # clip to frame
    W, H = G['W'], G['H']
    sx0, sy0 = max(0, -ox), max(0, -oy)
    sx1, sy1 = min(cw, W - ox), min(chh, H - oy)
    img.alpha_composite(cimg, (ox + sx0, oy + sy0), (sx0, sy0, sx1, sy1))
    tx, ty = G['TAILT']
    tip = (int(ox + tx), int(oy + ty))
    if G['CRYCFG']:
        img = draw_crying(img, tt)
    draw_title(img, tt)
    draw_tag(img, tt)
    for b in G['BUBBLES']:
        draw_bubble(img, b, tt, tip)
    return img.convert('RGB')

def frame_bytes(fi):
    return render_frame(fi).tobytes()

# ======================================================================
# Audio
# ======================================================================
KOKORO_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'

class Speaker:
    """Kokoro TTS; clips are cached on disk by (text, voice, speed, lang)."""
    SR = 24000

    def __init__(self, vcfg, model_dir):
        self.name = vcfg.get('name', 'af_heart')
        self.speed = float(vcfg.get('speed', 1.0))
        self.lang = vcfg.get('lang', 'en-us')
        self.dir = os.path.expanduser(model_dir)
        self.cache = os.path.join(self.dir, 'clips')
        os.makedirs(self.cache, exist_ok=True)
        self.k = None

    def model(self):
        if self.k is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError:
                sys.exit('The voice needs kokoro-onnx: pip install kokoro-onnx  (or pass --no-voice)')
            paths = []
            for f in ('kokoro-v1.0.onnx', 'voices-v1.0.bin'):
                p = os.path.join(self.dir, f); paths.append(p)
                if not os.path.exists(p):
                    print(f'downloading {KOKORO_URL + f} ...', file=sys.stderr)
                    urllib.request.urlretrieve(KOKORO_URL + f, p + '.part'); os.replace(p + '.part', p)
            self.k = Kokoro(*paths)
            if self.name not in self.k.get_voices():
                sys.exit(f"Unknown voice {self.name!r}; choose from: {', '.join(self.k.get_voices())}")
        return self.k

    def say(self, text):
        key = hashlib.sha1(json.dumps([text, self.name, self.speed, self.lang]).encode()).hexdigest()[:16]
        p = os.path.join(self.cache, key + '.npy')
        if os.path.exists(p):
            return np.load(p)
        print(f'  speaking: {text[:60]}{"..." if len(text) > 60 else ""}', file=sys.stderr)
        audio, sr = self.model().create(text, voice=self.name, speed=self.speed, lang=self.lang)
        assert sr == self.SR
        audio = audio.astype(np.float32); np.save(p, audio)
        return audio

def mouth_track(buf, sr, fps, nf):
    # RMS loudness per frame (taken a touch ahead of the sound, since lips move
    # first), thresholded into closed/mid/open; dips between syllables close it
    hop = sr / fps; lead = int(0.02 * sr)
    rms = np.zeros(nf, np.float32)
    for i in range(nf):
        seg = buf[int(i * hop) + lead:int((i + 1) * hop) + lead]
        if len(seg): rms[i] = np.sqrt((seg ** 2).mean())
    voiced = rms[rms > 1e-3]
    lvl = rms / max(np.percentile(voiced, 90) if len(voiced) else 1.0, 1e-6)
    states = np.where(lvl < 0.15, 0, np.where(lvl < 0.6, 1, 2)).astype(np.int8)
    # a frame clearly quieter than both neighbours is a syllable break: close a notch
    dip = (lvl[1:-1] < 0.75 * lvl[:-2]) & (lvl[1:-1] < 0.75 * lvl[2:])
    states[1:-1][dip] = np.maximum(0, states[1:-1][dip] - 1)
    return states

def write_speech(path):
    mix = G['SPEECH'] + (make_sfx(G['SR']) if G['CRYCFG'] else 0)
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(G['SR']); w.writeframes(pcm.tobytes())

def make_audio(path, pitch=1.0):
    sr = 44100
    n = int(G['TOTAL'] * sr); buf = np.zeros(n, np.float32)
    r = random.Random(11)
    L = int(0.045 * sr); t = np.arange(L) / sr
    env = np.exp(-t * 70) * np.minimum(1, t * 2000)
    for b in G['BUBBLES']:
        k = b['s']
        while k < b['r']:
            f = r.choice([520, 587, 659, 698, 784]) * pitch
            blip = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * env * 0.16
            i0 = int(k * sr); m = max(0, min(L, n - i0))
            buf[i0:i0 + m] += blip[:m]
            k += 1 / 13
    if G['CRYCFG']: buf += make_sfx(sr)[:n]
    pcm = (np.clip(buf, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())

# ======================================================================
def main():
    ap = argparse.ArgumentParser(description='Render a talking-character video from a text script.')
    ap.add_argument('script', help='script .txt file (see docstring for format)')
    ap.add_argument('-o', '--output', default=None, help='output .mp4 (default: script name + .mp4)')
    ap.add_argument('-c', '--character', default=os.path.join(HERE, 'character.json'), help='character JSON config')
    ap.add_argument('--size', type=int, default=1080, help='video height: 1080, 720, 480 ... (16:9)')
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--cps', type=float, default=32, help='typing speed, characters/sec (bubbles without speech)')
    ap.add_argument('--read-cps', type=float, default=38, help='extra hold time after typing = hold-base + chars/read-cps')
    ap.add_argument('--hold-base', type=float, default=0.9)
    ap.add_argument('--voice', default=None, help="Kokoro voice, e.g. af_heart, am_puck (default: character.json's voice.name)")
    ap.add_argument('--speed', type=float, default=None, help="speech speed multiplier (default: character.json's voice.speed)")
    ap.add_argument('--no-voice', action='store_true', help='no speech: voice blips and random mouth flaps instead')
    ap.add_argument('--kokoro-dir', default='~/.cache/talking-character', help='where Kokoro model files and cached speech go')
    ap.add_argument('--pitch', type=float, default=1.0, help='voice blip pitch multiplier (with --no-voice)')
    ap.add_argument('--no-audio', action='store_true', help='silent video (speech still sets timing and mouth)')
    ap.add_argument('--no-blink', action='store_true', help='keep the eyes open (ignore the eyes config)')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--crf', type=int, default=20)
    ap.add_argument('--preview', type=float, nargs='+', metavar='SEC', help='write PNG stills at these times (seconds) instead of a video')
    args = ap.parse_args()

    cv2.setNumThreads(1)
    setup(args)
    out = args.output or os.path.splitext(args.script)[0] + '.mp4'
    print(f"{len(G['BUBBLES'])} bubbles, {G['TOTAL']:.1f}s, {G['NF']} frames at {G['W']}x{G['H']}", file=sys.stderr)

    if args.preview:
        base = os.path.splitext(out)[0]
        for sec in args.preview:
            p = f'{base}_preview_{sec:g}s.png'
            render_frame(int(sec * args.fps)).save(p); print('wrote', p, file=sys.stderr)
        return

    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f"{G['W']}x{G['H']}", '-r', str(args.fps), '-i', '-']
    wav = None
    if G['AUDIO']:
        if G['SPEECH'] is not None:
            wav = out + '.speech.wav'; write_speech(wav)
        else:
            wav = out + '.blips.wav'; make_audio(wav, args.pitch)
        cmd += ['-i', wav, '-c:a', 'aac', '-b:a', '128k', '-shortest']
    cmd += ['-c:v', 'libx264', '-preset', 'medium', '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time(); NF = G['NF']
    ctx = mp.get_context('fork')
    with ctx.Pool(max(1, args.workers)) as pool:
        for i, buf in enumerate(pool.imap(frame_bytes, range(NF), chunksize=4)):
            ff.stdin.write(buf)
            if i % max(1, NF // 20) == 0 and i:
                el = time.time() - t0
                print(f'  {100 * i // NF:3d}%  ({el:.0f}s elapsed, ~{el / i * (NF - i):.0f}s left)', file=sys.stderr)
    ff.stdin.close(); ff.wait()
    if wav: os.remove(wav)
    print(f'done: {out}  ({time.time() - t0:.0f}s)', file=sys.stderr)

if __name__ == '__main__':
    main()
