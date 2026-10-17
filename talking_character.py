#!/usr/bin/env python3
"""
talking_character.py - turn a plain-text script into a talking-character video.

The character (from a single still image) flaps its mouth while each speech
bubble types out, drifts/bobs gently, and soft "voice blips" play while talking.

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
    > $ lines starting with "> " are code (monospace, dark box)
    [pause 1.5]                          on its own: hold with no bubble for N seconds
    // comment lines are ignored

Character settings (image, mouth shape, placement) live in a JSON file;
see character.json next to this script.

Requires: Python 3.9+, Pillow, NumPy, OpenCV (opencv-python-headless), ffmpeg.
"""
import argparse, json, math, multiprocessing as mp, os, random, re, subprocess, sys, time, wave
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

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
    srcw, srch = src.size
    a = np.asarray(src)
    bg = tuple(cfg.get('background') or np.median(
        np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3), a[-8:, :8].reshape(-1, 3)]), 0).astype(int).tolist())
    G['BG'] = bg
    mouth = cfg.get('mouth')
    if mouth:
        poly = [tuple(p) for p in mouth['cover_polygon']]
        lip = [tuple(p) for p in mouth['closed_line']]
        skin = tuple(mouth['skin_color']); line_col = tuple(mouth.get('line_color', [118, 20, 22]))
        def covered():
            b = src.copy(); ImageDraw.Draw(b).polygon(poly, fill=skin); return b
        closed = covered(); ImageDraw.Draw(closed).line(lip, fill=line_col, width=mouth.get('line_width', 4), joint='curve')
        mid = covered()
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        top = mouth.get('hinge_y', min(ys) - 6)
        box = (min(xs), top, max(xs) + 1, max(ys) + 1)
        reg = src.crop(box)
        msk = Image.new('L', src.size, 0); ImageDraw.Draw(msk).polygon(poly, fill=255)
        nh = max(1, int(reg.height * mouth.get('mid_ratio', 0.55)))
        mid.paste(reg.resize((reg.width, nh), Image.LANCZOS), box[:2],
                  msk.crop(box).resize((reg.width, nh), Image.LANCZOS))
        variants = [closed, mid, src]
    else:
        variants = [src, src, src]  # no mouth config: just bob

    def keyed(img):
        f = np.asarray(img).astype(np.float32)
        d = np.sqrt(((f - np.array(bg, np.float32)) ** 2).sum(-1))
        alpha = np.clip((d - 14) / 26, 0, 1) * 255
        return np.dstack([f, alpha]).astype(np.uint8)

    ch_h = int(H * cfg.get('height_frac', 1.06))
    sc = ch_h / srch
    ch_w = int(srcw * sc)
    G['mouths'] = [cv2.resize(keyed(v), (ch_w, ch_h), interpolation=cv2.INTER_AREA) for v in variants]
    G['CH'] = (ch_w, ch_h)
    G['CX0'] = W - ch_w + int(cfg.get('right_overhang', 40) * S)
    G['CY0'] = int(cfg.get('top_offset', -30) * S)
    px, py = cfg.get('pivot', [srcw * 0.55, srch * 0.88])
    G['PIVOT'] = (px * sc, py * sc)
    tx, ty = cfg.get('tail_target', [srcw * 0.3, srch * 0.68])
    G['TAILT'] = (tx * sc, ty * sc)
    mv = cfg.get('motion', {})
    G['MOTION'] = (mv.get('sway', 26) * S, mv.get('bob', 9) * S, mv.get('tilt_deg', 1.6))

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
    bubbles, t = [], title_s
    for it in items:
        if it['kind'] == 'pause':
            t += it['secs']; continue
        lay = layout_bubble(it['lines'])
        n = lay['nchars']
        rev = n / args.cps
        hold = args.hold_base + n / args.read_cps
        bubbles.append(dict(tag=it['tag'], lay=lay, s=t, r=t + rev, e=t + rev + hold))
        t += rev + hold + 0.25
    G['BUBBLES'] = bubbles
    G['TOTAL'] = t + 1.2
    G['NF'] = int(G['TOTAL'] * args.fps)

    # deterministic mouth sequence (changes every 3 frames, never repeats)
    r = random.Random(3); seq, prev = [], 0
    for _ in range(G['NF'] // 3 + 2):
        prev = r.choice([c for c in (0, 1, 2, 1, 2) if c != prev]); seq.append(prev)
    G['MSEQ'] = seq

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

    shown = int((tt - s) * G['CPS'])
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
    if any(b['s'] <= tt < b['r'] + 0.08 for b in G['BUBBLES']):
        return G['MSEQ'][fi // 3]
    return 0

def render_frame(fi):
    tt = fi / G['FPS']
    img = background(tt)
    sway, bob, tilt = G['MOTION']
    dx = sway * math.sin(tt * 0.45) + sway * 0.3 * math.sin(tt * 1.1)
    dy = bob * math.sin(tt * 1.6) + bob * 0.55 * math.sin(tt * 0.7)
    ang = tilt * math.sin(tt * 0.6 + 0.5)
    ch = G['mouths'][mouth_state(fi, tt)]
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
    ap.add_argument('--cps', type=float, default=32, help='typing speed, characters/sec')
    ap.add_argument('--read-cps', type=float, default=38, help='extra hold time after typing = hold-base + chars/read-cps')
    ap.add_argument('--hold-base', type=float, default=0.9)
    ap.add_argument('--pitch', type=float, default=1.0, help='voice blip pitch multiplier')
    ap.add_argument('--no-audio', action='store_true')
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
