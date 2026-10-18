# Notes for development

Single-file tool: `talking_character.py` (script parsing, Kokoro TTS, bubble layout, frame rendering with multiprocessing, audio, piping raw frames to ffmpeg). Character sprites are pre-rendered in `setup()` as `G['frames'][eye_state][mouth_state]` (3×3: open/half/closed eyes × closed/mid/open mouth); `eye_state()` follows the seeded blink schedule in `G['BLINKS']`. With voice (the default), `setup()` synthesizes each bubble via the `Speaker` class (kokoro-onnx), lays bubbles out to fit the speech, mixes `G['SPEECH']`, and `mouth_track()` turns its per-frame loudness into `G['MOUTH']`, which `mouth_state()` reads; with `--no-voice` it uses the seeded random sequence and blips. Character config lives in `character.json`; `make_character.py` generates one from an image (k-means color quantization, dark-blob eye pairs confirmed by a mouth-shaped hole in the skin below them, eye regions grown through small non-skin color patches) and prompts for the voice, playing Kokoro samples via `talking_character.Speaker`; its preview panels come from `talking_character.setup()`. Example input/output are `gil_script.txt` / `gil_explained.mp4`.

## Local environment

- Python via pyenv virtualenv `talking-character` (Python 3.13.15). `.python-version` in the repo root selects it automatically (it is gitignored).
- Install deps: `pip install -r requirements.txt`
- Kokoro model files and cached speech clips (`clips/*.npy`, keyed by text+voice+speed) live in `~/.cache/talking-character`. Delete `clips/` to force re-synthesis. First synthesis of the example script takes ~45s on CPU; cached runs are fast.
- ffmpeg is installed system-wide (`/usr/bin/ffmpeg`). `--preview` stills don't need it.
- Fonts are searched in `/usr/share/fonts`, `/usr/local/share/fonts`, `~/.fonts`, `~/.local/share/fonts`, and the repo dir; falls back to Pillow's default.

## Quick test

```bash
python talking_character.py gil_script.txt --size 480 --preview 0 5 20
python talking_character.py gil_script.txt --size 480 -o /tmp/test.mp4
python make_character.py Anima_20261016_00118_.png -o /tmp/imp.json --voice af_bella --speed 1   # compare with imp.json
```

Use `--size 480` or `--preview` while iterating; 1080p full renders are slow. With the example script and `character.json`, the first blink is at ~1.86s, so `--preview 1.9` shows closed eyes (an easy way to check the eye shapes).

## Conventions

- Keep dependencies minimal (numpy, opencv-python-headless, Pillow, kokoro-onnx — the ONNX build, not the torch `kokoro` package, which doesn't support Python 3.13); `requirements.txt` is unpinned.
- Large binaries (`gil_explained.mp4`, `character.png`) are committed; avoid committing more renders.
- `.gitignore` excludes `preview*.png`; name scratch stills accordingly.
