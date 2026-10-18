# Notes for development

Single-file tool: `talking_character.py` (script parsing, bubble layout, frame rendering with multiprocessing, audio blips, piping raw frames to ffmpeg). Character sprites are pre-rendered in `setup()` as `G['frames'][eye_state][mouth_state]` (3×3: open/half/closed eyes × closed/mid/open mouth); `eye_state()` follows the seeded blink schedule in `G['BLINKS']`, `mouth_state()` the seeded mouth sequence. Character config lives in `character.json`; example input/output are `gil_script.txt` / `gil_explained.mp4`.

## Local environment

- Python via pyenv virtualenv `talking-character` (Python 3.13.15). `.python-version` in the repo root selects it automatically (it is gitignored).
- Install deps: `pip install -r requirements.txt`
- ffmpeg is installed system-wide (`/usr/bin/ffmpeg`). `--preview` stills don't need it.
- Fonts are searched in `/usr/share/fonts`, `/usr/local/share/fonts`, `~/.fonts`, `~/.local/share/fonts`, and the repo dir; falls back to Pillow's default.

## Quick test

```bash
python talking_character.py gil_script.txt --size 480 --preview 0 5 20
python talking_character.py gil_script.txt --size 480 -o /tmp/test.mp4
```

Use `--size 480` or `--preview` while iterating; 1080p full renders are slow. With the example script and `character.json`, the first blink is at ~1.86s, so `--preview 1.9` shows closed eyes (an easy way to check the eye shapes).

## Conventions

- Keep dependencies minimal (numpy, opencv-python-headless, Pillow); `requirements.txt` is unpinned.
- Large binaries (`gil_explained.mp4`, `character.png`) are committed; avoid committing more renders.
- `.gitignore` excludes `preview*.png`; name scratch stills accordingly.
