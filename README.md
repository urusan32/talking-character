# Talking Character

Turn a plain-text script into a video of a character (made from a single still image) that flaps its mouth while speech bubbles type out, blinks every few seconds, sways and bobs gently, and plays soft "voice blips".

The included example, `gil_script.txt`, renders to `gil_explained.mp4`.

## Setup

Requirements: Python 3.9+ and [ffmpeg](https://ffmpeg.org/) on your `PATH`.

```bash
# install ffmpeg (Debian/Ubuntu example)
sudo apt install ffmpeg

# create and activate an environment, then install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python3 talking_character.py gil_script.txt -o out.mp4

# quick check: write PNG stills at 0s, 150s, and 900s instead of a video
python3 talking_character.py gil_script.txt --size 720 --preview 0 150 900
```

Useful options (`--help` lists all):

| Option | Meaning | Default |
|---|---|---|
| `-o, --output` | Output `.mp4` | script name + `.mp4` |
| `-c, --character` | Character JSON config | `character.json` |
| `--size` | Video height (16:9): 1080, 720, 480... | 1080 |
| `--fps` | Frames per second | 30 |
| `--cps` | Typing speed, characters/sec | 32 |
| `--pitch` | Voice blip pitch multiplier | 1.0 |
| `--no-audio` | Skip the voice blips | off |
| `--no-blink` | Keep the eyes open | off |
| `--workers` | Render processes | CPU count |
| `--crf` | Video quality (lower is better) | 20 |
| `--preview SEC...` | Write PNG stills at these times | |

## Script format

Plain UTF-8 text:

```text
# title: The GIL, explained!        optional title card (first lines only)
# subtitle: + free-threaded Python  optional, shown under the title
== PART 1 · WHAT IS THE GIL?        section tag shown top-left (until next ==)
Hi! This line is one bubble.

A blank line starts a new bubble.
A new line inside a bubble starts a new paragraph.
> $ lines starting with "> " are code (monospace, dark box)
[pause 1.5]                         hold with no bubble for N seconds
// comment lines are ignored
```

## Customizing the character

`character.json` points at the image (`character.png`) and defines its placement, motion, colors, the mouth (cover polygon, closed-mouth line, hinge), and the eyes. Copy it and the PNG to make a new character, then pass it with `-c`.

The optional `eyes` block makes the character blink. All coordinates are in source-image pixels:

| Key | Meaning | Default |
|---|---|---|
| `shapes` | One entry per eye: `cover_polygon` (area painted with skin when the eye closes, covering lashes and eyeball) and `closed_line` (control points of the closed-lid curve, smoothed into a spline) | required |
| `skin_color` | Lid color | the mouth's `skin_color` |
| `line_color`, `line_width` | Closed-lid line | near-black, 7 |
| `half_ratio` | How far the eye is squashed in the half-closed frames | 0.5 |
| `interval` | `[min, max]` seconds between blinks | `[2.5, 5.5]` |
| `duration` | Seconds per blink (half, closed, half) | 0.16 |
| `double_chance` | Probability that a blink is a double blink | 0.15 |

Leave out `eyes` (or pass `--no-blink`) for a character that never blinks. To check the shapes, render a still at a blink time; the first blink comes about 1-3 seconds in.
