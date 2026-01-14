# pywhisper-cli
pywhisper-cli is a command-line interface for transcribing audio using local Whisper models via `pywhispercpp`.

This project is not published to PyPI yet. Install from source:

```sh
git clone https://github.com/aljazceru/pywhisper-cli
cd pywhisper-cli
```

## Installation (uv)

```sh
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

## Installation (pip)

```sh
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Setup
Models download automatically to `~/.local/share/pywhispercpp/models/` on first use.
For non-WAV formats, install ffmpeg (Ubuntu/Debian: `sudo apt install ffmpeg`).

## Usage

To transcribe an audio file, run:

```sh
whisper transcribe <file_name>
```

With a specific model:
```sh
whisper transcribe <file_name> --model small
```

With a language hint:
```sh
whisper transcribe <file_name> --language en
```

Save output to a file:
```sh
whisper transcribe <file_name> --outfile output.txt
```

Silent mode (for piping):
```sh
whisper -s transcribe audio.webm 2>/dev/null | grep -v "^Progress:"
```

## Models

Available models: tiny, base, small, medium, large, large-v1, large-v2, large-v3

## Supported Audio Formats

WAV, WEBM, MP3, M4A, MP4, OGG, FLAC, AAC

## Development
If you'd like to contribute, you'll need Python 3.10+.

## Related
This was heavily inspired by [whisper-cli](https://github.com/vatsalaggarwal/whisper-cli).
