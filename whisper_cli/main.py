import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

import click
import numpy as np
from rich import print as rprint

try:
    from pywhispercpp.model import Model
except ImportError:
    from pywhispercpp import Model


silent_mode = False
MODEL_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
DEFAULT_MODEL = "base"


def log(message: str, level: str = "info"):
    if not silent_mode:
        if level == "error":
            rprint(f"[bold red]{message}[/bold red]")
        elif level == "warning":
            rprint(f"[yellow]{message}[/yellow]")
        elif level == "success":
            rprint(f"[green]{message}[/green]")
        else:
            rprint(f"[bold]{message}[/bold]")


def get_file_type(file_name: str) -> str:
    return file_name.split(".")[-1]


def check_ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_to_wav(input_file: str) -> str:
    if not check_ffmpeg_available():
        raise RuntimeError(
            "ffmpeg is required for audio conversion. Please install ffmpeg: sudo apt install ffmpeg"
        )

    with tempfile.NamedTemporaryFile(
        suffix=".wav", prefix="whisper_cli_", delete=False
    ) as temp_file:
        output_file = temp_file.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-y",
                output_file,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to convert audio: {e}")


def download_model(model_name: str, models_dir: Path) -> bool:
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"ggml-{model_name}.bin"
    url = f"{MODEL_BASE_URL}/ggml-{model_name}.bin"

    log(f"Downloading model: {model_name}...")
    try:
        urllib.request.urlretrieve(url, model_file)
        return True
    except Exception as e:
        if model_file.exists():
            try:
                model_file.unlink()
            except OSError:
                pass
        log(f"Failed to download model {model_name}: {e}", "warning")
        return False


def load_audio(file_path: str, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    file_type = get_file_type(file_path).lower()

    if file_type not in ["wav", "webm", "mp3", "m4a", "mp4", "ogg", "flac", "aac"]:
        raise ValueError(f"Unsupported file type: {file_type}")

    if file_type != "wav":
        log(f"Converting {file_type} to WAV...")
        file_path = convert_to_wav(file_path)
    else:
        file_path = str(Path(file_path).absolute())

    from scipy.io import wavfile

    try:
        sr, audio = wavfile.read(file_path)
        audio = audio.astype(np.float32) / 32768.0

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        if sr != sample_rate:
            from scipy import signal

            number_of_samples = round(len(audio) * float(sample_rate) / sr)
            audio = signal.resample(audio, number_of_samples)

        return audio, sample_rate
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file: {e}")


def load_model(model_name: str = DEFAULT_MODEL, n_threads: int = 4) -> Model:
    models_dir = Path.home() / ".local" / "share" / "pywhispercpp" / "models"
    candidate_names = [model_name]
    if not model_name.endswith(".en"):
        candidate_names.append(f"{model_name}.en")

    resolved_name = None
    for name in candidate_names:
        if (models_dir / f"ggml-{name}.bin").exists():
            resolved_name = name
            break

    if resolved_name is None:
        for name in candidate_names:
            if download_model(name, models_dir):
                resolved_name = name
                break

    if resolved_name is None:
        log(f"Error: Model not found: {model_name}", "error")
        click.echo(
            "Download models from https://huggingface.co/ggerganov/whisper.cpp/tree/main",
            err=True,
        )
        click.echo(f"Place model files in: {models_dir}", err=True)
        raise click.Abort()

    log(f"Loading model: {resolved_name}...")
    log_destination = None if silent_mode else False

    try:
        model = Model(
            model=resolved_name,
            n_threads=n_threads,
            redirect_whispercpp_logs_to=log_destination,
        )
        return model
    except Exception as e:
        log(f"Error loading model: {e}", "error")
        raise click.Abort()


@click.group()
@click.option(
    "-s",
    "--silent",
    is_flag=True,
    help="Silent mode - suppress progress output (use with 2>/dev/null for clean transcript)",
)
def cli(silent):
    global silent_mode
    silent_mode = silent


@cli.command()
@click.argument("file_name", type=click.Path(exists=True))
@click.option(
    "--model",
    default=DEFAULT_MODEL,
    help="Model size (tiny, base, small, medium, large, large-v1, large-v2, large-v3)",
    show_default=True,
)
@click.option(
    "--language", default=None, help="Language code (e.g., en, es, fr, de, ja, zh)"
)
@click.option(
    "--outfile", default=None, type=click.Path(), help="Save transcript to file"
)
def transcribe(
    file_name: str, model: str, language: Optional[str], outfile: Optional[str]
):
    log(f"Loading audio file: {file_name}")
    audio, sample_rate = load_audio(file_name)

    log(f"Audio loaded: {len(audio)} samples at {sample_rate}Hz")

    try:
        whisper_model = load_model(model)
    except Exception as e:
        log(f"Error loading model: {e}", "error")
        raise

    log("Transcribing...")
    try:
        if language:
            segments = whisper_model.transcribe(audio, language=language)
        else:
            segments = whisper_model.transcribe(audio)

        result = " ".join(seg.text for seg in segments).strip()

        if outfile:
            with open(outfile, "w") as f:
                f.write(result)
            log(f"Transcript saved to: {outfile}")
        else:
            if not silent_mode:
                log("\nTranscription:")
            click.echo(result)
    except Exception as e:
        log(f"Error during transcription: {e}", "error")
        raise


if __name__ == "__main__":
    cli()
