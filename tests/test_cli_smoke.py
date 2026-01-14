from __future__ import annotations

import numpy as np
from click.testing import CliRunner

from whisper_cli import main


class DummySeg:
    def __init__(self, text: str) -> None:
        self.text = text


class DummyModel:
    def transcribe(self, audio, language=None):
        return [DummySeg("hello"), DummySeg("world")]


def test_transcribe_stdout(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")

    monkeypatch.setattr(main, "load_audio", lambda path: (np.zeros(16000), 16000))
    monkeypatch.setattr(main, "load_model", lambda model: DummyModel())

    runner = CliRunner()
    result = runner.invoke(main.cli, ["-s", "transcribe", str(audio_file)])

    assert result.exit_code == 0
    assert result.output.strip() == "hello world"


def test_transcribe_outfile(monkeypatch, tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"")
    output_file = tmp_path / "output.txt"

    monkeypatch.setattr(main, "load_audio", lambda path: (np.zeros(16000), 16000))
    monkeypatch.setattr(main, "load_model", lambda model: DummyModel())

    runner = CliRunner()
    result = runner.invoke(
        main.cli, ["-s", "transcribe", str(audio_file), "--outfile", str(output_file)]
    )

    assert result.exit_code == 0
    assert output_file.read_text().strip() == "hello world"
