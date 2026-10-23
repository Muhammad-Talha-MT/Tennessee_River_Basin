"""Provider dispatch regression: NWM used to be defined but never called."""

import sys

import pytest


def test_forcing_calls_nwm(monkeypatch, tmp_path):
    pytest.importorskip("s3fs")
    pytest.importorskip("zarr")
    from tva_flood.downloads import forcing

    called = []
    monkeypatch.setattr(forcing.s3fs, "S3FileSystem", lambda **kwargs: object())
    monkeypatch.setattr(forcing, "download_nwm", lambda args, fs, out: called.append(out))
    monkeypatch.setattr(forcing, "local_year_complete", lambda *args: True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tva-flood",
            "--root",
            str(tmp_path),
            "--source",
            "all",
            "--start",
            "2018-01-01",
            "--stop",
            "2018-02-01",
        ],
    )
    forcing.main()
    assert called == [tmp_path / "raw"]
