"""
Tests for `statarb ingest --full-refresh` and the pure parts of extractor.py.

The extractor connects to an office-only SQL Server, but its pure machinery -
Manifest, price_shards, should_extract, _split_batches - has analytically known
answers and needs no connection. extractor.py imports arrow_odbc lazily (inside
the functions that connect), which is exactly what makes this file runnable in
CI: importing it must succeed without the `ingest` extra installed.
"""

from __future__ import annotations

import pytest

from statarb.ingest.factset import extract_raw_factset, extractor

# =============================================================================
# Manifest
# =============================================================================


def _completed_artifact(root, manifest, name="shard_a", content=b"parquet bytes"):
    """Write a real file and record it complete in the manifest."""
    f = root / "raw" / "prices" / f"{name}.parquet"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(content)
    manifest.append(
        {
            "name": name,
            "file": str(f.relative_to(root)),
            "bytes": f.stat().st_size,
            "sha256": extractor.sha256_file(f),
            "status": "complete",
        }
    )
    return f


def test_manifest_round_trip_and_checksum_gate(tmp_path):
    """Completion is recorded AND verified: the on-disk checksum must match."""
    manifest = extractor.Manifest(tmp_path / "_manifest" / "manifest.jsonl")
    assert not manifest.is_complete("shard_a", tmp_path)

    f = _completed_artifact(tmp_path, manifest)
    assert manifest.is_complete("shard_a", tmp_path)

    # A fresh Manifest over the same file sees the same state.
    reloaded = extractor.Manifest(tmp_path / "_manifest" / "manifest.jsonl")
    assert reloaded.is_complete("shard_a", tmp_path)

    # A truncated/corrupted file "exists" but must NOT count as complete:
    # that is the whole point of gating on the hash rather than existence.
    f.write_bytes(b"different bytes")
    assert not reloaded.is_complete("shard_a", tmp_path)

    f.unlink()
    assert not reloaded.is_complete("shard_a", tmp_path)


def test_manifest_skips_truncated_final_line(tmp_path):
    """A crash mid-append corrupts at most the last line; reading must survive."""
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        '{"name": "a", "status": "complete"}\n{"name": "b", "stat',  # torn write
        encoding="utf-8",
    )
    manifest = extractor.Manifest(path)  # must not raise
    assert not manifest.is_complete("b", tmp_path)


def test_manifest_superseding_records(tmp_path):
    """Full refresh appends fresh records; the newest complete record wins."""
    manifest = extractor.Manifest(tmp_path / "manifest.jsonl")
    f = _completed_artifact(tmp_path, manifest, content=b"old bytes")
    assert manifest.is_complete("shard_a", tmp_path)

    # Simulated refresh: the file is rewritten and a new record appended.
    _completed_artifact(tmp_path, manifest, content=b"new bytes")
    assert manifest.is_complete("shard_a", tmp_path)

    reloaded = extractor.Manifest(tmp_path / "manifest.jsonl")
    assert reloaded.is_complete("shard_a", tmp_path)
    # And if the refreshed file is corrupted, the gate still trips.
    f.write_bytes(b"tampered")
    assert not reloaded.is_complete("shard_a", tmp_path)


# =============================================================================
# should_extract: the full-refresh gate
# =============================================================================


def test_full_refresh_ignores_manifest_state(tmp_path):
    manifest = extractor.Manifest(tmp_path / "_manifest" / "manifest.jsonl")
    _completed_artifact(tmp_path, manifest)

    assert not extractor.should_extract("shard_a", manifest, tmp_path)
    assert not extractor.should_extract("shard_a", manifest, tmp_path, full_refresh=False)
    assert extractor.should_extract("shard_a", manifest, tmp_path, full_refresh=True)
    # Never-completed work is outstanding in both modes.
    assert extractor.should_extract("never_done", manifest, tmp_path)
    assert extractor.should_extract("never_done", manifest, tmp_path, full_refresh=True)


# =============================================================================
# price_shards
# =============================================================================


def test_price_shards_cover_the_sid_range_without_gaps():
    universe = [
        (sid, f"F{sid:06d}-R", None, None) for sid in (900, 5, 7, 401, 9)
    ]  # unsorted on purpose
    shards = extractor.price_shards(universe, per_shard=2)

    assert [name for name, _ in shards] == [
        "prices_sid_000005_000009",
        "prices_sid_000009_000900",
        "prices_sid_000900_000901",
    ]
    # Contiguous, half-open ranges: no sid is skipped or double-extracted.
    assert "u.sid >= 5 AND u.sid < 9" in shards[0][1]
    assert "u.sid >= 9 AND u.sid < 900" in shards[1][1]
    assert "u.sid >= 900 AND u.sid < 901" in shards[2][1]
    # Every shard carries the configured date window.
    for _, where in shards:
        assert "p.p_date >=" in where and "p.p_date <=" in where


def test_price_shards_single_shard_covers_everything():
    universe = [(sid, f"F{sid:06d}-R", None, None) for sid in (1, 2, 3)]
    shards = extractor.price_shards(universe, per_shard=400)
    assert len(shards) == 1
    name, where = shards[0]
    assert name == "prices_sid_000001_000004"
    assert "u.sid >= 1 AND u.sid < 4" in where


# =============================================================================
# _split_batches
# =============================================================================


def test_split_batches_strips_go_comments_and_blanks():
    sql = (
        "SET NOCOUNT ON;\n"
        "-- a line comment\n"
        "GO\n"
        "\n"
        "SET XACT_ABORT ON; /* a block\n"
        "comment spanning lines */ SET ANSI_WARNINGS ON;\n"
        "go\n"
    )
    assert extractor._split_batches(sql) == [
        "SET NOCOUNT ON",
        "SET XACT_ABORT ON",
        "SET ANSI_WARNINGS ON",
    ]


# =============================================================================
# CLI plumbing: the flag reaches extractor.run
# =============================================================================


def test_statarb_cli_parses_full_refresh():
    from statarb.cli.main import build_parser

    args = build_parser().parse_args(["ingest", "--full-refresh"])
    assert args.full_refresh

    # Combinable with --smoke: prove the refresh path on a few shards first.
    args = build_parser().parse_args(["ingest", "--smoke", "2", "--full-refresh"])
    assert args.smoke == 2 and args.full_refresh

    args = build_parser().parse_args(["ingest", "--execute"])
    assert not args.full_refresh


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["--full-refresh"], {"limit_shards": None, "full_refresh": True}),
        (["--execute"], {"limit_shards": None, "full_refresh": False}),
        (["--smoke", "2"], {"limit_shards": 2, "full_refresh": False}),
        (["--smoke", "2", "--full-refresh"], {"limit_shards": 2, "full_refresh": True}),
    ],
)
def test_script_routes_full_refresh_to_extractor(argv, expected, monkeypatch):
    """--full-refresh implies a real run and lands on extractor.run()."""
    calls = []
    monkeypatch.setattr(extractor, "run", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(
        extract_raw_factset, "run_extraction", lambda **kwargs: calls.append(kwargs)
    )

    extract_raw_factset.main(argv)

    assert calls == [expected]


def test_extractor_run_forwards_full_refresh(monkeypatch, tmp_path):
    """run() passes the flag to BOTH extraction stages."""
    seen = {}

    def fake_aux(password, universe, universe_hash, manifest, full_refresh=False):
        seen["aux"] = full_refresh

    def fake_prices(password, universe, universe_hash, manifest, full_refresh=False):
        seen["prices"] = full_refresh

    monkeypatch.setattr(extractor.config, "DESTINATION_ROOT", tmp_path)
    monkeypatch.setattr(extractor.config, "MANIFEST_DIR", tmp_path / "_manifest")
    monkeypatch.setattr(extractor.config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(extractor, "resolve_universe", lambda password: ([], "hash"))
    monkeypatch.setattr(extractor, "extract_aux", fake_aux)
    monkeypatch.setattr(extractor, "extract_prices", fake_prices)

    extractor.run(password="unused", full_refresh=True)  # noqa: S106

    assert seen == {"aux": True, "prices": True}
