"""
Path resolution and connection-config safety.

Two things are being locked in here:

* Data must never default to a location inside the repository. That default is
  how 1.4 GB of licensed vendor data came to sit in a git working tree whose
  ignore rules were one un-anchored pattern away from committing it.

* The password must never appear in anything intended for a log.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from statarb.paths import DEFAULT_ROOT, ENV_VAR, Locations, locations

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestLocations:
    def test_explicit_root_wins_over_environment(self, monkeypatch, tmp_path):
        """An explicit root is what lets a test avoid touching the environment."""
        monkeypatch.setenv(ENV_VAR, r"C:\somewhere\else")
        assert locations(tmp_path).root == tmp_path

    def test_environment_used_when_no_argument(self, monkeypatch, tmp_path):
        monkeypatch.setenv(ENV_VAR, str(tmp_path))
        assert locations().root == tmp_path

    def test_default_is_outside_the_repository(self, monkeypatch):
        """THE regression test for licensed data landing inside a git repo."""
        monkeypatch.delenv(ENV_VAR, raising=False)
        root = locations().root
        assert root == DEFAULT_ROOT.expanduser()
        assert REPO_ROOT not in root.parents and root != REPO_ROOT, (
            f"default data root {root} is inside the repository at {REPO_ROOT}; "
            "licensed vendor data must never default to a git working tree"
        )

    def test_user_home_is_expanded(self, monkeypatch):
        monkeypatch.setenv(ENV_VAR, "~/some-warehouse")
        assert "~" not in str(locations().root)

    def test_every_table_path_sits_under_the_root(self, tmp_path):
        loc = locations(tmp_path)
        for p in (loc.raw, loc.raw_prices, loc.curated, loc.prices,
                  loc.security, loc.corp_actions, loc.calendar,
                  loc.manifest, loc.discovery_output):
            assert tmp_path in p.parents or p == tmp_path

    def test_locations_is_immutable(self, tmp_path):
        """Frozen so a caller cannot repoint one table and not the others."""
        loc = locations(tmp_path)
        with pytest.raises(Exception):
            loc.root = Path("/elsewhere")  # type: ignore[misc]

    def test_raw_table_naming(self, tmp_path):
        assert locations(tmp_path).raw_table("splits").name == "splits.parquet"

    def test_accepts_a_locations_instance_unchanged(self, tmp_path):
        from statarb.data.api import DataAPI

        loc = locations(tmp_path)
        assert DataAPI(root=loc).loc is loc


class TestConnectionConfig:
    """`ingest.factset.config` — the module that once held a live password."""

    def test_no_hardcoded_host_or_user_defaults(self, monkeypatch):
        """A code default for the host is worse than a leak.

        It puts a production hostname into every clone AND means any machine
        running without a .env silently connects to it.
        """
        import importlib

        for var in ("FACTSET_SQL_SERVER", "FACTSET_SQL_DATABASE", "FACTSET_SQL_USER"):
            monkeypatch.delenv(var, raising=False)

        cfg = importlib.reload(
            importlib.import_module("statarb.ingest.factset.config"))

        assert cfg.SERVER_NAME == ""
        assert cfg.DATABASE == ""
        assert cfg.USER_NAME == ""

    def test_missing_settings_fail_loudly_at_connection_time(self, monkeypatch):
        """Not at import — validation at import is the bug this avoids."""
        import importlib

        for var in ("FACTSET_SQL_SERVER", "FACTSET_SQL_DATABASE", "FACTSET_SQL_USER"):
            monkeypatch.delenv(var, raising=False)

        cfg = importlib.reload(
            importlib.import_module("statarb.ingest.factset.config"))

        with pytest.raises(RuntimeError, match="FACTSET_SQL_SERVER"):
            cfg.require_connection_config()

    def test_safe_summary_never_contains_the_password(self, monkeypatch):
        """The security regression test.

        safe_connection_summary() exists to be logged. If it ever includes the
        password, every log line becomes a credential leak.
        """
        import importlib

        # A fake credential is the point of this test: to prove the summary does
        # not leak a password, one has to exist.
        secret = "hunter2-do-not-log-me"  # pragma: allowlist secret
        monkeypatch.setenv("FACTSET_SQL_SERVER", "test-host")
        monkeypatch.setenv("FACTSET_SQL_DATABASE", "test-db")
        monkeypatch.setenv("FACTSET_SQL_USER", "test-user")
        monkeypatch.setenv("FACTSET_SQL_PASSWORD", secret)

        cfg = importlib.reload(
            importlib.import_module("statarb.ingest.factset.config"))

        summary = cfg.safe_connection_summary()
        assert secret not in summary
        assert "PWD" not in summary.upper()
        # ...while the real connection string necessarily does contain it,
        # which is exactly why the two functions are separate.
        assert secret in cfg.connection_string()

    def test_destination_root_never_defaults_inside_the_repo(self, monkeypatch):
        import importlib

        monkeypatch.delenv("FACTSET_DEST_ROOT", raising=False)
        cfg = importlib.reload(
            importlib.import_module("statarb.ingest.factset.config"))

        assert REPO_ROOT not in cfg.DESTINATION_ROOT.parents
        assert cfg.DESTINATION_ROOT != REPO_ROOT
