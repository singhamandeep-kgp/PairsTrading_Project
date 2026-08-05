"""
Importing this package must do nothing but define names.

This is the permanent regression test for three defects that all had the same
root cause — work happening at import time:

1. `config/settings.py` called `pd.read_pickle(os.getcwd()/...)` at module scope,
   on a file that does not exist. Every import raised FileNotFoundError, which
   made the entire legacy strategy stack unimportable — including from tests that
   never wanted the calendar.

2. Five library modules called `logging.basicConfig()` at import, mutating the
   ROOT logger for the whole host process on a first-import-wins basis.

3. `curate/schema.py` resolved data paths from an environment variable at import,
   forcing tests into monkeypatch-before-import ordering games.

A test asserting "importing is inert" is cheap and catches all three classes
forever. Without it, the next one gets reintroduced and nobody notices until
something far downstream behaves oddly.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil

import pytest

# These genuinely require an optional extra that CI does not install.
# Listed explicitly so a NEW unimportable module is a failure, not a silent skip.
REQUIRES_OPTIONAL_EXTRA = {
    "statarb.ingest.crsp.puller",     # needs `wrds` (the crsp extra)
    "statarb.scripts.gics_export",    # imports the above
}


def _all_modules() -> list[str]:
    import statarb

    return [m.name for m in pkgutil.walk_packages(statarb.__path__, "statarb.")]


def test_every_module_imports_cleanly():
    """No module may fail to import for any reason other than a declared extra."""
    failures = []
    for name in _all_modules():
        if name in REQUIRES_OPTIONAL_EXTRA:
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")

    assert not failures, "modules failed to import:\n  " + "\n  ".join(failures)


def test_importing_everything_does_not_configure_the_root_logger():
    """A library must never call logging.basicConfig().

    basicConfig mutates the root logger, so importing one module silently changes
    logging for the entire process — including code with nothing to do with this
    package. Configuration is the application's job; see statarb.logging, called
    only from cli/.
    """
    root = logging.getLogger()
    before = list(root.handlers)

    for name in _all_modules():
        if name in REQUIRES_OPTIONAL_EXTRA:
            continue
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001, S110
            pass  # import failures are the other test's business

    added = [h for h in root.handlers if h not in before]
    assert not added, (
        "importing statarb added root logging handlers, so some module called "
        f"logging.basicConfig() at import time: {added}"
    )


def test_optional_extras_are_not_imported_at_module_scope():
    """The heavy vendor dependencies must stay behind lazy imports.

    `arrow_odbc` and `pyodbc` need a system ODBC driver that CI cannot install.
    Keeping them out of module-scope imports is exactly what lets CI install the
    package and exercise the whole read path on a clean Linux runner.
    """
    import sys

    for mod in ("arrow_odbc", "pyodbc", "wrds"):
        sys.modules.pop(mod, None)

    importlib.import_module("statarb")
    importlib.import_module("statarb.data.api")
    importlib.import_module("statarb.curate.build")
    importlib.import_module("statarb.cli.main")

    leaked = [m for m in ("arrow_odbc", "pyodbc", "wrds") if m in sys.modules]
    assert not leaked, (
        f"importing the core package pulled in optional extras {leaked}; they "
        "must be imported lazily inside the functions that need them"
    )


@pytest.mark.parametrize("module", [
    "statarb.paths",
    "statarb.curate.schema",
    "statarb.curate.factors",
    "statarb.data.api",
    "statarb.ingest.factset.config",
    "statarb.ingest.factset.queries",
])
def test_key_modules_import_without_credentials_or_data(module, monkeypatch):
    """The package must import on a machine with no FactSet access at all.

    This is why `ingest.factset.config` does NOT validate its required settings at
    import: `required=True` at module scope would make merely importing the module
    fail without a .env, which is the same import-time-crash bug as (1) above.
    Validation belongs at the point of use — see require_connection_config().
    """
    for var in ("FACTSET_SQL_SERVER", "FACTSET_SQL_DATABASE", "FACTSET_SQL_USER",
                "FACTSET_SQL_PASSWORD", "FACTSET_DEST_ROOT"):
        monkeypatch.delenv(var, raising=False)

    importlib.reload(importlib.import_module(module))
