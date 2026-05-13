# Codex Instructions

## Python Environment

Use the repo-local `.venv` entrypoint for all Python commands. It is a symlink to the Codex testing environment:

```bash
/Users/ioann/PycharmProjects/Ckit/.venv/bin/python
```

This environment is expected to be Python 3.11 and must satisfy the project requirement in `pyproject.toml`:

```toml
requires-python = ">=3.10,<3.15"
```


## Testing

Run tests through the environment's Python module entrypoint:

```bash
/Users/ioann/PycharmProjects/Ckit/.venv/bin/python -m pytest
```

For focused checks, pass the test path or node id:

```bash
/Users/ioann/PycharmProjects/Ckit/.venv/bin/python -m pytest tests/path/to/test_file.py
```

If documentation build tests are unrelated to the current change, use:

```bash
SKIP_DOCS_BUILD=true /Users/ioann/PycharmProjects/Ckit/.venv/bin/python -m pytest
```

## Environment Maintenance

If dependencies need to be refreshed, install from the current repo into the same environment:

```bash
/Users/ioann/PycharmProjects/Ckit/.venv/bin/python -m pip install -e ".[dev]"
```

Keep `.venv` as the single Codex-facing entrypoint, even if the backing interpreter is managed by conda.
