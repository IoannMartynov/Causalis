# Contributing

## Docstrings

Use NumPy-style docstrings across this repository.

### General Rules

- Use triple double quotes: `"""..."""`.
- Start with a short summary sentence on the first line.
- Keep docstrings accurate to the current implementation.
- Prefer describing behavior, side effects, constraints, and failure modes over repeating type hints.
- Do not use Google-style sections such as `Args:` or `Returns:`.

### When To Keep It Short

Use a one-line docstring for small helpers when the behavior is obvious from the signature and implementation.

Example:

```python
def _relative_posix_path(path: Path, start: Path) -> str:
    """Return the relative path from ``start`` to ``path`` using POSIX separators."""
```

### When To Use Full Sections

Use a multi-line NumPy docstring for public functions and for helpers with important side effects, filesystem changes, dependency requirements, or non-obvious behavior.

Use sections only when they add value, in this order:

1. Summary
2. `Parameters`
3. `Returns`
4. `Raises`
5. `Notes`

Example:

```python
def generate_api_reference() -> Path:
    """
    Generate the API reference HTML site under ``notebooks/api/html``.

    Returns
    -------
    Path
        Path to the generated HTML API reference directory.

    Raises
    ------
    FileNotFoundError
        If the package directory to document does not exist.
    RuntimeError
        If the Sphinx build finishes with a non-zero exit code.

    Notes
    -----
    This function builds temporary MyST source files in a staging area,
    preserves source snapshots under ``_sources`` in the HTML output, and
    swaps the staged HTML site into place only after generation succeeds.
    """
```

### Section Guidance

- `Parameters`: Explain what each argument means in the context of the function.
- `Returns`: Describe the returned value semantically, not just its type.
- `Raises`: Include meaningful exceptions a caller may need to handle.
- `Notes`: Document side effects such as file writes, temp directory usage, atomic replacement, or required optional dependencies.

### Avoid

- Repeating information already obvious from the type hints.
- Documenting every local implementation detail.
- Adding empty or low-value sections.
- Writing broader guarantees than the code actually provides.

## Releases

Production releases are automated through GitHub Actions and PyPI Trusted Publishing.

- Package versions are derived from Git tags via `setuptools-scm`; do not edit a version string by hand.
- Cut releases only from `main` after tests pass.
- Create an annotated `vX.Y.Z` tag, for example `git tag -a v0.5.1 -m "Release v0.5.1"`.
- Push the tag with `git push origin v0.5.1`.
- Pushing that tag triggers the `Release` workflow, which tests the project, builds and checks the package, publishes the same artifacts to PyPI, and then creates the GitHub Release.
- If a version has already been uploaded to PyPI, do not reuse it. Release a new patch, minor, or major version instead.

### PyPI Setup

Configure PyPI Trusted Publishing for this repository before the first automated release:

1. Open the `causalis` project on PyPI.
2. Add a trusted publisher for `causalis-causalcraft/Causalis`.
3. Set the workflow file to `release.yml`.
4. Set the environment name to `pypi`.

## Releasing

Production releases are created from annotated Git tags on `main`. The tag
drives the package version through `setuptools-scm`, then GitHub Actions builds,
checks, publishes to PyPI with Trusted Publishing, and creates the matching
GitHub Release.

```bash
git checkout main
git pull --ff-only origin main
python3 -m pytest

git tag -a v0.5.1 -m "Release v0.5.1"
git push origin v0.5.1
```