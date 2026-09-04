# Contributing

Thanks for your interest. This is a small project with deliberately strict
gates — reading this first will save you a review round.

Please open an issue before starting anything substantial.

## Development setup

`pyakima` uses [uv](https://docs.astral.sh/uv/) and requires CPython 3.10 or
newer; CI currently tests 3.10 through 3.14. Create the project environment
with:

```bash
uv sync --extra dev
```

Run the suite and the hooks before pushing:

```bash
uv run pytest
```

```bash
uv run prek run --all-files
```

The `prek` hooks include `ruff check --fix --preview`, `ruff format`, `ssort`,
`pyupgrade`, `pyrefly`, credential checks, and `actionlint`. CI additionally
runs strict `mypy`, `pyright`, `pylint`, `pydoclint`, a strict `skylos` job,
the supported Python/dependency matrix, distribution builds, documentation,
and branch coverage.

For the same 100% line-and-branch coverage gate used on pull requests to
`main`, run:

```bash
NUMBA_DISABLE_JIT=1 uv run --with coverage coverage run --branch --source=pyakima --omit='*/demos/*' -m pytest
uv run --with coverage coverage report -m --fail-under=100
```

The normal test run keeps Numba's JIT enabled. The coverage run disables it so
coverage.py can observe the Python bytecode; changes to compiled paths should
pass both runs.

## House rules

These are enforced by CI, so they are not merely stylistic preferences:

- **Strict typing.** Use the narrowest type that fits. Avoid `Any` and `object`
  unless the function genuinely accepts anything. Keep any `type: ignore` or
  `noqa` precise and document why the underlying Numba or typing boundary
  cannot be expressed directly.
- **Linting.** Ruff runs with `ALL` rules selected. Prefer restructuring code
  over adding a suppression, and do not broaden the configured global ignores
  for a local problem.
- **Docstrings.** Use concise NumPy-style docstrings, checked by `pydoclint`.
  Public functions need `Parameters`, `Returns`, and `Raises` sections where
  applicable.
- **Tests.** Use `pytest` style with non-vacuous assertions and maintain full
  line and branch coverage. Exercise relevant numerical boundaries, dtypes,
  shapes, extrapolation modes, and both JIT-enabled and pure-Python paths.
- **Compatibility.** Runtime support starts at NumPy 1.22 and Numba 0.56 on
  Python 3.10. Do not use newer APIs without either preserving those floors or
  proposing an explicit support-policy change. CI also tests current releases
  on every supported Python version.
- **Licensing.** New source files carry the SPDX header used throughout
  `pyakima/` and `tests/`. Contributions are accepted under Apache-2.0.

## Pull requests

Open contribution pull requests against `dev`, as documented in the README.
Keep each pull request narrowly scoped and explain the behavior being changed,
the tests that exercise it, and any compatibility implications. Update
`CHANGELOG.md` for user-visible changes.

Do not commit virtual environments, caches, coverage output, or build
artifacts. The committed `uv.lock` is maintained for reproducible development;
include its changes when an intentional dependency update requires them.

For performance changes, include before-and-after measurements, the Python,
NumPy, and Numba versions, and enough workload detail to reproduce the result.
The local benchmark is:

```bash
uv run --with scipy python -m pyakima.demos.speed_demo
```

The manual `Benchmark` GitHub Actions workflow records comparable artifacts on
Linux, macOS, and Windows. Benchmark numbers are not a merge gate because
shared hosted runners are noisy.

## Documentation changes

Public API or behavior changes should update the relevant README and `docs/`
pages. Build the documentation with warnings treated as errors:

```bash
uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Cutting a release

Maintainer checklist; the ordering matters:

1. Bump the version in both [pyproject.toml](pyproject.toml)
   (`project.version`) and [CITATION.cff](CITATION.cff) (`version:`), and update
   `date-released:` in `CITATION.cff`.
2. Add the new version and release date to [CHANGELOG.md](CHANGELOG.md), with
   links for any new release headings.
3. Publish a GitHub Release. Prereleases go only to TestPyPI; full releases go
   to PyPI through Trusted Publishing in
   [.github/workflows/publish.yml](.github/workflows/publish.yml).

## Use of AI tools

AI assistance is allowed. Unreviewed AI output is not. The bar is unchanged
whatever produced the diff: correctness, and a contributor who understands
what they submitted.

1. **You own the contribution.** Whatever generated it, you are the author of
   record and responsible for it being correct.
2. **Disclose material assistance.** If an AI tool meaningfully produced the
   code, tests, issue report, or review, say so in the pull request. A single
   line is enough. Disclosure is not held against you; undisclosed AI output
   that turns out to be wrong is.
3. **Be able to explain it.** Do not submit anything you cannot walk through
   and defend — why this approach, what the edge cases are, why the tests cover
   them.
4. **Expect to be asked.** For contributions that look machine-generated, the
   maintainer may ask for an explanation, a reproduction, or evidence the tests
   actually exercise the change before reviewing further. Unanswered, such a
   PR will be closed.

Assistive uses — spelling, formatting, looking things up, tightening prose —
need no disclosure.

PRs should be narrowly scoped for reviewability; bulky PRs against many files
at once or with many lines of churn may be closed without review.

## Reporting bugs

Please report non-security bugs through [GitHub
Issues](https://github.com/mcdigman/pyakima/issues). Include the affected
version or commit; Python, NumPy, and Numba versions; a minimal reproducer and
input arrays; and the actual versus expected output.

For anything security-relevant, follow [SECURITY.md](SECURITY.md) instead of
opening a public issue.
