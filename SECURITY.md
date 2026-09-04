# Security Policy

## Supported versions

`pyakima` is pre-1.0 and under active development. Only the latest release on
PyPI and the current `main` branch receive security fixes; there are no
backports to earlier versions.

## Reporting a vulnerability

Report privately through GitHub's **[private vulnerability
reporting](https://github.com/mcdigman/pyakima/security/advisories/new)** (the
"Report a vulnerability" button on the repository's Security tab). This opens
a draft advisory visible only to you and the maintainer.

Please do **not** open a public issue for a suspected vulnerability.

Include what you have: the affected version or commit, the Python, NumPy, and
Numba versions, a minimal reproducer or input, and what an attacker gains. A
proof of concept helps but is not required to file.

This is currently a single-maintainer project. We aim to acknowledge valid
reports within three days and complete initial triage within about a week.
Critical issues may be handled on an accelerated timeline.

There is no bug bounty.

## Scope

`pyakima` is an in-process numerical library. It accepts arrays and compiles
performance-critical functions with Numba; it is not a sandbox or a parser for
untrusted programs. Reports are in scope where the library crosses a security
boundary rather than only producing incorrect numerical output. Examples
include:

- A documented, supported input causing out-of-bounds memory access, arbitrary
  code execution, or disclosure of unrelated process memory through
  `pyakima`'s own indexing or type handling.
- An input-validation flaw that lets a caller read or write outside the arrays
  supplied to `pyakima`.
- A project-controlled build or release flaw that could publish artifacts that
  do not correspond to the reviewed source, or expose publishing credentials.
- Secrets leaking from project-controlled workflows, logs, or artifacts.

Out of scope:

- Incorrect interpolation, unexpected exceptions, or crashes on invalid or
  unsupported inputs without an exploitable security impact. Open a normal
  issue for those correctness bugs.
- Vulnerabilities in NumPy, Numba, SciPy, `pygsl_lite`, or another dependency;
  report those to the dependency's maintainers.
- Resource exhaustion proportional to intentionally enormous arrays or Numba
  compilation workloads.
- Treating `pyakima` as an isolation boundary for untrusted Python code. Calls
  execute inside the caller's process with that process's privileges.

If you are unsure whether a report is in scope, report it privately.
