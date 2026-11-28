---
name: pyasc-update-lit-checks
description: Use when lit tests fail in their CHECKs after a pass or emitter change, or when filling new test CHECKs
---

# PyAsc: Update Lit CHECKs

Regenerates the `CHECK` directives of `.mlir` lit tests from the actual output of the repo's tools — the
pyasc equivalent of LLVM's `update_llc_test_checks.py`, executed by you as a step-by-step protocol
(no script).

**Invariants:** only CHECK comment blocks change (IR, `RUN:` lines, headers, comments and line endings
preserved byte-for-byte); regenerated content passes the same `RUN:` pipeline + FileCheck before the file
is written; regeneration is idempotent; hand-written semantics survive.

**Read `references/protocol.md` and follow it exactly.** Its constraints are the point; do not work
around them.

## When to Use

* A pass or the emitter changed, and lit tests now fail only in their CHECK expectations.
* A new test needs its CHECK blocks filled from the real output.

## When NOT (static exclusions)

A match: keep the file hand-maintained and report it. Runtime stops — e.g. nondeterminism, duplicate
function names, vanished functions with no source change — are protocol-time, not classification-time.

* `-verify-diagnostics` tests (hand-written `expected-*` annotations);
* custom `-check-prefix` / `-check-prefixes`;
* multiple distinct `RUN:` pipelines in one file;
* no `RUN:` line or no FileCheck stage;
* `not ... 2>&1` error-output tests — their CHECKs match diagnostics, not stdout;
* `-debug-only` debug-log tests — CHECKs pin debug lines regeneration cannot place (e.g.
  `dispatch-alloc.mlir`);
* whole-output tests without input functions — intentional partial matching (e.g.
  `test/tools/ascendir-opt/dialects.mlir`);
* tests whose hand-written CHECKs cover content outside every function chunk — structs, `#include`,
  `#define` (e.g. `test/Target/AscendC/emitc.mlir` checks `#include` lines) — regeneration cannot
  reproduce that coverage; leftover content no CHECK covers is report-only, not an exclusion;
* CHECK blocks inline inside a function body or below the last function — corpus style: the block
  directly above its function.

## Quick Reference

| # | Stage | Gate |
|---|---|---|
| 1 | Classify (When NOT list above) | file is regenerable, else skip and report |
| 2 | Generate: pipeline ×3 (determinism), chunk output by function, rebuild CHECK blocks | stable reference output |
| 3 | Validate on a temp copy: pipeline ×3 + FileCheck | byte-identical output, exit 0 |
| 4 | Write the real file — CHECK lines only, preserve CRLF | — |
| 5 | Review the diff (old vs candidate) | deletions/reorders explained by the source change, else revert |
| 6 | Verify: regeneration idempotent (zero diff) + lit or pipeline+FileCheck | both pass |
| 7 | Report: diff signals, hand-maintained files, red flags, nondeterminism | — |

Never, beyond the protocol:

* rewrite a file's `RUN:` lines or strip hand-written directives to make an unsupported file regenerable;
* commit a file whose diff shows a change outside its CHECK comment lines;
* regenerate a file from the When NOT list.

The mechanics — chunking anchors, FileCheck escaping, CHECK-NOT placement, CRLF preservation, diff-review
predicates — are in `references/protocol.md` (Steps 1–8).
