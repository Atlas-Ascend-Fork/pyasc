# Update Lit Checks: Detailed Protocol

Detailed protocol for the `pyasc-update-lit-checks` skill — read together with its SKILL.md. Execute
Steps 1–8 in order; the stage gates and the absolute never-rules are in SKILL.md's Quick Reference.

Before anything: classify the file against the When NOT list in SKILL.md — if it matches, keep the file
hand-maintained and report it instead.

## Step 1: Produce the Reference Output

1. Read the `// RUN:` line. It has the shape `// RUN: <tool> [flags] %s [| <tool2> ...] | FileCheck %s [flags]`.
   `%s` means the test file itself. A trailing `2>&1` on a tool stage means "merge stderr into stdout"
   (debug-log tests).
2. Locate the tools: they live in `build/*/bin/` (`ascir-opt`, `ascir-translate`, `asctile-opt`), mirroring
   `test/lit.site.cfg.in`.
3. Locate FileCheck, matching the LLVM version the repo builds against: read `LLVM_PREFIX_PATH` from
   `build/*/CMakeCache.txt` and use `<LLVM_PREFIX_PATH>/bin/FileCheck`. Fallbacks: `$LLVM_INSTALL_PATH/bin/FileCheck`
   (the variable `test/build_llt.sh` uses), then `/usr/lib/llvm-19/bin/FileCheck` (the repo builds against LLVM 19;
   `/usr/lib/llvm-*/bin/FileCheck` may match several majors — pick 19, not 21).
4. Execute the tool stages in order, piping stdout into the next stage (`%s` → the test path). Save the final output
   to a temp file outside the repo. The build must be up to date — rebuild first if the
   emitter/pass under change was not compiled.
5. **Determinism check**: run the pipeline two more times and diff the three outputs. Same input must give
   byte-identical output; if it does not, STOP — that is a real pass bug (unordered container iteration), exact
   CHECKs are impossible. Report it to the user with the pass pipeline and file path; do not regenerate the file.

**Acceptance criteria**: a stable reference output file, and a one-sentence justification why the file is not in
the When NOT list (SKILL.md).

## Step 2: Split the Output Into Per-Function Chunks

* A chunk **starts** at an anchor line:
  * an MLIR function line `func[.func] @name(...)` at any indentation, with an **optional** visibility keyword
    `private`/`public`/`nested` (usually omitted for public functions — the corpus norm). This includes
    declarations without a body;
  * an emitted Ascend C signature at column 0 that ends with `{`, the function name being the identifier right
    before the last `(` — either `__inline__ __attribute__((always_inline)) __aicore__ void f(...) {` or
    `extern "C" __global__ __aicore__ void f(...) {`;
  * not anchors: struct definitions (`struct PyStruct1 {` — column 0, ends with `{`, but no checkable function
    identity) and `#pragma` lines — a chunk that starts at one is nameless.
* A chunk **ends** at the next anchor, at a `}` at column 0 (function/module close, included in the chunk), or at EOF.
* Content **before the first anchor** (e.g. `module {`) is not checked — that matches the existing corpus style.
  Structural noise to skip: `-----` / `// -----` `-split-input-file` separators, blank lines between chunks,
  repeated `module {` wrappers between splits.
* **Any other leftover content** — struct definitions, `#include`, `#define`, debug-log lines — must be reported
  in the Step 8 report even if no hand-written CHECK covers it. If hand-written CHECKs do cover it (e.g.
  `test/Target/AscendC/emitc.mlir` checks its `#include` lines), regenerating would lose that coverage → keep the
  file hand-maintained.

**Input↔output mapping is by function name.** If the same function name appears twice in the input (legal across
`// -----` splits), the pairing is ambiguous → stop, keep the file hand-maintained; do not guess.

**Acceptance criteria**: every chunk has a name; any nameless or leftover chunk is either classified as structural
noise or moves the file into the hand-maintained list (and is reported).

## Step 3: Generate the New CHECK Block per Function

### 3.1 Directive kinds

* First chunk line → `// CHECK-LABEL: <full line>` — including the `__inline__ __attribute__((always_inline))`
  `__aicore__` prefix on C signatures (hand-written tests often trimmed it).
* Every following non-empty line → `// CHECK-NEXT: <full line>`.
* Empty line → `// CHECK-EMPTY:`, followed by a plain `// CHECK-NEXT:` (CHECK-NEXT after CHECK-EMPTY is valid and
  stricter than `CHECK:` — it keeps the adjacency guarantee). FileCheck collapses runs of horizontal whitespace
  in patterns by default; generated patterns do not pin exact indentation or spacing — that is fine, do not add
  `-strict-whitespace`.
* The space after the directive colon (`CHECK-LABEL: pattern`) is this protocol's form; FileCheck also accepts
  no space (`CHECK-LABEL:pattern`, verified — both exit 0). Use the spaced form on every regeneration.

### 3.2 Escaping

FileCheck matches plain text as fixed strings — `(`, `)`, `*`, `+`, `[` alone need no escaping. Only two sequences
are special:

* `[[` opens a substitution block and `\[[` is **not** a valid escape (the FileCheck of LLVM 19 rejects it at
  parse time). Use the `{LITERAL}` directive modifier instead — `// CHECK-NEXT{LITERAL}: <full line>`;
  `CHECK-LABEL{LITERAL}:` works the same way. If a `{LITERAL}` directive cannot be used, keep that function's
  hand-written block.
* `{{` opens a regex block. A safe pattern is constructible (`{{[{][{]}}` — verified to match a literal `{{`) but
  error-prone; the default is to keep that function's hand-written block.

### 3.3 Hand-written directives in the old block

* `CHECK-NOT` / `CHECK-COUNT` / `CHECK-DAG`: keep verbatim. **Placement is semantics** — a `CHECK-NOT` asserts
  absence *between its two neighboring positive directives* (or before the first / after the last). Rebuild the
  placement mechanically:
  1. Map each old positive directive to the new directive generated from the same output line (match the old
     pattern text against the new generated lines).
  2. Place each kept `CHECK-NOT`/`COUNT`/`DAG` between the same pair of neighboring positive directives as in
     the old block.
  3. If either neighbor no longer exists (its output line disappeared), place it directly after the
     `CHECK-LABEL` (function-wide scope) and flag the file in the Step 8 report for human review.
  Never re-insert "somewhere approximately right" — two regenerations must agree on the exact placement, or
  idempotency (Step 6.1) breaks.
* `CHECK-SAME`: it pins a substring of the anchor line. If the substring is contained in your new full-line
  `CHECK-LABEL` pattern, the directive is redundant — drop it; if it is not, the hand-written intent cannot be
  reproduced — keep the whole old block.
* Functions whose output **embeds the test file's own path** (assert/printf source locations, e.g. `ascendc.printf
  {desc = "Assertion failed at <path>:<line>:<col>..."}`): the path changes with every file location and the line
  number shifts with every edit, so a literal pattern is wrong. Keep the hand-written block (it uses `{{.*}}`
  regexes) verbatim; regenerate only the other functions.

### 3.4 Functions without an old block

If an input function has no CHECK block above it, generate one and insert it directly above the function line,
adjacent to it (corpus style: the last CHECK line touches the `func.func` line; no blank line in between).

### 3.5 Output-only and vanished functions

* **Output-only functions** (in the output but not in the input — the pass synthesizes them, e.g. outlining):
  generate their blocks and append them after the last input function's block, in output order. Report this in
  Step 8.
* **Input functions absent from the output** (the pass removed them): delete their stale block **only when** the
  PR's source change explains the removal. Otherwise STOP and report — a vanishing function with no matching
  source change is a regression caught early.

**Acceptance criteria**: for every function you generated a complete new block, kept the old block (with a
reason), or handled it per 3.5 — nothing is silently dropped.

## Step 4: Rewrite the File

Rules for the rewrite — the write itself happens in Step 5, only after validation passes (SKILL.md's
stage order: validate, then write).

* Only lines inside CHECK comment blocks change; IR, `RUN:` lines, the copyright header and unrelated comments
  are copied byte-for-byte.
* Line endings are preserved: a number of corpus files are CRLF (check with `grep -c $'\r' <file>` — nonzero
  means CRLF). If your diff shows *every* line changed, you normalized CRLF→LF — redo the edit preserving the
  original endings.
* If a function's block was kept hand-written, the function and its block are left exactly as they were.

## Step 5: Validate Before Saving

1. Write the candidate content to a **temp copy** of the test, outside the repo (same pipeline input; content
   differs only in CHECK comments), with the original file's line endings.
2. Run the tool pipeline on the temp copy, three times, piping into FileCheck with the temp copy as the check
   file: `<tool> ... <temp-copy> | FileCheck <temp-copy>`.
3. All three runs must give byte-identical output (nondeterminism → Step 1.5) and FileCheck must exit 0.
4. Only then write the changes into the real test file. If validation fails, **do not save and do not hand-edit
   the generated block until it passes** — a validation failure means something is actually wrong (generator
   mistake, unstable output, or a real emitter bug).

**Acceptance criteria**: the saved file passes pipeline+FileCheck from a clean state.

## Step 6: Idempotency and Test Verification

1. Redo the generation (Steps 1–4) on the updated file — it must produce **zero diff**. If it does not, the output
   is unstable or your generation is nondeterministic (most often: inconsistent CHECK-NOT placement per Step 3.3)
   — investigate; do not "fix" by editing checks by hand.
2. Run the test. `lit` is **not guaranteed to be installed** (this repo's LLVM prefix ships FileCheck but not
   llvm-lit):
   * preferred, if `lit` is on PATH or at `<LLVM_PREFIX_PATH>/bin/llvm-lit` (the way `test/build_llt.sh` runs it,
     with the LLVM bin dir on PATH so bare tool names in RUN lines resolve): `lit -v <test path> --param
     ascir_tools_dir=build/*/bin`; if lit cannot resolve the tools, use the fallback instead of debugging lit;
   * otherwise re-run the Step 5 pipeline (tool | FileCheck, replicating all RUN flags) directly on the
     **saved** file from a clean shell.
   State in the Step 8 report which of the two you ran.
3. `git diff` on the file must show only `// CHECK` lines changing. Any change to IR, RUN lines, or headers is a
   bug — revert the file.

## Step 7: Semantic Review of the Diff

Compute the numbers mechanically — do not eyeball — from a diff of the old content (`git show HEAD:<path>`)
against the candidate (before writing; the real file is untouched) or the working-tree file (after writing);
always **before committing**. For a brand-new file there is no old content — every directive is an addition.

From that diff count three things: added CHECK lines, removed CHECK lines, and changed lines that are NOT
CHECK comment lines. A CHECK line is `//`, optional spaces, then `CHECK` — the corpus also has the unspaced
`//CHECK:` form; count it too, or removed lines will be undercounted. The non-CHECK count must be zero: a
nonzero count means IR, `RUN:` or header lines changed, which the invariants forbid — revert.

Classify by predicate (pattern text = the text after `CHECK...:`, trimmed; matching is substring containment
of the trimmed pattern text — an old pattern that survives inside a longer new line, such as a trimmed
hand-written LABEL becoming a full-line LABEL or the corpus's unspaced `CHECK-LABEL:void` rewritten in the
spaced form, is a Replacement, not a Deletion):

| Signal | Predicate | Verdict |
|---|---|---|
| Deletion | old CHECK pattern text appears in no new CHECK line | **red flag** — see below |
| Addition | new CHECK pattern text appears in no old line | expected — blocks were partial, regeneration is fuller |
| Replacement | only the `__inline__ ... __aicore__` prefix / signatures / spacing differ | format-only noise |
| Reorder | directive order changed | **red flag** — see below |
| Sparse→dense | directive count jumps (e.g. 7 → 79) | sparse test filled in — usually fine, mention it |

Reorders are checked on block anchors — the first CHECK line of each block (hand-written blocks start with
plain `CHECK:`, generated ones with `CHECK-LABEL:` — both count), because anchors are unique function
signatures, while short repeated body patterns (`return;`, `}`) give false reorders through pattern-level
matching. Map each new anchor to the old anchor whose pattern text it contains; the old indices must be
non-decreasing — a decrease is a Reorder. Intra-block line order is not checked here: the generated
`CHECK-NEXT` chain enforces it at validation time.

Deletions and reorders are red flags: each must be explained by the accompanying source change (e.g. an input
op deleted in this PR) — a deletion with no matching source change means the emitter lost output, an
unexplained reorder means the pass changed behavior on its own; stop and report in both cases. Sparse→dense
jumps usually mean a hand-written `-split-input-file` test that checks one line per split — fine, but tell
the user. Read the actual diff, not just the counts, for every file with deletions or a large number of
replacements.

## Step 8: Report to the User

Summarize: files updated with their diff signals (additions, deletions, reorders, sparse→dense jumps); every
red-flag deletion and how it is explained by the source change; files kept hand-maintained, each with a one-line
reason; synthesized functions (Step 3.5) and CHECK-NOT placements moved to function-wide scope (Step 3.3) — both
need human eyes; any nondeterminism discovered (pass pipeline + file path, so a pass bug can be filed); which
test verification you ran (lit or pipeline+FileCheck fallback).
