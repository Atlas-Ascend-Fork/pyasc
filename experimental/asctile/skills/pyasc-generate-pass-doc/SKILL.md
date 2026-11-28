---
name: pyasc-generate-pass-doc
description: Use when creating, filling in, or improving MLIR Passes.td summary and description fields after a PyAsc pass has been implemented, including automatic documentation in an end-to-end pass-development workflow.
---

# PyAsc: Generate Pass Documentation

Document the pass from verified evidence, then edit only the `summary` and `description` fields in its `Passes.td`
definition. Never change pass code, other TableGen fields, file formatting, or pass ordering.

If the existing `summary` and `description` already satisfy the documentation contract and match verified code, leave
them unchanged.

Default workflow:

1. Locate the pass, implementation, and matching tests.
2. Read the implementation and tests until behavior is provable.
3. Draft focused documentation.
4. Update only fields that need changes; if no changes are needed, stop.
5. Validate the edit.

The user already owns the requested change; do not ask again unless behavior is ambiguous, the pass cannot be located,
or the maintainer explicitly requested review. If the user specifies a documentation style or depth, honor it.

## Locate Evidence

Accept a TableGen pass name, CLI name, or implementation path. Convert a CLI name such as
`ascendc-hoist-tensor-allocation` to the `def` name `HoistTensorAllocation`.

Find all definitions with:

```sh
grep -R -n -E "def <PassName>|Pass<\"[^\"]*<cli-name>\"" include experimental/asctile/include --include="Passes.td"
```

For unexpected locations, use `grep -R -n -E "def <PassName>" include experimental/asctile/include
--include="Passes.td"`. Read the definition and extract the CLI name, operation type, options, and existing
documentation. Then locate the implementation and tests:

- `include/ascir/Dialect/X/Transforms/Passes.td` → `lib/Dialect/X/Transforms/<PassName>*.cpp`
- `include/ascir/Conversion/X/Passes.td` → `lib/Conversion/X/<PassName>*.cpp`
- `experimental/asctile/include/asctile/Dialect/X/Transforms/Passes.td` →
  `experimental/asctile/lib/Dialect/X/Transforms/<PassName>*.cpp`
- `experimental/asctile/include/asctile/Conversion/X/Passes.td` →
  `experimental/asctile/lib/Conversion/X/<PassName>*.cpp`
- `test/**/<cli-name>.mlir`
- `experimental/asctile/test/**/<cli-name>.mlir`

Implementation suffixes are not always exactly `<PassName>.cpp`; search the likely directory if the direct path is
absent. Read every relevant pattern or rewrite in the implementation. Tests clarify semantics but do not replace the
implementation. If behavior remains ambiguous after both sources, ask the maintainer rather than guessing.

## Documentation Contract

`summary` is one concise imperative sentence: **what the pass changes**. Start with a verb, name the affected IR
objects, and omit "this pass", "the pass", passive voice, and implementation mechanics.

`description` is complete but compact. Start with the user-visible change, then state the important match conditions
and resulting IR changes. Cover each option that changes the transformation, selects a materially different algorithm,
or changes hardware-dependent behavior; omit logging and low-value tuning flags. Do not document unimplemented or
planned features.

Use this structure, but do not include section headings:

```tablegen
let summary = "Hoist local tensor allocations out of loops";
let description = [{
  Hoists `LocalTensorAutoOp` allocations out of nested regions and places each
  allocation in the function entry block. A single allocation can then serve
  every loop iteration.

  When `exclude-in-out=true`, allocations marked `input` or `output` remain in
  place and only temporary allocations are hoisted. By default, all
  `LocalTensorAutoOp` allocations are hoisted.
}];
```

For a simple pass, one paragraph is enough. For multiple patterns, use a short bulleted list. Add one or two compact
before/after `mlir` examples only when prose cannot make non-obvious transformations clear; never invent examples that
are not supported by code or tests.

### Description Rules

- Wrap every generated line at 120 characters or less.
- Use backticks for operation, type, attribute, and option names.
- Allowed markdown: bullets, bold labels, numbered lists, and `mlir` blocks.
- Do not use headings, tables, links, or images.
- For 5 or more patterns, describe the 2–3 most important ones and summarize the remaining shared conditions.
- Describe hardware-specific behavior by its observable condition, not by a platform name: "when the target requires
  256-byte alignment...".
- Do not document performance motivation, error handling, debugging tips, or implementation internals unless the user
  explicitly asks for that context.

## Edit And Validate

Use a surgical patch. Preserve braces, indentation, spelling, ordering, and all fields other than `summary` and
`description`. In a `[{ }]` code block, escape an embedded `}]` as `} ]` or rephrase to avoid it.

After editing, verify:

```sh
git diff --check -- include/ascir/Dialect/Asc/Transforms/Passes.td
grep -n -E "let (summary|description)" <Passes.td>
```

Then inspect the diff and check that:

- Only `summary` and `description` changed.
- No generated line exceeds 120 characters.
- Summary is imperative and names the actual change.
- Every documented behavior and option is supported by code or tests.
- TableGen delimiters remain balanced and the documentation reads naturally.

For multiple passes, process one at a time so metadata does not leak between passes. Show a concise summary of the files
and passes changed at the end.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Inferring behavior from the pass name | Read the implementation before drafting |
| Copying stale docs | Verify every sentence against current code and tests |
| Exhaustively listing options | Document only options that materially alter IR behavior |
| Rewriting an entire `Passes.td` file | Apply a field-only surgical patch |
| Adding a trivial example | Explain the change in prose instead |
| Claiming platform-specific behavior | Tie behavior to a verified target condition |
