---
name: pyasc-generate-pass-doc
description: Generate or improve MLIR pass documentation (summary and description fields)
---

# PyAsc: Generate Pass Documentation

This skill generates or improves `summary` and `description` fields for MLIR passes in Passes.td files. Documentation is used for generated docs website and helps maintainers understand pass behavior.

**CRITICAL**: This skill does NOT modify pass implementation. Only updates TableGen documentation fields.

## When to Use

- Pass has been implemented but documentation is missing or incomplete
- Existing documentation needs improvement
- Maintainer wants to update pass documentation
- AI agent workflow: implement pass → generate documentation

## Batch Processing

This skill supports documenting multiple passes in a single run:

**Single pass**: User provides one pass name or file path
- Execute steps 1-8 once

**Multiple passes**: User provides a list of pass names/files or requests "all passes in Passes.td"
- **Before starting**: Ask user for review preference:
  - "Review each pass one-by-one" (interactive mode)
  - "Apply all changes without reviewing" (batch mode)
- **Interactive mode**: Execute steps 1-8 for each pass sequentially, asking for approval after each pass
- **Batch mode**: Execute steps 1-7 for all passes, then present summary of all changes for final approval
- Process passes one-by-one (do not batch multiple passes together in a single step)
- **Context refresh**: Every 3 passes, re-read this skill document to keep instructions fresh in context
- Maintain separate state for each pass (don't mix up metadata between passes)

**Examples**:
- "Document UnrollLoop, HoistTensorAllocation, and DetectKernelType"
- "Document all passes in include/ascir/Dialect/AscTile/Transforms/Passes.td"
- "Document these files: lib/Dialect/Asc/Transforms/Noop.cpp, lib/Dialect/Asc/Transforms/InsertSync.cpp"

---

## Step 1: Locate Pass and Gather Context

**Goal**: Find pass definition and all relevant source files

### Actions

1. Ask user for pass name (e.g., `UnrollLoop`) or path to `.cpp` implementation file
2. **Validate pass name format**: Pass name should be PascalCase (e.g., `UnrollLoop`, `HoistTensorAllocation`). If user provides kebab-case CLI name (e.g., `unroll-loop`), convert to PascalCase.
3. Search all `Passes.td` files for the pass definition:
   - `include/ascir/Dialect/Asc/Transforms/Passes.td`
   - `include/ascir/Dialect/AscTile/Transforms/Passes.td`
   - `include/ascir/Dialect/AscVF/Transforms/Passes.td`
   - `include/ascir/Conversion/LowerToAsc/Passes.td`

   **If pass not found in these locations**: Search all `Passes.td` files using glob pattern `include/ascir/**/Passes.td` to find passes in other dialects or conversions.
4. Extract metadata:
   - Pass name in TableGen (e.g., `def UnrollLoop`)
   - CLI name (e.g., `asctile-unroll-loop`)
   - Namespace (Asc/AscTile/AscVF/LowerToAsc)
   - Operation type (func::FuncOp or ModuleOp)
   - Options (if any)
5. Locate implementation `.cpp` file using this mapping:
   - For pass in `include/ascir/Dialect/X/Transforms/Passes.td`, implementation is in `lib/Dialect/X/Transforms/<PassName>.cpp`
   - For pass in `include/ascir/Conversion/X/Passes.td`, implementation is in `lib/Conversion/X/<PassName>.cpp`
   - Example: `include/ascir/Dialect/AscTile/Transforms/Passes.td` → `lib/Dialect/AscTile/Transforms/UnrollLoop.cpp`
   - **If implementation file not found**: Inform user that the file is missing and ask for the correct path, or ask if they want to skip this pass
6. Search for lit test files in both Dialect and Conversion directories:
   - `test/Dialect/*/Transforms/<cli-name>.mlir`
   - `test/Conversion/*/<cli-name>.mlir`
   - Also search: `test/**/<cli-name>.mlir` as fallback

### Acceptance Criteria

- Pass definition found in Passes.td
- Implementation file located
- Test files identified (may be none)
- All metadata extracted
- User informed of findings

---

## Step 2: Analyze Existing Documentation Quality

**Goal**: Evaluate current documentation and decide action

### Actions

1. Read existing `summary` field (may be empty or "TODO: add summary")
2. Read existing `description` field (may be empty)
3. **Auto-evaluate quality** using these criteria:

**Summary quality issues**:
- Empty or "TODO: add summary" → needs generation
- Not imperative form (e.g., "This pass unrolls..." or "Loop unrolling is...") → needs rewrite
- Contains "this pass", "the pass", or passive voice → needs rewrite
- Too long (>20 words) or too short (<5 words) → needs rewrite
- Describes HOW instead of WHAT → needs rewrite

**HOW vs WHAT examples**:
- ✗ HOW: "By walking the IR and finding scf.for ops with unroll_factor attributes, the pass unrolls them"
- ✓ WHAT: "Unroll scf.for loops by their unroll_factor attribute value"
- ✗ HOW: "The pass uses pattern matching to identify transpose operations and rewrites them"
- ✓ WHAT: "Remove transpose operations on cube operands and add transpose attributes to load/copy operations"
- ✗ HOW: "Iterates through all operations and checks if they have side effects"
- ✓ WHAT: "Hoist pure operations upward within their blocks to enable code motion optimizations"
- ✗ HOW: "Analyzes the data flow graph to find redundant computations"
- ✓ WHAT: "Eliminate redundant data transfer operations within VFGroupOp regions"

**Description quality issues**:
- Empty → needs generation
- Lacks transformation details → needs enhancement
- Options not documented → needs enhancement
- No examples for transformation pass → may need enhancement
- Exceeds 120-column limit → needs reformatting

4. **Present findings to user**:
   - Show existing documentation
   - List quality issues found
   - Recommend: "Generate new" or "Rewrite to improve"
   - Ask user to confirm action

### Acceptance Criteria

- Existing documentation evaluated
- Quality issues identified and listed
- User confirmed action (generate/rewrite)

---

## Step 3: Choose Documentation Depth

**Goal**: Determine level of detail

### Actions

Ask user to choose:
- **Simple and minimal**: Summary + 1-2 paragraph description, no examples
- **Detailed and comprehensive**: Summary + full description with transformation details, important options, examples (if they add value)

### Acceptance Criteria

- User selected depth
- Choice recorded for use in subsequent steps

---

## Step 4: Analyze Pass Implementation

**Goal**: Understand pass behavior completely

### Actions

1. Read implementation `.cpp` file completely
2. Identify:
   - **What operations** does it transform?
   - **What patterns** does it match?
   - **What attributes** does it add/remove/modify?
   - **Important options**: What do they control? (see criteria below)
   - **Conditions**: When does transformation apply?
   - **Hardware-specific**: Any platform-dependent logic?
3. If lit tests exist, read them to understand:
   - Input/output patterns
   - Key transformation examples
   - Edge cases (for understanding, not necessarily for docs)
4. **If no lit tests exist**: Rely solely on implementation analysis. Do not fabricate example transformations. Only document patterns you can verify from the code.

**Important Option Criteria**: An option is important if it:
- Changes the transformation pattern applied (different IR output)
- Enables/disables a major feature
- Affects hardware-specific behavior
- Controls a significant algorithmic choice

Options that are NOT important:
- Logging/debugging flags
- Minor threshold values
- Performance tuning parameters
- Experimental features

### Acceptance Criteria

- Can answer: (1) What operations are transformed? (2) What are the match conditions? (3) What IR changes occur? (4) What do important options control?
- Important options identified using criteria above
- Hardware-specific behavior noted (if any)
- Test patterns analyzed (if tests exist)

---

## Step 5: Generate Summary

**Goal**: Create one-line imperative description

### Rules

**MUST**:
- Imperative form: verb + object
- Describe WHAT the pass does, not HOW
- 8-15 words typical length
- Be concise and clear

**MUST NOT**:
- Use "this pass", "the pass", or similar phrases
- Use passive voice
- Describe implementation details
- Exceed 20 words

### Good Examples

✓ "Unroll scf.for loops by their unroll_factor attribute value"
✓ "Hoist pure operations upward within their blocks to enable code motion optimizations"
✓ "Convert tensor arithmetic operations with splat operands to scalar-splat variants for hardware efficiency"
✓ "Remove transpose operations on cube operands and add transpose attributes to load/copy operations"

### Bad Examples

✗ "This pass unrolls loops" (has "this pass")
✗ "Loop unrolling is performed using the unroll_factor attribute" (passive voice)
✗ "The UnrollLoop pass processes scf.for operations" (has "the pass")
✗ "By walking the IR and finding scf.for ops with unroll_factor attributes, the pass unrolls them" (describes HOW)

### Acceptance Criteria

- Imperative form (verb + object)
- Describes WHAT not HOW
- One line, concise (8-15 words typical)
- No forbidden phrases ("this pass", passive voice)

---

## Step 6: Generate Description

**Goal**: Create documentation following established patterns

### Structure for "Simple and Minimal"

Single paragraph (2-4 sentences):
- What the pass does
- Brief mention of how (if simple)
- Important option behavior (if applicable, integrate naturally)
- No examples unless absolutely necessary

### Structure for "Detailed and Comprehensive"

1. **Opening paragraph** (WHAT + WHY + context, 2-3 sentences):
   - What the pass does
   - Why it's needed (purpose, context)
   - When it's applied (pipeline position if relevant)

2. **Transformation details** (HOW, 1-2 paragraphs):
   - Patterns matched
   - Conditions for transformation
   - What changes in the IR
   - Use bullet lists for multiple patterns

3. **Important options** (1 paragraph, natural integration):
   - Only options that significantly change behavior
   - Integrate naturally: "The pass performs X, or performs Y if `option` is provided"
   - Don't list all options exhaustively

4. **Examples** (only if they enrich understanding, 1-2 max):
   - Before/After MLIR code blocks
   - Keep IR snippets short (5-10 lines max)
   - Add comments to explain key parts
   - Use pseudo-MLIR (simplified, not full test cases)
   - No obvious/trivial examples

5. **Hardware-specific behavior** (only if present in implementation, 1 paragraph):
   - Describe the behavior, not the platform
   - Don't mention platform names explicitly

### Formatting Rules (CRITICAL)

**120-column limit**: Count ALL characters per line (including leading whitespace/indentation), wrap at 120. This is strictly enforced.

**Allowed markdown**:
- `` ```mlir `` code blocks (for examples only)
- `-` bullet lists
- `**bold**` for emphasis
- `` `backticks` `` for operation/type/attribute names

**Forbidden markdown**:
- No headings (#, ##, ###)
- No tables
- No links
- No images

### Example Guidelines

**When to add examples**:
- Transformation is non-obvious
- Multiple patterns exist and example clarifies
- Before/after comparison adds value

**When NOT to add examples**:
- Transformation is trivial (e.g., "removes attribute X")
- Text description is already clear
- Example would just duplicate the text

**Example format**:
```mlir
// Before: brief description
%0 = op.x %arg {attr = "value"} : type1

// After: brief description
%0 = op.y %arg, %flag : type2
```

Keep examples concise. Use comments to explain, not verbose prose.

**Pseudo-MLIR guidelines**: Examples should use simplified, readable IR rather than exact test case syntax:
- Use placeholder operation names (e.g., `op.x`, `op.y`) if actual names are verbose
- Simplify types to focus on the transformation (e.g., `tensor<..., UB>` or `tensor` instead of full `tensor<..., #asctile.local<UB>>` signatures)
- Omit boilerplate (function signatures, module wrappers) unless relevant
- Focus on the key IR elements that change

**Multi-pattern passes**: For passes with 5+ transformation patterns:
- Document the 2-3 most common/important patterns in detail
- Mention that other patterns exist: "The pass also handles X, Y, and Z patterns."
- Don't exhaustively list all patterns unless user explicitly requests it

### Options Documentation

**Only mention options that significantly change behavior**.

**Good**: "The pass performs X, or performs Y if `opt-name` is provided."

**Bad**: "Options: opt-name (bool, default false): Controls whether..."

Integrate naturally into the description flow.

### Hardware-Specific Behavior

**Only document if implementation has explicit platform checks**.

**Good**: "When the target architecture requires 256-byte alignment for UB allocations, the pass additionally rounds up tensor sizes to the nearest multiple."

**Bad**: "On Ascend 910B1 hardware, the pass rounds up tensor sizes to 256 bytes." (don't mention platform names)

**How to document**: Describe the behavior and the condition that triggers it, not the platform name. Use phrases like:
- "When the target architecture requires..."
- "For hardware with specific alignment constraints..."
- "When certain buffer location constraints apply..."

Describe the behavior, not the platform.

### Acceptance Criteria

- Follows chosen depth structure (simple/detailed)
- **120-column limit maintained** (count characters per line)
- Markdown subset followed (no forbidden elements)
- Examples are concise and necessary (not obvious)
- Important options documented naturally (not exhaustive list)
- Hardware behavior documented (if applicable, without platform names)

---

## Step 7: Update Passes.td

**Goal**: Insert generated documentation

### Actions

1. Read current Passes.td file
2. Locate pass definition
3. Update `summary` field with generated summary
4. **If user chose "Detailed and comprehensive"**: Update `description` field with generated description. **If user chose "Simple and minimal"**: Update `description` field only if it was empty or "TODO".
5. **CRITICAL**: Do NOT change the position or ordering of the pass definition in the file. Leave it exactly where it is.
6. Preserve all other fields (constructor, options, dependentDialects) - do NOT modify anything except `summary` and `description`
7. Ensure TableGen syntax valid:
   - In TableGen `[{ }]` blocks, quotes don't need escaping
   - Only escape `}]` sequences (use `} ]` with space or rephrase)
   - Brackets balanced
   - No syntax errors

### Acceptance Criteria

- Passes.td updated with new documentation
- TableGen syntax valid
- Pass definition position unchanged in file
- Only `summary` and `description` fields modified
- No other fields modified

---

## Step 8: Verify and Get Approval

**Goal**: Ensure quality and get user confirmation

### Actions

1. Check 120-column compliance (count characters per line)
2. Verify markdown syntax (no forbidden elements)
3. Check TableGen syntax (brackets, escaping)
4. **Check grammar and spelling**: Review summary and description for grammar errors, typos, and unclear phrasing. Fix any issues found.
5. **Interactive mode**: Display final documentation to user in readable format and ask: "Approve this documentation or request changes?"
6. **Batch mode**: Skip individual approval, proceed to next pass. After all passes are processed, present summary of all changes for final approval.

### Acceptance Criteria

- All formatting rules satisfied
- No syntax errors
- No grammar or spelling errors
- User approved or requested specific changes (interactive mode) OR all passes processed and summary presented (batch mode)
- Documentation ready for use

---

## Critical Rules

**MUST**:
- Maintain 120-column limit strictly (count characters per line)
- Use imperative form for summary
- Auto-detect documentation quality issues
- Document important options naturally in description (not exhaustive list)
- Use backticks for operation/type/attribute names
- Keep examples concise and non-obvious
- Document hardware-specific behavior if present in code (without platform names)
- Ask user before modifying existing documentation
- Show final documentation for approval
- Preserve all other TableGen fields

**MUST NOT**:
- Use headings (#), tables, or links
- Exceed 120-column limit
- Use passive voice in summary
- Use "this pass" or "the pass" in summary
- Add verbose or obvious examples
- Document unimplemented features
- Modify pass implementation
- Analyze related passes (unless user explicitly requests)
- List all options exhaustively (only important ones)
- Mention platform names explicitly
- Change the position or ordering of pass definitions in Passes.td
- Modify any fields other than `summary` and `description`

**NEVER**:
- **Guess, infer, or fabricate pass behavior** - ONLY document what you can verify from the actual implementation code and test files. If you cannot determine behavior from code, ask the user for clarification. Do NOT make assumptions about what a pass "probably" does based on its name or similar passes.
- Add examples for trivial transformations
- Document error conditions or performance implications (too verbose)
- Use markdown headings or tables
- Break 120-column limit
- Generate documentation without user approval
- Modify fields other than summary and description

---

## Templates

### Simple and Minimal

```tablegen
def MyPass : Pass<"namespace-my-pass", "func::FuncOp"> {
  let summary = "Transform X operations to Y for Z purpose";
  let constructor = "mlir::namespace::createMyPassPass()";
  let description = [{
    Transforms X operations into Y operations when condition C is met. The
    transformation improves Z by doing W. If `option-name` is provided,
    performs alternative behavior B instead.
  }];
}
```

### Detailed and Comprehensive

```tablegen
def MyPass : Pass<"namespace-my-pass", "func::FuncOp"> {
  let summary = "Transform X operations to Y for Z purpose";
  let constructor = "mlir::namespace::createMyPassPass()";
  let options = [
    Option<"optName", "opt-name", "bool", "false", "Control behavior A vs B">,
  ];
  let description = [{
    Transforms X operations into Y operations to improve Z. Applied during
    the optimization pipeline after pass P1 and before pass P2.

    **Transformation patterns**:
    - Pattern 1: `op.x` with attribute `attr` → `op.y` with modified operand
    - Pattern 2: `op.x` in buffer location L0A → `op.y` with flag set

    The transformation applies when:
    1. Condition C1 is satisfied
    2. Condition C2 holds (operand has single use)
    3. Buffer location is L0A or L0B

    If `opt-name` is provided, performs aggressive transformation B instead
    of conservative behavior A.

    Example transformation:
    ```mlir
    // Before: X operation with attribute
    %0 = op.x %arg {attr = "value"} : type1

    // After: transformed to Y operation
    %0 = op.y %arg, %flag : type2
    ```
  }];
}
```

---

## User Interaction Points

1. **Step 1**: "Enter pass name or path to implementation file:"
2. **Batch mode (if multiple passes)**: "Review each pass one-by-one, or apply all changes without reviewing?"
3. **Step 2**: Show existing docs + quality issues, ask "Generate new or rewrite?"
4. **Step 3**: "Choose depth: Simple and minimal OR Detailed and comprehensive?"
5. **Step 8**: Show final docs, ask "Approve or request changes?" (skip in batch mode, show summary at end)

---

## Troubleshooting

**Issue**: Pass not found in any Passes.td file
- **Solution**: Verify pass name spelling, ask user to provide corresponding Passes.td file or clarify the pass name

**Issue**: Implementation file not found
- **Solution**: Check lib/ directory structure matches namespace, ask user to provide the path to a `.cpp` file

**Issue**: Description exceeds 120-column limit
- **Solution**: Reformat text, break lines at natural points, reduce verbosity

**Issue**: User rejects generated documentation
- **Solution**: Ask for specific feedback, adjust summary/description accordingly, iterate

**Issue**: Unclear pass behavior
- **Solution**: Read implementation more carefully, check lit tests, ask user for clarification

---

## Quality Checklist

Before presenting documentation to user, verify:

- [ ] Summary is imperative form (verb + object)
- [ ] Summary describes WHAT not HOW
- [ ] Summary has no "this pass" or passive voice
- [ ] Summary is 8-15 words (typical)
- [ ] Description follows chosen depth structure
- [ ] **All lines are ≤120 characters**
- [ ] No forbidden markdown (headings, tables, links)
- [ ] Examples are concise and necessary
- [ ] Important options documented naturally
- [ ] Hardware behavior documented (if applicable, no platform names)
- [ ] TableGen syntax valid
- [ ] Pass definition position unchanged in file
- [ ] Only `summary` and `description` fields modified
