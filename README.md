# Kernel Design Agents

Kernel Design Agents (KDA) is an agent-centric workflow for using coding agents to research, implement, verify, and iterate on performance-sensitive CUDA kernel tasks.

This repository documents an early research prototype and remains under active development. Community feedback and contributions are welcome. If you are interested in HAN Lab Mafia's solutions ranking #1–3 on tracks at the MLSys Kernel Contest, see [mit-han-lab/mlsys2026-flashinfer-contest](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest) for performance evaluation and reproduction.

## Contents

| Path | Purpose |
|---|---|
| `docs/agent-flow.md` | Minimal end-to-end KDA workflow. |
| `prompts/README.md` | How to use prompt templates. |
| `prompts/basic-flow.md` | Generic starter prompt for a new task. |
| `CLAUDE.md` | Repository-facing agent instructions. |
| `CONTRIBUTING.md` | Contribution process and DCO sign-off requirements. |
| `THIRD_PARTY_NOTICES.md` | Third-party component and license disclosures. |
| `third_party_licenses/` | Verbatim upstream license files for distributed third-party components. |

## Getting Started

Install the agent workflow dependencies before starting the agent session:

```bash
git clone --recurse-submodules --remote-submodules https://github.com/FangWu0314/kda.git
cd kda

# link skills
mkdir -p ~/.claude/skills
ln -s "$(pwd)/skills/ncu-report-skill" ~/.claude/skills/ncu-report-skill
ln -s "$(pwd)/skills/KernelWiki" ~/.claude/skills/KernelWiki
ln -s "$(pwd)/skills/IKET-skill" ~/.claude/skills/IKET-skill

# or clone the independently licensed ncu skill directly
mkdir -p ~/.claude/skills && cd ~/.claude/skills
git clone https://github.com/mit-han-lab/ncu-report-skill.git
```

`--remote-submodules` checks out the latest upstream commit of each submodule
instead of the revision recorded in this repository. To refresh an existing
checkout, run `git submodule update --remote`. To fall back to the recorded
revisions, run `git submodule update`.

The upstream KernelWiki checkout may contain artifact snapshots governed by
additional terms, such as the NVIDIA CuTe DSL license. Review those terms before
redistributing it.


## What You Need To Do

### 1. Create the Workspace Layout

Use this repository as reference material, then do implementation work elsewhere:

```text
task-workspace/
  docs/
    draft.md
    plan.md
  runs/
  outputs/
  profile/
  benchmark.csv
  candidates.jsonl
```

### 2. Define the Task Contract

| Item | What to provide | Example |
|---|---|---|
| Objective | One or two sentences on the user-facing goal, naming the kernel and the workload. | "Reduce latency of the FP8 grouped GEMM for MoE shapes on B300." |
| Correctness requirements | Required behavior, numerical tolerances, and invariants that must hold. | "Output matches the reference within 1e-3 relative error on all listed shapes." |
| Validation command | One command that proves correctness and exits non-zero on failure. | `python tests/test_fp8.py` |
| Evaluation command | One command that measures the target metric, if different from validation. | `python bench/run.py --shapes shapes.json` |
| Promotion criteria | What must be true before a candidate replaces the current best. | "Passes validation and is at least 3% faster on the geometric mean, with no shape more than 1% slower." |
| Constraints | Allowed languages, libraries, and APIs; files or interfaces that must not change; measurement rules such as warmup count and cache state. | "CUDA C++ only; public Python API unchanged; no new dependencies; 3 warmup calls and 5 measured calls, L2 cold." |

- These items fill the Task Contract placeholders in `prompts/basic-flow.md`; Constraints goes under "Allowed implementation approaches". The template also asks for a task name and a performance or quality target.
- The validation command and the promotion criteria are the easiest to leave out and the most important. Without an executable validation command, the agent cannot tell on its own whether a result is correct. Without quantified promotion criteria, it does not know when to stop or which candidate to keep.


## Workflow

1. Create a **separate implementation workspace** for the target task.
2. Define the task contract: objective, constraints, validation command, and promotion criteria.
3. Start an agent session in the implementation workspace.
4. Give the agent `prompts/basic-flow.md`, filled in with the task-specific details.
5. Ask the agent to write a short plan draft to `docs/draft.md` in the implementation workspace.
6. Convert the draft into an executable plan.
7. Implement in small iterations, verifying after each meaningful change.
8. Record candidates, benchmark or evaluation results, profiling evidence, and final promotion decisions.
