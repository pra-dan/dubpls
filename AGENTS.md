## Agents Overview

This project is configured to be used with AI coding agents inside Cursor. Agents should follow the rules defined in `.cursor/rules/` (if present) and respect the project's existing structure and tooling.


## Python environment

This project uses a Conda environment named `whisperx2`.

All Python commands must use the Conda environment interpreter directly.

Python executable: `/home/prashant/anaconda3/envs/whisperx2/bin/python`

Do NOT use `python` or `pip` without an explicit path.

Always run pip as:
`/home/prashant/anaconda3/envs/whisperx2/bin/python -m pip`

## Primary Agent

- **Purpose**: General development assistant for this repository (answering questions, editing code, running tools).
- **Scope**: Entire project under the workspace root.
- **Guidelines**:
  - Prefer small, targeted changes per edit.
  - Run tests or linters when modifying core logic, where practical.
  - Avoid committing or pushing changes unless explicitly requested by the user.

### Task
I need to improve the translation part of dubpls such that it uses multiple agents/subagents to run and improve the submodules of the pipeline. The following steps should preferrably be implemented, handled and improved by individual agents. These agents are in the form of those supported by Cursor and Claude Code. The tasks to be handled include:
- implement and improve gender classification using `/home/prashant/Documents/dubpls/audio_classification.py` as reference.
- extract context using `/home/prashant/Documents/dubpls/vlm.py` as reference.
- implement new features to improve translation.
- run the modules individually/together to obtain final translation, saved in the same json that contained the test English text. 
- compare the result with GT (french text present in same json as member of "gt_fr") and provides feedback on reducing the diff between the translation and GT. This feedback is used to run above steps to improve each submodule.

## Status
Currently, the pipeline runs sequentially in 2 stages because the available VRAM is only 12 GB and both VLM and Translation LLM cannot be loaded together.

### Automated (recommended)
Use `automate_pipeline.py` to run the full pipeline with zero manual intervention. It manages Docker container lifecycle, health checks, and stage sequencing automatically:

```bash
# Full pipeline (first 5 segments)
/home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --total 5

# Dry-run (validate paths & Docker without processing)
/home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --dry-run

# Skip VLM stage (reuse existing stage1 JSON, run translation only)
/home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --skip-stage1

# Skip translation (run only audio classification + VLM context)
/home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --skip-stage2
```

### Manual fallback
Stage 1: Context Extraction — Start the VLM container (`docker-compose-minicpm2.6.yaml`), then run: `conda run -n whisperx2 python eval_and_improve.py --stage 1`. This generates `media/en_fr_eval_stage1.json`.

Stage 2: Translation — Stop the VLM container, start the Translation LLM (`docker-compose.yaml`), then run: `conda run -n whisperx2 python eval_and_improve.py --stage 2`.

<!-- Run translation part in single go using `/home/prashant/anaconda3/envs/whisperx2/bin/python automate_pipeline.py --total 5` -->

