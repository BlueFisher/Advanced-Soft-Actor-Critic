# Python Typing Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and correct parameter and return annotations for every function under `algorithm/` and `tests/` without changing runtime behavior.

**Architecture:** Work from low-level dependencies toward SAC and Option-Critic orchestration. A repository-level AST test defines annotation completeness, while compilation and existing tests protect imports and behavior after each dependency layer.

**Tech Stack:** Python 3.10+, `ast`, pytest, NumPy, PyTorch, Gym/Unity adapters.

## Global Constraints

- Cover every `.py` file under `algorithm/` and `tests/`.
- Annotate every parameter except `self`, `cls`, and `**kwargs`.
- Annotate every function return value.
- Do not introduce an explicit `Any` annotation.
- Bare `dict`, `list`, and `tuple` annotations are permitted.
- `**kwargs` may remain unannotated.
- Preserve runtime behavior and the user's untracked `AGENTS.md`.

---

### Task 1: Annotation completeness contract and utilities

**Files:**
- Create: `tests/test_typing_completeness.py`
- Modify: `algorithm/utils/elapse_timer.py`
- Modify: `algorithm/utils/enums.py`
- Modify: `algorithm/utils/lock.py`
- Modify: `algorithm/utils/operators.py`
- Modify: `algorithm/utils/plot_episode.py`
- Modify: `algorithm/utils/transform.py`
- Modify: `algorithm/utils/visualization/image.py`
- Modify: `algorithm/utils/visualization/option.py`
- Modify: `algorithm/utils/visualization/ray.py`
- Modify: `algorithm/utils/visualization/v_over_options.py`

**Interfaces:**
- Consumes: Python AST nodes from all in-scope source files.
- Produces: `missing_annotations(paths: tuple[Path, ...]) -> list[str]`, a stable audit used by every later task.

- [ ] **Step 1: Add the completeness audit and a utility-layer test**

```python
import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


def missing_annotations(paths: tuple[Path, ...]) -> list[str]:
    missing: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
            if node.args.vararg is not None:
                args.append(node.args.vararg)
            for arg in args:
                if arg.arg not in {"self", "cls"} and arg.annotation is None:
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{arg.arg}")
            if node.returns is None:
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:return")
    return missing


def test_algorithm_utils_annotations_are_complete() -> None:
    paths = tuple((ROOT / "algorithm" / "utils").rglob("*.py"))
    assert missing_annotations(paths) == []
```

- [ ] **Step 2: Run the new test and confirm it fails on existing missing annotations**

Run: `python -m pytest tests/test_typing_completeness.py::test_algorithm_utils_annotations_are_complete -v`

Expected: failure listing utility parameters and return annotations that are currently absent.

- [ ] **Step 3: Annotate utility functions from implementations and callers**

Use `-> None` for mutators and constructors; concrete primitives for timer and enum helpers; `np.ndarray | torch.Tensor` where visualization helpers accept both; `Callable` for decorators and wrapped operations. Preserve bare containers when their heterogeneous structure is dynamic.

- [ ] **Step 4: Verify the utility layer**

Run: `python -m pytest tests/test_typing_completeness.py::test_algorithm_utils_annotations_are_complete -v`

Run: `python -m compileall -q algorithm/utils`

Expected: both commands exit successfully.

- [ ] **Step 5: Commit the utility layer**

```powershell
git add -- tests/test_typing_completeness.py algorithm/utils
git commit -m "type: annotate algorithm utilities"
```

### Task 2: Neural-network layers and model interfaces

**Files:**
- Modify: `tests/test_typing_completeness.py`
- Modify: `algorithm/nn_models/exploration.py`
- Modify: `algorithm/nn_models/policy.py`
- Modify: `algorithm/nn_models/predictions.py`
- Modify: `algorithm/nn_models/q.py`
- Modify: `algorithm/nn_models/representation.py`
- Modify: `algorithm/nn_models/layers/image_layers.py`
- Modify: `algorithm/nn_models/layers/linear_layers.py`
- Modify: `algorithm/nn_models/layers/seq_layers.py`

**Interfaces:**
- Consumes: `torch.Tensor`, `nn.Module`, project enums, observation-shape lists, optional sequence states and masks.
- Produces: concrete tensor, tensor-tuple, module, and shape return contracts used by SAC and test networks.

- [ ] **Step 1: Add `test_algorithm_nn_models_annotations_are_complete()` using `missing_annotations()` and `algorithm/nn_models/**/*.py`**
- [ ] **Step 2: Run that test and confirm the failure reports missing network annotations**
- [ ] **Step 3: Annotate constructors, `forward`, `__call__`, properties, and helpers; validate every tensor tuple against unpacking call sites and every override against its base class**
- [ ] **Step 4: Run the neural-model audit and `python -m compileall -q algorithm/nn_models`**
- [ ] **Step 5: Run `python -m pytest tests/test_typing_completeness.py -v` and commit with `type: annotate neural network models`**

### Task 3: Environment wrappers

**Files:**
- Modify: `tests/test_typing_completeness.py`
- Modify: `algorithm/env_wrapper/env_wrapper.py`
- Modify: `algorithm/env_wrapper/gym_wrapper.py`
- Modify: `algorithm/env_wrapper/obs_preprocessor_wrapper.py`
- Modify: `algorithm/env_wrapper/offline_wrapper.py`
- Modify: `algorithm/env_wrapper/test_offline_wrapper.py`
- Modify: `algorithm/env_wrapper/test_wrapper.py`
- Modify: `algorithm/env_wrapper/unity_wrapper.py`

**Interfaces:**
- Consumes: environment configs, `Path | None`, NumPy observations/actions, Unity decision/terminal steps, multiprocessing connections.
- Produces: consistent reset/step tuples and environment metadata consumed by `sac_main.py` and agent managers.

- [ ] **Step 1: Add an environment-wrapper completeness test**
- [ ] **Step 2: Run it and confirm it fails for missing wrapper annotations**
- [ ] **Step 3: Trace the abstract wrapper contract through every override; annotate lifecycle methods, worker functions, reset/step results, and callback parameters without narrowing valid backend values**
- [ ] **Step 4: Run the wrapper audit, `python -m compileall -q algorithm/env_wrapper`, and `python test_env.py` when its configured backend is available**
- [ ] **Step 5: Commit with `type: annotate environment wrappers`**

### Task 4: Configuration and replay infrastructure

**Files:**
- Modify: `tests/test_typing_completeness.py`
- Modify: `algorithm/config_helper.py`
- Modify: `algorithm/batch_buffer.py`
- Modify: `algorithm/replay_buffer.py`
- Modify: `algorithm/imitation_base.py`

**Interfaces:**
- Consumes: YAML-derived containers, NumPy arrays, PyTorch tensors, devices, replay indices, priorities.
- Produces: merged configuration values, sampled batches, replay updates, and imitation losses.

- [ ] **Step 1: Add a completeness test for these four modules**
- [ ] **Step 2: Run it and confirm current missing annotations are detected**
- [ ] **Step 3: Annotate config conversion recursion, replay tree operations, sampling/update methods, nested optimizer helpers, and all return tuples from their actual construction and unpacking sites**
- [ ] **Step 4: Run the scoped audit, compile the four modules, then run `python -m pytest tests/test_sac_params.py tests/test_oc_params.py -v`**
- [ ] **Step 5: Commit with `type: annotate configuration and replay infrastructure`**

### Task 5: Agent management and core SAC

**Files:**
- Modify: `tests/test_typing_completeness.py`
- Modify: `algorithm/agent.py`
- Modify: `algorithm/sac_base.py`
- Modify: `algorithm/sac_main.py`

**Interfaces:**
- Consumes: typed wrapper outputs, replay batches, network tensor contracts, config dictionaries, optional offline actions and sequence states.
- Produces: action-selection tuples, episode dictionaries, training losses, checkpoint operations, and the SAC main loop.

- [ ] **Step 1: Add a completeness test for the agent and SAC modules**
- [ ] **Step 2: Run it and confirm missing annotations are reported**
- [ ] **Step 3: Annotate from the complete data flow `sac_main.py -> agent.py -> sac_base.py`; explicitly verify optional defaults, episode dictionaries, recurrent states, policy/Q loss tuples, and nested optimizer functions**
- [ ] **Step 4: Run the scoped audit, compile these modules, and run `python -m pytest tests/test_sac_params.py tests/test_sac_main.py -v`**
- [ ] **Step 5: Commit with `type: annotate SAC training core`**

### Task 6: Option-Critic layer

**Files:**
- Modify: `tests/test_typing_completeness.py`
- Modify: `algorithm/oc/oc_agent.py`
- Modify: `algorithm/oc/oc_batch_buffer.py`
- Modify: `algorithm/oc/oc_main.py`
- Modify: `algorithm/oc/option_base.py`
- Modify: `algorithm/oc/option_selector_base.py`

**Interfaces:**
- Consumes: SAC contracts plus option indices, termination masks, low-level and key sequence states.
- Produces: option-selection actions, option batches, terminal losses, and hierarchical training results.

- [ ] **Step 1: Add an Option-Critic completeness test**
- [ ] **Step 2: Run it and confirm missing annotations are reported**
- [ ] **Step 3: Annotate overrides compatibly with SAC bases; trace option/termination tensors and long tuple returns through all unpacking sites**
- [ ] **Step 4: Run the scoped audit, compile `algorithm/oc`, and run `python -m pytest tests/test_oc_params.py tests/test_oc_main.py -v`**
- [ ] **Step 5: Commit with `type: annotate option critic modules`**

### Task 7: Test code and repository-wide verification

**Files:**
- Modify: `tests/test_typing_completeness.py`
- Modify: `tests/get_synthesis_data.py`
- Modify: `tests/nn_conv_attn.py`
- Modify: `tests/nn_conv_rnn.py`
- Modify: `tests/nn_conv_vanilla.py`
- Modify: `tests/nn_oc_combined.py`
- Modify: `tests/test_oc_main.py`
- Modify: `tests/test_oc_params.py`
- Modify: `tests/test_sac_main.py`
- Modify: `tests/test_sac_params.py`

**Interfaces:**
- Consumes: production network base classes and pytest fixtures.
- Produces: fully annotated synthetic-data helpers, test networks, test functions, and a repository-wide annotation gate.

- [ ] **Step 1: Add the final repository-wide test**

```python
def test_all_in_scope_annotations_are_complete() -> None:
    paths = tuple((ROOT / "algorithm").rglob("*.py")) + tuple(
        (ROOT / "tests").rglob("*.py")
    )
    assert missing_annotations(paths) == []
```

- [ ] **Step 2: Run it and confirm it fails on remaining test-code annotations**
- [ ] **Step 3: Annotate synthetic-data helpers, test network overrides, pytest tests, and local helpers; use `-> None` for assertion-only tests**
- [ ] **Step 4: Run final verification**

```powershell
python -m compileall -q algorithm tests
python -m pytest tests/test_typing_completeness.py -v
git diff --check
git diff -U0 | Select-String -Pattern '^\+.*\bAny\b'
python -m pytest tests
```

Expected: compilation, annotation audit, diff check, and tests exit successfully; the explicit-`Any` search prints no added annotation. If an existing environment dependency prevents a test from running, preserve its exact output and distinguish it from a typing regression.

- [ ] **Step 5: Review all changed signatures against the design acceptance criteria and commit with `type: complete Python annotations`**
