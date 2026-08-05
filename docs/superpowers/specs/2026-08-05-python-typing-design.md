# Python Typing Completion Design

## Objective

Complete and correct parameter and return type annotations for every function
in Python files under `algorithm/` and `tests/`, without changing runtime
behavior.

## Scope

- Cover all 50 Python files currently present under `algorithm/` and `tests/`.
- Cover module functions, nested functions, methods, test functions,
  constructors, `__call__`, and `forward` methods.
- Annotate every parameter except `self`, `cls`, and `**kwargs`.
- Annotate every return type, including explicit `-> None` for functions that
  do not return a value.
- Review existing annotations and correct them when they conflict with default
  values, function bodies, callers, overrides, or actual returned values.

## Type Rules

- Target Python 3.10 or newer.
- Continue the project's existing modern syntax, including `X | None` and
  built-in generics such as `list[T]`.
- Do not add an explicit `Any` annotation.
- Bare `dict`, `list`, and `tuple` annotations are permitted when their element
  structure is dynamic or not worth modeling for this task.
- `**kwargs` may remain unannotated.
- Prefer concrete project classes, enums, NumPy arrays, PyTorch tensors,
  callables, unions, and container element types when these are supported by
  the implementation and call sites.
- Preserve valid existing annotations where possible.

## Dependency-Layer Execution

The work will proceed from low-level dependencies to high-level orchestration:

1. Utilities and visualization helpers under `algorithm/utils/`.
2. Neural-network layers and model interfaces under `algorithm/nn_models/`.
3. Environment wrappers under `algorithm/env_wrapper/`.
4. Configuration, batch storage, replay storage, and imitation helpers.
5. Agent management and core SAC implementation.
6. Option-Critic implementation under `algorithm/oc/`.
7. Test helpers, test network definitions, and test functions under `tests/`.

Each layer will be reviewed using both function implementation and downstream
usage. Overrides will retain compatible parameter and return annotations with
their base definitions.

## Validation

Each dependency layer will be checked before moving to the next one:

- Parse all changed files to catch annotation syntax and import errors.
- Run an AST audit that reports any function with a missing required parameter
  annotation or missing return annotation.
- Run tests relevant to the changed layer when independently available.

After all layers are complete:

- Run `python -m compileall algorithm tests`.
- Run the annotation completeness audit across both directories.
- Search the diff for newly introduced explicit `Any` annotations.
- Run `python -m pytest tests`.
- Inspect the final diff to confirm that changes are limited to typing-related
  imports and function annotations, except for a necessary correction revealed
  by an invalid existing annotation.

## Non-Goals

- No algorithm changes, refactoring, formatting sweep, or dependency upgrade.
- No requirement to make the repository pass a newly introduced strict mypy or
  Pyright configuration.
- No edits to generated model artifacts, backups, caches, environment configs,
  or the user's untracked `AGENTS.md`.
- No attempt to eliminate implicit element types from every permitted bare
  container annotation.

## Acceptance Criteria

- Every in-scope function has all required parameter annotations and a return
  annotation.
- Existing annotations reviewed during the dependency-layer pass agree with
  observed runtime contracts.
- No explicit `Any` annotation is introduced.
- Python compilation succeeds for both directories.
- The full available test suite completes without regressions, or any
  environment/dependency-related baseline failures are reported precisely with
  their output.
