# Copilot Instructions for MonteCarloX

Always prioritize the repository concept in `.github/AI_SUMMARY.md`.

## Primary concept

MonteCarloX should remain **concise, modular, and compact**.

- Concise: keep APIs minimal and avoid overlap.
- Modular: maintain strict separation of concerns.
- Compact: prefer general composable primitives over specialized helpers.

## Practical defaults

- Reuse existing abstractions before introducing new ones.
- Favor one general function over multiple one-off convenience APIs.
- Keep algorithm code model-agnostic.
- Keep notebook/examples aligned with exported framework APIs.

When in doubt, choose the smallest change that preserves generality and composability.
