## What does this PR do?

<!-- One sentence: what changed and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / cleanup
- [ ] Documentation
- [ ] CI / tooling

## Verification

<!-- What did you run to confirm it works? Paste the output. -->

```
uv run ruff check app tests scripts
uv run pytest tests -q
uv run pytest tests --cov=app --cov-fail-under=65 -q
```

## Checklist

- [ ] Tests added or updated for changed behavior
- [ ] Documentation updated where relevant (README, USAGE.md, CHANGELOG.md)
- [ ] No API keys, tokens, local datasets, or generated model artifacts committed
- [ ] `uv lock --check` passes
- [ ] Follows the [Code of Conduct](../CODE_OF_CONDUCT.md)

## Related issues

<!-- Closes #NNN -->
