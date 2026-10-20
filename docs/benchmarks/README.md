# Scheduler capacity benchmark

Run the scheduler capacity baseline from the repository root:

```bash
uv run python -m tests.benchmarks.scheduler.capacity_benchmark
```

The benchmark uses a fake task runner and delivery sink, with no LLM, network,
or provider calls. The report is written to
`docs/benchmarks/scheduler-capacity-baseline.json`. Use `--output <path>` to
write to a different file.

See [the benchmark instructions](../../tests/benchmarks/README.md) for examples.
