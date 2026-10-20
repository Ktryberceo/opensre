# Scheduler capacity benchmark

Run the scheduler capacity baseline from the repository root:

```bash
uv run python -m tests.benchmarks.scheduler.capacity_benchmark
```

The benchmark uses a fake task runner and delivery sink, with no LLM, network,
or provider calls. It writes its report to
`docs/benchmarks/scheduler-capacity-baseline.json`.

To write a report elsewhere:

```bash
uv run python -m tests.benchmarks.scheduler.capacity_benchmark --output /tmp/scheduler-baseline.json
```
