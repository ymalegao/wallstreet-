# Pipeline v0 clarifications

No implementation questions remain for v0. These decisions resolve the design-boundary questions that came up during implementation.

- `MarketSnapshot` and `PortfolioState` are coordinator-supplied typed inputs, not source nodes. `SnapshotProvider` reads the preloaded point-in-time history outside the DAG.
- `Execution` also reads `MarketSnapshot` to record the last visible close used as each order's reference price.
- The acceptance suite stays on the synthetic fixture and uses a mocked HTTP transport. The CLI also supports a separate historical Alpaca download for a manual real-data run.
- Shared state stays behind explicit services: `BarHistory` is read-only, while `Broker` is mutable and is injected only into `Execution`. Nodes communicate through typed messages instead of a global mutable dictionary.
- The Alpaca downloader stamps daily bars with the project's modeled close-plus-15-minute availability. Alpaca's historical endpoint does not provide the vendor's original first-seen timestamp or a historical point-in-time adjustment archive; a downloaded historical run is a smoke test, not proof of point-in-time alpha.
