# Platform Biography Visualization — Data Gap

The platform biography figure requires per-platform per-step data:
- `platform_id, step, n_mainstream, n_extremist, utility_mainstream, utility_extremist`

Currently the model logs only aggregate per-governance counts, not
per-platform breakdowns. To enable this visualization, add per-platform
logging in `MiniTiebout._record_step_log()` that records community
count by type and utility by type for each platform at each step.
