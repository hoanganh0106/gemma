# Official scoring evidence observed 2026-09-23

Public endpoint: https://micro2026-api.duckdns.org:7777/api/leaderboard

This endpoint is referenced by the official leaderboard page source at https://raw.githubusercontent.com/micro2026-moa/micro2026-moa.github.io/main/leaderboard.html (API_URL).

Snapshot: `leaderboard-20260923T0650Z.json`, captured 2026-09-23 06:49:42 UTC (filename rounded to 06:50), 21,339 bytes, SHA256 `EC717B653B72D257E66B8E1236A8E4495DEA46C4F5ED2EF0664EB395C12A17FB`.

The API exposes one global `baselineCycles` tuple, not a per-version baseline map:

- `ops::sliding_project_qkv`: 250,514
- `ops::sliding_attention_output`: 404,633
- `ops::decoder_feedforward`: 3,703,473

The top published row at capture was `ff6c2353`, version `0.8.1`, with cycles 73,366 / 33,020 / 201,948 and score 9.155112220304128. Applying the geometric-mean formula to the exposed baseline reproduces that score exactly.

Read-only `moa-submitter status 7cffa30b` and `moa-submitter log 7cffa30b`, observed around 06:49 UTC, independently confirmed a current evaluator example from this team's completed submission. The grading log reports detected `furiosa-opt-std 0.8.1`, NPU job 77114, 3 cases x 3 independently seeded runs, every output comparison PASS, and:

- QKV: median 92,419 from [101675, 92419, 92007].
- Attention output: median 45,109 from [45725, 44268, 45109].
- FFN: median 221,526 from [222095, 220819, 221526].
- Official displayed score: 7.4077. Recalculation from the exposed baseline: 7.407717941283335.

Therefore this public global baseline tuple reproduces an observed current 0.8.1 full-nine-check score. It is appropriate for explicitly labeled illustrative score calculations. The endpoint does not prove separate baseline identities, baseline harness hashes, or a promised future final-evaluation baseline for each supported SDK.

The team's displayed leaderboard best was the older `b90219f5` (2026-09-20), cycles 81,824 / 38,961 / 213,727, score 8.198053797686157, and `skeleton: null`. Do not present that row as a measurement of the current source or current fixture.

The latest submission visible during the read-only status check was `724e79df`, still building. No MOA submission was made, and no status changes were requested by this research task.

Official rules: https://micro2026-moa.github.io/index.html and https://micro2026-moa.github.io/leaderboard.html state that final Round 1 evaluation uses each team's last submission before September 30, 2026, 23:59 AoE. Leaderboard best scores are reference values; official rankings are obtained by rerunning submissions under common conditions. Do not leave an experimental submission as the final entry.
