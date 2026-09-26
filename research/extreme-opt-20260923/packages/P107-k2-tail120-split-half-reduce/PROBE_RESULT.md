# P107 split half-reduce adaptation

P107 changed each half contract to reduce directly into a 60-row mapping before committing into the H/120 tile.

SDK 0.8.1 rejects the commit at compile time with `Slice size must be preserved`. The consumer cannot accept a reduced slice mapping as a partial commit. This proves the required half ownership adaptation is unavailable through the current commit API; no graph or Arena result exists.
