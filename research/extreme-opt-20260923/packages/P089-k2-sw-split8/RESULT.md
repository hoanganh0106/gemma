# P089 sw split8 probe

Parent: P078-k2-tail120. The `sw` narrow factor changed from `H/4%30, H%4` to `H/8%15, H%8` while keeping the same 120 elements and the existing widen.

Compiler gate: PASS at the identical 21,337 / 42 schedule, but the release binary SHA256 is exactly `390B0501301B94A400DBD5BD0DC0FB96151770D16B137DC0BD56224B970A34C6`, byte-identical to P078 sw-Sub. No runtime submission was made because this is compiler canonicalization, not a new candidate.
