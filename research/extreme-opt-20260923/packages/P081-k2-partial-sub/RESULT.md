# P081 partial on Sub

Parent: P078-k2-tail120. Only the sum-of-squares `partial` vector pipeline was moved from `device.main` to `device.sub`; mapping, arithmetic, epsilon hoist, `sw`-Sub path, inputs, and harness were preserved.

Compiler gate: PASS, but schedule worsened from P078 `21,337 / 42` to `21,341 / 42` cycles/instructions. Since the proposed overlap has no static advantage and adds a context handoff risk, reject before binary build or Arena measurement.
