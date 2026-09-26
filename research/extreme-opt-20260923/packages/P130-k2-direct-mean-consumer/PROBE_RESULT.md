# P130 K2 direct mean plus consumer rewrite

## Hypothesis

Change both sides of the P129 experiment: directly reduce the `H / 120`
partials into a scalar mean, then broadcast that scalar for the RMS and final
passes.

## Gate 1 result

The attempted producer/consumer rewrite did not compile. The direct reduction
introduced compiler padding and a different slice shape, and the following
errors remained:

```text
expected DmTensor<..., Vc, 1#8, 1#8>
found DmTensor<..., Broadcast<1>, Padding<...>>

PositionVectorFinal does not satisfy CanApplyCommit

expected VrfTensor<..., Vc, Tail, 1#8>
found VrfTensor<..., Pair<Padding<...>, Vr>, ...>
```

No schedule or runtime test was produced.

## Interpretation

Changing the consumer exposes two coupled constraints: reduction padding must
be represented in the type, and the broadcast result does not naturally have
the `Tail` slice geometry consumed by the final pass. The compiler-guided
retry got past the alias errors, but then failed DM allocation:

```text
DmTensor T42's slice extent 8 does not match the device config (slice = 256)
```

The old `switch` plus intra-slice reduction is therefore still the only legal
path found that produces a device-wide DM tensor. A direct-reduction family
would need to keep its result in VRF and add a legal broadcast into the final
consumer, without committing the partial scalar to DM.

The VRF-only follow-up also failed at the final consumer: the scalar VRF
operand does not implement the final stream's branched shape. The compiler
explicitly requires a `VrfTensor::reshape` to restate matching slices, but
the scalar result has only eight slices and cannot be reshaped into the
`Tail` geometry containing `H / 120` groups. This closes the current direct
reduction variant under SDK 0.8.1.
