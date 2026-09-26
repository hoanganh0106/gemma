# P108 K2 column half contract probe

## Result

**PASS: compile legality.** The SDK 0.8.1 release build completed for the
unused probe function.

The API can represent a half-Qs weight and activation tile
(`Qs%128`) while keeping the contract output at the full `H%120` tile. The
probe also type-checks the corresponding fetch, collect, contract, and commit
mapping.

## What this proves

- A column split is expressible at the input and contract type level.
- The existing H/120 output ownership can be retained.
- The result is only a compile gate; no runtime graph, correctness, schedule,
  or Arena result was produced.

## Accumulation gate

**PASS: compile legality.** Two full H/120 `bf16` partial results can be
fetched, cast, added in the vector unit, and committed back to the same full
H/120 ownership. The first attempt needed an explicit VRF shape annotation;
after that plumbing fix the release build completed.

The end-to-end compile chain also passed: two independent half-Qs contract
producers can feed the full H/120 accumulator in sequence. The latest release
build completed after adding that chain.

This proves only type-level composition. It does not prove graph legality,
producer/consumer lifetimes, schedule cost, correctness, or that the extra
contract and accumulation work beats P078.

## Next gate

Use graph construction on the two-producer chain to test axis ownership and
lifetimes before any Arena run. If graph construction passes, measure the
static schedule against P078 before considering runtime correctness.

## Graph integration attempt

The direct integration attempt was rejected at the API boundary:

- a `TensorView` of the full TRF cannot be passed where `contract_outer`
  requires a typed `TrfTensor` half;
- the proposed tile syntax for the multi-axis TRF is not a valid type-level
  tile declaration;
- a completed `DmTensor` cannot be copied into an output view with a plain
  `commit` call.

The control `contract_tile` was restored after this probe. This is a real
mechanism boundary: the next iteration needs an explicit half-Qs TRF/DM
intermediate with a legal producer and commit mapping, rather than slicing a
full tensor view at the contract call site.

## Explicit intermediate follow-up

The ownership probe then passed after using the SDK's required allocation
pattern:

- HBM weight view -> typed half-Qs DM via `to_dm`;
- activation DM view -> preallocated typed half-Qs DM via `to_dm_view`.

The release build completed. This establishes the materialization primitive;
the remaining work is a half-Qs quantizer producing a typed `TrfTensor`, then
graph construction with two legal half tensors.

The half-Qs quantizer and full typed chain now also compile:

`HBM weight view -> half DM weight + half DM activation -> half TRF -> half
contract -> full H/120 result`.

This is still compile-only. The next probe must invoke the chain from the
actual graph path and test the two-half lifetime/ownership interaction.

## Runtime graph probe

The isolated execution failed with:

`gather src residue has no live target axis`

However, a control run of the same P108 binary without the probe flag fails
with the identical message. Therefore this result cannot be attributed to the
half-contract path. The package graph harness is already failing before the
stage toggle can isolate the new operations. No Arena claim is made from this
probe; P078 remains the authoritative control and must be rerun from its own
frozen package for comparison.
