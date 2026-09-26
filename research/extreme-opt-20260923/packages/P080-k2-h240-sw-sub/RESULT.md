# P080 H/240 sw on Sub

Parent: P041-k2-direct-tail-dma. The only kernel source change is the `sw = weight_scale * rms_weight` vector pipeline: its producer runs on `device.sub` and terminates with Sub-context `to_vrf()`. The validated H/240 tail layout, arithmetic, inputs, and harness are preserved.

Static compiler gate: PASS, 21,290 cycles / 42 instructions. Schedule SHA256: `B52B97ABF770D180B13B94B3580BE9F2F6C6E731BC30AF9014A53DD936104FE7`.

Binary SHA256: `82CBA09E446FA6FD36B3442B6F822CBE071972E6429BB7736300D43CE7723105`.

Arena job 92337 passed all 15/15 checks. K2 median: 38,321 cycles `[38,321, 38,754, 37,962]`. Fresh P041 control 92348 was submitted immediately afterward and is pending; no promotion decision is made from the candidate alone.

Paired control completed: P041 job 92348 passed 15/15 at 37,322 cycles `[37,322, 37,052, 38,000]`. P080 loses by 999 cycles (2.68%); reject the H/240 sw-Sub hypothesis. The Sub context benefit seen with P078 does not transfer to this H/240 schedule.
