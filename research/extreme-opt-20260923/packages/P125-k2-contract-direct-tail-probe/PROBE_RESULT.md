# P125 K2 direct contract-to-tail probe

This probe replaced P122's unsafe post-contract reshape with a direct
`TailDm<bf16>` output allocation passed into `contract_tile`.

The SDK rejected the commit mapping at compile time: the contract pipeline's
`OutputClusters/Rows` destination view does not match the tail
`Vc/Tail` destination view. Direct contract-to-tail ownership is therefore
not legal without a new contraction/reduction mapping.

P122 remains the only candidate with the 20,495-cycle static result, but its
unsafe reshape still requires runtime correctness validation.
