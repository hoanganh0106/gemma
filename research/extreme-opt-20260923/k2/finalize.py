from pathlib import Path
import json,hashlib,difflib
root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
candidate=root/'candidates/C22-fixed-rms-ring16'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
changed=[]
for p in sorted((baseline/'src').rglob('*.rs')):
 rel=p.relative_to(baseline)
 q=candidate/'isolated'/rel
 assert q.exists(),rel
 if sha(p)!=sha(q): changed.append(str(rel).replace('\\','/'))
assert changed==['src/device/sliding/output31.rs'],changed
locks={}
for name in ['Cargo.toml','Cargo.lock','rust-toolchain.toml']:
 assert sha(baseline/name)==sha(candidate/'isolated'/name),name
 locks[name]=sha(baseline/name)
old=(baseline/'src/device/sliding/output31.rs').read_text().splitlines(True)
new=(candidate/'output31.rs').read_text().splitlines(True)
(candidate/'source.diff').write_text(''.join(difflib.unified_diff(old,new,fromfile='baseline/src/device/sliding/output31.rs',tofile='candidate/src/device/sliding/output31.rs')))
summary={
 'best_static_candidate':'C22-fixed-rms-ring16',
 'source_sha256':sha(candidate/'output31.rs'),
 'schedule_sha256':sha(candidate/'schedule.json'),
 'baseline_static_cycles':24718,'candidate_static_cycles':22317,
 'baseline_instructions':44,'candidate_instructions':42,
 'cycles_reduction_percent':100*(24718-22317)/24718,
 'illustrative_single_kernel_geomean_gain_percent':100*((24718/22317)**(1/3)-1),
 'changed_source_files':changed,'dependency_hashes':locks,
 'runtime_status':'Parent coordinates frozen integrated hardware packages; no Arena submissions made by K2 subtask.',
}
(root/'selection.json').write_text(json.dumps(summary,indent=2))
rows=json.loads((root/'ledger.json').read_text())
report='''# K2 campaign result

## Selection for integrated hardware validation

**C22-fixed-rms-ring16: 22,317 static cycles / 42 instructions**, versus the frozen source's 24,718 / 44. This is 2,401 cycles (9.713%) lower in the compiler model. Runtime correctness and stable speed must be decided using the parent's frozen full-kernel packages.

The candidate combines direct VRF for SW and inverse RMS, removal of one identity reshape, sixteen 240-value tail groups, and explicit fixed-topology gathering of all sixteen group partials before summation. The scalar epsilon/sqrt/div sequence, both FP8 levels, projection BF16 boundary, every channel scale, every RMSNorm weight and every residual input remain present.

Only `src/device/sliding/output31.rs` differs from the frozen baseline in the candidate's full source tree. Cargo.toml, Cargo.lock and rust-toolchain.toml are byte-identical. `selection.json` records hashes; `candidates/C22-fixed-rms-ring16/source.diff` contains the exact complete diff.

## Main evidence

- C14 proved the mechanism: implicit 2,579-cycle cross-slice reduction/scalar pass became an explicit 289-cycle ring8 gather+sum plus the unchanged 281-cycle scalar pass, reducing total to 22,400 cycles.
- C15 used ring16 with 240 values per group and reduced total further to 22,317 cycles.
- C22 kept that schedule length while replacing CustomBroadcast with fixed Broadcast1 and removing one routing-table DMA/instruction.
- A larger contiguous weight tile (60x512 instead of 120x256) worsened DMA and x replication costs.
- Exchanging only scalar means across clusters required extra synchronization and a costly strided final store, losing to the improved cluster-0 tail.
- Sum-based normalization and direct reciprocal arithmetic did not improve the schedule; neither is included.
- The x broadcast experiments attempted fewer HBM copies, but generic routing added setup and/or exposed excessive routing time. Fixed topology cannot simply treat padded uninitialized slices as actual source values.

## All screened candidates

'''
for r in rows:
 suffix=f"{r['cycles']:,} cycles / {r['instructions']} instructions" if 'cycles' in r else r['status']
 report+=f"- {r['candidate']}: {suffix}.\n"
report+='''
## Boundaries

No root source was edited by this subtask, no Arena or MOA job was submitted by it, and no fixture/harness/scoring/timing changes are part of any candidate. Dynamic tensor values are always consumed at runtime. Compiler PASS is separate from numerical PASS. Rejected sources remain isolated as experiment evidence.

Selected source, complete schedule, compile log, exact diff, full copied source tree, dependency locks, semantic derivation for the explicit-gather method, and the per-candidate ledger are retained in this directory.
'''
(root/'RESULTS.md').write_text(report)
print(json.dumps(summary,indent=2))
