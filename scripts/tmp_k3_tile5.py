from pathlib import Path


path = Path("src/device/shared/mlp.rs")
text = path.read_text(encoding="utf-8")
fn_start = text.index("fn project_down_matrix(")
block_start = text.index("{\nlet block_scale:", fn_start)
suffix_start = text.index("    // Keep both compensation parts", block_start)
delimiter = "\n}\n{\nlet block_scale:"
first_end = text.index(delimiter, block_start) + len("\n}")
template = text[block_start:first_end]

if "m![H % 15 = 3" not in template or ", 3," not in template:
    raise SystemExit("unexpected project_down_matrix tile-3 template")

blocks = []
for offset in (0, 5, 10):
    block = template.replace("m![H % 15 = 3", "m![H % 15 = 5")
    block = block.replace(", 3,", ", 5,")
    block = block.replace(">(0)", f">({offset})")
    blocks.append(block)

replacement = "\n".join(blocks) + "\n}\n\n\n"
new_text = text[:block_start] + replacement + text[suffix_start:]
path.write_text(new_text, encoding="utf-8")
