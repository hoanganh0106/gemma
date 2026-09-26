import json
import sys
from pathlib import Path


path = Path(sys.argv[1])
data = json.loads(path.read_text())
instructions = data["instructions"]

rows = []
for ins in instructions:
    begin = ins["lifetime"]["begin"]
    end = ins["lifetime"]["end"]
    rows.append(
        (
            end - begin,
            begin,
            end,
            ins["tpe"],
            ins["index"],
            ins.get("description", "").replace("\n", " "),
        )
    )

print(f"makespan={max(r[2] for r in rows)} instructions={len(rows)}")
print("\nLONGEST")
for row in sorted(rows, reverse=True)[:30]:
    print("\t".join(map(str, row)))

print("\nTAIL")
for row in sorted(rows, key=lambda r: r[2], reverse=True)[:30]:
    print("\t".join(map(str, row)))
