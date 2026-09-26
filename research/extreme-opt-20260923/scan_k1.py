import json, glob, os

base = "d:/Project/furiosa-opt-gemma4-12B-main/research/extreme-opt-20260923/packages"
rows = []
for f in glob.glob(base + "/*/arena-*.summary.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not d.get("all_checks_pass"):
        continue
    med = d.get("medians")
    if not med or len(med) != 3:
        continue
    pkg = os.path.basename(os.path.dirname(f))
    jid = os.path.basename(f).replace("arena-","").replace(".summary.json","")
    rows.append((pkg, jid, med[0], med[1], med[2]))

# Sort by K1 then K2 then K3
rows.sort(key=lambda r: (r[2], r[3], r[4]))
print(f"{'package':42} {'job':10} {'K1':>8} {'K2':>8} {'K3':>8}")
print("-"*80)
for r in rows[:30]:
    print(f"{r[0]:42} {r[1]:10} {r[2]:>8} {r[3]:>8} {r[4]:>8}")
