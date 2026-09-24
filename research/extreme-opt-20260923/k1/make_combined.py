from pathlib import Path
import ast, shutil
root=Path(__file__).resolve().parent
baseline=root.parent/'baseline'
tree=ast.parse((root/'make_fused.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fuse')
scope={}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'fuse','exec'),scope)
s=(root/'p14-rope-head-broadcast/src/device/sliding/qkv_head_local.rs').read_text()
s=scope['fuse'](s)
p=root/'p15-fused-rope-head-broadcast';p.mkdir(exist_ok=True)
shutil.copytree(baseline/'src',p/'src',dirs_exist_ok=True)
for f in ('Cargo.toml','Cargo.lock','rust-toolchain.toml'):shutil.copy2(baseline/f,p/f)
(p/'src/device/sliding/qkv_head_local.rs').write_text(s)
