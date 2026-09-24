"""Create a diagnostic-only host collector; all device source remains identical."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
parent = root / 'packages/P005-K1-081-K2K3-P005-final'
dest = root / 'packages/D001-parent-device-profile'
assert not dest.exists(), dest
dest.mkdir()
shutil.copytree(parent / 'src', dest / 'src')
for name in ['Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'fixtures.safetensors', 'remote_entrypoint.sh']:
    shutil.copy2(parent / name, dest / name)
p = dest / 'src/bin/test_kernels.rs'
s = p.read_text()
s = s.replace('struct FieldExtractor {\n', 'struct FieldExtractor {\n    name: String,\n')
old = '    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}'
assert s.count(old) == 1
s = s.replace(old, '''    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "name" { self.name = value.to_owned(); }
    }
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "name" { self.name = format!("{value:?}"); }
    }''')
marker = '                self.spans.lock().unwrap().push(Span { cluster, begin, end });'
assert s.count(marker) == 1
s = s.replace(marker, '                println!("PROFILE_SPAN\\t{cluster}\\t{begin}\\t{end}\\t{:?}", extractor.name);\n' + marker)
marker = '    fixture.assert_every_expectation_is_tested();'
s = s.replace(marker, '    println!("DIAGNOSTIC_ONLY device span logging; not an official score sample");\n' + marker)
p.write_text(s)
wrapper = dest / 'remote_entrypoint.sh'
wrapper.write_text(wrapper.read_text().replace('FURIOSA_OPT_PROFILE=info', 'FURIOSA_OPT_PROFILE=trace'))
(dest / 'HYPOTHESIS.md').write_text('Diagnostic only. Device sources, test inputs, references and output comparisons are unchanged from canonical parent. Host collector logs span name/cluster/begin/end; runtime profiling level is trace. Trace instrumentation can change timings, so these cycles must never enter official performance comparisons. Purpose: identify resource costs hidden by the static model.\n')
print(dest)
