from pathlib import Path
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'scripts'))
import generate_references as generator

generator.FIXTURE = Path(__file__).resolve().parent / 'fixtures.safetensors'
generator.generate()
