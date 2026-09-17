"""Download pinned official assets without modifying the AirSim environment."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'downloads' / 'local_llm'
REVISION = '23749fefcc72300e3a2ad315e1317431b06b590a'
ASSETS = {
    'llama-b10941-bin-win-cpu-x64.zip': 'https://github.com/ggml-org/llama.cpp/releases/download/b10941/llama-b10941-bin-win-cpu-x64.zip',
    'Qwen3-0.6B-Q8_0.gguf': f'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/{REVISION}/Qwen3-0.6B-Q8_0.gguf',
    'MODEL_LICENSE': f'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/{REVISION}/LICENSE',
}

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = {'model_revision': REVISION, 'runtime': 'b10941 cpu x64', 'assets': []}
    for name, url in ASSETS.items():
        path = DEST / name
        if not path.exists():
            print('Downloading ' + name, flush=True)
            partial = path.with_suffix(path.suffix + '.partial')
            with urllib.request.urlopen(url, timeout=60) as src, partial.open('wb') as dst:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    dst.write(block)
            partial.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if name == 'Qwen3-0.6B-Q8_0.gguf' and digest != '9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031':
            raise ValueError('Model SHA256 differs from pinned official LFS object')
        manifest['assets'].append({'file': name, 'url': url, 'sha256': digest, 'bytes': path.stat().st_size})
        if name.endswith('.zip'):
            with zipfile.ZipFile(path) as archive:
                runtime = DEST / 'runtime'
                for member in archive.infolist():
                    if not (runtime / member.filename).resolve().is_relative_to(runtime.resolve()):
                        raise ValueError('Archive path outside runtime')
                archive.extractall(runtime)
        print('Ready ' + name, flush=True)
    (DEST / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
