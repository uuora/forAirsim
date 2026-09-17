# 本机 MessagePack 编译扩展重建

2026-09-15。环境为Windows x64、CPython 3.10、Visual Studio C++工具链。保持`msgpack-python==0.5.6`及`msgpack-rpc-python==0.4.1`兼容，重新生成旧Cython源码解决其原生成C++不兼容当前Python的问题。直接pip安装旧源码可能“成功”但没有编译扩展，必须检查wheel含`.pyd`。

以下在`D:\forAirsim`运行；只构建wheel，最后安装命令单列。现成wheel可直接按《传感器验收与限制.md》重新安装。

```powershell
.venv\python.exe -m pip download msgpack-python==0.5.6 --no-deps --dest downloads\python-wheels
.venv\python.exe -m pip install Cython==0.29.37 --no-deps --target downloads\msgpack-build-tools
@'
import hashlib, os, runpy, sys, tarfile, zipfile
from pathlib import Path
root = Path.cwd().resolve()
archive = root / 'downloads/python-wheels/msgpack-python-0.5.6.tar.gz'
expected = '378cc8a6d3545b532dfd149da715abae4fda2a3adb6d74e525d0d5e51f46909b'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
destination = root / 'downloads/msgpack-source'
destination.mkdir(exist_ok=True)
with tarfile.open(archive) as bundle:
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents or member.issym() or member.islnk():
            raise RuntimeError('Unsafe archive member')
    bundle.extractall(destination)
sys.path.insert(0, str(root / 'downloads/msgpack-build-tools'))
from Cython.Compiler.Main import compile
os.chdir(destination / 'msgpack-python-0.5.6')
for source in Path('msgpack').glob('*.pyx'):
    result = compile(str(source), cplus=True)
    if result.num_errors:
        raise RuntimeError('Cython generation failed')
sys.argv = ['setup.py', 'bdist_wheel', '--dist-dir', str(root / 'downloads/python-wheels')]
runpy.run_path('setup.py', run_name='__main__')
wheel = root / 'downloads/python-wheels/msgpack_python-0.5.6-cp310-cp310-win_amd64.whl'
with zipfile.ZipFile(wheel) as bundle:
    binaries = [name for name in bundle.namelist() if name.endswith('.pyd')]
    if len(binaries) != 2:
        raise RuntimeError('Build fell back to pure Python')
print(wheel, binaries)
'@ | .venv\python.exe -
.venv\python.exe -m pip install --no-deps --force-reinstall downloads\python-wheels\msgpack_python-0.5.6-cp310-cp310-win_amd64.whl
.venv\python.exe -c "import msgpack; print(msgpack.Packer, msgpack.Unpacker)"
.venv\python.exe -m pip check
```

需要重新启动已经导入msgpack的Python进程。验收脚本如果仍检测到`msgpack.fallback.Unpacker`，会在浮点深度采集前拒绝执行，避免再次长时间卡住。该检查不阻止普通传感器读取。

本轮没有用中断的慢路径算加速倍数，因为慢路径没有完整结束。最终报告仅给出修复后实测RPC耗时，不把它当作控制频率、双机同步性能或模型推理速度。
