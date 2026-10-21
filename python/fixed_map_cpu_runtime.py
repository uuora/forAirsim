"""Explicit per-process v20 math budget, applied before NumPy imports."""
import json
import os
import sys
from pathlib import Path
_APPLIED_BUDGET=None


def configure_math_runtime(config):
    global _APPLIED_BUDGET
    if not config.get('math_thread_budget_v20'):return None
    if config.get('torch_threads')!=2:raise ValueError('V20_MATH_BUDGET_REQUIRES_TWO_TORCH_THREADS')
    values=dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='1')
    if 'numpy' in sys.modules:
        if _APPLIED_BUDGET!=values:raise ValueError('CPU_RUNTIME_MUST_PRECEDE_NUMPY_IMPORT')
        if any(os.environ.get(key)!=value for key,value in values.items()):raise ValueError('CPU_RUNTIME_ENVIRONMENT_CHANGED_AFTER_NUMPY_IMPORT')
        return dict(values)
    os.environ.update(values)
    _APPLIED_BUDGET=dict(values)
    return dict(values)


def configure_from_argv(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if '--config' not in args:return None
    index=args.index('--config')
    if index+1>=len(args):raise ValueError('CONFIG_PATH_MISSING')
    path=Path(args[index+1])
    return configure_math_runtime(json.loads(path.read_text(encoding='utf-8-sig')))
