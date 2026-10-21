import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MathRuntimeTests(unittest.TestCase):
    def execute(self,code):
        result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,text=True,capture_output=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        return result.stdout.strip()
    def test_budget_is_applied_before_actual_numpy_import(self):
        value=json.loads(self.execute("from fixed_map_cpu_runtime import configure_math_runtime; import os,json; configure_math_runtime(dict(math_thread_budget_v20=True,torch_threads=2)); import numpy; print(json.dumps(dict(openblas=os.environ['OPENBLAS_NUM_THREADS'],omp=os.environ['OMP_NUM_THREADS'],dot=float(numpy.array([1.,2.])@numpy.array([3.,4.])))))"))
        self.assertEqual(value,dict(openblas='1',omp='2',dot=11.))
    def test_late_budget_cannot_claim_it_was_applied(self):
        self.assertEqual(self.execute("import os; os.environ['OPENBLAS_NUM_THREADS']='1'; import numpy; from fixed_map_cpu_runtime import configure_math_runtime\ntry:configure_math_runtime(dict(math_thread_budget_v20=True,torch_threads=2))\nexcept ValueError as e:print(e)"),'CPU_RUNTIME_MUST_PRECEDE_NUMPY_IMPORT')
    def test_nested_driver_and_sensor_imports_keep_the_same_attested_budget(self):
        self.assertEqual(self.execute("from fixed_map_cpu_runtime import configure_math_runtime; c=dict(math_thread_budget_v20=True,torch_threads=2); configure_math_runtime(c); import numpy; print(configure_math_runtime(c)['OPENBLAS_NUM_THREADS'])"),'1')
    def test_changed_environment_after_import_is_rejected(self):
        self.assertEqual(self.execute("from fixed_map_cpu_runtime import configure_math_runtime; import os; c=dict(math_thread_budget_v20=True,torch_threads=2); configure_math_runtime(c); import numpy; os.environ['OPENBLAS_NUM_THREADS']='6'\ntry:configure_math_runtime(c)\nexcept ValueError as e:print(e)"),'CPU_RUNTIME_ENVIRONMENT_CHANGED_AFTER_NUMPY_IMPORT')
    def test_legacy_profiles_leave_existing_environment_unchanged(self):
        self.assertEqual(self.execute("import os; os.environ['OPENBLAS_NUM_THREADS']='3'; from fixed_map_cpu_runtime import configure_math_runtime; configure_math_runtime({}); print(os.environ['OPENBLAS_NUM_THREADS'])"),'3')
    def test_incompatible_torch_budget_refuses_silent_override(self):
        self.assertEqual(self.execute("from fixed_map_cpu_runtime import configure_math_runtime\ntry:configure_math_runtime(dict(math_thread_budget_v20=True,torch_threads=6))\nexcept ValueError as e:print(e)"),'V20_MATH_BUDGET_REQUIRES_TWO_TORCH_THREADS')
    def test_explicit_config_path_controls_process_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'config.json';path.write_text(json.dumps(dict(math_thread_budget_v20=True,torch_threads=2)))
            output=self.execute(f"from fixed_map_cpu_runtime import configure_from_argv; print(configure_from_argv(['--config',{str(path)!r}])['OPENBLAS_NUM_THREADS'])")
            self.assertEqual(output,'1')
    def test_library_import_does_not_consume_an_unrelated_program_config_argument(self):
        self.assertEqual(self.execute("import os,sys; os.environ['OPENBLAS_NUM_THREADS']='1'; sys.argv=['another_program','--config','nonexistent_config.json']; import fixed_map_perception_process; print('IMPORTED')"),'IMPORTED')


if __name__=='__main__':unittest.main()
