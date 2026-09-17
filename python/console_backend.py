"""Local experiment-console services; model text is never evaluated as code."""
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from language_plan import validate_plan, preview_steps
from run_language_planner import local_server, infer

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.venv/python.exe'

def snapshot_id(instruction, plan):
    validate_plan(plan)
    return hashlib.sha256(json.dumps([instruction, plan], ensure_ascii=False,
                                     sort_keys=True).encode('utf-8')).hexdigest()

def execution_command(instruction, record, approved_snapshot):
    if record.get('instruction') != instruction or not record.get('schema_valid'):
        raise ValueError('指令已变化或模型计划未通过校验，请重新解析。')
    plan = validate_plan(record.get('plan'))
    if plan['disposition'] != 'ACCEPT':
        raise ValueError('当前计划需要澄清或不受支持。')
    if not approved_snapshot or approved_snapshot != snapshot_id(instruction, plan):
        raise ValueError('请核对当前指令与完整计划。')
    command = [str(PYTHON), '-u', str(ROOT / 'python/run_visual_alignment.py')]
    if plan['routine'] == 'visual_range':
        command.append('--range')
    return command

@contextmanager
def workspace_lease():
    """OS lock is released even if a console process crashes."""
    import msvcrt
    path = ROOT / 'logs/console_mission.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('另一个实验控制台正在执行任务。') from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)

def rpc_ready():
    try:
        with socket.create_connection(('127.0.0.1', 41451), timeout=0.4):
            return True
    except OSError:
        return False

@contextmanager
def sim_client():
    if not rpc_ready():
        raise RuntimeError('AirSim尚未连接。请先启动Blocks并等待加载。')
    import airsim
    client = airsim.MultirotorClient(ip='127.0.0.1', timeout_value=5)
    try:
        yield client
    finally:
        client.client.close()

def read_snapshot(include_images=True):
    from setup_scene import vector
    import airsim
    with sim_client() as client:
        names = client.listVehicles()
        if not {'DroneA', 'DroneB'}.issubset(names):
            raise RuntimeError('需要当前项目的DroneA和DroneB。')
        output = dict(paused=client.simIsPause(), sampled_at=datetime.now().astimezone().isoformat(), vehicles={})
        for name in ('DroneA', 'DroneB'):
            state = client.getMultirotorState(vehicle_name=name)
            speed = math.sqrt(sum(v*v for v in vector(state.kinematics_estimated.linear_velocity)))
            output['vehicles'][name] = dict(position=vector(client.simGetObjectPose(name).position),
                                           speed=speed, api_control=client.isApiControlEnabled(vehicle_name=name))
            if include_images:
                raw = client.simGetImage('front_center', airsim.ImageType.Scene, vehicle_name=name)
                output['vehicles'][name]['image'] = bytes(raw) if raw else None
        return output

def require_ground_ready(snapshot):
    for name in ('DroneA', 'DroneB'):
        item = snapshot['vehicles'][name]
        if not all(math.isfinite(v) for v in item['position']) or not math.isfinite(item['speed']):
            raise ValueError('飞机状态无效。')
        if item['api_control'] or not 0.65 < item['position'][2] < 1.1 or item['speed'] >= 0.08:
            raise ValueError('两机须在已知地面静止且释放API控制；空中暂停现场请先使用回收。')

class ConsoleBackend:
    def __init__(self, emit):
        self.emit = emit
        self.folder = ROOT / 'logs' / datetime.now().strftime('console_%Y%m%d_%H%M%S_%f')
        self.folder.mkdir(parents=True)
        self._mutex = threading.Lock()
        self._state_lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._epoch = 0
        self._process = None
        self._stop_requested = False
        self._flight_active = False

    def event(self, name, **data):
        item = dict(event=name, time=datetime.now().astimezone().isoformat(), **data)
        with self._event_lock:
            with (self.folder/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(item, ensure_ascii=False)+'\n')
        return item

    def plan(self, instruction):
        with self._mutex:
            folder = self.folder / datetime.now().strftime('plan_%H%M%S_%f')
            folder.mkdir()
            self.emit('模型加载中，指令在本机处理……')
            with local_server(folder) as base:
                result = infer(base, instruction)
            result['model'] = 'Qwen3-0.6B-Q8_0 CPU'
            result['fine_tuned'] = False
            result['source_sha256'] = {name: hashlib.sha256((ROOT/'python'/name).read_bytes()).hexdigest()
                                       for name in ('language_plan.py', 'run_language_planner.py')}
            manifest = ROOT/'downloads/local_llm/manifest.json'
            if manifest.is_file():
                result['asset_manifest'] = json.loads(manifest.read_text(encoding='utf-8'))
            (folder/'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            self.event('plan_completed', report=str(folder/'report.json'), schema_valid=result['schema_valid'])
            return result

    def start_simulator(self):
        from launch_demo import EDITOR, PROJECT, editor_running
        if rpc_ready():
            return 'AirSim已连接。'
        if not EDITOR.is_file() or not PROJECT.is_file():
            raise FileNotFoundError('找不到当前Blocks项目或UE4Editor。')
        if not editor_running():
            subprocess.Popen([str(EDITOR), str(PROJECT), '-game', '-windowed', '-ResX=1280', '-ResY=720'],
                             cwd=ROOT, creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        deadline = time.monotonic()+90
        while not rpc_ready():
            if time.monotonic()>deadline:
                raise TimeoutError('AirSim连接超时，请检查Blocks窗口。')
            time.sleep(0.5)
        self.event('simulator_connected')
        return 'AirSim已连接，可以读取画面或准备固定气球。'

    def _run(self, command, kind):
        self.emit('开始：'+kind)
        env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')
        with (self.folder/(kind+'.log')).open('a', encoding='utf-8') as log:
            with self._state_lock:
                if self._stop_requested:
                    raise RuntimeError('任务已取消。')
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self._process = process
            self.event('process_started', kind=kind, command=command, pid=process.pid)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self.emit(line.rstrip())
                code = process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                if self._flight_active:
                    with sim_client() as client:
                        client.simPause(True)
                raise
            finally:
                process.stdout.close()
                with self._state_lock:
                    self._process = None
            stopped = self._stop_requested
            self.event('process_finished', kind=kind, returncode=code, interrupted=stopped)
            if stopped:
                raise RuntimeError('任务已暂停并中断；这不是落地。请读取状态或回收。')
            if code:
                raise RuntimeError(f'{kind}退出码 {code}，请查看日志与仿真状态。')

    def request_epoch(self):
        with self._state_lock:
            return self._epoch

    def cancel_pending(self):
        """Invalidate already queued requests before a pause worker can be delayed."""
        with self._state_lock:
            self._epoch += 1
            self._stop_requested = True

    def _begin(self, request_epoch=None):
        with self._state_lock:
            if request_epoch is not None and request_epoch != self._epoch:
                raise RuntimeError('请求已被暂停操作取消。请重新发起任务。')
            self._stop_requested = False

    def prepare_scene(self, request_epoch=None):
        with self._mutex, workspace_lease():
            self._begin(request_epoch)
            require_ground_ready(read_snapshot(False))
            self._run([str(PYTHON), '-u', str(ROOT/'python/setup_scene.py')], 'prepare_scene')
            return '固定气球已准备。'

    def execute(self, instruction, record, approved_snapshot, request_epoch=None):
        command = execution_command(instruction, record, approved_snapshot)
        with self._mutex, workspace_lease():
            self._begin(request_epoch)
            require_ground_ready(read_snapshot(False))
            self.event('plan_approved_for_execution', instruction=instruction, plan=record['plan'],
                       snapshot=approved_snapshot, preview=preview_steps(record['plan']))
            # Preparing first is safe only after the ground check above.
            self._run([str(PYTHON), '-u', str(ROOT/'python/setup_scene.py')], 'prepare_scene')
            self._flight_active = True
            try:
                self._run(command, 'visual_mission')
            finally:
                self._flight_active = False
            return '任务脚本结束。请核查任务报告中的落地证据。'

    def recover(self, request_epoch=None):
        with self._mutex, workspace_lease():
            self._begin(request_epoch)
            self._flight_active = True
            try:
                self._run([str(PYTHON), '-u', str(ROOT/'python/recover_airborne.py')], 'recovery')
            finally:
                self._flight_active = False
            return '回收脚本结束，请刷新双机状态。'

    def pause(self):
        # Stop our command producer FIRST, then pause and verify. The current
        # controllers also pause on bounded failures; never disarm here.
        self.cancel_pending()
        with self._state_lock:
            process = self._process
            if process is not None and process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        # Even when RPC is unavailable, stop sending commands. A failure below
        # is reported as unconfirmed pause; it must never be reported as landed.
        with sim_client() as client:
            client.simPause(True)
            if not client.simIsPause():
                raise RuntimeError('暂停请求未被确认，请检查Blocks。')
        self.event('operator_pause', motors_disarmed=False)
        return '仿真已暂停；未解除武装，未宣称落地。可检查后回收。'
