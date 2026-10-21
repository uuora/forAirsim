"""Bind runtime scene to its RPC engine lifetime; contains no target positions."""
import copy
import psutil

def current_engine_identity():
    owners={c.pid for c in psutil.net_connections(kind='tcp') if c.laddr and c.laddr.port==41454 and c.status=='LISTEN'}
    if len(owners)!=1:raise ValueError('SCENE_PORT_OWNER_AMBIGUOUS_OR_MISSING')
    process=psutil.Process(next(iter(owners)))
    return dict(pid=process.pid,created=process.create_time())

def engine_identity_alive(identity):
    try:return isinstance(identity,dict) and psutil.Process(identity['pid']).create_time()==identity['created']
    except (KeyError,TypeError,psutil.Error):return False

def bind_scene_config(config,receipt):
    if not receipt.get('status','').startswith('PASS_RUNTIME_'):raise ValueError('SCENE_RECEIPT_NOT_VALID')
    identity=current_engine_identity()
    if receipt.get('engine_identity')!=identity:raise ValueError('SCENE_ENGINE_LIFETIME_CHANGED')
    result=copy.deepcopy(config);result.update(runtime_scene_version=receipt['scene_version'],runtime_engine_identity=identity,scene_epoch_required=True)
    return result
