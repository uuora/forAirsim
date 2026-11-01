"""Protocol16: immutable two-target references and future C2 handoffs.

No simulator calls here. A sampled/model check is not a flight guarantee.
"""
from dataclasses import asdict
import hashlib,json
import numpy as np
from trajectory_research.parameterizations import BoundaryStates,curve_from_parameters
from trajectory_research.curve_task import subdivided_derivative_norm_bound
from .route_curve import RouteCurve


def target_stream(start,seed,count=32):
    """Precommitted constrained stream; never depends on executed positions."""
    rng=np.random.default_rng(seed);points=[];previous=np.asarray(start,dtype=float);heading=np.array([1.,0.,0.])
    for _ in range(count):
        for attempt in range(20000):
            angle=rng.uniform(-np.pi,np.pi);d=rng.uniform(6.,9.)
            delta=np.array([d*np.cos(angle),d*np.sin(angle),rng.uniform(-1.,1.)])
            point=previous+delta
            if np.any(point<[-24,-24,4]) or np.any(point>[24,24,10]):continue
            direction=delta/np.linalg.norm(delta)
            if direction@heading<-.4:continue
            points.append(point.tolist());previous=point;heading=direction;break
        else:raise ValueError('无法在开发边界内生成连续目标；请更换起点或种子')
    return points


def stream_hash(points):return hashlib.sha256(json.dumps(points,sort_keys=True).encode()).hexdigest()


def make_request(boundary,targets,method,*,seed=1201,limits=(6.,6.,30.),anchor=None):
    p,v,a=np.asarray(boundary,dtype=float);goals=np.asarray(targets,dtype=float)
    if goals.shape!=(2,3):raise ValueError('滚动视野必须正好有当前和下一目标')
    incoming=goals[0]-p;outgoing=goals[1]-goals[0]
    u=incoming/max(np.linalg.norm(incoming),1e-9);w=outgoing/max(np.linalg.norm(outgoing),1e-9)
    tangent=u+w
    if np.linalg.norm(tangent)<.1:raise ValueError('两点视野近掉头，当前开发协议不接受')
    middle=tangent/np.linalg.norm(tangent)*1.2
    boundaries=[asdict(BoundaryStates(p0=tuple(p),p1=tuple(goals[0]),v0=tuple(v),v1=tuple(middle),a0=tuple(a))),
                asdict(BoundaryStates(p0=tuple(goals[0]),p1=tuple(goals[1]),v0=tuple(middle),v1=(0.,0.,0.)))]
    if anchor is None:
        durations=[np.clip(np.linalg.norm(incoming)/1.5+1,3.,7.),np.clip(np.linalg.norm(outgoing)/1.5+1,3.,7.)]
        anchor=[0.,0.,0.,float(durations[0]),0.,0.,0.,float(durations[1])]
    return dict(boundaries=boundaries,targets=goals.tolist(),method=method,seed=int(seed),limits=list(limits),anchor=list(anchor),
                lower=[-.8,-1.4,-.8,3.]*2,upper=[.8,1.4,.8,7.]*2,dt=.04,velocity_lag_s=.9)


def decode(request,vector):
    x=np.asarray(vector,dtype=float).reshape(2,4)
    segments=[curve_from_parameters(row,family='single_bump',representation=request['method'].split('_')[0],boundary=BoundaryStates(**b)) for row,b in zip(x,request['boundaries'])]
    events=[dict(index=i,arrival_time_s=float(sum(s.duration_s for s in segments[:i+1])),evaluation_time_s=float(sum(s.duration_s for s in segments[:i+1])),point=request['targets'][i]) for i in range(2)]
    return RouteCurve(segments,events,dict(start=request['boundaries'][0]['p0'],waypoints=request['targets']))


def reference_checks(request,vector):
    curve=decode(request,vector)
    bounds=[max(subdivided_derivative_norm_bound(s,d,depth=2) for s in curve.segments) for d in (1,2,3)]
    # All polynomial control points inside this box imply continuous containment.
    hull=np.vstack([s.bezier_control_points() for s in curve.segments])
    inside=bool(np.all(hull>=[-45,-45,3]) and np.all(hull<=[45,45,23]))
    return dict(passed=bool(inside and all(x<=limit for x,limit in zip(bounds,request['limits']))),derivative_bounds=bounds,inside_control_hull=inside)


def fallback(request):
    vector=np.asarray(request['anchor'],float).reshape(2,4).copy();vector[:,:3]=0
    for _ in range(12):
        check=reference_checks(request,vector.ravel())
        if check['passed']:return vector.ravel().tolist(),check
        if np.all(vector[:,3]>=7):break
        vector[:,3]=np.minimum(7,vector[:,3]*1.1)
    raise ValueError('没有通过运动学检查的后备曲线；保持原曲线并终止本次新目标规划')


class ReferenceTimeline:
    def __init__(self,request,vector):
        self.epoch=0;self.windows=[];self.add(0.,request,vector,'fallback')

    def at(self,t):
        window=next(w for w in reversed(self.windows) if not w.get('cancelled') and w['start_s']<=t+1e-10)
        curve=window['_curve'];u=min(max(0.,t-window['start_s']),curve.duration_s)
        return [curve.sample(u,derivative=d) for d in (0,1,2)],window

    def add(self,start,request,vector,source,*,now=None):
        curve=decode(request,vector);errors=[0.,0.,0.]
        if self.windows:
            if now is None or start<now+.05:raise ValueError('切换点必须位于未来至少50毫秒')
            old,_=self.at(start);errors=[float(np.linalg.norm(old[d]-curve.sample(0.,derivative=d))) for d in (0,1,2)]
            if max(errors)>1e-7:raise ValueError('C2边界不连续')
            for w in self.windows:
                if w['start_s']>=start:w['cancelled']=True
        self.epoch+=1
        self.windows.append(dict(id=len(self.windows),epoch=self.epoch,start_s=float(start),request=request,parameters=list(vector),source=source,C2_errors=errors,_curve=curve))
        return self.windows[-1]

    def records(self):return [{k:v for k,v in w.items() if k!='_curve'} for w in self.windows]


def evaluate(request,vector):
    """Declared lagged point-mass surrogate, fixed feedback, not AirSim."""
    curve=decode(request,vector);checks=reference_checks(request,vector)
    times=np.unique(np.r_[np.arange(0,curve.duration_s,request['dt']),curve.ends]);qs=[curve.sample(times,derivative=d) for d in (0,1,2)]
    p=qs[0][0].copy();v=qs[1][0].copy();command=v.copy();errors=[];cap,slew,_=request['limits'];tau=request['velocity_lag_s']
    for i,t in enumerate(times):
        errors.append(float(np.linalg.norm(p-qs[0][i])))
        if i==len(times)-1:break
        dt=times[i+1]-t;raw=qs[1][i]+2*(qs[0][i]-p)+.8*(qs[1][i]-v)+.9*qs[2][i]
        speed=raw*min(1.,cap/max(np.linalg.norm(raw),1e-300));delta=speed-command
        command=command+delta*min(1.,slew*dt/max(np.linalg.norm(delta),1e-300));decay=np.exp(-dt/tau)
        p=p+command*dt+(v-command)*tau*(1-decay);v=command+(v-command)*decay
    rmse=float(np.sqrt(np.trapz(np.square(errors),times)/curve.duration_s));peak=max(errors)
    terminal=errors[-1];terminal_v=float(np.linalg.norm(v));middle=errors[int(np.searchsorted(times,curve.ends[0]))]
    excess=sum(max(0.,x/lim-1) for x,lim in zip(checks['derivative_bounds'],request['limits']))+(0 if checks['inside_control_hull'] else 1)
    excess+=max(0.,peak/.5-1)+max(0.,terminal/.25-1)+max(0.,terminal_v/.5-1)+max(0.,middle/.5-1)
    feasible=excess<=0
    quality=.7*curve.duration_s/(curve.duration_s+10)+.1*curve.jerk_energy/(curve.jerk_energy+60)+.2*rmse/(rmse+.25)
    return dict(feasible=feasible,score=float(quality if feasible else 1+excess),position_rmse_m=rmse,peak_error_m=float(peak),terminal_error_m=float(terminal),terminal_velocity_mps=terminal_v,checks=checks)
