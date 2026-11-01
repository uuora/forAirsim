"""Protocol14 reference geometry and declared waypoint rules.

Stop/pass-through is fixed by the experiment. The policy schema reserves speed
and braking variables for a later optimization ablation; neither is secretly
selected by a curve basis. No single-polynomial coefficients are invented for a
piecewise curve.
"""
from dataclasses import asdict
import numpy as np
from trajectory_research.parameterizations import BoundaryStates,curve_from_parameters
from trajectory_research.curves import PolynomialSegment
from trajectory_research.curve_task import jerk_energy

POLICIES={'stop':'逐点停稳','pass':'连续过点'}


def route_spec(points,policy='stop',seed=1201,*,pass_speed=1.2,dwell_s=1.,time_scale=1.25,max_speed=6.,max_slew=6.):
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not 1<=len(p)<=10 or not np.isfinite(p).all():raise ValueError('请输入1至10个有限的三维航点')
    start=np.array([0.,0.,5.]);chain=np.vstack([start,p]);delta=np.diff(chain,axis=0)
    if np.any(np.linalg.norm(delta,axis=1)<2.):raise ValueError('相邻航点间距至少2米')
    if np.any(p<[-45,-45,3]) or np.any(p>[45,45,23]):raise ValueError('航点须位于x/y±45米、高度3至23米内')
    if policy not in POLICIES:raise ValueError('未知过点规则')
    if not .5<=float(time_scale)<=2 or not .5<=float(pass_speed)<=3 or not .5<=float(dwell_s)<=5:raise ValueError('倍率/过点速度/停留时间超出开发范围')
    if not .5<=float(max_speed)<=15 or not .5<=float(max_slew)<=15 or pass_speed>max_speed:raise ValueError('指令上限超出范围或过点速度大于指令上限')
    velocities=np.zeros_like(chain)
    if policy=='pass':
        for i in range(1,len(chain)-1):
            a=delta[i-1]/np.linalg.norm(delta[i-1]);b=delta[i]/np.linalg.norm(delta[i])
            if a@b<-.85:raise ValueError('连续过点组暂不接受接近原地掉头的航点；请增大转弯空间')
            velocities[i]=(a+b)/np.linalg.norm(a+b)*float(pass_speed)
    return dict(schema_version=1,seed=int(seed),policy=policy,policy_name=POLICIES[policy],start=start.tolist(),waypoints=p.tolist(),
        waypoint_velocities=velocities[1:].tolist(),pass_speed_mps=float(pass_speed),dwell_s=float(dwell_s),time_scale=float(time_scale),
        waypoint_radius_m=.5,stop_speed_limit_mps=.3,minimum_pass_speed_mps=.5,
        command_limits=dict(max_speed_mps=float(max_speed),max_slew_mps2=float(max_slew)),
        future_optimization=dict(policy_selection='fixed_by_experiment',pass_speed_scale=dict(value=1.,bounds=[.5,1.5],optimized=False),
            braking_position_scale=dict(value=1.,bounds=[.7,1.3],optimized=False)),
        scope='Same ordered points and rules for six methods; last point always at rest')


def route_boundaries(spec):
    positions=[spec['start'],*spec['waypoints']];velocities=[[0.,0.,0.],*spec['waypoint_velocities']]
    factor=spec['time_scale']
    return [asdict(BoundaryStates(p0=tuple(positions[i]),p1=tuple(positions[i+1]),
        v0=tuple(np.asarray(velocities[i])/factor),v1=tuple(np.asarray(velocities[i+1])/factor))) for i in range(len(positions)-1)]


class RouteCurve:
    def __init__(self,segments,waypoint_times,spec):
        self.segments=tuple(segments);self.ends=np.cumsum([s.duration_s for s in segments]);self.duration_s=float(self.ends[-1])
        self.waypoint_times=list(waypoint_times);self.spec=spec
        self.jerk_energy=sum(float(jerk_energy(s)) for s in self.segments)

    def sample(self,t,derivative=0):
        values=np.asarray(t,dtype=float);flat=values.reshape(-1)
        if not np.isfinite(flat).all() or np.any(flat<0) or np.any(flat>self.duration_s+1e-9):raise ValueError('路线时间超出定义域')
        indices=np.searchsorted(self.ends,np.minimum(flat,self.duration_s),side='right').clip(0,len(self.segments)-1)
        result=np.empty((len(flat),3));starts=np.r_[0.,self.ends[:-1]]
        for index in np.unique(indices):
            mask=indices==index;s=self.segments[index]
            result[mask]=s.sample(np.clip(flat[mask]-starts[index],0,s.duration_s),derivative=derivative)
        return result.reshape(values.shape+(3,))


def decode_route(plan,factor=None):
    spec=plan['route'];factor=spec['time_scale'] if factor is None else float(factor)
    if abs(factor-spec['time_scale'])>1e-9:raise ValueError('多点路线倍率已计入规划；修改倍率后请重新生成方案')
    segments=[];times=[];total=0.
    for i,item in enumerate(plan['segments']):
        c=curve_from_parameters(item['parameters'],family='single_bump',representation=plan['representation'],boundary=BoundaryStates(**item['boundary']))
        c=PolynomialSegment(c.coefficients.copy(),c.duration_s/factor);segments.append(c);total+=c.duration_s
        event=dict(index=i,arrival_time_s=total,evaluation_time_s=total,point=spec['waypoints'][i])
        if spec['policy']=='stop' and i<len(plan['segments'])-1:
            hold=PolynomialSegment(np.array([spec['waypoints'][i]],dtype=float),spec['dwell_s']);segments.append(hold);total+=hold.duration_s
            event['evaluation_time_s']=total
        times.append(event)
    return RouteCurve(segments,times,spec)


def waypoint_results(curve,times,positions,velocities):
    times=np.asarray(times);positions=np.asarray(positions);velocities=np.asarray(velocities);rows=[]
    for event in curve.waypoint_times:
        t=event['evaluation_time_s'];i=int(np.searchsorted(times,t))
        if i>=len(times):rows.append(dict(index=event['index'],status='NOT_REACHED',passed=False));continue
        j=max(0,i-1);fraction=0. if times[i]==times[j] else (t-times[j])/(times[i]-times[j])
        p=positions[j]*(1-fraction)+positions[i]*fraction;v=velocities[j]*(1-fraction)+velocities[i]*fraction
        error=float(np.linalg.norm(p-event['point']));speed=float(np.linalg.norm(v));last=event['index']==len(curve.waypoint_times)-1
        speed_ok=(speed<=.5 if last else speed<=curve.spec['stop_speed_limit_mps'] if curve.spec['policy']=='stop' else speed>=curve.spec['minimum_pass_speed_mps'])
        rows.append(dict(index=event['index'],status='MEASURED',reference_time_s=t,position_error_m=error,actual_speed_mps=speed,
            passed=bool(error<=curve.spec['waypoint_radius_m'] and speed_ok),rule='terminal_stop' if last else curve.spec['policy']))
    return rows


def route_metrics(curve,states,commands):
    t=np.array([s['reference_elapsed_s'] for s in states]);p=np.array([s['position_zup_m'] for s in states]);v=np.array([s['velocity_zup_mps'] for s in states]);T=curve.duration_s
    if len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0) or t[-1]<T:raise ValueError('路线记录尚未完整覆盖参考时间')
    end=int(np.searchsorted(t,T));left=max(0,end-1)
    if t[end]-t[left]>.1:raise ValueError('终端插值间隔过大')
    f=0 if t[end]==t[left] else (T-t[left])/(t[end]-t[left]);pe=p[left]*(1-f)+p[end]*f;ve=v[left]*(1-f)+v[end]*f
    keep=t<T;mt=np.r_[t[keep],T];mp=np.vstack([p[keep],pe]);err=np.linalg.norm(curve.sample(mt)-mp,axis=1)
    rmse=float(np.sqrt(np.sum(.5*(err[:-1]**2+err[1:]**2)*np.diff(mt))/T));peak=float(err.max())
    terminal=float(np.linalg.norm(curve.sample(T)-pe));terminal_v=float(np.linalg.norm(curve.sample(T,derivative=1)-ve))
    waypoints=waypoint_results(curve,t,p,v);checks=dict(sampled_peak_position=peak<=.5,terminal_position=terminal<=.25,terminal_velocity=terminal_v<=.5,waypoints=all(r['passed'] for r in waypoints))
    weights=np.maximum(0,np.minimum(T,t[1:])-t[:-1]);n=len(curve.spec['waypoints']);energy=curve.jerk_energy
    return dict(reference_duration_s=T,position_rmse_m=rmse,max_position_error_m=peak,terminal_position_error_m=terminal,
        terminal_velocity_error_mps=terminal_v,control_error_checks=checks,control_error_gates_passed=all(checks.values()),task_success=all(checks.values()),
        speed_clipped_time_fraction=float(np.dot(weights,[c['speed_clipped'] for c in commands])/T),
        command_slew_clipped_time_fraction=float(np.dot(weights,[c['slew_clipped'] for c in commands])/T),
        max_observed_state_gap_s=float(np.diff(t).max()),jerk_energy_m2_s5=energy,waypoint_results=waypoints,
        quality_term=.7*T/(T+5*n)+.1*energy/(energy+30*n)+.2*rmse/(rmse+.25))
