"""Display-only route landmarks. Visited is not a control-gate verdict."""
from copy import deepcopy
import numpy as np


class RouteMilestones:
    def __init__(self,spec):
        self.spec=deepcopy(spec);self.contact={};self.events=[]
        self.markers=[dict(id='S',kind='start',position=spec['start'],state='fixed')]
        self.markers += [dict(id=f'W{i+1}',kind='via',position=p,state='pending') for i,p in enumerate(spec['waypoints'][:-1])]
        self.markers.append(dict(id='E',kind='goal',position=spec['waypoints'][-1],state='pending'))

    def update(self,t,position,route_index,waypoint_times,goal_reached=False):
        p=np.asarray(position,dtype=float)
        for i,marker in enumerate(self.markers[1:-1]):
            near=float(np.linalg.norm(p-marker['position']))
            if t>=max(0.,waypoint_times[i]['arrival_time_s']-.5) and near<=self.spec['waypoint_radius_m']:
                self.contact.setdefault(i,dict(reference_time_s=float(t),distance_m=near,actual_position=p.tolist()))
            if route_index>i and marker['state']=='pending':
                marker['state']='visited' if i in self.contact else 'unconfirmed'
                self.events.append(dict(id=marker['id'],state=marker['state'],reference_time_s=float(t),
                    contact=self.contact.get(i),scope='Display contact/progress only, not waypoint success grading'))
        if goal_reached:self.markers[-1]['state']='reached'
        return deepcopy(self.markers)


def glyph_lines(marker):
    """Metre-sized debug glyphs; no UE mesh, collision object or physics edit."""
    p=np.asarray(marker['position'],dtype=float)
    if marker['kind']=='start':
        r=.24;v=[p+[-r,-r,0],p+[r,-r,0],p+[r,r,0],p+[-r,r,0]]
        return [q.tolist() for a,b in ((0,1),(1,2),(2,3),(3,0)) for q in (v[a],v[b])]
    if marker['kind']=='goal':
        top=p+[0,0,.7];tip=top+[.5,0,-.13];bottom=top+[0,0,-.29]
        return [q.tolist() for a,b in ((p,top),(top,tip),(tip,bottom),(bottom,top)) for q in (a,b)]
    return []
