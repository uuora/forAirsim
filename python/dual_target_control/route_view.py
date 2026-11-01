"""Task-sized orthographic route monitor; fixed camera during execution."""
import math
import numpy as np
from .ui import FlightView


class RouteFlightView(FlightView):
    def __init__(self,config,parent=None):
        super().__init__(config,parent);self.setMinimumSize(350,200)
        from .route_scene import RouteScene
        self.scene=RouteScene(self)

    def reset_view(self):
        self.view_plane='3d';self.azimuth=math.radians(-51);self.elevation=math.radians(28)
        self.zoom=1.;self.pan=np.zeros(2);self.update()

    def _display_bounds(self):
        task=self.config.get('preparation',{}).get('task')
        if not task:return super()._display_bounds()
        points=np.asarray([task['start'],*task['points']],dtype=float)
        low=points.min(axis=0)-[1.8,1.8,1.];high=points.max(axis=0)+[1.8,1.8,1.]
        return (low[0],high[0],low[1],high[1],max(0.,low[2]),high[2])

    def _camera(self):
        bounds=self._display_bounds()
        signature=(bounds,self.width(),self.height(),self.azimuth,self.elevation,self.zoom,*self.pan)
        if signature==self._camera_signature:return
        self._camera_signature=signature
        az,el=self.azimuth,self.elevation
        self.right=np.array([-math.sin(az),math.cos(az),0.])
        self.up=np.array([-math.sin(el)*math.cos(az),-math.sin(el)*math.sin(az),math.cos(el)])
        self.depth=np.cross(self.right,self.up)
        xmin,xmax,ymin,ymax,zmin,zmax=bounds
        self.center=np.array([(xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2])
        task=self.config.get('preparation',{}).get('task')
        floor=np.array([[xmin,ymin,zmin],[xmax,ymin,zmin],[xmax,ymax,zmin],[xmin,ymax,zmin]])
        frame=[*floor,*(floor-[0,0,.23]),[xmin,ymin,zmax]]
        if task:frame.extend([task['start'],*task['points']])
        self.corners=np.asarray(frame,dtype=float)
        projected=(self.corners-self.center)@np.column_stack((self.right,self.up))
        span=np.maximum(np.ptp(projected,axis=0),.1)
        self.scale=min(max(60,self.width()-110)/span[0],max(60,self.height()-116)/span[1])*self.zoom
        self.screen_center=np.array([self.width()/2,self.height()/2+3])+self.pan
        self.projection_matrix=np.column_stack((self.right,-self.up))*self.scale
        projected_center=(projected.min(axis=0)+projected.max(axis=0))/2
        self.projection_offset=self.screen_center-self.center@self.projection_matrix-projected_center*np.array([1.,-1.])*self.scale


    def _landmarks(self):
        markers=self.snapshot.get('route_markers')
        if markers:return markers
        task=self.config.get('preparation',{}).get('task')
        if not task:return []
        return [dict(id='S',kind='start',position=task['start'],state='fixed'),
            *[dict(id=f'W{i+1}',kind='via',position=p,state='pending') for i,p in enumerate(task['points'][:-1])],
            dict(id='E',kind='goal',position=task['points'][-1],state='pending')]


    def paintEvent(self,event):
        self.scene.paint()

    def mouseMoveEvent(self,event):
        super().mouseMoveEvent(event)
        if self._mouse is None:
            self._camera();nearest=None
            position=self.snapshot.get('position')
            if position is not None and self.snapshot.get('connected'):
                p=self.project(position)
                if math.hypot(p.x()-event.pos().x(),p.y()-event.pos().y())<28:
                    self.setToolTip('四旋翼无人机：保留原机臂与旋翼图标，橙色箭头表示机头偏航。\n图标为放大示意；轨迹坐标与网格仍按米绘制。');return
            for marker in self._landmarks():
                p=self.project(marker['position']);d=math.hypot(p.x()-event.pos().x(),p.y()-event.pos().y())
                if d<18:nearest=marker;break
            self.setToolTip((nearest['id']+'  '+str(nearest['position'])+' m\n显示经过标记不等于控制判据通过。') if nearest else '左拖旋转，右拖平移，滚轮缩放，双击复位；运行时不自动缩放。')
