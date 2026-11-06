"""Native Qt scene styling. All geometry is telemetry or coordinate guides."""
import math
import numpy as np
from PyQt5.QtCore import QPointF,QRectF,Qt,QSettings
from PyQt5.QtGui import QColor,QFont,QLinearGradient,QPainter,QPainterPath,QPixmap,QPolygonF,QRadialGradient
from PyQt5.QtWidgets import QPushButton
from .ui import FlightView,_vec

PALETTES={
 'dark':dict(bg='#111e2b',bg2='#182a39',text='#e4edf4',muted='#96aabd',plane='#293e50',plane2='#203342',side='#172937',
     border='#567286',minor='#344e62',major='#47657b',label='#172a3b',label_border='#425e74',reference='#65d4ae',actual='#f178cc',drone='#d6e5ef',start='#37bfb5',via='#68aaf0',goal='#e8b45a',current='#ef8b92'),
 'light':dict(bg='#ecf2f6',bg2='#dfe9f0',text='#213b50',muted='#637e93',plane='#f3f7fa',plane2='#d7e3ec',side='#b6c8d6',
     border='#8da8bc',minor='#d8e3ec',major='#bacddc',label='#f7fafc',label_border='#c3d3df',reference='#169b79',actual='#c92a9b',drone='#28445c',start='#087f83',via='#316fb7',goal='#ab6c0c',current='#b74b5c')}


def alpha(value,opacity):
    c=QColor(value);c.setAlpha(opacity);return c


class RouteScene:
    def __init__(self,view):
        self.v=view;self.settings=QSettings('forAirsim','RouteMapAppearance')
        self.theme=str(self.settings.value('theme','dark'))
        if self.theme not in PALETTES:self.theme='dark'
        self.cache=None;self.cache_key=None;self.labels=[]
        self.toggle=QPushButton(view);self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.setToolTip('切换地图外观；保持当前视角、轨迹和飞行参数。\n平台为标注高度的坐标参考面，不是UE地面。')
        self.toggle.clicked.connect(lambda:self.set_theme('light' if self.theme=='dark' else 'dark',True))
        self.set_theme(self.theme)

    @property
    def p(self):return PALETTES[self.theme]

    def set_theme(self,name,persist=False):
        if name not in PALETTES:raise ValueError('Unknown map theme')
        self.theme=name;self.cache_key=None
        self.toggle.setText('浅色地图' if name=='dark' else '深色地图')
        self.toggle.setStyleSheet('QPushButton{font-size:11px;min-height:0;padding:4px 9px;border-radius:5px;'
            f'color:{self.p["text"]};background:{self.p["label"]};border:1px solid {self.p["label_border"]};'
            '}QPushButton:hover{border-color:#65b5bd;}')
        self.toggle.adjustSize()
        if persist:self.settings.setValue('theme',name)
        self.v.update()

    def polygon(self,points):return QPolygonF([self.v.project(p) for p in points])

    def line(self,painter,points,color,width=1.,style=Qt.SolidLine):
        self.v._polyline(painter,points,QColor(color),width,style)

    def note(self,painter,point,text,color=None,offset=(12,-29),small=False):
        v=self.v;q=v.project(point);painter.setFont(QFont('Microsoft YaHei',8 if small else 9))
        fm=painter.fontMetrics();width=fm.horizontalAdvance(text)+18;h=24
        candidates=[offset,(12,15),(-width-14,-29),(-width-14,15),(12,-56)]
        chosen=None;best=None
        for dx,dy in candidates:
            rect=QRectF(min(max(9,q.x()+dx),max(9,v.width()-width-9)),min(max(52,q.y()+dy),max(52,v.height()-h-43)),width,h)
            overlap=sum(rect.intersected(r).width()*rect.intersected(r).height() for r in self.labels if rect.intersects(r))
            if best is None or overlap<best:best=overlap;chosen=rect
            if overlap==0:break
        self.labels.append(chosen.adjusted(-4,-4,4,4))
        end=QPointF(min(max(q.x(),chosen.left()),chosen.right()),min(max(q.y(),chosen.top()),chosen.bottom()))
        painter.setPen(v._pen(alpha(color or self.p['muted'],120),.8));painter.drawLine(q,end)
        painter.setPen(v._pen(alpha(self.p['label_border'],190),.7));painter.setBrush(alpha(self.p['label'],237));painter.drawRoundedRect(chosen,5,5)
        painter.setPen(QColor(color or self.p['text']));painter.drawText(chosen.adjusted(9,0,-5,0),Qt.AlignVCenter,text)

    def stage(self,painter):
        v=self.v;p=self.p;x0,x1,y0,y1,z0,z1=v._display_bounds()
        step=v._grid_step(max(x1-x0,y1-y0));minor=step/5
        floor=np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]])
        if v.view_plane!='xz':
            # The shallow plinth is a coordinate reference, never scene physics.
            base=floor-[0,0,.23]
            shadow=self.polygon(base)
            for i in range(8,0,-1):
                path=QPainterPath();path.addPolygon(shadow.translated(0,i*1.5));painter.setPen(v._pen(QColor(0,0,0,2+i),i*2.))
                painter.setBrush(QColor(0,0,0,2+i));painter.drawPath(path)
            painter.setPen(v._pen(QColor(p['side']),.8));painter.setBrush(QColor(p['side']))
            for i in range(4):painter.drawPolygon(self.polygon([floor[i],floor[(i+1)%4],base[(i+1)%4],base[i]]))
            gradient=QLinearGradient(0,50,v.width(),v.height()-40);gradient.setColorAt(0,QColor(p['plane']));gradient.setColorAt(1,QColor(p['plane2']))
            painter.setBrush(gradient);painter.setPen(v._pen(QColor(p['border']),1.));painter.drawPolygon(self.polygon(floor))
            for x in np.arange(math.ceil(x0/minor)*minor,x1+1e-7,minor):
                self.line(painter,[[x,y0,z0],[x,y1,z0]],p['minor'],.65)
            for y in np.arange(math.ceil(y0/minor)*minor,y1+1e-7,minor):
                self.line(painter,[[x0,y,z0],[x1,y,z0]],p['minor'],.65)
            for x in np.arange(math.ceil(x0/step)*step,x1+1e-7,step):
                self.line(painter,[[x,y0,z0],[x,y1,z0]],p['major'],.95)
                self.tick(painter,[x,y0,z0],f'{x:g}',(-4,17))
            for y in np.arange(math.ceil(y0/step)*step,y1+1e-7,step):
                self.line(painter,[[x0,y,z0],[x1,y,z0]],p['major'],.95)
                self.tick(painter,[x0,y,z0],f'{y:g}',(-23,3))
        else:
            for x in np.arange(math.ceil(x0/minor)*minor,x1+1e-7,minor):self.line(painter,[[x,y0,z0],[x,y0,z1]],p['minor'],.7)
            hstep=v._grid_step(z1-z0)
            for z in np.arange(math.ceil(z0/hstep)*hstep,z1+1e-7,hstep):
                self.line(painter,[[x0,y0,z],[x1,y0,z]],p['major'],.9);self.tick(painter,[x0,y0,z],f'{z:g}',(-26,3))
            for x in np.arange(math.ceil(x0/step)*step,x1+1e-7,step):self.tick(painter,[x,y0,z0],f'{x:g}',(-5,18))
        origin=[x0,y0,z0]
        for end,name,offset in [([x1,y0,z0],'X / m',(8,9)),([x0,y1,z0],'Y / m',(4,-8)),([x0,y0,z1],'H / m',(-12,-12))]:
            if np.linalg.norm((np.array(end)-origin)@v.projection_matrix)<4:continue
            self.line(painter,[origin,end],p['border'],1.2);self.tick(painter,end,name,offset)
        if v.view_plane=='3d':
            self.tick(painter,origin,f'{z0:g}',(-22,3));self.tick(painter,[x0,y0,z1],f'{z1:g}',(-22,3))
        return step,z0

    def tick(self,painter,point,text,offset):
        q=self.v.project(point);painter.setFont(QFont('Microsoft YaHei',8));painter.setPen(QColor(self.p['muted']))
        painter.drawText(QPointF(q.x()+offset[0],q.y()+offset[1]),text)

    def background(self):
        v=self.v;dpr=v.devicePixelRatioF();key=(v._camera_signature,self.theme,dpr)
        if key==self.cache_key:return
        pix=QPixmap(int(v.width()*dpr),int(v.height()*dpr));pix.setDevicePixelRatio(dpr)
        painter=QPainter(pix);painter.setRenderHint(QPainter.Antialiasing)
        gradient=QLinearGradient(0,0,v.width()*.75,v.height());gradient.setColorAt(0,QColor(self.p['bg']));gradient.setColorAt(1,QColor(self.p['bg2']))
        painter.fillRect(QRectF(0,0,v.width(),v.height()),gradient)
        self.step,self.z0=self.stage(painter)
        painter.setPen(v._pen(alpha(self.p['border'],95),.7));painter.drawLine(QPointF(18,44),QPointF(v.width()-18,44))
        painter.end();self.cache=pix;self.cache_key=key

    def marker(self,painter,m):
        v=self.v;q=v.project(m['position']);kind=m['kind'];state=m['state'];targets=v.snapshot.get('targets') or []
        current=bool(targets) and kind!='start' and np.allclose(targets[0],m['position'],atol=1e-6)
        c=QColor(self.p['start'] if kind=='start' else self.p['goal'] if kind=='goal' else self.p['via'] if state=='visited' else self.p['muted'])
        glow=QRadialGradient(q,18);glow.setColorAt(0,alpha(c,50));glow.setColorAt(1,alpha(c,0))
        painter.setPen(Qt.NoPen);painter.setBrush(glow);painter.drawEllipse(q,18,18)
        if current:
            painter.setPen(v._pen(QColor(self.p['current']),1.5));painter.setBrush(Qt.NoBrush);painter.drawEllipse(q,11,11)
        painter.setPen(v._pen(QColor(self.p['bg']),3));painter.setBrush(c if kind!='via' or state=='visited' else QColor(self.p['label']))
        if kind=='start':
            rect=QRectF(q.x()-5.5,q.y()-5.5,11,11);painter.drawRoundedRect(rect,1.5,1.5)
            painter.setPen(v._pen(c,1.5));painter.drawRoundedRect(rect,1.5,1.5)
        elif kind=='goal':
            diamond=QPolygonF([QPointF(q.x()+dx,q.y()+dy) for dx,dy in ((0,-8),(7,0),(0,8),(-7,0))])
            painter.drawPolygon(diamond);painter.setPen(v._pen(c,1.3));painter.drawPolygon(diamond)
        else:
            painter.setPen(v._pen(c,1.6));painter.drawEllipse(q,4,4)
        text=m['id']+('  起点' if kind=='start' else '  终点' if kind=='goal' else '  已经过' if state=='visited' else '  未确认' if state=='unconfirmed' else '')
        if current:text+=' · 当前'
        self.note(painter,m['position'],text,self.p['current'] if current else c)

    def vehicle(self,painter):
        v=self.v;value=v.snapshot.get('position')
        if value is None or not v.snapshot.get('connected'):return
        p=_vec(value);q=v.project(p)
        if not (22<q.x()<v.width()-22 and 66<q.y()<v.height()-55):
            self.note(painter,p,f'无人机在视野外 · H={p[2]:.1f}m',self.p['muted'],(-160,-25));return
        drop=p.copy();drop[2]=self.z0
        if v.view_plane!='xy':
            self.line(painter,[p,drop],alpha(self.p['muted'],150),1.,Qt.DotLine)
            painter.setPen(v._pen(alpha(self.p['muted'],170),1));painter.setBrush(alpha(self.p['bg'],85));painter.drawEllipse(v.project(drop),7,3.5)
        glow=QRadialGradient(q,30);glow.setColorAt(0,alpha(self.p['drone'],28));glow.setColorAt(1,alpha(self.p['drone'],0))
        painter.setBrush(glow);painter.setPen(Qt.NoPen);painter.drawEllipse(q,30,30)
        FlightView._drone(v,painter,arm_pixels=12.,show_label=False,ink_color=self.p['drone'])
        text=f'无人机  H={p[2]:.2f}m'
        velocity=v.snapshot.get('velocity')
        if velocity is not None:text+=f'  {np.linalg.norm(_vec(velocity)):.1f}m/s'
        self.note(painter,p,text,self.p['text'],(-125,-42))

    def orientation(self,painter):
        v=self.v;origin=np.array([v.width()-46,v.height()-64]);painter.setFont(QFont('Microsoft YaHei',8))
        for axis,name,c in [(np.array([1.,0,0]),'X','#d79791'),(np.array([0.,1,0]),'Y','#8aafe0'),(np.array([0.,0,1]),'H','#80bfaf')]:
            direction=axis@np.column_stack((v.right,-v.up));end=origin+direction*19
            painter.setPen(v._pen(QColor(c),1.8));painter.drawLine(QPointF(*origin),QPointF(*end));painter.drawText(QPointF(*(end+np.array([2,-2]))),name)

    def header(self,painter):
        v=self.v;painter.setPen(QColor(self.p['text']));painter.setFont(QFont('Microsoft YaHei',10,QFont.DemiBold))
        title={'3d':'飞行轨迹  /  三维','xy':'飞行轨迹  /  XY俯视','xz':'飞行轨迹  /  高度剖面'}.get(v.view_plane,'飞行轨迹')
        painter.drawText(QPointF(19,28),title)
        self.toggle.move(v.width()-self.toggle.width()-14,11)
        if v.width()>700:
            painter.setFont(QFont('Microsoft YaHei',8));painter.setPen(QColor(self.p['muted']))
            painter.drawText(QRectF(v.width()-430,13,320,21),Qt.AlignRight,f'细格 {self.step/5:g}m  ·  主格 {self.step:g}m  ·  参考面 H={self.z0:g}m')
        painter.setPen(v._pen(alpha(self.p['border'],90),.7));painter.drawLine(QPointF(18,v.height()-33),QPointF(v.width()-18,v.height()-33))
        painter.setFont(QFont('Microsoft YaHei',8));x=19;spacing=min(114,(v.width()-30)/6)
        goal_label='当前/下一点' if v.config.get('online') else 'E 终点'
        via_label='W 经过点' if v.config.get('online') else 'W 已经过'
        for color,label,style in [(self.p['start'],'S 起点','square'),(self.p['via'],via_label,'dot'),(self.p['goal'],goal_label,'diamond'),(self.p['reference'],'参考','dash'),(self.p['actual'],'实测','line'),('#ec9840','机头','line')]:
            y=v.height()-16;painter.setPen(v._pen(QColor(color),2.,Qt.DashLine if style=='dash' else Qt.SolidLine));painter.setBrush(QColor(color))
            if style=='square':painter.drawRect(QRectF(x,y-4,6,6))
            elif style=='dot':painter.drawEllipse(QPointF(x+3,y-1),3,3)
            elif style=='diamond':painter.drawPolygon(QPolygonF([QPointF(x+dx,y+dy) for dx,dy in ((3,-5),(7,-1),(3,3),(-1,-1))]))
            else:painter.drawLine(QPointF(x,y-1),QPointF(x+12,y-1))
            painter.setPen(QColor(self.p['muted']));painter.drawText(QPointF(x+18,y+3),label);x+=spacing

    def paint(self):
        v=self.v;v._camera();self.background();self.labels=[]
        painter=QPainter(v);painter.setRenderHint(QPainter.Antialiasing);painter.drawPixmap(0,0,self.cache)
        painter.setClipRect(QRectF(3,46,v.width()-6,v.height()-81))
        plan=v.display_points(v.snapshot.get('planned_path',[]),600);trail=v.display_points(v.snapshot.get('trail',[]),600)
        if len(trail)>1 and v.view_plane!='xz':
            shadow=trail.copy();shadow[:,2]=self.z0
            self.line(painter,shadow,alpha(self.p['bg'],100),4.)
        self.line(painter,plan,self.p['reference'],1.9,Qt.DashLine)
        self.line(painter,trail,alpha(self.p['actual'],28),8.)
        self.line(painter,trail,self.p['actual'],2.6)
        position=v.snapshot.get('position')
        if position is not None and v.snapshot.get('connected'):
            q=v.project(position);self.labels.append(QRectF(q.x()-26,q.y()-24,70,47))
        for marker in v._landmarks():self.marker(painter,marker)
        self.vehicle(painter);painter.setClipping(False)
        self.orientation(painter);self.header(painter);painter.end()
