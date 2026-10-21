"""Vector scene geometry and importable coordinates; no simulated result claims."""
import argparse,csv,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from t1_obstacle_scene import ROOT,load_scene,route,box

def export(scene_path,output):
    scene=load_scene(scene_path);planned=route(scene);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.linewidth':.7,
        'xtick.direction':'in','ytick.direction':'in','svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(4.8,4.0),layout='constrained')
    for i,b in enumerate(scene['buildings']):
        low,high=box(b,scene['clearance_m'])
        ax.add_patch(Rectangle(low[:2],high[0]-low[0],high[1]-low[1],facecolor='none',edgecolor='#777777',linestyle=(0,(3,2)),linewidth=.7,label='Clearance boundary' if i==0 else None))
        low,high=box(b);ax.add_patch(Rectangle(low[:2],high[0]-low[0],high[1]-low[1],facecolor='#d3d3d3',edgecolor='#454545',linewidth=.8,label='Static building' if i==0 else None))
        ax.text(*b['center_ned_m'][:2],b['name'].replace('T1Building_',''),ha='center',va='center',fontsize=9)
    start=scene['start_hover_ned_m'];goal=scene['observation_goal_ned_m'];target=scene['target']['position_ned_m']
    ax.plot([start[0],goal[0]],[start[1],goal[1]],color='#9b9b9b',linestyle=':',linewidth=1,label='Blocked direct segment')
    points=planned['waypoints_ned_m'];ax.plot([p[0] for p in points],[p[1] for p in points],'-o',color='#286196',linewidth=1.3,markersize=3,label='Planned route')
    ax.scatter(*start[:2],marker='s',s=35,c='#2e7050',edgecolor='black',linewidth=.5,zorder=4,label='UAV start')
    ax.scatter(*goal[:2],marker='D',s=30,c='#286196',edgecolor='black',linewidth=.5,zorder=4,label='Observation goal')
    ax.scatter(*target[:2],marker='o',s=40,c='#b84a48',edgecolor='black',linewidth=.5,zorder=4,label='White-stripe target')
    ax.set(xlabel='World NED X (m)',ylabel='World NED Y (m)',xlim=(scene['workspace']['low_ned_m'][0],scene['workspace']['high_ned_m'][0]),ylim=(scene['workspace']['low_ned_m'][1],scene['workspace']['high_ned_m'][1]))
    ax.set_aspect('equal',adjustable='box');ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,fontsize=8,handlelength=1.7)
    for suffix in ('svg','pdf','png'):fig.savefig(output/('configured_scene.'+suffix),dpi=600,facecolor='white')
    plt.close(fig)
    with (output/'route_waypoints.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['waypoint_index','x_ned_m','y_ned_m','z_ned_m','data_role'])
        writer.writerows([i,*p,'planned_not_measured'] for i,p in enumerate(points))
    with (output/'building_geometry.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['name','x_center_m','y_center_m','z_center_m','x_size_m','y_size_m','z_size_m','clearance_m'])
        writer.writerows([b['name'],*b['center_ned_m'],*b['size_m'],scene['clearance_m']] for b in scene['buildings'])
    (output/'provenance.json').write_text(json.dumps(dict(scene_config=str(Path(scene_path).resolve()),scene_sha256=hashlib.sha256(Path(scene_path).read_bytes()).hexdigest(),route=planned,
        scope='configured geometry and planned static route; no measured trajectory or completed avoidance claim',new_flights=0,ai_assisted=True),ensure_ascii=False,indent=2),encoding='utf-8')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',type=Path,default=ROOT/'configs/t1_building_scene_v1.json');p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(export(a.scene,a.output))
