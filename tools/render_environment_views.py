"""Reconstruct two evaluation scenes from saved poses and robot trajectories."""
from pathlib import Path
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'docs/results/stage3_687935'
OUT=ROOT/'docs/images'

def main():
    layouts={x['seed']:x for x in json.loads((DATA/'evaluation_layouts.json').read_text())['episodes']}
    episodes={int(x['seed']):x for x in csv.DictReader((DATA/'evaluation_episodes.csv').open())}
    tracks={seed:[] for seed in (140000,140002)}
    with (DATA/'evaluation_trajectories.csv').open() as f:
        for row in csv.DictReader(f):
            seed=int(row['seed'])
            if seed in tracks:tracks[seed].append((float(row['robot_x']),float(row['robot_y'])))
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
    for seed,suffix,title,color in [(140000,'success','Successful navigation','#167c63'),(140002,'timeout','Timeout after hesitation','#b16c0a')]:
        scene=layouts[seed]
        row=episodes[seed]
        points=tracks[seed]
        fig,ax=plt.subplots(figsize=(7.8,7.6),facecolor='#f4f7fb')
        ax.set_facecolor('#ffffff')
        ax.add_patch(Rectangle((-4.5,-4.5),9,9,fill=False,lw=3,ec='#34445a',zorder=3))
        for x,y,z,yaw in scene['object_layout'].values():
            assert abs(yaw)<1e-9
            ax.add_patch(Rectangle((x-.5,y-.5),1,1,fc='#b7c5d7',ec='#687d98',lw=1.2,zorder=4))
        gx,gy=scene['goal']
        ax.add_patch(Circle((gx,gy),.25,fc='#78d5a1',ec='#167c63',lw=1.8,zorder=6))
        ax.plot([p[0] for p in points],[p[1] for p in points],color='#2f70ca',lw=2.1,zorder=5,label='Recorded robot path')
        sx,sy=points[0]
        yaw=scene['start'][2]
        triangle=[]
        for dx,dy in ((.23,0),(-.17,.15),(-.17,-.15)):
            triangle.append((sx+dx*math.cos(yaw)-dy*math.sin(yaw),sy+dx*math.sin(yaw)+dy*math.cos(yaw)))
        ax.add_patch(Polygon(triangle,fc='#245aa0',ec='white',lw=1,zorder=8))
        if row['outcome']=='timeout':ax.scatter([points[-1][0]],[points[-1][1]],s=70,c=color,marker='X',zorder=9)
        ax.annotate('Start',(sx,sy),xytext=(7,8),textcoords='offset points',color='#245aa0',weight='bold',fontsize=10)
        ax.annotate('Goal',(gx,gy),xytext=(7,8),textcoords='offset points',color='#167c63',weight='bold',fontsize=10)
        ax.set(xlim=(-4.8,4.8),ylim=(-4.8,4.8),aspect='equal',xlabel='x (m)',ylabel='y (m)')
        ax.set_xticks(range(-4,5,2));ax.set_yticks(range(-4,5,2))
        ax.grid(color='#e6edf4',lw=.8,zorder=0)
        for spine in ax.spines.values():spine.set_visible(False)
        ax.tick_params(length=0,labelcolor='#65758b')
        fig.suptitle(title,fontsize=19,fontweight='bold',color='#24344a',y=.96)
        ax.set_title(f"Stage 3 · {row['randomized_object_count']} boxes · seed {seed} · {row['length']} steps",fontsize=11,color='#64748b',pad=12)
        fig.text(.5,.035,'Recorded layout reconstruction · 9 × 9 m arena',ha='center',fontsize=10,color='#64748b')
        fig.tight_layout(rect=(0,.055,1,.95))
        fig.savefig(OUT/f'environment_{suffix}.png',dpi=160,facecolor=fig.get_facecolor())
        plt.close(fig)
    print('Created two recorded-layout environment views in',OUT)

if __name__=='__main__':main()
