import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":8.5})
AZ="#1f5fa8"; AZc="#cfe0f3"; CI="#6b6b6b"; LA="#d9622b"; ES="#0B2545"
fig=plt.figure(figsize=(7.16,2.25)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,32.6); ax.axis("off")
def seta(x0,y0,x1,y1,c=CI,lw=0.9):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=7,color=c,lw=lw,shrinkA=0,shrinkB=0))
def mapa(xc,yc,h):
    w=h*0.55
    for i,dx in enumerate([-1.2,0,1.2]):
        ax.add_patch(Polygon([(xc+dx-w/2,yc-h/2),(xc+dx+w/2,yc-h/2+1),(xc+dx+w/2,yc+h/2+1),(xc+dx-w/2,yc+h/2)],
                     closed=True,fc=AZc if i<2 else "#a9c6e8",ec=AZ,lw=0.6))
yA=24
# imagem
g=np.linspace(0,1,32); im=np.dstack([0.55+0.35*g[:,None]*np.ones(32),0.70+0.2*g[:,None]*np.ones(32),0.95*np.ones((32,32))])
yy,xx=np.mgrid[0:32,0:32]; im[((xx-16)**2/90+(yy-19)**2/30)<1]=[0.85,0.45,0.25]; im[((xx-16)**2+(yy-12)**2)<18]=[0.80,0.40,0.22]
ax.imshow(im,extent=(3,11,yA-4,yA+4),zorder=3); ax.add_patch(Rectangle((3,yA-4),8,8,fill=False,ec=CI,lw=0.6))
ax.text(7,yA+5.2,"$x$",ha="center",fontsize=10)
xs=[24,42,68]; nm=["1","2","B"]; hs=[9,7.5,5.5]
seta(11.4,yA,19.5,yA)
C=[[.27,.25,.26,.22],[.18,.17,.43,.22],[.02,.02,.06,.90]]
yS=11.5
for i,(x,b,h) in enumerate(zip(xs,nm,hs)):
    mapa(x,yA,h); ax.text(x,yA+6.2,r"$\phi_{%s}$"%b,ha="center",fontsize=10)
    # sonda
    seta(x,yA-h/2-0.8,x,yS+3.2,c=AZ,lw=0.8); ax.text(x+0.8,(yA-h/2-0.8+yS+3.2)/2,r"$\mathbf{a}_{%s}$"%b,fontsize=8,color=AZ,va="center")
    ax.add_patch(FancyBboxPatch((x-3,yS),6,2.9,boxstyle="round,pad=0.2",fc="#fff4ec",ec=LA,lw=0.8))
    ax.text(x,yS+1.45,r"$g_{%s}$"%b,ha="center",va="center",fontsize=10)
    y0=3.6
    for k in range(4):
        ax.add_patch(Rectangle((x-2.9+k*1.5,y0),1.35,6.4*C[i][k],fc=ES if k==3 else "#bdbdbd",ec="none"))
    ax.plot([x-3.2,x+3.2],[y0,y0],color=CI,lw=0.5)
    ax.text(x-4.2,y0+3,r"$\mathbf{c}_{%s}$"%b,ha="right",va="center",fontsize=9)
    seta(x,yS-0.3,x,10.4,c=LA,lw=0.8)
seta(28.5,yA,36.5,yA); seta(46.5,yA,51,yA); ax.text(55,yA,r"$\cdots$",ha="center",va="center",fontsize=12); seta(59,yA,63.5,yA)
ax.text(55,6.5,r"$\cdots$",ha="center",va="center",fontsize=12)
# cabeça
seta(72.5,yA,77,yA); ax.add_patch(FancyBboxPatch((77.2,yA-2.6),7.2,5.2,boxstyle="round,pad=0.25",fc="white",ec=ES,lw=0.8))
ax.text(80.8,yA,"cabeça",ha="center",va="center",fontsize=8); seta(84.8,yA,88,yA); ax.text(88.6,yA,r"$p(x)$",va="center",fontsize=10,color=ES)
# caminho
ax.annotate("",xy=(74,1.6),xytext=(18,1.6),arrowprops=dict(arrowstyle="-|>",color=LA,lw=1.0,mutation_scale=8,linestyle=(0,(4,2))),zorder=0)
ax.text(76,1.8,"caminho da decisão",fontsize=8.5,color=LA,va="center")
fig.savefig("/home/claude/gh_incerteza_fuzzy/artigo_ijcnn/figuras/blocos.png",dpi=300)
fig.savefig("/home/claude/gh_incerteza_fuzzy/artigo_ijcnn/figuras/blocos.pdf")
