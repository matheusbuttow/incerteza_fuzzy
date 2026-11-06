import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":8})
AZ="#1f5fa8"; AZc="#cfe0f3"; CI="#6b6b6b"; LA="#d9622b"; ES="#0B2545"
fig=plt.figure(figsize=(7.16,3.55)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,50); ax.axis("off")

def seta(x0,y0,x1,y1,c=CI,lw=0.9):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=7,color=c,lw=lw,shrinkA=0,shrinkB=0))
def mapa(xc,yc,h,d,n=3):
    """pilha de mapas de características: altura h (resolução), profundidade d (canais)"""
    w=h*0.55
    for i in range(n):
        dx=-d/2+i*d/(n-1) if n>1 else 0
        ax.add_patch(Polygon([(xc+dx-w/2,yc-h/2),(xc+dx+w/2,yc-h/2+1.0),(xc+dx+w/2,yc+h/2+1.0),(xc+dx-w/2,yc+h/2)],
                     closed=True,fc=AZc if i<n-1 else "#a9c6e8",ec=AZ,lw=0.6))

# ---- imagem de entrada (sintética)
yA=41
g=np.linspace(0,1,32); im=np.dstack([0.55+0.35*g[:,None]*np.ones(32),0.70+0.2*g[:,None]*np.ones(32),0.95*np.ones((32,32))])
yy,xx=np.mgrid[0:32,0:32]; m=((xx-16)**2/90+(yy-19)**2/30)<1; im[m]=[0.85,0.45,0.25]
m2=((xx-16)**2+(yy-12)**2)<18; im[m2]=[0.80,0.40,0.22]
ax.imshow(im,extent=(1.5,9.5,yA-4,yA+4),zorder=3); ax.add_patch(Rectangle((1.5,yA-4),8,8,fill=False,ec=CI,lw=0.6))
ax.text(5.5,yA+5.2,"imagem $x$",ha="center")

# ---- blocos
xs=[18,31,44,62,75]; nomes=[r"$\phi_1$",r"$\phi_2$",r"$\phi_3$",r"$\phi_{B-1}$",r"$\phi_B$"]
hs=[9,8,7,5.5,5]; ds=[1.5,2.2,2.8,3.6,4.0]
seta(9.8,yA,13.6,yA)
for i,(x,nm,h,d) in enumerate(zip(xs,nomes,hs,ds)):
    mapa(x,yA,h,d)
    ax.text(x,yA+6.6,nm,ha="center",fontsize=9)
    ax.text(x,yA-6.6,r"$\mathbf{Z}_{%s}(x)$"%(["1","2","3","B-1","B"][i]),ha="center",fontsize=7,color=CI)
    if i<len(xs)-1:
        nx=xs[i+1]
        if i==2:
            seta(x+4,yA,50.5,yA); ax.text(53,yA,r"$\cdots$",ha="center",va="center",fontsize=11); seta(55.5,yA,nx-4,yA)
        else: seta(x+4,yA,nx-4,yA)
# cabeça e saída
seta(79.5,yA,83.4,yA)
ax.add_patch(FancyBboxPatch((83.6,yA-2.8),7.4,5.6,boxstyle="round,pad=0.25",fc="white",ec=ES,lw=0.8))
ax.text(87.3,yA,"cabeça\n+ softmax",ha="center",va="center",fontsize=6.5)
seta(91.3,yA,93.4,yA)
ax.text(93.8,yA,r"$p(x)$",va="center",fontsize=9,color=ES)

# ---- sondas e opiniões parciais
K=4
C=[[.27,.25,.26,.22],[.18,.17,.43,.22],[.12,.10,.30,.48],[.05,.04,.11,.80],[.02,.02,.06,.90]]
yS=25.5
for i,x in enumerate(xs):
    b=["1","2","3","B-1","B"][i]
    seta(x,yA-7.9,x,yS+3.5,c=AZ,lw=0.8)
    ax.text(x+0.7,(yA-7.6+yS+3.3)/2,r"$\mathbf{a}_{%s}(x)$"%b,fontsize=7,color=AZ,va="center")
    ax.add_patch(FancyBboxPatch((x-3.4,yS),6.8,3.0,boxstyle="round,pad=0.2",fc="#fff4ec",ec=LA,lw=0.8))
    ax.text(x,yS+1.5,r"sonda $g_{%s}$"%b,ha="center",va="center",fontsize=7)
    seta(x,yS-0.3,x,yS-1.9,c=LA,lw=0.8)
    # barras c_b(x)
    y0=yS-9.0; bw=1.35
    for k in range(K):
        v=C[i][k]; cc=ES if k==3 else "#b9b9b9"
        ax.add_patch(Rectangle((x-2.9+k*1.5,y0),bw,6.6*v,fc=cc,ec="none"))
    ax.plot([x-3.2,x+3.2],[y0,y0],color=CI,lw=0.5)
    ax.text(x,y0-1.4,r"$\mathbf{c}_{%s}(x)$"%b,ha="center",fontsize=7.5)
ax.text(53,yS+1.5,r"$\cdots$",ha="center",va="center",fontsize=11); ax.text(53,yS-6,r"$\cdots$",ha="center",va="center",fontsize=11)
ax.text(0.8,yS+1.5,"o que a rede\ndiria se parasse\nno bloco $b$",fontsize=6.5,va="center",color=LA)
ax.text(0.8,yS-5.8,"opinião parcial\n(prob. de\ncada classe)",fontsize=6.5,va="center",color=CI)

# ---- caminho da decisão
ax2=fig.add_axes([0.20,0.095,0.56,0.15])
B=16; b=np.arange(1,B+1)
cam=np.array([.27,.24,.22,.48,.41,.38,.55,.58,.62,.70,.74,.80,.84,.87,.89,.90])
ax2.axvspan(B-B/3+0.5,B+0.5,color="#eef3fa",lw=0)
ax2.plot(b,cam,color=ES,lw=1.2,marker="o",ms=2.6)
ax2.axhline(0.5,color=CI,lw=0.5,ls=":")
Dd=7; ax2.axvline(Dd,color=LA,lw=0.9,ls="--"); ax2.text(Dd+0.2,0.06,r"$D$: a decisão firma",color=LA,fontsize=6.5)
ax2.text(B-B/3+0.8,0.16,r"último terço ($A$)",fontsize=6.5,color=AZ)
ax2.annotate("troca de opinião",xy=(4,0.48),xytext=(1.6,0.80),fontsize=6.5,color=CI,arrowprops=dict(arrowstyle="-",color=CI,lw=0.5))
ax2.set_xlim(0.5,B+0.5); ax2.set_ylim(0,1); ax2.set_xticks([1,4,8,12,16]); ax2.set_xticklabels(["1","","","","$B$"])
ax2.set_yticks([0,0.5,1]); ax2.tick_params(labelsize=6.5,length=2)
for s in ["top","right"]: ax2.spines[s].set_visible(False)
ax2.set_xlabel("bloco $b$",fontsize=7,labelpad=0); ax2.set_ylabel(r"$c_b(x)_{\hat y}$",fontsize=7.5)
fig.text(0.78,0.17,"caminho da decisão:\nconfiança de cada sonda\nna resposta final $\\hat y$\nda rede",fontsize=7,va="center")
fig.savefig("/home/claude/gh_incerteza_fuzzy/artigo_ijcnn/figuras/blocos.png",dpi=300)
fig.savefig("/home/claude/gh_incerteza_fuzzy/artigo_ijcnn/figuras/blocos.pdf")
