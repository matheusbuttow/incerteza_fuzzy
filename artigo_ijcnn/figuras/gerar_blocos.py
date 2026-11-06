import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle, Circle, FancyArrowPatch
plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":8.5})
AZ="#1f5fa8"; AZc="#dbe7f5"; CI="#7a7a7a"; LA="#d9622b"; LAc="#fdeee4"; ES="#0B2545"; CZ="#c4c4c4"
fig=plt.figure(figsize=(7.16,2.0)); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,100); ax.set_ylim(0,28); ax.axis("off")
def seta(x0,y0,x1,y1,c=CI,lw=0.9):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=7,color=c,lw=lw,shrinkA=0,shrinkB=0))
def caixa(x,y,w,h,fc,ec,txt,fs=8.5,c="k"):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0,rounding_size=0.8",fc=fc,ec=ec,lw=0.9))
    ax.text(x,y,txt,ha="center",va="center",fontsize=fs,color=c)
yT, yS, yB = 22.5, 13.5, 2.5
# imagem (ícone)
ax.add_patch(FancyBboxPatch((2.5,yT-3.5),7,7,boxstyle="round,pad=0,rounding_size=0.6",fc="#eef3fa",ec=CI,lw=0.8))
ax.add_patch(Polygon([(3.2,yT-3),(5.6,yT+0.6),(7.4,yT-1.6),(8.3,yT-0.6),(8.8,yT-3)],closed=True,fc="#9fb9d8",ec="none"))
ax.add_patch(Circle((7.6,yT+1.8),0.8,fc="#f0b35a",ec="none"))
ax.text(6,yT-5.3,"imagem $x$",ha="center",fontsize=8.5)
cols=[(22,"bloco 1"),(38,"bloco 2"),(66,r"bloco $B$")]
C=[[.27,.25,.26,.22],[.18,.17,.43,.22],[.03,.03,.07,.87]]
seta(9.9,yT,17,yT)
topos=[]
for i,((x,nm),c) in enumerate(zip(cols,C)):
    caixa(x,yT,10,5.2,AZc,AZ,nm)
    seta(x,yT-2.7,x,yS+2.1,c=AZ,lw=0.8)
    caixa(x,yS,7.2,3.8,LAc,LA,r"sonda $g_{%s}$"%["1","2","B"][i],fs=8)
    seta(x,yS-2.0,x,yB+6.9,c=LA,lw=0.8)
    for k in range(4):
        ax.add_patch(Rectangle((x-3.0+k*1.55,yB),1.3,6.2*c[k],fc=ES if k==3 else CZ,ec="none"))
    ax.plot([x-3.3,x+3.3],[yB,yB],color=CI,lw=0.5)
    topos.append((x+1.65,yB+6.2*c[3]))
seta(27.2,yT,32.8,yT); seta(43.2,yT,48,yT); ax.text(52,yT,r"$\cdots$",ha="center",va="center",fontsize=12); seta(56,yT,60.8,yT)
ax.text(52,yS,r"$\cdots$",ha="center",va="center",fontsize=12)
caixa(82,yT,8,5.2,"white",ES,"saída")
seta(71.2,yT,77.8,yT); seta(86.2,yT,90.5,yT); ax.text(91.2,yT,r"$p(x)$",va="center",fontsize=9.5,color=ES)
# caminho: liga o topo da barra da resposta final em cada sonda
xs_,ys_=zip(*topos)
ax.plot(xs_,ys_,color=LA,lw=1.1,ls=(0,(3,1.6)),marker="o",ms=2.5,zorder=5)
ax.text(71,yB+3.2,"caminho\nda decisão",fontsize=8.5,color=LA,va="center")
ax.text(14.5,yB+3.2,"opinião\nparcial $\\mathbf{c}_b$",fontsize=8,color=CI,ha="right",va="center")
fig.savefig("/home/claude/gh_incerteza_fuzzy/artigo_ijcnn/figuras/blocos.png",dpi=300)
fig.savefig("/home/claude/gh_incerteza_fuzzy/artigo_ijcnn/figuras/blocos.pdf")
