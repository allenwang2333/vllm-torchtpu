import re, sys, hashlib
def loops(path):
    src=[re.sub(r"loc\(.*","",l.rstrip("\n")) for l in open(path)]
    i=[n for n,l in enumerate(src) if "func.func @sc_topk" in l][0]
    src=src[i:]
    out=[]
    for n,l in enumerate(src):
        if "scf.for" not in l: continue
        d=0; started=False
        for m in range(n, len(src)):
            d += src[m].count("{") - src[m].count("}")
            if src[m].count("{"): started=True
            if started and d<=0:
                out.append((n+i, m+i, src[n:m+1])); break
    return out
def norm(body):
    t="\n".join(body)
    t=re.sub(r"%[A-Za-z0-9_]+", "%v", t)
    t=re.sub(r"#map\d*", "#map", t)
    return re.sub(r"\s+"," ",t).strip()
res={}
for p in sys.argv[1:]:
    print("="*78); print(p)
    res[p]=[]
    for s,e,b in loops(p):
        txt="\n".join(b)
        tag=[]
        if "vector_store_idx" in txt: tag.append("SCATTER")
        if re.search(r"tpu\.vector_store\b", txt): tag.append("STORE")
        if "tpu.scan" in txt: tag.append("SCAN")
        if "tpu.all_reduce" in txt: tag.append("AR")
        if "tpu.enqueue_dma" in txt: tag.append("DMA")
        h=hashlib.md5(norm(b).encode()).hexdigest()[:10]
        res[p].append((h,e-s+1," ".join(tag)))
        print(f"  {s:5d}-{e:5d} {e-s+1:4d}L md5={h}  {' '.join(tag) or '-'}")
if len(sys.argv)==3:
    a=[x[0] for x in res[sys.argv[1]]]; b=[x[0] for x in res[sys.argv[2]]]
    print("="*78)
    print("shared loop fingerprints:", len(set(a)&set(b)))
    print("base-only:", [x for x in res[sys.argv[1]] if x[0] not in set(b)])
    print("v3-only  :", [x for x in res[sys.argv[2]] if x[0] not in set(a)])
