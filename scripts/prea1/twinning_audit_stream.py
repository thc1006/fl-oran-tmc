"""Streaming fragmentation audit of the Twinning -kpm tar.xz (read via Python lzma; no disk extract).
entity=UE/IMSI, run=(cluster,slicing,sched,RESERVATION), step=row index in each bs/<IMSI>_metrics.csv.
Per (run,UE) group: intact (whole group->1 client) windows are all contiguous (frag 1.0);
row-level (random_split / dirichlet) scatter the group's rows across clients -> windows span gaps."""
import sys, re, tarfile
import numpy as np
SEQ_LEN, N, SEED = 5, 8, 0
RE = re.compile(r"cluster_(\d+)/slicing_(\d+)/scheduling_?(\d+)/RESERVATION-([0-9]+)/bs/(\d+)_metrics\.csv")
def cw(steps):
    n=len(steps)
    if n<SEQ_LEN: return 0,0
    sp=np.sort(steps); spans=sp[SEQ_LEN-1:]-sp[:n-SEQ_LEN+1]
    return int((spans==SEQ_LEN-1).sum()), len(spans)
def main():
    rng=np.random.default_rng(SEED); pdir=rng.dirichlet([1.0]*N)
    acc={"entity/run-level (intact)":[0,0],"random_split (row)":[0,0],"dirichlet (row,a=1)":[0,0]}
    t=tarfile.open(fileobj=sys.stdin.buffer, mode="r|xz")
    ncsv=ngrp=tot=0
    for m in t:
        if not m.isfile(): continue
        if not RE.search(m.name.replace("\\","/")): continue
        ncsv+=1
        data=t.extractfile(m).read()
        nrows=max(0, data.count(b"\n")-1)
        if nrows<SEQ_LEN: continue
        ngrp+=1; tot+=nrows; steps=np.arange(nrows)
        acc["entity/run-level (intact)"][0]+=nrows-SEQ_LEN+1; acc["entity/run-level (intact)"][1]+=nrows-SEQ_LEN+1
        for label,cl in (("random_split (row)",rng.integers(0,N,size=nrows)),("dirichlet (row,a=1)",rng.choice(N,size=nrows,p=pdir))):
            for c in range(N):
                a,b=cw(steps[cl==c]); acc[label][0]+=a; acc[label][1]+=b
        if ncsv%500==0: print(f"  ...{ncsv} bs CSVs processed", flush=True)
    print(f"\nDONE: {ncsv} bs CSVs, {ngrp} usable (run,UE) groups, {tot:,} total rows")
    print(f"{'mode':>26} | {'fragmentation_score':>19} | windows")
    for k,(c,w) in acc.items():
        print(f"{k:>26} | {(c/w if w else float('nan')):>19.4f} | {w:,}")
if __name__=="__main__": main()
