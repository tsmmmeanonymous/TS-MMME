#!/usr/bin/env python3
"""Build TS-MMME master/train/val/test for the FULL MMM-Trend benchmark.
Mirrors bench_common.load_rows() EXACTLY (same filter + label rule) so rows/labels/
splits are identical to the BERT baselines, but also captures the extra fields TS-MMME
needs (Location_std -> location, topics_new -> cluster_id, day/time, Title).
Split = train <=2023-Q4 (10% val, seed 2378), test = each 2024 quarter (+ pooled)."""
import openpyxl, re, os, json, collections, random, datetime, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO, "baselines"))
import bench_common as bc   # reuse cell/qof/_body/IMG_DIRS/IMG_COLS/SEED

OUTDIR=os.path.join(REPO,"ts_mmme/Dataset/Experiment"); os.makedirs(OUTDIR,exist_ok=True)
MASTER=os.path.join(REPO,"ts_mmme/Dataset/experimental_master.json")
TOPIC_ID={
 "General social narratives and mixed-content claims":0,
 "Urban governance, civic issues, and local administrative claims":1,
 "Political statements and public discourse-related-narratives":2,
 "Science, health, and environmental misinformation":3,
 "Digital media, online narratives, and platformdriven misinformation":4,
 "Government schemes, infrastructure projects, and policy-related claims":5,
 "Climate, weather, air pollution, and environmental condition reports":6}
def cell(v): return bc.cell(v)

# ---- mirror load_rows, capturing extra fields ----
have=[set(os.listdir(d)) if os.path.isdir(d) else set() for d in bc.IMG_DIRS]
wb=openpyxl.load_workbook(bc.XLSX, read_only=True); ws=wb["Complete-Dataset"]
it=ws.iter_rows(values_only=True); H=[h for h in next(it) if h is not None]; idx={h:i for i,h in enumerate(H)}
R=[]
for r in it:
    q=bc.qof(r[idx["dates"]])
    if not q or not (2012<=q[0]<=2024): continue
    b=bc._body(r[idx["File"]])
    txt=b if (b and len(b)>=20) else (cell(r[idx["Title"]])+". "+cell(r[idx["Summary"]])).strip()
    img=None
    for k in bc.IMG_COLS:
        v=cell(r[idx[k]])
        if not v: continue
        for d,h in zip(bc.IMG_DIRS,have):
            if v in h: img=os.path.join(d,v); break
        if img: break
    mm=re.match(r"(\d{2})/(\d{2})/(\d{4})",cell(r[idx["dates"]]))
    R.append({"text":txt,"auth_raw":cell(r[idx["Authenticity"]]),"yq":q,"month":int(mm[2]),"day":int(mm[1]),
              "year":str(q[0]),"img":img,"aug":cell(r[idx["File"]]).startswith("ID-AUG-"),
              "loc":cell(r[idx["Location_std"]]) or "Pan_India",
              "loc_bn":cell(r[idx["Precise Locations"]]),
              "cluster":TOPIC_ID.get(cell(r[idx["topics_new"]]),0)})
wb.close()
# label consolidation (identical to bench_common)
FIXED={"True":"real","Real":"real","False":"fake","Fake":"fake","Problematic":"fake"}
for x in R: x["lab"]=FIXED.get(x["auth_raw"])
qc=collections.defaultdict(lambda:{"real":0,"fake":0})
for x in R:
    if x["lab"]: qc[x["yq"]][x["lab"]]+=1
for x in sorted([x for x in R if x["lab"] is None],key=lambda x:(x["yq"][0],x["yq"][1],x["auth_raw"])):
    c=qc[x["yq"]]; cls="real" if c["real"]<c["fake"] else "fake"; c[cls]+=1; x["lab"]=cls
for x in R: x["y"]=0 if x["lab"]=="real" else 1

tr,val,tests=bc.splits(R)
print(f"rows={len(R)}  train={len(tr)} val={len(val)}  test2024: "+", ".join(f"Q{k}={len(tests[k])}" for k in range(1,5)))
print(f"train label: real={sum(1 for i in tr if R[i]['y']==0)} fake={sum(1 for i in tr if R[i]['y']==1)}")
print(f"images resolved: {sum(1 for x in R if x['img'])}/{len(R)}")
print(f"location nodes: {len(set(x['loc'] for x in R))}")

def rec(i,rid):
    x=R[i]; y,m,d=x["yq"][0],x["month"],x["day"]; s=(m-1)//3+1
    try: dt=datetime.datetime(y,m,d)
    except ValueError:
        try: dt=datetime.datetime(y,m,1)
        except ValueError: dt=datetime.datetime(y,1,1)
    return {"content":x["text"],"image_path":x["img"] or "","cluster_id":x["cluster"],
            "time":int(dt.timestamp()*1000),"label":"truth" if x["y"]==0 else "rumor",
            "year":y,"season":s,"month":m,"id":rid,"weight":1,"day":d,
            "year_month":f"{y}-{m:02d}","year_season":f"{y}-{s:02d}",
            "location":x["loc"],"location_bengali":x["loc_bn"],"original_location":x["loc"]}
def dumpl(recs,path):
    with open(path,"w",encoding="utf-8") as f:
        for x in recs: f.write(json.dumps(x,ensure_ascii=False)+"\n")

rid=1; train_recs=[]
for i in tr: train_recs.append(rec(i,rid)); rid+=1
val_recs=[]
for i in val: val_recs.append(rec(i,rid)); rid+=1
q_recs={}
for k in range(1,5):
    q_recs[k]=[]
    for i in tests[k]: q_recs[k].append(rec(i,rid)); rid+=1
pooled=[r for k in range(1,5) for r in q_recs[k]]
dumpl(train_recs, os.path.join(OUTDIR,"train_dataset.json"))
dumpl(pooled,     os.path.join(OUTDIR,"test_dataset.json"))         # default test = pooled 2024
for k in range(1,5): dumpl(q_recs[k], os.path.join(OUTDIR,f"test_2024_Q{k}.json"))
dumpl(train_recs+pooled, MASTER)
dumpl(val_recs, os.path.join(OUTDIR,"val_dataset.json"))   # held-out 10% of pre-2024 data (same rows as the baselines' validation split)
print(f"wrote train({len(train_recs)}) + pooled-test({len(pooled)}) + per-quarter test files to {OUTDIR}")
