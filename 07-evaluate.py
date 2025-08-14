import os, itertools, pandas as pd, numpy as np
from rdkit import Chem, RDLogger
from tqdm import tqdm
RDLogger.DisableLog("rdApp.*");  tqdm.pandas()
from eval_functions import read_results, flatten, get_stereocenters, per_stereocenter

SEEDS           = [0, 1, 42]
FULL_MODELS     = ["c1","r1","a2","a5","a10","a20","a50","npstereo","rp","m65"]
PARTIAL_MODELS  = ["npstereo","rp","m65"]
RESULT_DIR      = "results"; os.makedirs(RESULT_DIR, exist_ok=True)

# 1 SMILES validity 
def smiles_validity(path):
    try:
        with open(os.path.join(path, "pred-test.txt")) as fh:
            smi = [ln.strip() for ln in fh if ln.strip()]
        ok = sum(Chem.MolFromSmiles(s) is not None for s in smi)
        return ok / len(smi) if smi else 0.0
    except FileNotFoundError:
        return np.nan

def collect_validity():
    rows=[]
    for mdl, suf in itertools.product(FULL_MODELS, ["canonical", "randomized"]):
        vals = [smiles_validity(f"predictions/seed-{s}/{mdl}-{suf}") for s in SEEDS]
        mean = np.nanmean(vals)
        std  = np.nanstd(vals)
        rows.append(dict(
            augmentation = f"{mdl}-{suf}",
            validity_pct = f"{mean*100:.2f} ± {std*100:.2f}"
        ))
    return pd.DataFrame(rows)

# 2 full‑assignment
def fa_acc(path):
    df = read_results(path)
    groups = df.groupby("source")["target"].apply(list).to_dict()
    f = lambda r, ks: any(r[k] in groups.get(r["source"], []) for k in ks)
    return (df.apply(f, ks=["beam_1"], axis=1).mean(),
            df.apply(f, ks=["beam_1","beam_2"], axis=1).mean(),
            df.apply(f, ks=["beam_1","beam_2","beam_3"], axis=1).mean())

def aggregate_full():
    rows=[]
    for mdl in FULL_MODELS:
        for suf in ["canonical","randomized"]:
            tmp=[fa_acc(f"predictions/seed-{s}/{mdl}-{suf}/") for s in SEEDS]
            arr=pd.DataFrame(tmp, columns=["top1","top2","top3"])
            rows.append(dict(
                augmentation=f"{mdl}-{suf}",
                top1=f"{arr.top1.mean()*100:.2f} ± {arr.top1.std()*100:.2f}",
                top2=f"{arr.top2.mean()*100:.2f} ± {arr.top2.std()*100:.2f}",
                top3=f"{arr.top3.mean()*100:.2f} ± {arr.top3.std()*100:.2f}",
            ))
    return pd.DataFrame(rows)

# 3 partial full‑assignment
def _read_partial(path):
    src = pd.read_csv(f"{path}/src-test.txt", header=None, names=["source"])
    tgt = pd.read_csv(f"{path}/tgt-test.txt", header=None, names=["target"])
    with open(f"{path}/pred-test.txt") as fh:
        preds = [l.strip() for l in fh]
    beams = list(zip(*[iter(preds)]*3))
    df = pd.concat([src, tgt], axis=1)
    for i, b in enumerate(zip(*beams), 1):
        df[f"beam_{i}"] = b
    return df.apply(lambda c: c.str.replace(" ", ""), axis=0)

def partial_fa_acc(path):
    df = _read_partial(path); df["flat"] = df.source.apply(flatten)
    groups = df.groupby("flat")["target"].apply(list).to_dict()
    f = lambda r, ks: any(r[k] in groups.get(r["flat"], []) for k in ks)
    return (df.apply(f, ks=["beam_1"], axis=1).mean(),
            df.apply(f, ks=["beam_1","beam_2"], axis=1).mean(),
            df.apply(f, ks=["beam_1","beam_2","beam_3"], axis=1).mean())

def aggregate_partial_fa():
    rows=[]
    for mdl in PARTIAL_MODELS:
        for suf in ["canonical-partial","randomized-partial"]:
            tmp=[partial_fa_acc(f"predictions/seed-{s}/{mdl}-{suf}/") for s in SEEDS]
            arr=pd.DataFrame(tmp, columns=["top1","top2","top3"])
            rows.append(dict(
                augmentation=f"{mdl}-{suf}",
                top1=f"{arr.top1.mean()*100:.2f} ± {arr.top1.std()*100:.2f}",
                top2=f"{arr.top2.mean()*100:.2f} ± {arr.top2.std()*100:.2f}",
                top3=f"{arr.top3.mean()*100:.2f} ± {arr.top3.std()*100:.2f}",
            ))
    return pd.DataFrame(rows)

# 4 per‑stereocenter
def stereo_acc(path):
    df = read_results(path)
    df["n"] = df.source.apply(lambda s: len(get_stereocenters(Chem.MolFromSmiles(s))))
    df = df[df.n > 0]
    if df.empty: return 0,0,0
    groups = df.groupby("source")["target"].apply(list).to_dict()
    for i in (1,2,3):
        df[f"a{i}"] = df.apply(
            lambda r: max(per_stereocenter(r.source, t, r[f"beam_{i}"]) for t in groups[r.source]),
            axis=1
        )
    return df.a1.mean(), df[["a1","a2"]].max(axis=1).mean(), df[["a1","a2","a3"]].max(axis=1).mean()

def aggregate_stereo():
    rows=[]
    for mdl in FULL_MODELS:
        for suf in ["canonical","randomized"]:
            tmp=[stereo_acc(f"predictions/seed-{s}/{mdl}-{suf}/") for s in SEEDS]
            arr=pd.DataFrame(tmp, columns=["top1","top2","top3"])
            rows.append(dict(
                augmentation=f"{mdl}-{suf}",
                top1=f"{arr.top1.mean()*100:.2f} ± {arr.top1.std()*100:.2f}",
                top2=f"{arr.top2.mean()*100:.2f} ± {arr.top2.std()*100:.2f}",
                top3=f"{arr.top3.mean()*100:.2f} ± {arr.top3.std()*100:.2f}",
            ))
    return pd.DataFrame(rows)

# 5 partial per‑stereocenter
def partial_stereo_acc(path):
    df = read_results(path)
    df["n"] = df.source.apply(lambda s: len(get_stereocenters(Chem.MolFromSmiles(s))))
    df = df[df.n > 0]
    if df.empty: return 0,0,0
    df["flat"] = df.source.apply(flatten)
    groups = df.groupby("flat")["target"].apply(list).to_dict()

    cache = {}
    def sc(src, tgt, beam):
        key = (src, tgt, beam)
        if key not in cache:
            cache[key] = per_stereocenter(src, tgt, beam)
        return cache[key]

    for i in (1,2,3):
        df[f"a{i}"] = df.apply(
            lambda r: max(sc(r.source, t, r[f"beam_{i}"]) for t in groups[r.flat]),
            axis=1
        )
    return df.a1.mean(), df[["a1","a2"]].max(axis=1).mean(), df[["a1","a2","a3"]].max(axis=1).mean()

def aggregate_partial_stereo():
    rows=[]
    for mdl in PARTIAL_MODELS:
        for suf in ["canonical-partial","randomized-partial"]:
            tmp=[partial_stereo_acc(f"predictions/seed-{s}/{mdl}-{suf}/") for s in SEEDS]
            arr=pd.DataFrame(tmp, columns=["top1","top2","top3"])
            rows.append(dict(
                augmentation=f"{mdl}-{suf}",
                top1=f"{arr.top1.mean()*100:.2f} ± {arr.top1.std()*100:.2f}",
                top2=f"{arr.top2.mean()*100:.2f} ± {arr.top2.std()*100:.2f}",
                top3=f"{arr.top3.mean()*100:.2f} ± {arr.top3.std()*100:.2f}",
            ))
    return pd.DataFrame(rows)

if __name__ == "__main__":
    collect_validity()          .to_csv(f"{RESULT_DIR}/smiles_validity.csv",                index=False)
    aggregate_full()            .to_csv(f"{RESULT_DIR}/full_assignment_accuracy.csv",      index=False)
    aggregate_partial_fa()      .to_csv(f"{RESULT_DIR}/partial_full_assignment_accuracy.csv", index=False)
    aggregate_stereo()          .to_csv(f"{RESULT_DIR}/per_stereocenter_accuracy.csv",     index=False)
    aggregate_partial_stereo()  .to_csv(f"{RESULT_DIR}/partial_per_stereocenter_accuracy.csv", index=False)
    print("✅  metrics written to", RESULT_DIR)
