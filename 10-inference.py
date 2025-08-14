#!/usr/bin/env python3
# 10-inference.py
from __future__ import annotations
from pathlib import Path
import argparse, subprocess, shutil, sys, re
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem

RDLogger.DisableLog('rdApp.*')

def parse_args():
    p = argparse.ArgumentParser(description="Run ONMT inference and majority-vote predictions.")
    p.add_argument("dataset", help="Path to dataset .xlsx (e.g., data/literature-dataset-reduced.xlsx)")
    p.add_argument("--model-key", default="npstereo")
    p.add_argument("--seeds", default="0,1,42", help="Comma-separated list, e.g. 0,1,42")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--out-dir", default="data/inference")
    p.add_argument("--n-best", type=int, default=1)
    p.add_argument("--beam-size", type=int, default=1)
    return p.parse_args()

def ensure_tools():
    if not shutil.which("onmt_translate"):
        sys.exit("ERROR: 'onmt_translate' not found in PATH.")

def tokenize_smiles(series: pd.Series) -> pd.Series:
    pattern = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\!|\$|\%[0-9]{2}|[0-9])'
    tok = re.compile(pattern).findall
    return series.apply(lambda s: ' '.join(tok(s)))

def canonize_smiles(series: pd.Series) -> pd.Series:
    def _canon(s):
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m, isomericSmiles=True, kekuleSmiles=False) if m else ""
    return series.apply(_canon)

def flatten(smi: str) -> str:
    """
    Remove all stereochemistry (atom chirality + double-bond E/Z) and return
    a normalized non-isomeric SMILES suitable for the model input.
    """
    m = Chem.MolFromSmiles(smi)
    if not m:
        return ""
    # clear atom chirality
    for a in m.GetAtoms():
        a.SetChiralTag(rdchem.ChiralType.CHI_UNSPECIFIED)
        if a.HasProp("_CIPCode"): a.ClearProp("_CIPCode")
        if a.HasProp("_ChiralityPossible"): a.ClearProp("_ChiralityPossible")
    # clear double-bond stereo and wedge directions
    for b in m.GetBonds():
        b.SetStereo(rdchem.BondStereo.STEREONONE)
        b.SetBondDir(rdchem.BondDir.NONE)
    # reassign to ensure nothing lingers
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    # non-isomeric SMILES (no @, /, \, or E/Z)
    return Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=False)

def majority_vote(smiles_list: list[str]) -> str:
    """Robust: ignores empty/invalid SMILES; majority on valid subset."""
    mols = []
    for s in smiles_list:
        if not s:
            continue
        m = Chem.MolFromSmiles(str(s))
        if m is not None:
            mols.append(m)

    if not mols:
        return ""  # nothing valid

    # Assign stereo on valid mols only
    for m in mols:
        Chem.AssignStereochemistry(m, cleanIt=True, force=True)

    # Use first valid as reference
    ref = mols[0]
    ref_n = ref.GetNumAtoms()

    # Keep only same-size mols to compare stereo features reliably
    mols = [m for m in mols if m.GetNumAtoms() == ref_n]
    if len(mols) < 2:
        # Not enough comparable structures → just return canonical SMILES of ref
        return Chem.MolToSmiles(ref, isomericSmiles=True, kekuleSmiles=False)

    # Majority on chiral centers
    for a in ref.GetAtoms():
        if a.HasProp('_CIPCode'):
            idx = a.GetIdx()
            tags = [m.GetAtomWithIdx(idx).GetChiralTag() for m in mols]
            if tags:  # defensive
                maj = max(set(tags), key=tags.count)
                for m in mols:
                    m.GetAtomWithIdx(idx).SetChiralTag(maj)

    # Majority on E/Z double bonds
    from rdkit.Chem import rdchem
    for b in ref.GetBonds():
        if b.GetBondType() == rdchem.BondType.DOUBLE and b.GetStereo() != rdchem.BondStereo.STEREONONE:
            idx = b.GetIdx()
            ster = [m.GetBondWithIdx(idx).GetStereo() for m in mols]
            if ster:  # defensive
                maj = max(set(ster), key=ster.count)
                for m in mols:
                    m.GetBondWithIdx(idx).SetStereo(maj)

    return Chem.MolToSmiles(ref, isomericSmiles=True, kekuleSmiles=False)

def main():
    args = parse_args()
    ensure_tools()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        sys.exit(f"ERROR: Dataset not found: {dataset_path}")
    name_dataset = dataset_path.stem

    model_key = args.model_key
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    base_model = Path(args.models_dir)
    out_root  = Path(args.out_dir) / model_key / name_dataset
    out_root.mkdir(parents=True, exist_ok=True)

    # Load & prep data
    df = pd.read_excel(dataset_path)
    if "smiles" not in df.columns:
        sys.exit("ERROR: dataset must contain a 'smiles' column.")
    df["smiles"] = canonize_smiles(df["smiles"])        # keep canonical record
    df["source"] = df["smiles"].apply(flatten)          # **flattened** for the model
    df["src_tok"] = tokenize_smiles(df["source"])

    src_txt = out_root / "src.txt"
    df["src_tok"].to_csv(src_txt, index=False, header=False)

    # Run predictions (one file per seed)
    for seed in seeds:
        ckpt = base_model / f"seed-{seed}" / model_key / f"{model_key}_step_100000.pt"
        if not ckpt.exists():
            sys.exit(f"ERROR: missing checkpoint: {ckpt}")
        out_pred = out_root / f"pred_seed{seed}.txt"
        if not out_pred.exists():
            cmd = [
                "onmt_translate",
                "-model", str(ckpt),
                "-src", str(src_txt),
                "-output", str(out_pred),
                "-n_best", str(args.n_best),
                "-beam_size", str(args.beam_size),
            ]
            subprocess.run(cmd, check=True)

    # Read predictions and vote
    vote_cols = []
    for idx, seed in enumerate(seeds):
        col = f"pred_{idx}"
        vote_cols.append(col)
        path = out_root / f"pred_seed{seed}.txt"
        if path.exists():
            df[col] = pd.read_csv(path, header=None, sep="\t", engine="python").iloc[:, 0].str.replace(" ", "", regex=False)
        else:
            df[col] = ""

    if sum(df[c].astype(bool).any() for c in vote_cols) < 1:
        sys.exit("ERROR: No prediction columns found.")

    df["majority_prediction"] = [majority_vote(vals) for vals in df[vote_cols].values.tolist()]

    out_csv = out_root / "majority_predictions.csv"
    keep_cols = [c for c in ("id", "name", "smiles") if c in df.columns]
    (df[keep_cols + ["majority_prediction"]] if keep_cols else df[["smiles","majority_prediction"]]).to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")

if __name__ == "__main__":
    main()