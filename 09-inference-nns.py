# Input
#   • data/literature-dataset-reduced.xlsx
#   • models/seed-{0,1,42}/{MODEL_KEY}/{MODEL_KEY}_step_100000.pt
#   • onmt_translate in PATH
#
# Output
#   data/inference/{MODEL_KEY}/pred_seed*.txt               (raw preds)
#   data/inference/{MODEL_KEY}/majority_predictions.csv
# --------------------------------------------------------------------

from io import BytesIO
from pathlib import Path
import subprocess, re, sys
import base64
import pickle

import pandas as pd
from rdkit.Chem import Draw, rdchem
from mapchiral import mapchiral
from eval_functions import *

from IPython.display import HTML
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# Config
MODEL_KEY = 'npstereo'           
SEEDS     = [0, 1, 42]

BASE_MODEL = Path('models')
OUT_DIR    = Path('data/inference') / MODEL_KEY
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_BEST    = 1      
BEAM_SIZE = 1

# Load dataset
df = pd.read_excel('data/literature-dataset-reduced.xlsx') # Change dataset path as needed
df['smiles'] = df['smiles'].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True, kekuleSmiles=False))
df['source']  = df['smiles'].apply(flatten)

pattern = r'(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\!|\$|\%[0-9]{2}|[0-9])'
tokenizer = re.compile(pattern).findall
df['src_tok'] = df['source'].apply(lambda s: ' '.join(tokenizer(s)))

SRC_TXT = OUT_DIR / 'src.txt'
df['src_tok'].to_csv(SRC_TXT, index=False, header=False)

# Calculate fingerprints for nearest neighbor search
fp_path = Path('data/inference/fingerprints.pkl')
coconut = pd.read_csv('data/coconut/coconut-split-0.csv')
coconut = coconut[coconut['split'] != 'test']

if fp_path.exists():
    with open(fp_path, 'rb') as f:
        coconut['map4c'] = pickle.load(f)
else:
    coconut['map4c'] = coconut['smiles'].apply(lambda x: mapchiral.encode(Chem.MolFromSmiles(x)))
    with open(fp_path, 'wb') as f:
        pickle.dump(coconut['map4c'].tolist(), f)

# Run predictions
for seed in SEEDS:
    ckpt = BASE_MODEL / f'seed-{seed}' / MODEL_KEY / f'{MODEL_KEY}_step_100000.pt'
    out_pred = OUT_DIR   / f'pred_seed{seed}.txt'

    if out_pred.exists():
        print(f'[✓] Seed {seed} predictions already exist – skip')
        continue

    cmd = [
        'onmt_translate',
        '-model',     str(ckpt),
        '-src',       str(SRC_TXT),
        '-output',    str(out_pred),
        '-n_best',    str(N_BEST),
        '-beam_size', str(BEAM_SIZE),
        '-verbose',
    ]
    print('»', ' '.join(cmd))
    subprocess.run(cmd, check=True)

# Read predictions
vote_cols: list[str] = []

for idx, seed in enumerate(SEEDS):
    col_name = f'pred_{idx}'
    vote_cols.append(col_name)

    path = OUT_DIR / f'pred_seed{seed}.txt'
    if path.exists():
        df[col_name] = (pd.read_csv(path, header=None, sep='\t', engine='python').iloc[:, 0].str.replace(' ', '', regex=False))
    else:
        print(f'{path} missing - column filled with empty strings', file=sys.stderr)
        df[col_name] = ''     

# Need at least one non-empty column
if sum(df[c].astype(bool).any() for c in vote_cols) < 1:
    sys.exit('No prediction columns - abort')

# Majority vote for three predictions
def majority_vote(smiles_list):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles_list if smi]
    for mol in mols:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    ref = mols[0]
    ref_natoms = ref.GetNumAtoms()

    # Only keep molecules with same atom count
    mols = [mol for mol in mols if mol.GetNumAtoms() == ref_natoms]
    if len(mols) < 2:
        raise ValueError('Too few structurally matching molecules to vote')

    # Handle chiral centers
    for atom in ref.GetAtoms():
        if atom.HasProp('_CIPCode'):
            idx = atom.GetIdx()
            tags = [mol.GetAtomWithIdx(idx).GetChiralTag() for mol in mols]
            majority = max(set(tags), key=tags.count)
            for mol in mols:
                mol.GetAtomWithIdx(idx).SetChiralTag(majority)

    # Handle double bonds (E/Z)
    for bond in ref.GetBonds():
        if bond.GetBondType() == rdchem.BondType.DOUBLE and bond.GetStereo() != rdchem.BondStereo.STEREONONE:
            idx = bond.GetIdx()
            stereos = [mol.GetBondWithIdx(idx).GetStereo() for mol in mols]
            majority = max(set(stereos), key=stereos.count)
            for mol in mols:
                mol.GetBondWithIdx(idx).SetStereo(majority)

    return Chem.MolToSmiles(ref, isomericSmiles=True, kekuleSmiles=True)

def safe_majority(vals: list[str]) -> str:
    '''Wrapper: never raises - returns first non-empty SMILES on error.'''
    try:
        return majority_vote(vals)
    except Exception as e:
        if safe_majority.errs < 3:
            print(' [!] majority_vote failed → fallback:', e)
            safe_majority.errs += 1
        for x in vals:
            if x:
                return x
        return ''

safe_majority.errs = 0

df['majority_prediction'] = [safe_majority(r) for r in df[vote_cols].values.tolist()]

# Calculate accuracy
df['top1_wt'] = [
    per_stereocenter(src, tgt, pred)
    for src, tgt, pred in zip(df['source'], df['smiles'], df['majority_prediction'])
]

df['#stereo']     = df['smiles'].apply(add_stereocenters)
df['#unassigned'] = df['source'].apply(add_unassigned_stereocenters)

# Determine MAP4C nearest neighbours of majority predictions
df['map4c'] = df['majority_prediction'].apply(lambda x: mapchiral.encode(Chem.MolFromSmiles(x)))

nn1, nn2, nn3 = [], [], []
sim1, sim2, sim3 = [], [], []

for qfp in df['map4c']:
    sims = [(row.smiles, mapchiral.jaccard_similarity(qfp, rfp))
            for row, rfp in zip(coconut.itertuples(), coconut['map4c'])]
    sims.sort(key=lambda x: x[1], reverse=True)

    nn1.append(sims[0][0] if len(sims) > 0 else '')
    sim1.append(sims[0][1] if len(sims) > 0 else 0.0)
    nn2.append(sims[1][0] if len(sims) > 1 else '')
    sim2.append(sims[1][1] if len(sims) > 1 else 0.0)
    nn3.append(sims[2][0] if len(sims) > 2 else '')
    sim3.append(sims[2][1] if len(sims) > 2 else 0.0)

df['nn-1'] = nn1
df['sim-1'] = sim1
df['nn-2'] = nn2
df['sim-2'] = sim2
df['nn-3'] = nn3
df['sim-3'] = sim3

# Save results
out_csv = OUT_DIR / 'majority_predictions.csv'
df[['id', 'name', 'smiles', 'majority_prediction', 'top1_wt', '#stereo', '#unassigned', 'nn-1', 'sim-1', 'nn-2', 'sim-2', 'nn-3', 'sim-3']].to_csv(out_csv, index=False)

# HTML report with images
def mol_png_base64(mol, size=(300, 300)):
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" width="{size[0]}" height="{size[1]}"/>'

if 'target_img' not in df.columns:
    df['target_img'] = df['smiles'].apply(
        lambda s: mol_png_base64(Chem.MolFromSmiles(s))
    )
    df['pred_img']   = df['majority_prediction'].apply(
        lambda s: mol_png_base64(Chem.MolFromSmiles(s))
    )

cols_for_html = [
    'id', 'name',
    'source', 'smiles', 'majority_prediction',
    'top1_wt',
    'target_img', 'pred_img', 
    'nn-1', 'sim-1', 'nn-2', 'sim-2', 'nn-3', 'sim-3',
]

html_report = HTML(df[cols_for_html].to_html(escape=False))
html_path = OUT_DIR / 'majority_predictions.html'
with open(html_path, 'w', encoding='utf-8') as fh:
    fh.write(html_report.data)

print(f' Full HTML report written to →  {html_path}')
print(f' Saved: {out_csv}   ({len(df)} molecules)')
print(df[['id', 'name', 'smiles', 'majority_prediction', 'top1_wt']].head())