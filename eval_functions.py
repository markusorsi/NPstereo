import re
import numpy as np
import pandas as pd
from rdkit import Chem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from tqdm import tqdm
tqdm.pandas()

##### DATA PROCESSING #####

def read_results(directory):
    source = pd.read_csv(directory + 'src-test.txt', header=None, names=['source'])
    target = pd.read_csv(directory + 'tgt-test.txt', header=None, names=['target'])
    df = pd.concat([source, target], axis=1)

    predictions = [[] for i in range(3)]
    with open(directory + 'pred-test.txt', 'r') as f:
        for i, line in enumerate(f.readlines()):
            predictions[i % 3].append(line.strip())

    for i in range(3):
        df['beam_' + str(i+1)] = predictions[i]

    df = df.apply(lambda x: x.str.replace(' ', ''))
    return df

def flatten(smiles):
    substitutions = {
        r'\[K[@,H]*\]': '[K]',
        r'\[B[@,H]*\]': 'B',
        r'\[Na[@,H,+,-]*\]': '[Na]',
        r'\[C[@,H]*\]': 'C',
        r'\[N[@,H]*\]': 'N',
        r'\[O[@,H]*\]': 'O',
        r'\[S[@,H]*\]': 'S',
        r'\[P[@,H]*\]': 'P',
        r'\[F[@,H]*\]': 'F',
        r'\[Cl[@,H]*\]': '[Cl]',
        r'\[Br[@,H]*\]': '[Br]',
        r'\[I[@,H]*\]': 'I',
        r'@': '',
        r'/': '',
        r'\\': '',
        r'\[C\]': 'C'
    }

    for pattern, replacement in substitutions.items():
        smiles = re.sub(pattern, replacement, smiles)

    return smiles

##### SMILES VALIDITY #####

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def check_smiles_validity(directories):
    df_smiles = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')
        
        beam1_smiles = df_augmentation.apply(lambda x: is_valid_smiles(x['beam_1']), axis=1)
        beam2_smiles = df_augmentation.apply(lambda x: is_valid_smiles(x['beam_2']), axis=1)
        beam3_smiles = df_augmentation.apply(lambda x: is_valid_smiles(x['beam_3']), axis=1)
        
        top1 = beam1_smiles.mean()
        top2 = (beam1_smiles | beam2_smiles).mean()
        top3 = (beam1_smiles | beam2_smiles | beam3_smiles).mean()

        df_smiles.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
    
    return df_smiles

##### FULL ASSIGNMENT ACCURACY #####

def full_assignment_accuracy(directories):
    df_full_assignment = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')
        grouped_targets = df_augmentation.groupby('source')['target'].apply(list).to_dict()
        
        top1 = df_augmentation.apply(lambda row: row['beam_1'] in grouped_targets.get(row['source'], []), axis=1).mean()
        top2 = df_augmentation.apply(lambda row: (row['beam_1'] in grouped_targets.get(row['source'], [])) or (row['beam_2'] in grouped_targets.get(row['source'], [])), axis=1).mean()
        top3 = df_augmentation.apply(lambda row: (row['beam_1'] in grouped_targets.get(row['source'], [])) or (row['beam_2'] in grouped_targets.get(row['source'], [])) or (row['beam_3'] in grouped_targets.get(row['source'], [])), axis=1).mean()

        df_full_assignment.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
        
    return df_full_assignment

def partial_full_assignment_accuracy(directories):
    df_full_assignment = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')
        
        df_augmentation['flattened'] = df_augmentation['source'].apply(flatten)
        grouped_targets = df_augmentation.groupby('flattened')['target'].apply(list).to_dict()

        top1 = df_augmentation.apply(lambda row: row['beam_1'] in grouped_targets.get(row['flattened'], []), axis=1).mean()
        top2 = df_augmentation.apply(lambda row: (row['beam_1'] in grouped_targets.get(row['flattened'], [])) or (row['beam_2'] in grouped_targets.get(row['flattened'], [])), axis=1).mean()
        top3 = df_augmentation.apply(lambda row: (row['beam_1'] in grouped_targets.get(row['flattened'], [])) or (row['beam_2'] in grouped_targets.get(row['flattened'], [])) or (row['beam_3'] in grouped_targets.get(row['flattened'], [])), axis=1).mean()
        
        df_full_assignment.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
    
    return df_full_assignment

##### PER STEREOCENTER ACCURACY #####

def get_stereocenters(mol):
    stereo_info = Chem.FindPotentialStereo(mol)
    chiral_centers = []
    for info in stereo_info:
        chiral_centers.append(f'{info.descriptor}')
    return chiral_centers

def per_stereocenter(source, smiles1, smiles2):

    try: 
        mol_source = Chem.MolFromSmiles(source)
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        flat1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
        flat2 = Chem.MolToSmiles(mol2, isomericSmiles=False)
        
        chirality_source = get_stereocenters(mol_source)
        chirality1 = get_stereocenters(mol1)
        chirality2 = get_stereocenters(mol2)

        chirality1, chirality2 = (
        [x for x, y in zip(chirality1, chirality_source) if y == 'NoValue'],
        [x for x, y in zip(chirality2, chirality_source) if y == 'NoValue']
        ) # remove stereocenters that are already defined in source for evaluation

        if flat1 != flat2:
            return 0
        elif len(chirality1) == 0 and len(chirality2) == 0:
            return 1 
        elif len(chirality1) != 0 and len(chirality2) == 0:
            return 0
        elif len(chirality1) == 0 and len(chirality2) != 0:
            return 0
        elif len(chirality1) != len(chirality2):
            return 0
        else:
            return sum([1 for c1, c2 in zip(chirality1, chirality2) if c1 == c2]) / len(chirality1)
    except:
        return 0
    
def per_stereocenter_accuracy(directories):
    df_chirality = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')

        df_augmentation['num_stereocenters'] = df_augmentation['source'].apply(lambda x: len(get_stereocenters(Chem.MolFromSmiles(x))))
        df_augmentation = df_augmentation[df_augmentation['num_stereocenters'] > 0].copy() # ensure we only work with molecules containing stereocenters
        
        grouped_targets = df_augmentation.groupby('source')['target'].apply(list).to_dict()

        df_augmentation['wt_acc_1'] = df_augmentation.apply(lambda row: max([per_stereocenter(row['source'], target, row['beam_1']) for target in grouped_targets.get(row['source'], [])]),axis=1)
        df_augmentation['wt_acc_2'] = df_augmentation.apply(lambda row: max([per_stereocenter(row['source'], target, row['beam_2']) for target in grouped_targets.get(row['source'], [])]),axis=1)
        df_augmentation['wt_acc_3'] = df_augmentation.apply(lambda row: max([per_stereocenter(row['source'], target, row['beam_3']) for target in grouped_targets.get(row['source'], [])]),axis=1)

        top1 = df_augmentation['wt_acc_1'].mean()
        top2 = df_augmentation.apply(lambda row: max(row['wt_acc_1'], row['wt_acc_2']), axis=1).mean()
        top3 = df_augmentation.apply(lambda row: max(row['wt_acc_1'], row['wt_acc_2'], row['wt_acc_3']), axis=1).mean()

        df_chirality.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
    
    return df_chirality

def partial_per_stereocenter_accuracy(directories):
    df_chirality = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')
        df_augmentation['num_stereocenters'] = df_augmentation['source'].apply(lambda x: len(get_stereocenters(Chem.MolFromSmiles(x))))
        df_augmentation = df_augmentation[df_augmentation['num_stereocenters'] > 0].copy()
        
        df_augmentation['flattened'] = df_augmentation['source'].apply(flatten)
        grouped_targets = df_augmentation.groupby('flattened')['target'].apply(list).to_dict()

        accuracy_cache = {}
        def get_accuracy(source, target, beam):
            key = (source, target, beam)
            if key not in accuracy_cache:
                accuracy_cache[key] = per_stereocenter(source, target, beam)
            return accuracy_cache[key]
        
        for beam in ['beam_1', 'beam_2', 'beam_3']:
            df_augmentation[f'wt_acc_{beam[-1]}'] = df_augmentation.apply(
                lambda row: max(get_accuracy(row['source'], target, row[beam]) 
                                for target in grouped_targets.get(row['flattened'], [])), axis=1)
            
        top1 = df_augmentation['wt_acc_1'].mean()
        top2 = df_augmentation[['wt_acc_1', 'wt_acc_2']].max(axis=1).mean()
        top3 = df_augmentation[['wt_acc_1', 'wt_acc_2', 'wt_acc_3']].max(axis=1).mean()

        df_chirality.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
    return df_chirality

##### ACCURACIES FOR PLOTTING #####

def get_accuracies(directory): 

    df = read_results(f'data/opennmt/{directory}/')

    df['top1'] = (df['target'] == df['beam_1']).astype(int)
    df['top2'] = ((df['target'] == df['beam_1']) | (df['target'] == df['beam_2'])).astype(int)
    df['top3'] = ((df['target'] == df['beam_1']) | (df['target'] == df['beam_2']) | (df['target'] == df['beam_3'])).astype(int)

    df['wt_acc_1'] = df.apply(lambda x: per_stereocenter(x['source'], x['target'], x['beam_1']), axis=1)
    df['wt_acc_2'] = df.apply(lambda x: per_stereocenter(x['source'], x['target'], x['beam_2']), axis=1)
    df['wt_acc_3'] = df.apply(lambda x: per_stereocenter(x['source'], x['target'], x['beam_3']), axis=1)

    df['top1_wt'] = df['wt_acc_1']
    df['top2_wt'] = np.maximum(df['wt_acc_1'], df['wt_acc_2'])
    df['top3_wt'] = np.maximum.reduce([df['wt_acc_1'], df['wt_acc_2'], df['wt_acc_3']])

    grouped_df = df.groupby('source').agg({
        'target': list,
        'beam_1': 'first',
        'beam_2': 'first',
        'beam_3': 'first',
        'top1': 'max',
        'top2': 'max',
        'top3': 'max',
        'top1_wt': 'max',
        'top2_wt': 'max',
        'top3_wt': 'max'
    }).reset_index()

    return grouped_df

def get_partial_accuracies(directory):
    
    df = read_results(f'data/opennmt/{directory}/')
    df['flattened'] = df['source'].apply(flatten)
    flattened_to_targets = df.groupby('flattened')['target'].apply(list).to_dict()
    
    df['top1'] = df.apply(lambda row: row['beam_1'] in flattened_to_targets.get(row['flattened'], []), axis=1).astype(int)
    df['top2'] = df.apply(lambda row: (row['beam_1'] in flattened_to_targets.get(row['flattened'], [])) or (row['beam_2'] in flattened_to_targets.get(row['flattened'], [])), axis=1).astype(int)
    df['top3'] = df.apply(lambda row: (row['beam_1'] in flattened_to_targets.get(row['flattened'], [])) or (row['beam_2'] in flattened_to_targets.get(row['flattened'], [])) or (row['beam_3'] in flattened_to_targets.get(row['flattened'], [])), axis=1).astype(int)
    
    accuracy_cache = {}
    def get_accuracy(source, target, beam):
        key = (source, target, beam)
        if key not in accuracy_cache:
            accuracy_cache[key] = per_stereocenter(source, target, beam)
        return accuracy_cache[key]

    df['wt_acc_1'] = df.apply(lambda row: max(get_accuracy(row['source'], target, row['beam_1']) for target in flattened_to_targets.get(row['flattened'], [])), axis=1)
    df['wt_acc_2'] = df.apply(lambda row: max(get_accuracy(row['source'], target, row['beam_2']) for target in flattened_to_targets.get(row['flattened'], [])), axis=1)
    df['wt_acc_3'] = df.apply(lambda row: max(get_accuracy(row['source'], target, row['beam_3']) for target in flattened_to_targets.get(row['flattened'], [])), axis=1)

    df['top1_wt'] = df['wt_acc_1']
    df['top2_wt'] = np.maximum(df['wt_acc_1'], df['wt_acc_2'])
    df['top3_wt'] = np.maximum.reduce([df['wt_acc_1'], df['wt_acc_2'], df['wt_acc_3']])

    grouped_df = df.groupby('source').agg({
        'target': list,
        'beam_1': 'first',
        'beam_2': 'first',
        'beam_3': 'first',
        'top1': 'max',
        'top2': 'max',
        'top3': 'max',
        'top1_wt': 'max',
        'top2_wt': 'max',
        'top3_wt': 'max'
    }).reset_index()

    return grouped_df

##### SUMMARIZING ACCURACIES #####

def stereocenter_accuracies(df): 
    accuracies = {  'top1': [],
                    'top2': [],
                    'top3': [],
                    'top1_wt': [],
                    'top2_wt': [],
                    'top3_wt': []
                  }
    categories = np.sort(df['#stereo'].unique())
    for category in categories: 
        df_cat = df[df['#stereo'] == category]
        accuracies['top1'].append(df_cat['top1'].mean())
        accuracies['top2'].append(df_cat['top2'].mean())
        accuracies['top3'].append(df_cat['top3'].mean())
        accuracies['top1_wt'].append(df_cat['top1_wt'].mean())
        accuracies['top2_wt'].append(df_cat['top2_wt'].mean())
        accuracies['top3_wt'].append(df_cat['top3_wt'].mean())
    
    accuracies = pd.DataFrame(accuracies)
    accuracies.index = categories
    accuracies = accuracies.sort_index()
    accuracies = accuracies.round(2)
    return accuracies

def unassigned_accuracies(df): 
    accuracies = {  'top1': [],
                    'top2': [],
                    'top3': [],
                    'top1_wt': [],
                    'top2_wt': [],
                    'top3_wt': []
                  }
    categories = np.sort(df['#unassigned'].unique())
    for category in categories: 
        df_cat = df[df['#unassigned'] == category]
        accuracies['top1'].append(df_cat['top1'].mean())
        accuracies['top2'].append(df_cat['top2'].mean())
        accuracies['top3'].append(df_cat['top3'].mean())
        accuracies['top1_wt'].append(df_cat['top1_wt'].mean())
        accuracies['top2_wt'].append(df_cat['top2_wt'].mean())
        accuracies['top3_wt'].append(df_cat['top3_wt'].mean())
    
    accuracies = pd.DataFrame(accuracies)
    accuracies.index = categories
    accuracies = accuracies.sort_index()
    accuracies = accuracies.round(2)
    return accuracies

def chemical_class_accuracies(df):
    accuracies = {  
        'top1': [],
        'top2': [],
        'top3': [],
        'top1_wt': [],
        'top2_wt': [],
        'top3_wt': []
    }
    
    categories = np.sort(df['chemical_class_new'].astype(str).unique())
    
    for category in categories: 
        df_cat = df[(df['chemical_class_new'].astype(str) == category) & (df['#stereo'] != 0)]
        accuracies['top1'].append(df_cat['top1'].mean())
        accuracies['top2'].append(df_cat['top2'].mean())
        accuracies['top3'].append(df_cat['top3'].mean())
        accuracies['top1_wt'].append(df_cat['top1_wt'].mean())
        accuracies['top2_wt'].append(df_cat['top2_wt'].mean())
        accuracies['top3_wt'].append(df_cat['top3_wt'].mean())
    
    accuracies = pd.DataFrame(accuracies)
    accuracies.index = categories
    accuracies = accuracies.sort_index()
    accuracies = accuracies.round(3)
    return accuracies