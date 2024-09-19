import re
import numpy as np
import pandas as pd
from rdkit import Chem

############################################
# HELPER FUNCTIONS
############################################

# Read preddiction files from OpenNMT with beam size 3
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

# Check if SMILES is valid
def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# Extract chirality from SMILES: CW, CCW, CIS, TRANS
def get_stereocenters(mol):
    stereo_info = Chem.FindPotentialStereo(mol)
    chiral_centers = []
    for info in stereo_info:
        chiral_centers.append(f'{info.descriptor}')
    return chiral_centers

# Calculate chirality weighted accuracy
def chirality_weighted_accuracy(smiles1, smiles2):

    try: 
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)

        flat1 = Chem.MolToSmiles(mol1, isomericSmiles=False)
        flat2 = Chem.MolToSmiles(mol2, isomericSmiles=False)

        chirality1 = get_stereocenters(mol1)
        chirality2 = get_stereocenters(mol2)

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

# Extract highest accuracy from top 2 or top 3
def extract_highest_accuracy(row, top=2):
    if top == 2:
        return max(row['wt_acc_1'], row['wt_acc_2'])
    elif top == 3:
        return max(row['wt_acc_1'], row['wt_acc_2'], row['wt_acc_3'])


############################################
# MAIN FUNCTIONS TO CHECK PERFORMANCE
############################################

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


def check_overall_accuracy(directories):
    df_ov_acc = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')
        
        top1 = (df_augmentation['target'] == df_augmentation['beam_1']).mean()
        top2 = ((df_augmentation['target'] == df_augmentation['beam_1']) | (df_augmentation['target'] == df_augmentation['beam_2'])).mean()
        top3 = ((df_augmentation['target'] == df_augmentation['beam_1']) | (df_augmentation['target'] == df_augmentation['beam_2']) | (df_augmentation['target'] == df_augmentation['beam_3'])).mean()

        df_ov_acc.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
    
    return df_ov_acc


def check_chirality_weighted_accuracy(directories):
    df_chirality = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')
        
        df_augmentation['wt_acc_1'] = df_augmentation.apply(lambda x: chirality_weighted_accuracy(x['target'], x['beam_1']), axis=1)
        df_augmentation['wt_acc_2'] = df_augmentation.apply(lambda x: chirality_weighted_accuracy(x['target'], x['beam_2']), axis=1)
        df_augmentation['wt_acc_3'] = df_augmentation.apply(lambda x: chirality_weighted_accuracy(x['target'], x['beam_3']), axis=1)

        top1 = df_augmentation['wt_acc_1'].mean()
        top2 = df_augmentation.apply(lambda x: extract_highest_accuracy(x, top=2), axis=1).mean()
        top3 = df_augmentation.apply(lambda x: extract_highest_accuracy(x, top=3), axis=1).mean()

        df_chirality.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
    
    return df_chirality


def check_overall_accuracy(directories):
    df_ov_acc = pd.DataFrame(columns=['augmentation', 'top1', 'top2', 'top3'])
    
    for idx, directory in enumerate(directories):
        df_augmentation = read_results(f'data/opennmt/{directory}/')
        
        top1 = (df_augmentation['target'] == df_augmentation['beam_1']).mean()
        top2 = ((df_augmentation['target'] == df_augmentation['beam_1']) | (df_augmentation['target'] == df_augmentation['beam_2'])).mean()
        top3 = ((df_augmentation['target'] == df_augmentation['beam_1']) | (df_augmentation['target'] == df_augmentation['beam_2']) | (df_augmentation['target'] == df_augmentation['beam_3'])).mean()

        df_ov_acc.loc[idx] = {'augmentation': directory, 'top1': top1, 'top2': top2, 'top3': top3}
    
    return df_ov_acc


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


def chemical_class_accuracies(df):
    accuracies = {  'top1': [],
                    'top2': [],
                    'top3': [],
                    'top1_wt': [],
                    'top2_wt': [],
                    'top3_wt': []
                  }
    categories = np.sort(df['chemical_class_new'].unique())
    for category in categories: 
        df_cat = df[df['chemical_class_new'] == category]
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

############################################
# MAIN FUNCTIONS FOR TMAP
############################################

def calculate_accuracies(directory):
    
        df = read_results(f'data/opennmt/{directory}/')

        df['top1'] = (df['target'] == df['beam_1']).astype(int)
        df['top2'] = ((df['target'] == df['beam_1']) | (df['target'] == df['beam_2'])).astype(int)
        df['top3'] = ((df['target'] == df['beam_1']) | (df['target'] == df['beam_2']) | (df['target'] == df['beam_3'])).astype(int)
        
        df['wt_acc_1'] = df.apply(lambda x: chirality_weighted_accuracy(x['target'], x['beam_1']), axis=1)
        df['wt_acc_2'] = df.apply(lambda x: chirality_weighted_accuracy(x['target'], x['beam_2']), axis=1)
        df['wt_acc_3'] = df.apply(lambda x: chirality_weighted_accuracy(x['target'], x['beam_3']), axis=1)

        df['top1_wt'] = df['wt_acc_1']
        df['top2_wt'] = df.apply(lambda x: extract_highest_accuracy(x, top=2), axis=1)
        df['top3_wt'] = df.apply(lambda x: extract_highest_accuracy(x, top=3), axis=1)
    
        return df[['source', 'target', 'beam_1', 'beam_2', 'beam_3', 'top1', 'top2', 'top3', 'top1_wt', 'top2_wt', 'top3_wt']]