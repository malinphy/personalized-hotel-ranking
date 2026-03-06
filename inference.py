import torch
import pandas as pd
import numpy as np
import os
import random
import json
from models.esmm import ESMM, CTRTower, CVRTower, ESMMEmbeddingLayer
from src.utils.config import get_config

def load_config(checkpoint_dir):
    """Loads the config object from the saved JSON."""
    config_json_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_json_path):
        with open(config_json_path, 'r') as f:
            cfg_dict = json.load(f)
        
       
        class SimpleConfig:
            def __init__(self, dictionary):
                for k, v in dictionary.items():
                    setattr(self, k, v)
        return SimpleConfig(cfg_dict)
    return None

def run_inference(srch_id=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    checkpoint_dir = os.path.join("src", "training", "weights")
    test_path = os.path.join("data", "test_split.csv")

    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found at {checkpoint_dir}")
        return

    
    config = load_config(checkpoint_dir)
    
    
    checkpoint_name = "esmm_epoch_4.pth"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: {checkpoint_name} not found. Attempting to load latest checkpoint.")
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if not checkpoints:
            print("Error: No .pth files found in checkpoints directory.")
            return
        checkpoint_name = sorted(checkpoints)[-1]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    print(f"Loading weights from: {checkpoint_name}")

    
    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        return
    
    print("Loading test dataset...")
    df = pd.read_csv(test_path)

    
    if srch_id is None:
        srch_id = random.choice(df['srch_id'].unique())
    
    print(f"Performing inference for Search ID: {srch_id}")
    query_group = df[df['srch_id'] == srch_id].copy().reset_index(drop=True)

    if len(query_group) == 0:
        print(f"Error: No data found for srch_id {srch_id}")
        return

    
    if config is None:
        print("Warning: config.json not found, attempting dynamic config from test data (may fail if vocab mismatch).")
        config = get_config(df)

    emb_layer = ESMMEmbeddingLayer(config)
    ctr_tower = CTRTower(input_dim=emb_layer.output_dim)
    cvr_tower = CVRTower(input_dim=emb_layer.output_dim)
    model = ESMM(emb_layer, ctr_tower, cvr_tower)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    
    cols = [
        'srch_id_encoded', 'srch_adults_count', 'srch_children_count', 
        'srch_room_count', 'srch_saturday_night_bool', 'prop_id_encoded', 
        'prop_country_id_encoded', 'prop_starrating', 'prop_brand_bool', 
        'promotion_flag'
    ]
    

    query_group[cols] = query_group[cols].fillna(0).astype(np.int64)

    
    
    x = torch.stack([
        torch.tensor(query_group['srch_id_encoded'].values),
        torch.tensor(query_group['srch_adults_count'].values),
        torch.tensor(query_group['srch_children_count'].values),
        torch.tensor(query_group['srch_room_count'].values),
        torch.tensor(query_group['srch_saturday_night_bool'].values),
        torch.tensor(query_group['prop_id_encoded'].values),
        torch.tensor(query_group['prop_country_id_encoded'].values),
        torch.tensor(query_group['prop_starrating'].values),
        torch.tensor(query_group['prop_brand_bool'].values),
        torch.zeros(len(query_group), dtype=torch.long), 
        torch.tensor(query_group['promotion_flag'].values)
    ], dim=1).to(device)

    
    with torch.no_grad():
        _, _, p_ctcvr = model(x)
        scores = p_ctcvr.squeeze().cpu().numpy()
        
    
    if scores.ndim == 0:
        scores = np.array([scores])

    
    query_group['prediction_score'] = scores
    output = query_group.sort_values(by='prediction_score', ascending=False)
    
    print("\n" + "="*50)
    print(f"RANKING RESULTS FOR SEARCH ID: {srch_id}")
    print("="*50)
    print(output[['prop_id', 'prediction_score', 'click_bool', 'booking_bool']].to_string(index=False))
    print("="*50)
    print(f"Top Recommended Hotel: {output.iloc[0]['prop_id']}")

if __name__ == "__main__":

    run_inference(srch_id=None)
