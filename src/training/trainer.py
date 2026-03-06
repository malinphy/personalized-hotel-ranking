import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.metrics import compute_mean_ndcg
import os
import json

def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, config):
    model.train()
    train_total_loss = 0
    train_ctr_loss   = 0
    train_ctcvr_loss = 0

    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]'):
        user_feats = batch['user_features']
        item_feats = batch['item_features']

        x = torch.stack([
            user_feats['srch_id_encoded'],
            user_feats['srch_adults_count'],
            user_feats['srch_children_count'],
            user_feats['srch_room_count'],
            user_feats['srch_saturday_night_bool'],
            item_feats['prop_id_encoded'],
            item_feats['prop_country_id_encoded'],
            item_feats['prop_starrating'],
            item_feats['prop_brand_bool'],
            torch.zeros_like(item_feats['prop_brand_bool']),
            item_feats['promotion_flag']
        ], dim=1).to(device)

        labels            = batch['labels'].to(device)
        click_labels      = labels[:, 0].unsqueeze(1)
        conversion_labels = labels[:, 1].unsqueeze(1)

        optimizer.zero_grad()
        p_ctr, p_cvr, p_ctcvr = model(x)

        loss_ctr   = criterion(p_ctr,   click_labels)
        loss_ctcvr = criterion(p_ctcvr, click_labels*conversion_labels)
        loss       = loss_ctr + loss_ctcvr

        loss.backward()
        optimizer.step()

        train_total_loss += loss.item()
        train_ctr_loss   += loss_ctr.item()
        train_ctcvr_loss += loss_ctcvr.item()

    n = len(train_loader)
    print(f'\n--- Epoch {epoch+1} Summary ---')
    print(f'Train Loss: {train_total_loss/n:.4f} (CTR: {train_ctr_loss/n:.4f}, CTCVR: {train_ctcvr_loss/n:.4f})')


def evaluate_one_epoch(model, test_loader, device, epoch, config):
    model.eval()
    all_scores = []
    all_indices = []
    
    # Storage for "Entire Space" (All Impressions)
    all_click_probs = []
    all_click_labels = []
    
    all_ctcvr_probs = []
    all_ctcvr_labels = []
    
    # Storage for CVR tower specific analysis
    all_cvr_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Eval]'):
            user_feats = batch['user_features']
            item_feats = batch['item_features']

            # Reconstruct input tensor (matching your training stack)
            x = torch.stack([
                user_feats['srch_id_encoded'],
                user_feats['srch_adults_count'],
                user_feats['srch_children_count'],
                user_feats['srch_room_count'],
                user_feats['srch_saturday_night_bool'],
                item_feats['prop_id_encoded'],
                item_feats['prop_country_id_encoded'],
                item_feats['prop_starrating'],
                item_feats['prop_brand_bool'],
                torch.zeros_like(item_feats['prop_brand_bool']), # Padding/Placeholder
                item_feats['promotion_flag']
            ], dim=1).to(device).long()

            # Forward pass
            p_ctr, p_cvr, p_ctcvr = model(x)
            all_scores.extend(p_ctcvr.squeeze().cpu().numpy())
            # Extract ground truth
            labels = batch['labels'].to(device)
            click_l = labels[:, 0].cpu().numpy()
            conv_l = labels[:, 1].cpu().numpy()
            ctcvr_l = click_l * conv_l  # y * z (The ground truth for CTCVR)
            
            # Store results for Entire Space metrics
            all_click_probs.extend(p_ctr.squeeze().cpu().numpy())
            all_click_labels.extend(click_l)
            
            all_ctcvr_probs.extend(p_ctcvr.squeeze().cpu().numpy())
            all_ctcvr_labels.extend(ctcvr_l)
            
            # Store raw CVR tower output for post-click analysis
            all_cvr_probs.extend(p_cvr.squeeze().cpu().numpy())

    # --- Data Processing for Visualization ---
    all_click_probs = np.array(all_click_probs)
    all_click_labels = np.array(all_click_labels)
    all_cvr_probs = np.array(all_cvr_probs)
    all_ctcvr_probs = np.array(all_ctcvr_probs)
    all_ctcvr_labels = np.array(all_ctcvr_labels)

    # 1. CTR Predictions (Standard 0.5 threshold)
    ctr_preds_bin = (all_click_probs > 0.5).astype(int)

    # 2. CVR Predictions (Filtered for Clicked Samples only)
    click_mask = (all_click_labels == 1)
    filtered_cvr_probs = all_cvr_probs[click_mask]
    filtered_cvr_labels = all_ctcvr_labels[click_mask] # If y=1, then y*z == z
    
    # Use mean as threshold to handle sparsity in Confusion Matrix
    cvr_thresh = np.mean(filtered_cvr_probs) if len(filtered_cvr_probs) > 0 else 0.5
    cvr_preds_bin = (filtered_cvr_probs > cvr_thresh).astype(int)

    # 3. CTCVR Predictions (Entire Space)
    ctcvr_thresh = np.mean(all_ctcvr_probs) if len(all_ctcvr_probs) > 0 else 0.5
    ctcvr_preds_bin = (all_ctcvr_probs > ctcvr_thresh).astype(int)

    # --- Metrics Calculation ---
    ctr_auc = roc_auc_score(all_click_labels, all_click_probs)
    ctcvr_auc = roc_auc_score(all_ctcvr_labels, all_ctcvr_probs)
    print(f"\nEvaluation Results - CTR AUC: {ctr_auc:.4f} | CTCVR AUC: {ctcvr_auc:.4f}")

    # --- Plotting Heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # CTR Matrix
    sns.heatmap(confusion_matrix(all_click_labels, ctr_preds_bin, normalize='true'), 
                annot=True, fmt='.2%', ax=axes[0], cmap='Blues')
    axes[0].set_title(f'CTR Matrix (Entire Space)\nAUC: {ctr_auc:.4f}')

    # CVR Matrix (The post-click view)
    if len(filtered_cvr_labels) > 0:
        sns.heatmap(confusion_matrix(filtered_cvr_labels, cvr_preds_bin, normalize='true'), 
                    annot=True, fmt='.2%', ax=axes[1], cmap='Oranges')
        axes[1].set_title(f'CVR Matrix (Clicked Only)\nThresh: {cvr_thresh:.4f}')
    else:
        axes[1].text(0.5, 0.5, 'No Clicked Samples', ha='center')

    # CTCVR Matrix
    sns.heatmap(confusion_matrix(all_ctcvr_labels, ctcvr_preds_bin, normalize='true'), 
                annot=True, fmt='.2%', ax=axes[2], cmap='Greens')
    axes[2].set_title(f'CTCVR Matrix (Entire Space)\nAUC: {ctcvr_auc:.4f}')

    plt.tight_layout()
    plt.show()

    test_split['pred_score'] = all_scores

    # Compute NDCG@38
    mean_ndcg, per_query_ndcg = compute_mean_ndcg(
        test_df=test_split,
        pred_scores=all_scores,
        query_col='srch_id',   # raw (un-encoded) srch_id column
        k=38
    )
    
    print(f"NDCG@38: {mean_ndcg:.4f}")


def run_training(model, train_loader, test_loader, config, device):
    checkpoint_dir = config.checkpoint_dir
    
    # Optional: Save config so you know how to rebuild the model later
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(vars(config), f)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(config.epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, config)
        evaluate_one_epoch(model, test_loader, device, epoch, config)
        
        # --- Saving Logic ---
        checkpoint_name = f"esmm_epoch_{epoch+1}.pth"
        save_path = os.path.join(checkpoint_dir, checkpoint_name)
        
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config # Optional: save the config object directly
        }, save_path)
        
        print(f"Successfully saved checkpoint to: {save_path}")


def load_esmm_model(checkpoint_path, config, device):
    # 1. Re-initialize the sub-modules using the same config
    emb_layer = ESMMEmbeddingLayer(config)
    ctr_tower = CTRTower(input_dim=emb_layer.output_dim)
    cvr_tower = CVRTower(input_dim=emb_layer.output_dim)
    
    # 2. Assemble the main model
    model = ESMM(emb_layer, ctr_tower, cvr_tower)
    
    # 3. Load the saved weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval() # Set to evaluation mode
    return model


run_training(model, train_dataloader, test_dataloader, config, device)