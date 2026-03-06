import torch
import torch.nn as nn

class CTRTower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.ctr_tower = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.ctr_tower(x)
class CVRTower(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.cvr_tower = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.cvr_tower(x)
class ESMMEmbeddingLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.srch_id_embedding = nn.Embedding(config.srch_id_vocab_size, config.embedding_dim)
        self.srch_adults_count_embedding = nn.Embedding(config.srch_adults_count_vocab_size, config.embedding_dim)
        self.srch_children_count_embedding = nn.Embedding(config.srch_children_count_vocab_size, config.embedding_dim)
        self.srch_room_count_embedding = nn.Embedding(config.srch_room_count_vocab_size, config.embedding_dim)
        self.srch_saturday_night_bool_embedding = nn.Embedding(config.srch_saturday_night_bool_vocab_size, config.embedding_dim)
        self.prop_id_embedding = nn.Embedding(config.prop_id_vocab_size, config.embedding_dim)
        self.prop_country_id_embedding = nn.Embedding(config.prop_country_id_vocab_size, config.embedding_dim)
        self.prop_starrating_embedding = nn.Embedding(config.prop_starrating_vocab_size, config.embedding_dim)
        self.prop_brand_bool_embedding = nn.Embedding(config.prop_brand_bool_vocab_size, config.embedding_dim)
        self.prop_location_score1_embedding = nn.Embedding(config.prop_location_score1_vocab_size, config.embedding_dim)
        self.prop_promotion_flag_embedding = nn.Embedding(config.prop_promotion_flag_vocab_size, config.embedding_dim)
        
        self.output_dim = config.embedding_dim * 11
    def forward(self, x):
        embeddings = [
            self.srch_id_embedding(x[:, 0]),
            self.srch_adults_count_embedding(x[:, 1]),
            self.srch_children_count_embedding(x[:, 2]),
            self.srch_room_count_embedding(x[:, 3]),
            self.srch_saturday_night_bool_embedding(x[:, 4]),
            self.prop_id_embedding(x[:, 5]),
            self.prop_country_id_embedding(x[:, 6]),
            self.prop_starrating_embedding(x[:, 7]),
            self.prop_brand_bool_embedding(x[:, 8]),
            self.prop_location_score1_embedding(x[:, 9]),
            self.prop_promotion_flag_embedding(x[:, 10]),
        ]
        return torch.cat(embeddings, dim=1)
class ESMM(nn.Module):
    def __init__(self, embedding_layer, ctr_tower, cvr_tower):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.ctr_tower = ctr_tower
        self.cvr_tower = cvr_tower
    def forward(self, x):
        shared_repr = self.embedding_layer(x)
        ctr_logits = self.ctr_tower(shared_repr)
        cvr_logits = self.cvr_tower(shared_repr)
        p_ctr = torch.sigmoid(ctr_logits)
        p_cvr = torch.sigmoid(cvr_logits)
        p_ctcvr = torch.mul(p_ctr, p_cvr)
        return p_ctr, p_cvr, p_ctcvr