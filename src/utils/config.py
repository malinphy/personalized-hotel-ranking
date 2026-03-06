import os

class Config:
    def __init__(self, train_df):
        # --- Paths ---
        self.checkpoint_dir = os.path.join("src", "training", "weights")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- Model Hyperparameters ---
        self.embedding_dim = 96
        self.learning_rate = 0.001
        self.batch_size = 256
        self.epochs = 10

        # Dynamic Vocab Sizes (Calculated from data)
        self.srch_id_vocab_size = int(train_df['srch_id_encoded'].max() + 1)
        self.prop_id_vocab_size = int(train_df['prop_id_encoded'].max() + 1)
        self.prop_country_id_vocab_size = int(train_df['prop_country_id_encoded'].max() + 1)

        # Static or small Vocab Sizes
        self.srch_adults_count_vocab_size = int(train_df['srch_adults_count'].max() + 1)
        self.srch_children_count_vocab_size = int(train_df['srch_children_count'].max() + 1)
        self.srch_room_count_vocab_size = int(train_df['srch_room_count'].max() + 1)
        self.srch_saturday_night_bool_vocab_size = 2
        self.prop_starrating_vocab_size = int(train_df['prop_starrating'].max() + 1)
        self.prop_brand_bool_vocab_size = 2
        self.prop_location_score1_vocab_size = 1000
        self.prop_promotion_flag_vocab_size = 2

def get_config(df):
    return Config(df)