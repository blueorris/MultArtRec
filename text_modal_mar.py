import cornac
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality
import numpy as np

# Load data
feedback = np.load('data/feedback.npy')
features = np.load('data/text_features_bert_base.npy')
# features = np.load('data/text_features_poems_sentiment_final.npy')
# features = np.load('data/text_features_bert_base_emotion.npy')
item_ids = np.load('data/item_ids.npy')

item_text_modality = TextModality(features=features, ids=item_ids)

# Define an evaluation method to split feedback into train and test sets
ratio_split = RatioSplit(
    data=feedback,
    test_size=0.2,
    exclude_unknowns=True,
    item_text=item_text_modality,
    verbose=True,
    seed=123,
    rating_threshold=0.5,
)

# Instantiate MultArtRec model
mar = cornac.models.MAR(
    input_size=len(features[0]),
    K=50,  # latent topic size
    mlp_layers=[200],  # latent layer size of MLP
    training_epochs=300, 
    lambda_n=1000,  # bias for reconstruction loss
    lambda_r=1,  # bias for regularization loss
    lambda_v=1,  # bias for item loss
    lambda_c=1,  # bias for rating loss
    lambda_t=0.1,  # bias for topic loss
    learning_rate=0.001,
    batch_size=128,
)

k = 10
rec_10 = cornac.metrics.Recall(k=k)
pre_10 = cornac.metrics.Precision(k=k)
ndcg_10 = cornac.metrics.NDCG(k=k)

metrics = [ndcg_10, pre_10, rec_10]
cornac.Experiment(eval_method=ratio_split, models=[mar], metrics=metrics).run()