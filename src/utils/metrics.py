import numpy as np

def compute_ndcg_at_k(relevances, k=38):
    """Compute NDCG@k for a single query."""
    relevances = np.array(relevances[:k], dtype=float)
    
    if len(relevances) == 0:
        return 0.0
    
    # DCG@k
    positions = np.arange(1, len(relevances) + 1)
    dcg = np.sum((2**relevances - 1) / np.log2(positions + 1))
    
    # Ideal DCG@k (sort descending)
    ideal_relevances = np.sort(relevances)[::-1]
    idcg = np.sum((2**ideal_relevances - 1) / np.log2(positions + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def assign_relevance(click_bool, booking_bool):
    """
    Relevance grades:
      5 - purchased (booking_bool == 1)
      1 - clicked but not purchased (click_bool == 1)
      0 - neither
    """
    if booking_bool == 1:
        return 5
    elif click_bool == 1:
        return 1
    else:
        return 0


def compute_mean_ndcg(test_df, pred_scores, query_col='srch_id', k=38):
    """
    test_df   : DataFrame with ground truth columns (click_bool, booking_bool, srch_id)
    pred_scores: array-like of model scores aligned with test_df rows
    query_col : column name identifying each query/search
    k         : cutoff rank
    """
    df = test_df.copy()
    df['pred_score'] = pred_scores
    df['relevance'] = df.apply(
        lambda row: assign_relevance(row['click_bool'], row['booking_bool']), axis=1
    )
    
    ndcg_scores = []
    
    for srch_id, group in df.groupby(query_col):
        # Sort hotels by predicted score descending (model's ranking)
        group_sorted = group.sort_values('pred_score', ascending=False)
        relevances = group_sorted['relevance'].tolist()
        ndcg = compute_ndcg_at_k(relevances, k=k)
        ndcg_scores.append(ndcg)
    
    mean_ndcg = np.mean(ndcg_scores)
    return mean_ndcg, ndcg_scores