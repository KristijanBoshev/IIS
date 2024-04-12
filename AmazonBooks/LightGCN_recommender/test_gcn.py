def precision_at_k(model, test_data, k):
    """
    Compute precision at k on the test
    """
    precision_sum = 0.0
    total_users = len(test_data)

    for user_id, test_items in test_data.items():
        recommendations = model.get_recommendations(user_id, k)
        relevant_items = set(test_items)
        recommended_items = set(recommendations[:k])
        precision_sum += len(recommended_items.intersection(relevant_items)) / k

    return precision_sum / total_users


def recall_at_k(model, test_data, k):
    """
    Compute recall@k for the given model and test data.
    """
    recall_sum = 0.0
    total_users = len(test_data)

    for user_id, test_items in test_data.items():
        recommendations = model.get_recommendations(user_id, k)
        relevant_items = set(test_items)
        recommended_items = set(recommendations[:k])
        recall_sum += len(recommended_items.intersection(relevant_items)) / len(relevant_items)

    return recall_sum / total_users


def f1_at_k(model, test_data, k):
    """
    Compute F1@k for the given model and test data.
    """
    precision = precision_at_k(model, test_data, k)
    recall = recall_at_k(model, test_data, k)

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)