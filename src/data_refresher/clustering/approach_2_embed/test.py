from src.data_refresher.clustering.approach_2_embed.utils.utils import *
import pandas as pd
import joblib


def test_approach_2(test_df, model):
    df_embedding = embedding_creation(test_df)

    predicted_df = predict(test_df, df_embedding, model)

    col = 'cluster'

    clus_level_eval = calculate_cluster_distances(
        predicted_df, 
        df_embedding,
        model,
        col
    )
    # Calculate overall evaluation metrics
    overall_eval = calculate_overall_distance(clus_level_eval)


    # Save results with versioning
    FULL_DATA_WITH_CLUSTER_DIR = RESULTS_DIR + '/full_data_with_cluster'
    CLUS_LEVEL_EVAL_DIR = RESULTS_DIR + '/clus_level_eval'
    OVERALL_EVAL_DIR = RESULTS_DIR + '/overall_eval'

    # full_data_with_cluster
    save_csv_file(FULL_DATA_WITH_CLUSTER_DIR, predicted_df, 'full_data_with_cluster')
    # clus_level_eval
    save_csv_file(CLUS_LEVEL_EVAL_DIR, clus_level_eval, 'clus_level_eval')
    # overall_eval
    save_csv_file(OVERALL_EVAL_DIR, overall_eval, 'overall_eval')

    return predicted_df, clus_level_eval, overall_eval


if __name__ == "__main__":
    test_df = pd.read_csv('src/data_refresher/data/T0/test_df.csv')
    MODEL_DIR = "src/data_refresher/clustering/approach_2_embed/model/model_app2_v4.pkl"
    RESULTS_DIR = "src/data_refresher/clustering/approach_2_embed/pred_result"


    model = joblib.load(MODEL_DIR)
    predicted_df, clus_level_eval, overall_eval = test_approach_2(test_df, model)

    print(predicted_df)
    print(clus_level_eval)
    print(overall_eval)