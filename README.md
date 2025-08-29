# Package management

## Installation

Please install `uv`, a python package manager [(link)](https://astral.sh/blog/uv), using the following command

```zsh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Directory setup

1. Activate the virtual environment

```zsh
source .venv/bin/activate
```

2. Find a directory of `site-packages`. It's something like `/Users/pakhapoomsarapat/Workspaces/simulating-agents/.venv/lib/python3.10/site-packages` for example.

```zsh
python -m site
```

3. Add a path to `custom_path.pth` inside the `site-packages`'s directory.

```zsh
echo path/to/project > path/to/site-packages/custom_path.pth
```

For example,

```zsh
echo "/Users/pakhapoomsarapat/Workspaces/simulating-agents" > /Users/pakhapoomsarapat/Workspaces/simulating-agents/.venv/lib/python3.10/site-packages/custom_path.pth
```

## Exploration

For running a python script, please specify `uv run` followed by the python file, such as

```zsh
uv run path/to/py/file.py
```

# Project structure

## Client Component

```
src/
└── client/
    ├── config.py         # Configuration management and environment setup
    ├── llm.py            # Azure OpenAI client interface
    └── qdrant.py         # Qdrant vector database client interface
```

## Data Refresher

### High-Level Structure

```
src/data_refresher/
├── clustering/             # Customer segmentation approaches
├── data/                   # Data storage and management
├── data_prep/              # Data preparation pipelines
├── evaluation/             # Performance evaluation framework
├── prediction/             # Prediction generation system
└── rag/                    # Retrieval-Augmented Generation components
```

### Clustering

```
src/data_refresher/clustering/
├── approach_1_trad/                        # Traditional clustering approach
│   ├── model/                              # Trained models
│   ├── result/                             # Training results
│   ├── utils/                              # Utility functions
│   └── train.py                            # Training script
│
├── approach_2_embed/                       # ✅ SELECTED APPROACH (best performance)
│   ├── model/                              # Trained models
│   ├── pred_result/                        # Prediction results (test data)
│   │   ├── clus_level_eval/                # Cluster-level evaluation
│   │   ├── full_data_with_cluster/         # Full data with cluster assignments
│   │   └── overall_eval/                   # Overall evaluation metrics
│   ├── result/                             # Training results
│   │   ├── clus_explain/                   # Cluster explanations/descriptions
│   │   ├── clus_level_eval/                # Cluster-level evaluation
│   │   ├── full_data_with_cluster/         # Full training data with clusters
│   │   └── overall_eval/                   # Overall evaluation metrics
│   ├── utils/                              # Utility functions
│   ├── train.py                            # Training script
│   └── test.py                             # Testing/inference script
│
├── approach_3_embed_num/                   # Numerical embedding approach
│   ├── model/
│   ├── result/
│   ├── utils/
│   └── train.py
│
└── change_analysis/                        # Cluster change analysis over time
│   ├── result/
│   ├── utils/
│   └── change_analysis.py
```

### Data and Data Preparation

```
src/data_refresher/data/
├── preprocessed_data/                  # Cleaned and processed datasets
├── raw_data/                           # Original, unprocessed source data
├── summary_reasoning/                  # Generated summaries and reasoning outputs
├── T0/                                 # Time period T0 train and test datasets
└── T1/                                 # Time period T1 train and test datasets
```

```
src/data_refresher/data_prep/
├── data_prep_rag/              # Specialized preparation for RAG systems
├── eda/                        # Exploratory Data Analysis scripts and outputs
└── preprocess.py               # Main data preprocessing pipeline
```

### Evaluation

```
src/data_refresher/evaluation/
├── error_analysis/             # Detailed error analysis by approach
├── combined_metrics.csv        # Aggregated performance metrics
├── cal_demog_accuracy.py       # Demographic accuracy calculator
└── merge_eval.py               # Evaluation results merger
```

### Prediction

```
src/data_refresher/prediction/
├── pred_results/               # Prediction outputs from various models
├── similar_cust_results/       # Similar customer retrieval results
├── utils/                      # Prediction utilities and helpers
├── prompts.py                  # AI prompt templates for prediction
└── run_predictions.py          # Main prediction orchestration script
```

### RAG

```
src/data_refresher/rag/
├── utils/                      # RAG utility functions
└── store.py                    # Vector store operations and management
```

## Recommendation

### High-Level Structure

```
src/recommendation/
├── data/                       # Comprehensive data storage for recommendation system
├── data_prep/                  # Data preparation and preprocessing pipelines
├── evaluation/                 # Comprehensive evaluation framework
│   ├── baseline/               # Baseline model evaluations
│   ├── eval_results/           # Evaluation results storage
│   ├── cal_eval.py             # Evaluation metric calculation
│   └── merge_eval.py           # Results merging and aggregation
├── metrics/                    # Performance metrics tracking
├── models/                     # Trained model storage
├── predictions/                # Prediction outputs
├── rag/                        # RAG-specific components
├── utils/                      # Utility functions and helpers
├── prompts.py                  # AI prompt templates
├── run_predictions.py          # Prediction orchestration
└── train_model.py              # Model training pipeline
```

### Data and Data Preparation

```
src/recommendation/data/
├── ans_key/                    # Answer keys for evaluation
├── preprocessed_data/          # Processed data for modeling
│   ├── embedded_demog/         # Embedded demographic data
│   ├── test_T0_demog_summ/     # Test data with demographic summaries (T0)
│   └── train_T0_demog_summ/    # Training data with demographic summaries (T0)
├── rag/                        # RAG-specific data
├── raw_data/                   # Original source data
├── rl/                         # Reinforcement learning data
├── T0/                         # Time period T0 datasets
├── T1/                         # Time period T1 datasets
└── T1_predicted/               # T1 data with predictions
```

```
src/recommendation/data_prep/
├── eda/                        # Exploratory Data Analysis
├── rag/                        # RAG-specific data preparation
├── rl/                         # RL-specific data preparation
├── T1_predicted/               # Preparation for predicted T1 data
├── ans_key_prep.py             # Answer key preparation
├── merge_with_txn.py           # Transaction data merging
└── preprocess.py               # Main preprocessing pipeline
```
