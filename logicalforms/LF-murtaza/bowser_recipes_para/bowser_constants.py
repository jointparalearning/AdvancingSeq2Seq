import os

class constants:
    vocab_file = "data/processed/vocab_recipes_v1.pkl"
    data_dir = "data/processed"
    train_data_file = "recipes_train_v1.json"
    test_data_file = "recipes_test_v1.json"

    results_dir = "./results"
    log_dir = os.path.join("results", "run_logs")
    tb_log_dir = os.path.join("results", "tb_logs")
    model_dir = os.path.join("results", "saved_models")
    pkl_dir = os.path.join(results_dir, "pkl_files")

    version = "train_v2"
    add_info = ""       # Can put things like DLaaS, server, etc. here. Please add underscore after the word
