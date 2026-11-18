"""
Master End-to-End Training, Experimentation, Bayesian Optimisation, HMM+Viterbi Decoding,
and Visualisation Pipeline for PathSnap.
"""

import json
import os
import time
import joblib
import pandas as pd

from data_processing.eda import run_eda
from hmm_viterbi.viterbi_decoder import HMMConfig, ViterbiRouteDecoder
from hyperparameter_optimisation.bayesian_opt import run_bayesian_optimisation
from model_training.train_and_compare import build_partition_cache, run_model_comparison
from visualisations.plots_and_maps import (
    generate_eda_plots,
    generate_folium_map_for_trip,
    generate_model_comparison_plots,
    generate_optimisation_and_hmm_plots,
    generate_static_map_comparison_figure,
)


def main() -> None:
    t_start = time.perf_counter()
    root_dir = os.path.dirname(os.path.abspath(__file__))
    student_folder = os.path.join(root_dir, "Student folder")
    results_dir = os.path.join(root_dir, "supporting_results")
    fig_dir = os.path.join(results_dir, "figures")
    maps_dir = os.path.join(results_dir, "maps")
    models_dir = os.path.join(root_dir, "models")

    print("=" * 80)
    print("STEP 1: Exploratory Data Analysis (EDA) across 80 Train & 10 Student-Test Trips")
    print("=" * 80)
    inv_csv = os.path.join(results_dir, "dataset_inventory.csv")
    if os.path.exists(inv_csv):
        df_inv = pd.read_csv(inv_csv)
        print(f"Loaded existing EDA inventory for {len(df_inv)} trips.")
    else:
        df_inv = run_eda(student_folder, results_dir)
        print(f"Completed EDA on {len(df_inv)} trips.")

    print("\n" + "=" * 80)
    print("STEP 2: Candidate Generation (K=8) & 27-Feature Extraction")
    print("=" * 80)
    t_f0 = time.perf_counter()
    train_cache = build_partition_cache(student_folder, "train", k_candidates=8)
    test_cache = build_partition_cache(student_folder, "student_test", k_candidates=8)
    print(f"Extracted features for {len(train_cache)} train trips and {len(test_cache)} test trips in {time.perf_counter() - t_f0:.2f}s.")

    print("\n" + "=" * 80)
    print("STEP 3: Training & Comparing 6 Models Across 6 Algorithm Families")
    print("=" * 80)
    df_comp, detailed_results = run_model_comparison(train_cache, test_cache, results_dir)

    print("\n" + "=" * 80)
    print("STEP 4: Bayesian Hyperparameter Optimisation (Optuna TPE) & HMM+Viterbi Decoding")
    print("=" * 80)
    opt_summary = run_bayesian_optimisation(
        train_cache=train_cache,
        test_cache=test_cache,
        output_dir=results_dir,
        models_dir=models_dir,
        n_trials=15,
    )

    print("\n" + "=" * 80)
    print("STEP 5: Generating Plots & Interactive Folium Geographical Maps")
    print("=" * 80)
    generate_eda_plots(df_inv, test_cache, fig_dir)
    generate_model_comparison_plots(df_comp, detailed_results, fig_dir)

    df_trials = pd.read_csv(os.path.join(results_dir, "bayesian_optimisation_trials.csv"))
    df_imp = pd.read_csv(os.path.join(results_dir, "feature_importances.csv"))
    df_per_trip = pd.read_csv(os.path.join(results_dir, "per_trip_student_test_results.csv"))
    generate_optimisation_and_hmm_plots(df_trials, df_imp, df_per_trip, opt_summary, fig_dir)

    # Load final saved model artifact and generate Folium HTML maps for all 10 student_test trips
    artifact = joblib.load(os.path.join(models_dir, "final_model.joblib"))
    clf_final = artifact["classifier"]
    decoder = ViterbiRouteDecoder(HMMConfig(**artifact["hmm_config"]))

    tuned_probs_test = [
        clf_final.predict_proba(td["X"].reshape(-1, td["X"].shape[-1]))[:, 1].reshape(td["X"].shape[:2])
        for td in test_cache
    ]
    for idx, tdata in enumerate(test_cache):
        map_path = generate_folium_map_for_trip(tdata, tuned_probs_test[idx], decoder, maps_dir, fig_dir)
        print(f"  Generated interactive Folium map: {os.path.basename(map_path)}")

    generate_static_map_comparison_figure(test_cache, tuned_probs_test, decoder, fig_dir)

    total_elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE in {total_elapsed:.1f}s!")
    print(f"Final Model Artifact saved to: {os.path.join(models_dir, 'final_model.joblib')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
