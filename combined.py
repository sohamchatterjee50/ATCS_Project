"""
Compare multiple summary.csv files from different models side-by-side.

Usage:
    python compare_summaries.py \
        --model_names "LLaMA-3.1-8B" "DeepSeek-R1-8B" "Phi-4" "Phi-4-Plus" \
        --summary_files llama3_summary.csv deepseek_summary.csv phi4_summary.csv phi4plus_summary.csv \
        --output_dir ./comparison_results

This script loads each summary.csv, attaches a model label, concatenates,
then generates comparison plots for accuracy, avg_length, expl_rate,
stereotype pick rate, and mention rates.
Saves combined CSV and figures to the output directory.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare summary.csv files across multiple models with multiple runs per model"
    )
    parser.add_argument(
        "--llama3_files", nargs='*', default=[],
        help="Summary CSV files for Meta LLaMA 3.1 8B"
    )
    parser.add_argument(
        "--deepseek_files", nargs='*', default=[],
        help="Summary CSV files for DeepSeek-R1 Distill LLaMA 8B"
    )
    parser.add_argument(
        "--phi4_files", nargs='*', default=[],
        help="Summary CSV files for Phi-4-Reasoning"
    )
    parser.add_argument(
        "--phi4plus_files", nargs='*', default=[],
        help="Summary CSV files for Phi-4-Reasoning-Plus"
    )
    parser.add_argument(
        "--output_dir", default="./comparison_results",
        help="Directory to save combined comparisons and plots"
    )
    return parser.parse_args()


def load_and_average(files):
    """
    Load multiple summary CSVs for a model and average metrics per sample type.
    Returns a DataFrame with columns including 'is_stereotypical_sample',
    averaged numeric metrics.
    """
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    # average metrics across runs by sample type
    avg = combined.groupby('is_stereotypical_sample').mean().reset_index()
    return avg


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_sets = {
        'Meta LLaMA-3.1-8B': args.llama3_files,
        'DeepSeek-R1-Distill-Llama-8B': args.deepseek_files,
        'Phi-4': args.phi4_files,
        'Phi-4-Reasoning-Plus': args.phi4plus_files,
    }

    results = []
    for model_name, files in model_sets.items():
        if not files:
            continue
        df_avg = load_and_average(files)
        df_avg['model'] = model_name
        results.append(df_avg)

    if not results:
        raise ValueError('No summary files provided for any model.')

    compare_df = pd.concat(results, ignore_index=True)
    compare_df.to_csv(os.path.join(args.output_dir, 'combined_summary.csv'), index=False)

    # Plot settings
    sns.set_theme(style='whitegrid')
    #is_stereotypical_sample,total,extracted,accuracy_extracted,truncation_rate,mean_length,mean_length_no_trunc,expl_rate,stereo_pred_rate,stereo_mentions_per_sample,anti_mentions_per_sample
    metrics = [
        ('accuracy_extracted', 'Accuracy'),
        ('mean_length', 'Average Response Length'),
        ('mean_length_no_trunc', 'Average Response Length of not Truncated Samples'),
        ('truncation_rate', 'Truncation Rate'),
        ('expl_rate', 'Explanation Rate'),
        ('stereo_pred_rate', 'Stereotype Pick Rate'),
        ('stereo_mentions_per_sample', 'Stereotype Mentions per Sample'),
        ('anti_mentions_per_sample', 'Anti-Stereotype Mentions per Sample')
    ]
    for col, title in metrics:
        plt.figure(figsize=(8,5))
        sns.barplot(
            data=compare_df,
            x='model',
            y=col,
            hue='is_stereotypical_sample'
        )
        plt.title(f'{title} by Model and Sample Type')
        plt.xlabel('Model')
        plt.ylabel(title)
        plt.legend(
            title='Stereotypical Sample',
            loc='center left',
            bbox_to_anchor=(1, 0.5),
            borderaxespad=0.5,
            frameon=False,
            prop={'size':8}
        )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out_path = os.path.join(args.output_dir, f'{col}_comparison.png')
        plt.savefig(out_path)
        plt.close()

    print(f'Comparison complete. Results in {args.output_dir}')


if __name__ == '__main__':
    main()
