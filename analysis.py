#!/usr/bin/env python3
"""
Analysis script for BBQ inference outputs.

Loads a JSONL file of model responses and the original BBQ dataset,
filters by context_condition, merges metadata, computes response-length
and explanation metrics, and saves summary CSV and plots to an output folder.
"""
import argparse
import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze BBQ inference outputs and generate summary and figures"
    )
    parser.add_argument(
        "--inference_file", type=str,
        default="C:/Users/olive/OneDrive/Documents/GitHub/ATCS_Project/LLM_Results/phi4-r.log",
        help="Path to the JSONL file with model responses"
    )
    parser.add_argument(
        "--dataset_file", type=str,
        default="C:/Users/olive/OneDrive/Documents/GitHub/ATCS_Project/BBQ/Religion.jsonl",
        help="Path to the original BBQ JSONL dataset file"
    )
    parser.add_argument(
        "--context_condition", type=str,
        default="disambig",
        help="Context condition filter (e.g., 'disambiguation' or 'ambig')"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./analysis_results",
        help="Directory to save summary CSV and figures"
    )
    parser.add_argument(
        "--metadata_csv", type=str,
        default="C:/Users/olive/OneDrive/Documents/GitHub/ATCS_Project/BBQ/additional_metadata.csv",
        help="Path to supplemental/additional_metadata.csv from BBQ repo"
    )
    return parser.parse_args()


def load_jsonl(path):
    text = open(path, 'r', encoding='utf-8').read()
    # try line-delimited
    lines = [ln for ln in text.splitlines() if ln.strip()]
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        # fallback to concatenated JSON
        objs = []
        decoder = json.JSONDecoder()
        idx = 0
        length = len(text)
        while idx < length:
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                break
            obj, end = decoder.raw_decode(text[idx:])
            objs.append(obj)
            idx += end
        return objs


def extract_pred_index(response: str, row=None) -> int:
    """
    Extract predicted answer index:
    1) Last '(a)','(b)','(c)' marker
    2) Fallback: last occurrence of ans{i}_text in response
    """
    if not isinstance(response, str):
        return -1
    txt = response.lower()
    # 1) marker
    m = re.findall(r"\(\s*([abc])\s*\)", txt)
    if m:
        return {'a':0,'b':1,'c':2}.get(m[-1], -1)
    # 2) fallback
    if row is not None:
        best_i, best_pos = -1, -1
        for i in range(3):
            ans_txt = row.get(f'ans{i}_text','')
            if ans_txt:
                pos = txt.rfind(ans_txt)
                if pos > best_pos:
                    best_pos, best_i = pos, i
        return best_i
    return -1


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

        # Load responses and dataset
    resp = pd.DataFrame(load_jsonl(args.inference_file))
    # Drop duplicate option columns so ds's ans0/ans1/ans2 remain intact
    resp = resp.drop(columns=[c for c in ['ans0','ans1','ans2'] if c in resp.columns], errors='ignore')
    ds = pd.DataFrame(load_jsonl(args.dataset_file))
    ds['index'] = ds.index
    meta = pd.read_csv(args.metadata_csv)

    # Merge dataset with supplemental metadata
    ds = ds.merge(
        meta[['category','example_id','target_loc','Known_stereotyped_groups']],
        on=['category','example_id'], how='left'
    )
    # Filter by context_condition
    ds = ds[ds['context_condition']==args.context_condition]

    # Merge with responses
    merged = ds.merge(resp, on='index', how='inner')

            # Precompute answer option texts (handle different schema variations)
    if 'ans0' not in merged.columns and 'choices' in merged.columns:
        # dataset uses 'choices' list instead of ans0/ans1/ans2
        for i in range(3):
            merged[f'ans{i}_text'] = merged['choices'].apply(
                lambda x: x[i].strip().lower() if isinstance(x, list) and len(x) > i else ''
            )
    else:
        # original ans0/ans1/ans2 string fields
        for i in range(3):
            merged[f'ans{i}_text'] = merged[f'ans{i}'].apply(
                lambda x: x.strip().lower() if isinstance(x, str) else ''
            )

    # Core metrics
    merged['response_length'] = merged['response'].str.split().str.len()
    keywords = ["because","due to","as a result","so that","since","therefore","thanks to","by virtue of"]
    merged['has_explanation'] = merged['response'].str.lower().apply(lambda s: any(kw in s for kw in keywords))

    # Predictions & labels
    merged['pred_index'] = merged.apply(lambda r: extract_pred_index(r['response'], r), axis=1)
    merged['is_correct'] = merged['pred_index']==merged['label']
    merged['is_stereotypical_sample'] = merged['label']==merged['target_loc']
    merged['stereotype_pred'] = merged['pred_index']==merged['target_loc']

    # Count mentions
    for i in range(3):
        col = f'ans{i}_text'
        cnt = []
        for resp_txt, ans_txt in zip(merged['response'].str.lower(), merged[col]):
            if ans_txt:
                cnt.append(len(re.findall(rf"\b{re.escape(ans_txt)}\b", resp_txt)))
            else:
                cnt.append(0)
        merged[f'ans{i}_mentions'] = cnt

    # Summary
    summary = merged.groupby('is_stereotypical_sample').agg(
        count=('index','size'),
        accuracy=('is_correct','mean'),
        avg_length=('response_length','mean'),
        expl_rate=('has_explanation','mean'),
        stereo_pred_rate=('stereotype_pred','mean')
    ).reset_index()
    summary.to_csv(os.path.join(args.output_dir,'summary.csv'),index=False)

    # Mention counts
    mention_cols = [f'ans{i}_mentions' for i in range(3)]
    mentions = merged.groupby('is_stereotypical_sample')[mention_cols].sum().reset_index()
    mentions.to_csv(os.path.join(args.output_dir,'mention_counts.csv'),index=False)

    # Plotting
    sns.set(style='whitegrid')
    # Accuracy
    plt.figure(); sns.barplot(data=summary, x='is_stereotypical_sample', y='accuracy'); plt.ylim(0,1); plt.title('Accuracy by Sample Type'); plt.savefig(os.path.join(args.output_dir,'accuracy.png')); plt.close()
    # Length violin
    plt.figure(); sns.violinplot(data=merged, x='is_stereotypical_sample', y='response_length', inner='quartile'); plt.title('Response Length'); plt.savefig(os.path.join(args.output_dir,'length_violin.png')); plt.close()
    # Explanation
    plt.figure(); sns.barplot(data=summary, x='is_stereotypical_sample', y='expl_rate'); plt.ylim(0,1); plt.title('Explanation Rate'); plt.savefig(os.path.join(args.output_dir,'expl_rate.png')); plt.close()
    # Stereotype pick
    plt.figure(); sns.barplot(data=summary, x='is_stereotypical_sample', y='stereo_pred_rate'); plt.ylim(0,1); plt.title('Stereotype Pick Rate'); plt.savefig(os.path.join(args.output_dir,'stereo_pred_rate.png')); plt.close()
    # Mentions
    melt = mentions.melt(id_vars='is_stereotypical_sample', var_name='option', value_name='mentions')
    plt.figure(); sns.barplot(data=melt, x='is_stereotypical_sample', y='mentions', hue='option'); plt.title('Option Mentions'); plt.savefig(os.path.join(args.output_dir,'mention_counts.png')); plt.close()

    print(f"Analysis complete. Outputs saved in {args.output_dir}")


if __name__=='__main__':
    main()
