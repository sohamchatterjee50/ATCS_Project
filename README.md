# Explanatory Bias in Large Language Models  
**Comparing Reasoning and Non-Reasoning Models on Stereotypical Prompts**  
_Soham Chatterjee, Pearl Owusu, Oliver Savolainen, Yushuang Wang_  

This repository contains all code, data, and analyses for the experiments in our paper: **“Explanatory Bias in Large Language Models: Comparing Reasoning and Non-Reasoning Models on Stereotypical Prompts”**.  

---

## ⚙️ Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-org/ATCS_PROJECT.git
   cd ATCS_PROJECT
````

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Run vLLM inference

Generate model outputs for a BBQ JSONL file.

```bash
python run_vllm.py \
  --hf_token YOUR_HF_TOKEN \
  --model_name_or_path microsoft/phi-4-reasoning-plus \
  --data_file BBQ/Gender_identity.jsonl \
  --log_file llm_logs/Gender/phi4_reasoning.log \
  --batch_size 16 \
  --device cuda \
  --max_new_tokens 1024 \
  --temperature 0.7 \
  --msg_format 1
```

* `--msg_format`:
  1 = question + options + context (full disambiguation)
  2 = context only (continuation)
  3 = question only (no options)

### 2. Analyze inference logs

Turn raw JSONL logs into summary CSVs and diagnostic plots.

```bash
python log_analysis.py \
  --inference_file llm_logs/Gender/phi4_reasoning.log llm_logs/Gender/phi4_plus.log \
  --dataset_file BBQ/Gender_identity.jsonl \
  --metadata_csv BBQ/additional_metadata.csv \
  --output_dir main_results_per_category_and_model/Gender_Results
```

### 3. Combine across models

Aggregate multiple summary CSVs (from `log_analysis.py`) for side-by-side comparison.

```bash
python combined_analysis.py \
  --llama3_files main_results_per_category_and_model/Gender_Results/llama3*.csv \
  --deepseek_files main_results_per_category_and_model/Gender_Results/distil_deepseek*.csv \
  --phi4_files main_results_per_category_and_model/Gender_Results/phi4*.csv \
  --phi4plus_files main_results_per_category_and_model/Gender_Results/phi4_plus*.csv \
  --output_dir combined_results_across_categories/Gender
```

---

## 📊 Results & Artifacts

* **`main_results_per_category_and_model/`**
  CSV summaries and plots broken out by demographic category (Age, Gender, Religion) and model.

* **`combined_results_across_categories/`**
  Cross-category, cross-model comparison tables & figures.

* **`LIB/`**
  Reproduce the Linguistic Bias (LIB) score analyses via the Jupyter notebook.

* **`qualitative/`**
  Curated examples, error analyses, and thematic observations.

```
```
