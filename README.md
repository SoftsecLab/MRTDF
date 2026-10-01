# MRTDF

## Setup

Run from this directory:

```bash
pip install -r requirements.txt
```

## Data Preparation

The datasets used in the experiments are available at:

- Machine-revised text: [MixSet](https://github.com/Dongping-Chen/MixSet/tree/f37eea51fcb7fd490391431c52adf79848cb1e89/data/MixSet/separated_files)
- Machine-generated text: [DetectRL](https://github.com/NLP2CT/DetectRL/tree/3472f0ae3cc81133c821e901f6d5c8f79c60dd23/Benchmark/Benchmark_Data/Direct_Prompt)

After downloading the corresponding data, use the following scripts to reconstruct the test splits used in this study:

```bash
python prepare_mrt_test.py
python prepare_mgt_test.py
```

## Configuration

Set the following parameters in `config.py`:

- Neutralization API: `NEUTRALIZATION_API_KEY`, `NEUTRALIZATION_API_URL`, `NEUTRALIZATION_MODEL`
- Model paths: `Mistral_PATH`, `DEBERTA_PATH`
- Checkpoint paths: `OUTPUT_MISTRAL_SAVE_DIR`, `REDUCER_SAVE_PATH`, `MAIN_CLASSIFIER_DIR`
- Neutralization files: `NEUTRALIZATION_INPUT`, `NEUTRALIZATION_OUTPUT`
- Recovery files: `RECOVERY_INPUT`, `RECOVERY_OUTPUT`
- Evaluation directories: `HUMAN_TEST_DIR`, `MODIFY_TEST_DIR`, pointing to the corresponding recovery output directories
- Optional training directories: `HUMAN_TRAIN_DIR`, `MODIFY_TRAIN_DIR`, needed only for classifier retraining

## Style Neutralization

```bash
python run.py "Restore module/step0_neutralize.py"
```

## Style Recovery

```bash
python run.py "Restore module/step5_infer.py"
```

Repeat these steps for each human-written and machine-revised test file, preserving the original filenames and saving the recovered files in separate new output directories.

## Feature Extraction and Evaluation

Extract style-difference features and evaluate using the supplied seed-41 classifier checkpoint.

```bash
python run.py task_result/task1/extract_style_diff_features.py
python run.py run_main.py --task 1
```
