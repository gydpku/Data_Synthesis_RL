# 🚀 Data-Synthesis-RL: Efficient Few-Shot RL Fine-Tuning with Synthetic Data Generation

## Introduction

![System Overview](img/final-one.png)

This project implements an efficient approach for fine-tuning a base language model ($\mathcal{M}_{base}$) when only a few demonstration examples ($\mathcal{D}$) are available. It leverages a powerful teacher large language model ($\mathcal{T}$), external knowledge retrieval, and strategic synthetic data generation to create an effective training dataset. The base model is then trained on the most informative subset of this synthetic data using reinforcement learning.

## 🛠️ Components

The system utilizes four key components:

1. **🔍 Passage Retriever ($\mathcal{P}$)**:
   * Takes keywords (extracted from the task instruction $\mathcal{I}$ and demos $\mathcal{D}$ using the teacher model $\mathcal{T}$) as input.
   * Searches a large text library ($\mathcal{L}$, e.g., Wikipedia) to find relevant passages ($\mathcal{R}$) that provide external context.

2. **📊 LLM Data Generator ($\text{LLM}_{generator}$)**:
   * Uses the teacher model ($\mathcal{T}$) to synthesize new training data ($\mathcal{S}_{initial}$).
   * Takes task instructions ($\mathcal{I}$), summarized sample patterns ($P$), demonstration examples ($\mathcal{D}$), and retrieved passages ($\mathcal{R}$) as input.
   * Incorporates a verification step (e.g., majority voting) to ensure data quality.

3. **✍️ LLM Data Re-writer ($\text{LLM}_{writer}$)**:
   * Also uses the teacher model ($\mathcal{T}$).
   * Takes existing synthetic samples and modifies them to be either harder ($\mathcal{S}\_{harder}$) or easier ($\mathcal{S}_{easier}$) based on the base model's performance.
   * Includes a verification step to filter low-quality outputs.

4. **📈 Trainer ($T$)**:
   * Implements a reinforcement learning algorithm.
   * Trains the base model ($\mathcal{M}_{base}$) specifically on a selected subset of high-potential synthetic samples.

## ⚙️ Workflow

![Workflow Step 1](img/final1.4.png)
![Workflow Step 2](img/final2.4.png)

Given a base model ($\mathcal{M}_{base}$), task instruction ($\mathcal{I}$), and a few demonstration examples ($\mathcal{D}$), the training process involves four main steps:

1. **Keyword Extraction and Passage Retrieval**:
   * The teacher model $\mathcal{T}$ extracts domain-specific keywords ($\mathcal{K}$) from $\mathcal{D}$ and $\mathcal{I}$.
   * The Passage Retriever $\mathcal{P}$ uses $\mathcal{K}$ to fetch relevant passages $\mathcal{R}$ from the library $\mathcal{L}$.

2. **Sample Pattern Summarization and Initial Data Generation**:
   * The teacher model $\mathcal{T}$ summarizes the underlying pattern ($P$) from the demonstration examples $\mathcal{D}$.
   * The $\text{LLM}\_{generator}$ creates an initial set of $N$ synthetic samples ($\mathcal{S}_{initial}$) using $\mathcal{R}$, $P \cup \mathcal{D}$, and $\mathcal{I}$.

3. **Difficulty-Adaptive Sample Generation**:
   * The base model $\mathcal{M}\_{base}$ attempts to solve the samples in $\mathcal{S}_{initial}$.
   * Samples are split into solved ($\mathcal{S}\_{solved}$) and unsolved ($\mathcal{S}_{unsolved}$) sets.
   * The $\text{LLM}\_{writer}$ generates harder samples ($\mathcal{S}\_{harder}$) from $\mathcal{S}\_{solved}$ and easier samples ($\mathcal{S}\_{easier}$) from $\mathcal{S}_{unsolved}$.
   * All generated samples are combined: $\mathcal{S}\_{synth} = \mathcal{S}\_{initial} \cup \mathcal{S}\_{harder} \cup \mathcal{S}_{easier}$.

4. **Training with High-Potential Samples**:
   * Each sample $s \in \mathcal{S}_{synth}$ is scored based on the base model's consistency in solving it. Lower scores indicate inconsistency (higher potential).
   * The top $M$ samples with the lowest scores (those the model can solve occasionally but not always, or never solves) are selected.
   * The Trainer $T$ fine-tunes $\mathcal{M}\_{base}$ on this selected subset using reinforcement learning, resulting in the final trained model $\mathcal{M}_{trained}$.

## Get started

### 1. Create and activate a virtual environment

```bash
conda create -n data_rl python=3.10
conda activate data_rl
```

### 2. Install related libraries.

```bash
sh activate.sh
```

### 3. Put your OpenAI Key

Place your openai key in ```model_inference/openai_call.py```

### 4. Create Your Task-Specific Evaluation Folder

Create a new directory for your task within `src/eval/tasks/`. This folder handles task-specific logic for the data generation and evaluation process and must include the following five Python scripts:

* `process_label.py`: Extracts the ground truth label from the human-labeled output of a test sample.
    * *Example (GSM8K):* Extracts `72` from `<human COT> #### 72`.
    * *Example (LogiQA):* Transforms `'2'` into `'C'`.
* `process_prediction.py`: Extracts the model's prediction from its full response to a test sample.
    * *Example (LogiQA):* Extracts `'A'` from `<model COT> <result>A</result>`.
* `eval_function.py`: Compares the extracted prediction with the ground truth label. Returns `True` if they match according to task criteria, `False` otherwise.
* `get_output_instruction.py`: Provides the specific output format instruction for the model.
    * *Example (GSM8K):* `"Let's think step by step and output the final result after '####'."`
* `process_and_save_dataset.py`: Transforms a list of raw training data examples (e.g., `[{...},{...},...]`) into the format required for the Reinforcement Learning (RL) training  dataset and download and transform your test dataset for evaluation.

### 5. Define the Reward Function for RL Training

To guide the Reinforcement Learning (RL) process for your task, you need to create a custom reward scoring function:

1. **Create the Reward Script:** Add a new Python file (e.g., `your_task_reward.py`) inside the `verl/utils/reward_score/` directory.
2. **Implement Scoring Logic:** Within this file, define a function that calculates a reward score based on the model's output. This function typically considers:
    * **Format Score:** How well the output matches the required format.
    * **Result Score:** Whether the final answer is correct.
3. **Register the Function:** Modify the `verl/trainer/main_ppo.py` script. Specifically, import it at line 20 and update the `_select_rm_score_fn` function  to map your task's identifier (passed via the `data_source` argument) to your newly created reward function. This ensures the PPO trainer uses the correct scoring logic for your task.

### 6. Prepare Passage Libraries for Retrieval

The retriever component requires access to text corpora (passage libraries). You need to place these within the `src/retriever/passages/` directory.

**Using Standard Corpora (Example: CRAFT):**

1. **Download:** Obtain corpus files, for example, the Wikipedia, Wikihow, and StackExchange archives (`.tar.gz`) recommended in Step 0 of the CRAFT repository: [https://github.com/ziegler-ingo/CRAFT](https://github.com/ziegler-ingo/CRAFT).
2. **Extract:** Unzip the downloaded archives directly into the `src/retriever/passages/` directory.
3. **Rename (if necessary):** Ensure the resulting directories containing the corpus data are named appropriately. For the CRAFT examples, the expected structure would be:
    * `src/retriever/passages/wiki/`
    * `src/retriever/passages/wikihow/`
    * `src/retriever/passages/stackexchange/`

**Using Custom Corpora:**

* You can add your own text libraries to the `src/retriever/passages/` directory.
* Custom libraries must be in the **`.jsonl`** format (JSON Lines).
* Each line in the `.jsonl` file must be a valid JSON object (dictionary) containing at least a `'text'` key, where the value is the passage content string.
    * *Example line:* `{"text": "This is the content of a single passage."}`

### 7. Configure the Demonstration Example

You can set a specific input/output example for demonstration or quick testing purposes directly within the main script.

1. **Locate:** Open the `src/main.py` file. Find the section where the demonstration example is defined (the original documentation points near **line 575**, but note that this line number may change as the code evolves).
2. **Set Example:** Modify the variable assignment to include your desired example(s). The required format is a Python list containing one or more dictionaries. Each dictionary must have an `'input'` key and an `'output'` key.

    * **Format:**
        ```python
        [
          {'input': 'Your example input prompt or question here', 'output': 'The corresponding desired or example output here'},
          # You can add more examples if needed
          # {'input': 'Another input', 'output': 'Another output'}
        ]
        ```
### 8. Running an Experiment

To execute an experiment, use the `src/main.py` script. You can specify the target GPUs using the `CUDA_VISIBLE_DEVICES` environment variable and configure the experiment using the following command-line arguments:

* `--base_model_path`: Specifies the file path to the foundational language model directory (e.g., `./src/model/Qwen2.5-7B`).
* `--task_name`: Identifies the specific task being run (e.g., `logiqa`).
* `--task_instruction`: Provides the textual prompt defining the task objective for the model (e.g., `'You should answer logical reasoning questions accurately based on the provided context.'`).
* `--dataset_path`: Points to the directory containing the dataset for this task (e.g., `'./src/data_logqa'`).
* `--work_model_paths`: Specifies the file path to the specific model checkpoint(s) being used or evaluated in this run (e.g., `'./TinyZero/checkpoints/TinyZero'`).

**Example Command:**

This example runs the `gsm8k` task on GPUs 0 through 4.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Execute the script
python src/main.py \
  --base_model_path ./src/model/Qwen2.5-7B \
  --task_name gsm8k \
  --task_instruction 'You are given a word problem involving basic arithmetic, algebra, or geometry. Your task is to carefully read the problem and provide a step-by-step solution for it' \
  --dataset_path './src/data_gsm8k' \
  --work_model_paths './TinyZero/checkpoints/TinyZero'
```

This example runs the `logiqa` task on GPUs 0 through 4.

```bash
# Set target GPUs (e.g., GPUs 0, 1, 2, 3, 4)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# Run the experiment script
python src/main.py \
    --base_model_path ./src/model/Qwen2.5-7B \
    --task_name logiqa \
    --task_instruction 'You should answer logical reasoning questions accurately based on the provided context.' \
    --dataset_path './src/data_logqa' \
    --work_model_paths './TinyZero/checkpoints/TinyZero'
```

### Citation

Please consider citing our paper if you find this approach useful.
