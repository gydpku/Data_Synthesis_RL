# 🚀 Data-Synthesis-RL: Efficient Few-Shot RL Fine-Tuning with Synthetic Data Generation

## Introduction

This project implements an efficient approach for fine-tuning a base language model ($\mathcal{M}_{base}$) when only a few demonstration examples ($\mathcal{D}$) are available. It leverages a powerful teacher large language model ($\mathcal{T}$), external knowledge retrieval, and strategic synthetic data generation to create an effective training dataset. The base model is then trained on the most informative subset of this synthetic data using reinforcement learning.

## 🛠️ Components

The system utilizes four key components:

1.  **🔍 Passage Retriever ($\mathcal{P}$)**:
    * Takes keywords (extracted from the task instruction $\mathcal{I}$ and demos $\mathcal{D}$ using the teacher model $\mathcal{T}$) as input.
    * Searches a large text library ($\mathcal{L}$, e.g., Wikipedia) to find relevant passages ($\mathcal{R}$) that provide external context.

2.  **📊 LLM Data Generator ($\text{LLM}_{generator}$)**:
    * Uses the teacher model ($\mathcal{T}$) to synthesize new training data ($\mathcal{S}_{initial}$).
    * Takes task instructions ($\mathcal{I}$), summarized sample patterns ($P$), demonstration examples ($\mathcal{D}$), and retrieved passages ($\mathcal{R}$) as input.
    * Incorporates a verification step (e.g., majority voting) to ensure data quality.

3.  **✍️ LLM Data Re-writer ($\text{LLM}_{writer}$)**:
    * Also uses the teacher model ($\mathcal{T}$).
    * Takes existing synthetic samples and modifies them to be either harder ($\mathcal{S}\_{harder}$) or easier ($\mathcal{S}_{easier}$) based on the base model's performance.
    * Includes a verification step to filter low-quality outputs.

4.  **📈 Trainer ($T$)**:
    * Implements a reinforcement learning algorithm.
    * Trains the base model ($\mathcal{M}_{base}$) specifically on a selected subset of high-potential synthetic samples.

## ⚙️ Workflow

Given a base model ($\mathcal{M}_{base}$), task instruction ($\mathcal{I}$), and a few demonstration examples ($\mathcal{D}$), the training process involves four main steps:

1.  **Keyword Extraction and Passage Retrieval**:
    * The teacher model $\mathcal{T}$ extracts domain-specific keywords ($\mathcal{K}$) from $\mathcal{D}$ and $\mathcal{I}$.
    * The Passage Retriever $\mathcal{P}$ uses $\mathcal{K}$ to fetch relevant passages $\mathcal{R}$ from the library $\mathcal{L}$.

2.  **Sample Pattern Summarization and Initial Data Generation**:
    * The teacher model $\mathcal{T}$ summarizes the underlying pattern ($P$) from the demonstration examples $\mathcal{D}$.
    * The $\text{LLM}\_{generator}$ creates an initial set of $N$ synthetic samples ($\mathcal{S}_{initial}$) using $\mathcal{R}$, $P \cup \mathcal{D}$, and $\mathcal{I}$.

3.  **Difficulty-Adaptive Sample Generation**:
    * The base model $\mathcal{M}\_{base}$ attempts to solve the samples in $\mathcal{S}_{initial}$.
    * Samples are split into solved ($\mathcal{S}\_{solved}$) and unsolved ($\mathcal{S}_{unsolved}$) sets.
    * The $\text{LLM}\_{writer}$ generates harder samples ($\mathcal{S}\_{harder}$) from $\mathcal{S}\_{solved}$ and easier samples ($\mathcal{S}\_{easier}$) from $\mathcal{S}_{unsolved}$.
    * All generated samples are combined: $\mathcal{S}\_{synth} = \mathcal{S}\_{initial} \cup \mathcal{S}\_{harder} \cup \mathcal{S}_{easier}$.

4.  **Training with High-Potential Samples**:
    * Each sample $s \in \mathcal{S}_{synth}$ is scored based on the base model's consistency in solving it. Lower scores indicate inconsistency (higher potential).
    * The top $M$ samples with the lowest scores (those the model can solve occasionally but not always, or never solves) are selected.
    * The Trainer $T$ fine-tunes $\mathcal{M}\_{base}$ on this selected subset using reinforcement learning, resulting in the final trained model $\mathcal{M}_{trained}$.


## Get started

### 1. Create a virtual environment

```bash
conda create -n data_rl python=3.10
conda activate data_rl
```

### 2. Install related libraries.

```bash
pip install -r requirements.txt
```
### 3. Put your OpenAI Key

Place your openai key in ```model_inference/openai_call.py```

### Running the examples

To deploy our agent, you can run this:

```bash
cd src
python run.py --experiment_name <experiment_name> \
              --target_model_path <target_model_path> \
              --model <model_name> \
              --base_model_path <base_model_path> \
              --task_name <task_name> \
              --task_instruction <task_instruction>
```
Arguments
```--experiment_name``` (str): Name of the experiment.

```--target_model_path``` (str): Directory where the trained model will be saved.

```--model (str)```: Name of the model (e.g., "Llama-3.1-8B-Instruct").

```--base_model_path``` (str): Path to the initial pre-trained model.

```--task_name``` (str): Name of the task being performed.

```--task_instruction``` (str): Instruction for the task.

For a example:
```
python src/run.py --experiment_name "exp1" \
              --target_model_path "./checkpoints/exp1" \
              --model "LLaMa-3.1-8B-Instruct" \
              --base_model_path "./models/llama-3.1-8B-instruct" \
              --task_name "MedNLI" \
              --task_instruction "The domain is Medical. The TASK: Please classify the relationship between the given premise and hypothesis into one of the following labels: entailment, contradiction, or neutral."
```
To train and test on your task, follow these two steps:

1. Store your training and test data in the following format using the Hugging Face dataset:

```
from datasets import Dataset, DatasetDict

# Step 1: Define your training and test data
train_data = [
    {"Input": "Example input 1", "Output": "Example output 1"},
    {"Input": "Example input 2", "Output": "Example output 2"},
]

test_data = [
    {"Input": "Test input 1", "Output": "Test output 1"},
    {"Input": "Test input 2", "Output": "Test output 2"},
]

# Step 2: Convert to Hugging Face datasets
train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

# Step 3: Store datasets in a DatasetDict
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# Step 4: Save the dataset (e.g., to disk)
dataset.save_to_disk("task_dataset_path")
```
## Citation

Please consider citing our paper if you find this approach useful.

Specify the dataset path using ```--dataset_path``` and define the task's domain or topic with ```--domain```. 
