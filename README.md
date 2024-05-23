# Biomedical Knowledge Encoding and Learning Dynamics in LLMs

<table style="width:100%;">
  <tr>
    <td valign="top" style="width:30%; padding-right: 20px;">
      <img src="https://github.com/NikolajMA/biomedical-llm-scaling/blob/main/image.jpeg" style="width:100%; height:auto;">
    </td>
    <td valign="top" style="width:70%;">
      <p>This repository contains the evaluation results and notebooks to replicate all experiments, analyses, tables, and figures from the paper: 
      <em>Evaluating Biomedical Knowledge Encoding in Large Language Models: The Effects of Scaling and Pre-Training Data</em>. The main contributions are</p>
      <ul>
        <li> :pencil: MultiMedQA evals for OLMo, Pythia, Mamba, Qwen, OpenLlama</li>
        <li>:chart_with_upwards_trend: Analysis of how task accuracy increase as a function of total model parameters within each model suite </li>
        <li>:chart_with_upwards_trend: Analysis of biomedical learning trajectories and performance scaling as a function of training data volume using intermediate checkpoint evals (OLMo + Pythia)</li>
        <li>:books: Analysis of how biomedical term frequencies in pre-training corpora (Dolma, RedPajama, The Pile) affect biomedical performance</li>
        <li>⚖️ Paloma1B evals: model performance deviations based solely on pre-training corpora</li>
      </ul>
    </td>
  </tr>
</table>

All LLM evaluations were done using the [LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) CLI from EleutherAI, using the following flags and parameters:
```bash
lm_eval --model hf \ #Use "--model mamba-ssm" for Mamba models
    --model_args pretrained=Link/tomodel \ #HuggingFace Model Link
    --tasks medmcqa,medqa_4options,pubmedqa,mmlu_college_medicine,mmlu_college_biology,mmlu_clinical_knowledge,mmlu_anatomy,mmlu_medical_genetics,mmlu_professional_medicine \ #All tasks from MultiMedQA
    --device cuda:0 \
    --batch_size auto \
    --log_samples \
    --wandb_args project=wandb-project-name \ #Requires wandb library and api key
    --trust_remote_code \
    --output_path output-path #Specify path where eval results should be saved
    #--num_fewshot 5 for few-shot evaluations with 5 examples
```
All evaluation results were logged to a Weights & Biases project, and exported to (insert repo path). Raw logged data is found in (insert repo path).


When running OLMo and Mamba it is necesarry to 
```bash
pip install ai2-olmo lm_eval[mamba]
```
and 
```python
from hf_olmo import OLMoForCausalLM
```
After installing the harness
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

For the evaluation on intermediate checkpoints for OLMo, model revisions were assessed using
```python

from huggingface_hub import list_repo_refs

out = list_repo_refs("allenai/OLMo-7B")
branches = [b.name for b in out.branches]

# Extract the step number from the branch name
def get_step_number(branch_name):
    return int(branch_name.split('-')[0].replace('step', ''))

sorted_branches = sorted(branches, key=lambda x: get_step_number(x) if x != "main" else float('inf'))
