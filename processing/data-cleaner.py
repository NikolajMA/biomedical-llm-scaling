import pandas as pd 
import numpy as np
import re
import json
import os

# Functions to make plotting easier later

def convert_model_name_to_param_count(model_name):
    match = re.search(r'(\d+(\.\d+)?)[mMbB]', model_name)
    if match:
        number = float(match.group(1))
        magnitude = match.group(0)[-1].lower()
        if magnitude == 'm':
            return int(number * 1_000_000)
        elif magnitude == 'b':
            return int(number * 1_000_000_000)
    return 0

def get_model_family(model_name):
    if 'pythia' in model_name.lower():
        return 'Pythia'
    elif 'qwen' in model_name.lower():
        return 'Qwen'
    elif 'mamba' in model_name.lower():
        return 'Mamba'
    elif 'olmo' in model_name.lower():
        return 'OLMo'
    elif 'paloma' in model_name.lower():
        return 'Paloma'
    elif 'openllama' in model_name.lower():
        return 'OpenLlama'
    elif 'llm360' in model_name.lower():
        return 'LLM360'
    elif 'meditron' in model_name.lower():
        return 'Meditron'
    elif 'biomistral' in model_name.lower():
        return 'Biomistral'
    else:
        return 'Unknown'
    

#First clean the zero-shot results
results = pd.read_csv('results/wandb-logs/wandb_export_results.csv')

accuracy_columns = [col for col in results.columns if 'acc' in col and 'acc_norm' not in col]
acc_results = results[['Name'] + accuracy_columns]
acc_results.insert(1, 'param_count', acc_results['Name'].apply(convert_model_name_to_param_count))
acc_results.insert(1, 'model_family', acc_results['Name'].apply(get_model_family))


acc_scale_results = acc_results[acc_results['Name'].str.contains('pythia|qwen|mamba|olmo|openllama', case=False)] #all models included to study scaling
acc_data_results = acc_results[acc_results['Name'].str.contains('paloma', case=False)] #all models included to study data variation
acc_misc_results = acc_results[acc_results['Name'].str.contains('meditron|biomistral', case=False)] #misc models to study systematic error in qa performance

#fetching pythia mmlu results from jsons, as they weren't logged with wandb. this causes some NAs in the acc_scale_results

data_folder = 'results/raw_outputs/eval-results/pythia-mmlu'
json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]

data_list = []

for json_file in json_files:
    with open(os.path.join(data_folder, json_file), 'r') as file:
        data = json.load(file)

    model_name = json_file.replace('.json', '')
    model_data = {'model_name': model_name}

    for task, results in data['results'].items():
        acc_key = f'{task}/acc'
        acc_stderr_key = f'{task}/acc_stderr'
        if 'acc,none' in results:
            model_data[acc_key] = results['acc,none']
        if 'acc_stderr,none' in results:
            model_data[acc_stderr_key] = results['acc_stderr,none']

    data_list.append(model_data)

pythia_mmlu = pd.DataFrame(data_list)

# Not using these benchmarks in the analysis so dropping them, they were made with another branch of the eval harness, and are not part of the standardized mmlu biomedical benchmarks
columns_to_remove = ['mmlu_clinical_workflow/acc', 'mmlu_clinical_workflow/acc_stderr',
 'mmlu_human_aging/acc', 'mmlu_human_aging/acc_stderr', 'mmlu_human_sexuality/acc', 'mmlu_human_sexuality/acc_stderr'
 , 'mmlu_nutrition/acc', 'mmlu_nutrition/acc_stderr', 'mmlu_professional_psychology/acc', 'mmlu_professional_psychology/acc_stderr'
 , 'mmlu_virology/acc', 'mmlu_virology/acc_stderr', 'prob_list_summ/acc', 'prob_list_summ/acc_stderr']

pythia_mmlu = pythia_mmlu.drop(columns=columns_to_remove)

#updating to fill na values with pythia mmlu results

acc_scale_results.rename(columns={'Name': 'model_name'}, inplace=True)

pythia_mmlu.set_index('model_name', inplace=True)
acc_scale_results.set_index('model_name', inplace=True)

# Fill NaN values in df2 with values from df1
acc_scale_results = acc_scale_results.combine_first(pythia_mmlu)

# Reset the index if you want 'model_name' back as a column
acc_scale_results.reset_index(inplace=True)

#save cleaned zero-shot results to csv
acc_data_results.to_csv('results/wandb-logs/cleaned/acc_data_results.csv')
acc_misc_results.to_csv('results/wandb-logs/cleaned/acc_misc_results.csv')
acc_scale_results.to_csv('results/wandb-logs/cleaned/acc_scale_results.csv')

#cleaning the few-shot results
fewshot_results = pd.read_csv('results/wandb-logs/wandb_export_results_fewshot.csv')

fs_accuracy_columns = [col for col in fewshot_results.columns if 'acc' in col and 'acc_norm' not in col]
fs_acc_results = fewshot_results[['Name'] + fs_accuracy_columns]
fs_acc_results.insert(1, 'param_count', fs_acc_results['Name'].apply(convert_model_name_to_param_count))
fs_acc_results.insert(1, 'model_family', fs_acc_results['Name'].apply(get_model_family))


fs_acc_scale_results = fs_acc_results[fs_acc_results['Name'].str.contains('pythia|qwen|mamba|olmo', case=False)]
fs_acc_scale_results.to_csv('results/wandb-logs/cleaned/fs_acc_scale_results.csv')