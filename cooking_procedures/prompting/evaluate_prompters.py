from cooking_procedures.json_utils import extract_results_json
import pandas as pd


def evaluate_classification(data):
    tp = fp = tn = fn = 0
    incorrect_entries = []

    for entry in data:
        response, correct_response = entry.get('response', '').strip().lower(), entry.get('correct_response', '').strip().lower()
        predicted_yes, predicted_no = "yes" in response, "no" in response
        actual_yes, actual_no = "yes" in correct_response, "no" in correct_response

        if predicted_yes:
            tp += actual_yes
            fp += not actual_yes
        elif predicted_no:
            tn += actual_no
            fn += not actual_no

        if (predicted_yes != actual_yes) or (predicted_no != actual_no):
            incorrect_entries.append({'question': entry.get('question', 'No question specified'),
                                      'response': response, 'correct_response': correct_response})

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    accuracy = (tp + tn) / total if total else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'metrics': pd.Series({'ratio': (tp + fp) / (tn + fn) if tn + fn else 0,
                                                                          'acc': accuracy, 'prec': precision, 'rec': recall,
                                                                          'spec': specificity, 'f1': f1}).to_frame().T,
            'incorrect_entries': incorrect_entries}


def calculate_overall_metrics(results_list):
    total_tp, total_fp, total_tn, total_fn = sum(res['tp'] for res in results_list), sum(res['fp'] for res in results_list), \
                                              sum(res['tn'] for res in results_list), sum(res['fn'] for res in results_list)

    total = total_tp + total_fp + total_tn + total_fn
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
    specificity = total_tn / (total_tn + total_fp) if total_tn + total_fp else 0
    accuracy = (total_tp + total_tn) / total if total else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return pd.Series({'tn': total_tn, 'tp': total_tp, 'fn': total_fn, 'fp': total_fp,
                      'ratio': (total_tp + total_fp) / (total_tn + total_fn) if total_tn + total_fn else 0,
                      'acc': accuracy, 'prec': precision, 'rec': recall, 'spec': specificity, 'f1': f1}).to_frame().T


def evaluate(prompters):
    comb_result = pd.DataFrame(columns=['model', 'tn', 'tp', 'fn', 'fp', 'ratio', 'acc', 'prec', 'rec', 'spec', 'f1'])

    for prompter in prompters:
        model_results = []

        for correct_answer in ["Yes", "No"]:
            for temporal_relation in ["after", "before"]:
                for recipe_number in range(1, 5):
                    evaluation_file = f'cooking_procedures/results/{correct_answer}/{prompter.model_name}/{temporal_relation}_{recipe_number}.json'

                    try:
                        results = extract_results_json(evaluation_file)

                        if isinstance(results, list):
                            metrics_dict = evaluate_classification(results)
                            model_results.append({'tp': metrics_dict['tp'], 'fp': metrics_dict['fp'],
                                                  'tn': metrics_dict['tn'], 'fn': metrics_dict['fn']})

                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error processing {evaluation_file}: {e}")

        if model_results:
            overall_metrics = calculate_overall_metrics(model_results)
            overall_metrics['model'] = prompter.model_name
            comb_result = pd.concat([comb_result, overall_metrics], ignore_index=True)

    comb_result.to_csv('cooking_procedures/results/model_overview.csv', index=False)
