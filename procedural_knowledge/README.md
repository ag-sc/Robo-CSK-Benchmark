# Procedural Knowledge Task

The idea behind this task is that a cooking recipe title and two instructions from a recipe are provided.
The commonsense problem is to decide, which instructions happens before or after the other. 
This problem is encapsulated by the following two temporal commonsense questions the robot should be able to answer:

- In the recipe {title}, did {step_1} occur before {step_2}? 
- In the recipe {title}, did {step_1} occur after {step_2}?

To ensure a balanced dataset, we alternate the order of step_1 and step_2 in the queries.


## Data Generation

The data generation process is automated using gpt-4o-2024-08-06, sourced from the Recipe 1M+ Dataset (https://paperswithcode.com/dataset/recipe1m-1) [1]. 
For each recipe, GPT selects the two steps with the highest temporal relationship. For the Multiple Choice Task GPT select the three steps with the highest temporal relationsship. 
These selected steps are then saved, along with the recipe title and all other steps, in the *data_generation/questions_components_binary* or *data_generation/questions_components_multi* folder. 
Afterward, we manually review each file from the binary data_generation to remove any steps that are not temporally related.

Example:
```json
{
  "goal": "World's Best Mac and Cheese",
  "step_1": "Cook pasta for less time than directed.",
  "step_2": "Bake pasta with a cheese topping."
}
```
To generate questions based on your own recipes or recreate these questions, 
simply download the Recipe 1M+ dataset or your recipes into the *data_generation/* folder and execute the *askgpt_temporalorder.py* in this folder.

## Experiments

To evaluate this tasks, we provide the LLMs with the following prompt:

### Prompt for Binary Questions:

---
**System**: Imagine you are a robot tasked with determining the temporal order of two steps from one recipe. Based on the recipe title and the two steps provided, identify whether one action occurred before another.

**User**: Answer only with 'Yes' or 'No'.

Question: [Question]

---
If the question contains the temporal relation "after", we modify the temporal relation in the system prompt accordingly. 
For our experiment, we selected 776 recipes during the data generation process. 
For each recipe, we ask two questions: one regarding the "before" temporal relation and one regarding the "after" temporal relation. 
Additionally, we also reverse the order of the steps in each question to have a positive and negative correct answer. This results in a total of 776 * 4 = 3104 questions.

### Prompt for Multiple Choice Questions:

---
**System**: Imagine you are a robot tasked with determining the temporal order of steps in a recipe. Based on the recipe title and the provided steps, identify which step occurred before another.

**User**: Answer only with your chosen step.

Question: [List of 5 Answer Options]

---
If the question contains the temporal relation "after", we modify the temporal relation in the system prompt accordingly. 
For our experiment, we selected 800 recipes during the data generation process. 
For each recipe, we ask two questions: one regarding the "before" temporal relation and one regarding the "after" temporal relation. This results in a total of 800 * 2 = 1600 questions.

## Results

### Binary
We analyse the results for each model using the following metrics:
- Amount of true positives (TPs), true negatives (TNs), false positives (FPs) & false negatives (FNs)
- Ratio of positive to negative answers by the model (Formula: (TPs + FPs) / (TNs + FNs))
- Accuracy, Precision, Recall, Specificity
- F1-Score

| LLM                    | TNs      | TPs      | FNs   | FPs    | Ratio | Acc       | Prec      | Rec       | Spec      | F1        |
|------------------------|----------|----------|-------|--------|-------|-----------|-----------|-----------|-----------|-----------|
| gpt-4o-2024-08-06      | **1362** | 1511     | 41    | **190**| 1.212 | **0.926** | **0.888** | 0.974     | **0.878** | **0.929** |
| Llama-3.3-70B-Instruct | 1339     | 1501     | 51    | 213    | 1.233 | 0.915     | 0.876     | 0.967     | 0.863     | 0.919     |
| gemma-2-27b-it         | 754      | **1543** | **9** | 798    | 3.068 | 0.740     | 0.659     | **0.994** | 0.486     | 0.793     |


### Multiple Choice
We analyse the results for each model using accuracy.

| LLM                    | Total | Correct  | Accuracy  |
|------------------------|-------|----------|-----------|
| gpt-4o-2024-08-06      | 1600  | **1462** | **0.914** |
| Llama-3.3-70B-Instruct | 1600  | 1357     | 0.848     |
| gemma-2-27b-it         | 1600  | 1294     | 0.809     | 


## References

[1] J. Marin, et al., ‘Recipe1M+: a dataset for learning cross-modal embeddings for cooking recipes and food images’, in 2018 arXiv preprint arXiv:1810.06553
