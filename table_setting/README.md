# Table Setting Task

In this task, the embodied agent should set the dinner table for a specific meal. 
This includes choosing the correct type of cutlery and the correct type of plate, so we provide a list of meals with their respective cutlery and plate.
This tasks is an example for task-specific commonsense knowledge since the knowledge here is not generally applicable as the knowledge covered by the other tasks.
We thus pose the following two task-specific commonsense questions that the agent should be able to answer:

- What cutlery is needed to eat a given meal?
- On what kind of plate is the given meal served?

To answer these questions, we used crowdsourcing to gather data about 100 recipes from 51 different participants, asking them to choose their preferred cutlery as well as their preferred plate.
For the cutlery, the participants could choose as many as they preferred from a given list of available tools (*Hands, Tongs, Knife, Fork, Skewer, Chopsticks & Spoon*).
For the plate, they had to choose exactly one from the following list: *Dinner Plate, Dessert Plate, Bowl & Coupe Plate*.
In addition to these choices, the participants were provided with a one sentence explanation for each meal and for each type of plate.
The results are summarized in [results.md](results.md).

## Data Structure

We provide two files with the data from our user study in Prolific:
In *prolific_user_data.csv*, we save the decisions of each participant for each meal.
In *combined_prolific_data.csv*, we combine the results for each meal.
The first column states the ID of the recipe since all 100 were randomly chosen from the Recipe1M+ dataset [1], followed by the meal name in the second column.
The remaining columns count how many participants chose the respective cutlery or type of plate during their participation.

For the evaluation, we use the type of plate with the most votes as the correct answer.
For the cutlery, we include every tool that received at least 10 votes (20%) in the list of potentially correct choices.

## Experiments

To evaluate this tasks, we provide the LLMs with the following prompt for asking about the cutlery...

---
**System**: Imagine you are a robot setting a table for a meal.

**User**: What are the types of cutlery you would use to eat that meal? Please choose from the following and only answer with your choices: [List of utensils]

Meal: [Meal]

Cutlery:

---
... and for asking about the plate:

---
**System**: Imagine you are a robot setting a table for a meal.

**User**: What is the type of plate you would use to eat that meal? Please choose one from the following and only answer with your choice: [List of plates]

Meal: [Meal]

Plate:

---

We analyse the results for each model using different metrics for the plates and the cutlery:
- For the plates, we calculate the accuracy of the model prediction
- For the cutlery, we calculate the Jaccard Coefficient based on the predicted and the correct list

## References

[1] J. Marín et al., ‘Recipe1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images’, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 1, pp. 187–203, Jan. 2021, doi: 10.1109/TPAMI.2019.2927476.
