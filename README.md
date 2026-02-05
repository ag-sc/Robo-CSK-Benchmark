# RoboCSKBench: Benchmarking Embodied Commonsense Capabilities of Large Language Models

Recent technological advancements have led to an increased use of autonomous agents in households for simpler, navigation-focused tasks since these functions are relatively straightforward.
However, deploying more advanced embodied agents for complex manipulation tasks remains challenging due to the open-world nature of home environments, where commonsense knowledge plays a crucial role in adapting to diverse situations.

In this paper, we are proposing the RoboCSKBench, a language-based multi-task benchmark to assess embodied commonsense knowledge capabilities of agents and systems interacting in dynamic household environments.
Our benchmark combines varying resources (e.g. knowledge graphs, manipulation benchmarks, public datasets) to provide data for five different, commonly encountered household tasks: Tidy Up, Tool Usage, Meta-Reasoning, Table Setting and Cooking Procedures.
For each task, we provide the extracted data and a set of metrics for the evaluation, but we envision our benchmark to be extendable through the addition of new commonsense-based tasks or different applications of the already existing data.

As an exemplary application, we assess the embodied commonsense capabilities of three state-of-the-art large language models with this benchmark.
Our results indicate a diverse foundation, with model performance varying across different tasks, suggesting that no single model serves as the definitive solution.
On the contrary, all models exhibit limitations, leaving room for further optimization and improvement.

## Publication

The accompanying paper has been presented at the **22nd International Conference on Ubiquitous Robots (UR 2025)**. 
You can find the paper on [IEEEXplore](https://ieeexplore.ieee.org/abstract/document/11078036).
Please use the following citation when working with the benchmark:
```
@inproceedings{Toberg2025RoboCSKBenchBenchmarking,
  title = {RoboCSKBench: Benchmarking Embodied Commonsense Capabilities of Large Language Models},
  booktitle = {2025 22nd International Conference on Ubiquitous Robots (UR)},
  author = {Töberg, Jan-Philipp and Kenneweg, Svenja and Cimiano, Philipp},
  year = {2025},
  pages = {199--206},
  address = {College Station, Texas, USA},
  doi = {10.1109/UR65550.2025.11078036}
}
```

## Embodied Commonsense

**Embodied commonsense knowledge** covers all information that an embodied agent needs to seamlessly interact with and understand its dynamic Everyday environment. 
 This includes but is not limited to:
 - Knowledge about participating objects, their (task-specific) properties and affordances[^1]
 - Knowledge about intuitive physics and causality[^2]
 - Knowledge about intuitive psychology and the mindfulness of other agents[^2]
 - Temporal knowledge about typical tasks (e.g. ordering, duration)[^3]

## Benchmark Overview

We propose **RoboCSKBench**, which is a language-driven benchmark for assessing the embodied commonsense capabilities of large language models as well as any approach working in the robotic household domain.
The benchmark currently focuses on the five most common commonsense aspects investigated by robotics researchers in the past ten years[^4].

We envision the proposed benchmark to be an extendable framework focusing not only on the five exemplary tasks we propose here, but also on future tasks to be incorporated under the *RoboCSKBench* umbrella, similar to the GLUE benchmark for natural language understanding[^5].
We encourage other researchers and practitioners alike to benchmark their approaches and models, either on a single tasks or on the whole benchmark.

The five tasks currently included in the benchmark are the following:
- **[Tidy Up](./tidy_up/README.md)**: This task is centred around the problem of identifying prototypical locations for objects inside a household environment.
The main focus is on providing the model with a specific household object and returning a ranked list of locations in a household where the object can be expected.
The data can also be used to reason about objects that are out of place at their current location and thus need to be moved to a different place.
- **[Tool Usage](./tool_usage/README.md)**: The second task focuses on the concept of affordances, which describe what an object or the environment offers an agent[^6].
We provide a list of objects with their respective affordances and a list of example tasks for each affordance. 
- **[Meta-Reasoning](./meta_reasoning/README.md)**: To evaluate the models capabilities in evaluating whether a specific robotic hardware setup is capable of executing a specific task, we propose this task.
In the data, we provide different tasks written as single sentence natural language instructions and a robotic setup that has was demonstrated to be capable of executing the task.
For each robotic setup, we provide the following information: Is the robot stationary or can it move? How many arms does it have? How many degrees of freedom does the arms have? What type of gripper is used? Are the grippers soft or rigid?
- **[Table Setting](./table_setting/README.md)**: In this task, the embodied agent should set the dinner table for a specific meal. 
This includes choosing the correct type of cutlery and the correct type of plate, so we provide a list of meals with their respective cutlery and plate.
- **[Procedural Knowledge](./procedural_knowledge/README.md)**: The last task evaluates a models ability to reason about the correct chronological sequence of two steps from a cooking recipe. 
Inspired by the *Event Ordering* task[^7], the model must decide in the case of a binary answer whether on step is done before or after another. In the case of a multiple choice answer the model must decide which step out of five is done before or after another.
Therefore, we provide the recipe title and specific cooking steps from this recipe.

Below you can find an overview over the five tasks with their variations, question type, size and link to the results:

| Abbr  | Task                    | Question Type   | # Questions | Link to Results                                                                                                  |
|-------|-------------------------|-----------------|-------------|------------------------------------------------------------------------------------------------------------------|
| TU-G  | Tidy Up                 | Generative      | 383         | [tidy_up/results_open/model_overview.csv](tidy_up/results_open/model_overview.csv)                               |
| TU-MC | Tidy Up                 | Multiple Choice | 383         | [tidy_up/results_multi/model_overview.csv](tidy_up/results_multi/model_overview.csv)                             |
| ToUs  | Tool Usage              | Multiple Choice | 291         | [tool_usage/results/model_overview.csv](tool_usage/results/model_overview.csv)                                   |
| MR-S  | Meta-Reasoning          | Binary          | 9,576       | [meta_reasoning/results_binary/model_overview.csv](meta_reasoning/results_binary/model_overview.csv)             |
| MR-MC | Meta-Reasoning          | Multiple Choice | 4,788       | [meta_reasoning/results_multi/model_overview.csv](meta_reasoning/results_multi/model_overview.csv)               |
| TS-U  | Table Setting (Utensils | Multiple Choice | 100         | [table_setting/results/model_overview.csv](table_setting/results/model_overview.csv)                             |
| TS-P  | Table Setting (Plates)  | Multiple Choice | 100         | [table_setting/results/model_overview.csv](table_setting/results/model_overview.csv)                             |
| PK-S  | Procedural Knowledge    | Binary          | 3,104       | [procedural_knowledge/results_binary/model_overview.csv](procedural_knowledge/results_binary/model_overview.csv) |
| PK-MC | Procedural Knowledge    | Multiple Choice | 1,600       | [procedural_knowledge/results_multi/model_overview.csv](procedural_knowledge/results_multi/model_overview.csv)   |

You have any new results to report? Please reach out to [Jan-Philipp Töberg](https://www.uni-bielefeld.de/fakultaeten/technische-fakultaet/arbeitsgruppen/semantic-computing/team/jan-philipp-toeberg/) (jtoeberg(at)techfak(dot)uni-bielefeld(dot)de).

## Setup

Apart from installing the [required](requirements.txt) Python packages, you need to create the "credentials.json" file in the root folder in which you save the OpenAI API key:
```json
{
    "api_key": "api-key-here"
}
```
You can then adapt the settings (Which LLMs to use? Which tasks to evaluate) in the main.py and start benchmarking!

[^1]: R. Gupta and M. J. Kochenderfer, ‘Common Sense Data Acquisition for Indoor Mobile Robot’, in Proceedings of the 19th national conference on Artifical intelligence, in AAAI’04. San Jose, California: AAAI Press, Jul. 2004, pp. 605–610.

[^2]: B. M. Lake, T. D. Ullman, J. B. Tenenbaum, and S. J. Gershman, ‘Building Machines That Learn and Think Like People’, The Behavioral and Brain Sciences, vol. 40, pp. 1–72, 2017, doi: 10.1017/S0140525X16001837.

[^3]: B. Zhou, D. Khashabi, Q. Ning, and D. Roth, ‘“Going on a vacation” takes longer than “Going for a walk”: A Study of Temporal Commonsense Understanding’, in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China: Association for Computational Linguistics, 2019, pp. 3361–3367. doi: 10.18653/v1/D19-1332.

[^4]: J.-P. Töberg, A.-C. N. Ngomo, M. Beetz, and P. Cimiano, ‘Commonsense knowledge in cognitive robotics: a systematic literature review’, Front. Robot. AI, vol. 11, 2024, doi: 10.3389/frobt.2024.1328934.

[^5]: A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman, ‘GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding’, in ICLR 2019 · The Seventh International Conference on Learning Representations, New Orleans, LA, USA, 2019. doi: 10.48550/ARXIV.1804.07461.

[^6]: M. H. Bornstein and J. J. Gibson, ‘The Ecological Approach to Visual Perception’, The Journal of Aesthetics and Art Criticism, vol. 39, no. 2, p. 203, 1980, doi: 10.2307/429816.

[^7]: L. Zhang, Q. Lyu, and C. Callison-Burch, ‘Reasoning about Goals, Steps, and Temporal Ordering with WikiHow’, in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Online: Association for Computational Linguistics, 2020, pp. 4630–4639. doi: 10.18653/v1/2020.emnlp-main.374.
