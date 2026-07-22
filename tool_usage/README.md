# Tool Substitution Task

The idea behind this task is that the robot is tasked with a specific instruction and should choose the most suitable tool from a list of available objects.
This problem is encapsulated by the following commonsense questions the robot should be able to answer:

- What kind of tool do I need for the task at hand?
- What objects are similar to each other? What can I use as a tool substitute?
- What are the affordances of an object?

To answer these questions, we provide a list of 494 household objects and their associated affordances (*affordance_data.csv*).
For each unique affordance, we provide between 1 and 11 household tasks where this affordance is needed (*affordance_task_map.json*).
The results are summarized in [results.md](results.md).

## Data Structure

The data in the *affordance_data.csv* is collected in two columns:
The first column (*Object*) contains all object names.
The second column (*Affordances*) contains a Python dictionary with a ranked list of associated affordances.
The key in the dictionary describes the rank of the affordance.
The value in the dictionary is the identifier of the affordance as well as a trust value on which the ranking is based.
The trust value depends on the data source from which this affordance is gathered.

Example:
```sink basin,"{1: ('wash', 1.0), 2: ('dispose', 1.0)}"```
The object *sink basin* affords *wash*(ing) or *dispose*(ing).


Additionally, we provide a list of tasks for each affordance in the *affordance_task_map.json*.
This file contains a Python dictionary where each key describes one (unique) affordance and the value is a list of strings where each string describes a single task.

Example:
```"illuminate":  ["increasing brightness of the room", "lighting up a room", "decreasing darkness"],```
The affordance *illuminate* is employed for the three tasks *increasing (the) brightness of the room*, *lighting up a room* and *decreasing darkness*.

## Data Sources

We combine six different data sources to gather the information on object affordances.
Additionally, we extend the affordance-task-mapping from [3] to include at least 3 exemplary tasks for each unique affordance.
For each source, we provide a single file focused on the data extraction.
We use WordNet [1] to ensure that the found object candidates in the CSKG [4] are actually household objects.
Additionally, to create the ranking of the locations, we manually assign each source a specific trust value based on the reliability.
If an object-affordance-tuple occurs in multiple sources, their trust value is accumulated.

| Source               | Trust Value | Filtered Objects | Affordances | Unique Affordances | Ref |
|----------------------|-------------|------------------|-------------|--------------------|-----|
| AI2Thor              | 1.00        | 115              | 201         | 11                 | [2] |
| COAT                 | 1.00        | 100              | 166         | 22                 | [3] |
| CSKG                 | 0.50        | 973              | 8215        | 1259               | [4] |
| Narrative Objects    | 0.75        | 376              | 4511        | 101                | [5] |
| RoCS                 | 0.75        | 11               | 108         | 10                 | [6] |
| Visual Affordance DS | 1.00        | 37               | 80          | 15                 | [7] |
| **SUM**              | -           | **557**          | **3222**    | **86**             |     |

## Experiments

To evaluate this tasks, we provide the LLMs with the following prompt:

---
**System**: Imagine you are a robot in a household environment being confronted with a task and a list of tools.

**User**: What is the single tool from the given list that you think is most suitable to help you execute your task? Please only answer with the tool you chose.

Task: [Task Name]

Tools: [List of Objects to choose from]

Your Choice:

---

In each prompt we provide a list of five objects to choose from.
One object has the correct affordance necessary for executing the given task.
The other four objects are randomly chosen from the list of objects without the necessary affordance. 

We analyse the results for each model using the following metric:
- Accuracy (of choosing the correct tool)

## References

[1] G. A. Miller, ‘WordNet: A Lexical Database for English’, Communications of the ACM, vol. 38, no. 11, pp. 39–41, 1995, doi: 10.1145/219717.219748.

[2] E. Kolve et al., ‘AI2-THOR: An Interactive 3D Environment for Visual AI’, 2017, arXiv. doi: 10.48550/ARXIV.1712.05474.

[3] A. Agrawal, R. Prabhakar, A. Goyal, and D. Liu, ‘Physical Reasoning and Object Planning for Household Embodied Agents’, Transactions on Machine Learning Research, May 2024, doi: 10.48550/ARXIV.2311.13577.

[4] F. Ilievski, P. Szekely, and B. Zhang, ‘CSKG: The CommonSense Knowledge Graph’, in The Semantic Web, vol. 12731, R. Verborgh, K. Hose, H. Paulheim, P.-A. Champin, M. Maleshkova, O. Corcho, P. Ristoski, and M. Alam, Eds., in Lecture Notes in Computer Science, vol. 12731. , Cham: Springer International Publishing, 2021, pp. 680–696. doi: 10.1007/978-3-030-77385-4_41.

[5] M. Pomarlan and R. Porzel, ‘Narrative Objects’, in Proceedings of the IJCAI/ECAI Workshop on Semantic Techniques for Narrative-based Understanding, Jul. 2022.

[6] M. Thosar et al., ‘From Multi-Modal Property Dataset to Robot-Centric Conceptual Knowledge About Household Objects’, Frontiers in Robotics and AI, vol. 8, 2021, doi: 10.3389/frobt.2021.476084.

[7] Z. Khalifa and S. A. A. Shah, ‘A Large Scale Multi-View RGBD Visual Affordance Learning Dataset’, in 2023 IEEE International Conference on Image Processing (ICIP), Kuala Lumpur, Malaysia: IEEE, Oct. 2023, pp. 1325–1329. doi: 10.1109/ICIP49359.2023.10222906.