# Tidy Up Task

The idea behind this task is that the robot is encountering a cluttered household environment where different objects are misplaced.
This problem is encapsulated by the following two commonsense questions the robot should be able to answer:

- What objects in my environment are out of place?
- What are the prototypical locations for these objects?

To answer these questions, we provide a list of 868 household objects and their associated locations.

## Data Structure

The data in the *tidy_up_data_csv* is collected in two columns:
The first column (*Object*) contains all object names.
The second column (*Locations*) contains a Python dictionary with a ranked list of possible locations.
The key in the dictionary describes the rank of the location.
The value in the dictionary is the name of the location as well as a trust value on which the ranking is based.
The trust value depends on the data source from which this location is gathered.

Example:
```ladle,"{1: ('kitchen', 1.5), 2: ('drawer', 0.5)}"```

The object *ladle* is probably found in the *kitchen* otherwise it is found in a *drawer*.

## Data Sources

We combine five different data sources to gather the information on prototypical object locations.
For each source, we provide a single file focused on the data extraction.
We use WordNet [1] to ensure that the found candidates are actually household objects and household locations.
Additionally, to create the ranking of the locations, we manually assign each source a specific trust value based on the reliability.

| Source         | Trust Value | # Triples | # Objects | Ref |
|----------------|-------------|-----------|-----------|-----|
| AI2Thor        | 1.00        | -         | 118       | [2] |
| Ascent++       | 0.50        | 4309      | 131       | [3] |
| CSKG           | 0.50        | 9264      | 462       | [4] |
| Housekeep      | 1.00        | -         | 268       | [5] |
| Microsoft COCO | 0.75        | -         | 7         | [6] |
| **SUM**        | -           | **13966** | **986**   |     |
| **Filtered**   |             |           | **383**   |     |

After removing duplicates from the list and filtering the 986 extracted objects, *383* distinct objects and their locations remain.

## Experiments

For this task, we evaluate the LLM capabilities in two varying ways:
In the first experiment, we prompt the model to provide a prototypical location *without* giving it any options to choose from (= Open Question). 
In a second experiment, we provide the model with a list of 5 locations to choose from (= Multi Choice Question). 
Here are the two prompts we have employed:

### Prompt for Open Questions

---
**System**: Imagine you are a robot tidying up a household.

**User**: What are the prototypical locations in a household where the following object can be found? Please only answer with a comma separated & ranked list of locations.

Object: [Object] 

Locations:

---

### Prompt for Multi Choice Questions

---
**System**: Imagine you are a robot tidying up a household environment, being confronted with an object and a possible list of locations to put it.

**User**: What is the single location from the given list that you think is the most suitable place to put the object? Please only answer with the location you chose.

Object: [Object] 

Locations: [List of 5 Locations]

Your Choice:

---

## Results

We analyse the results for each model using different metrics depending on the type of question in the experiment.
For the open ended questions, we rely on the following ranking-based metrics:
- Mean Reciprocal Rank (MRR) of the generated answers when compared to the gold standard ranked list
- Mean Average Precision @k (MAP@k) over all objects (evaluated for k=1, k=3 & k=5)
- Mean Average Recall @k (MAR@k) over all objects (evaluated for k=1, k=3 & k=5)

For the multiple choice questions, we evaluate the Accuracy (Acc) of the model answers.

| LLM                    | MRR       | MAP@1     | MAP@3     | MAP@5     | MAR@1     | MAR@3     | MAR@5     | Acc       |
|------------------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| gpt-4o-2024-08-06      | 0.661     | 0.580     | 0.628     | 0.580     | 0.147     | 0.221     | 0.249     | 0.509     |
| Llama-3.3-70B-Instruct | **0.680** | **0.608** | **0.645** | **0.584** | **0.158** | **0.225** | **0.256** | 0.496     |
| gemma-2-27b-it         | 0.348     | 0.274     | 0.321     | 0.307     | 0.096     | 0.144     | 0.157     | **0.522** |

## References

[1] G. A. Miller, ‘WordNet: A Lexical Database for English’, Communications of the ACM, vol. 38, no. 11, pp. 39–41, 1995, doi: 10.1145/219717.219748.

[2] E. Kolve et al., ‘AI2-THOR: An Interactive 3D Environment for Visual AI’, 2017, arXiv. doi: 10.48550/ARXIV.1712.05474.

[3] T.-P. Nguyen, S. Razniewski, J. Romero, and G. Weikum, ‘Refined Commonsense Knowledge from Large-Scale Web Contents’, IEEE Trans. Knowl. Data Eng., pp. 1–16, 2022, doi: 10.1109/TKDE.2022.3206505.

[4] F. Ilievski, P. Szekely, and B. Zhang, ‘CSKG: The CommonSense Knowledge Graph’, in The Semantic Web, vol. 12731, R. Verborgh, K. Hose, H. Paulheim, P.-A. Champin, M. Maleshkova, O. Corcho, P. Ristoski, and M. Alam, Eds., in Lecture Notes in Computer Science, vol. 12731. , Cham: Springer International Publishing, 2021, pp. 680–696. doi: 10.1007/978-3-030-77385-4_41.

[5] Y. Kant et al., ‘Housekeep: Tidying Virtual Households Using Commonsense Reasoning’, in Computer Vision – ECCV 2022, S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, and T. Hassner, Eds., in Lecture Notes in Computer Science, vol. 13699. Cham: Springer Nature Switzerland, 2022, pp. 355–373. doi: 10.1007/978-3-031-19842-7_21.

[6] T.-Y. Lin et al., ‘Microsoft COCO: Common Objects in Context’, in Proceedings of the 13th European Conference on Computer Vision (ECCV 2014), D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars, Eds., in Lecture Notes in Computer Science, vol. 8693. Zurich, Switzerland: Springer International Publishing, 2014, pp. 740–755. doi: 10.1007/978-3-319-10602-1_48.