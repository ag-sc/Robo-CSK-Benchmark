# Meta-Reasoning Task

The idea behind this task is that a specific task as well as a specific robotic hardware setup is provided.
The commonsense problem is to decide, whether the given robotic setup is equipped with the necessary hardware to execute this task.
This problem is encapsulated by the following two commonsense questions the robot should be able to answer:

- Can I execute the manipulation task that is asked from me (depending on my setup)?
- Can I evaluate which other agent could help me?

To answer these questions, we provide a list of 4788 manipulation tasks and information about the minimal robotic setup necessary for executing this task.

## Robotic Capabilities

The following questions are answered for each task in our dataset:
- Is the robot stationary or can it move through its environment?
- How many arms does the robot have?
- How many degrees of freedom (DoF) does each arm have?
- What is the gripper used by the arm(s)? *(classified using [1])*
- Are the grippers rigid or soft?

## Data Structure

The data in the *meta_reasoning_data.csv* is collected in six columns:
The first column (*Task*) describes the task to perform as a single natural language instruction sentence.
The second boolean column (*Mobile?*) states whether the robot has the capability to move (*True*) or is stationary (*False*).
The third column (*Arms*) counts the number of arms the robot needs to have.
In the fourth column (*DoFs*), the degrees of freedom for these arms is counted.
The fifth column (*Gripper Config*) displays the class of the gripper that is used, as is classified by [1].
The last boolean column (*Rigid Gripper?*) states whether the aforementioned gripper is rigid (*True*) or soft (*False*).

Example:
```chop the cucumber into shreds,False,1,6,Robot Grippers with 2 Fingers,True```

For executing the task *chop the cucumber into shreds*, the robot can be stationary, needs one arm with 6 DoFs and a rigid gripper with 2 fingers.

## Data Sources

We combine four different data sources to gather the information on tasks and their associated robotic hardware setups.
For each source, we provide a single file focused on the data extraction.
To combine the different tasks, we use SentenceBERT-based embeddings and clustering [2] in combination with hand-written rules to choose the minimal robotic capabilities for executing the specific task.

| Source         | Tasks    | Ref |
|----------------|----------|-----|
| ALFRED         | 21050    | [3] |
| DROID          | 2807     | [4] |
| RH20T          | 147      | [5] |
| Virtual Home   | 49545    | [6] |
| **SUM**        | **4788** |     |

## Experiments

To evaluate this tasks, we provide the LLMs with the following prompt:
```
System: Imagine you are a robot with a given hardware configuration and you should decide whether you are capable of executing a task.
User: Please only answer with Yes or No.
Task: [Task]
Hardware: [Hardware Configuration] 
```

For each of the 4788 tasks, which consists of a robot hardware configuration capable of executing it, we also create a negative sample by setting all robotic capabilities to the minimal configuration.
So each model is provided with 9576 samples.
We analyse the results for each model using the following metrics:
- Amount of true positives (TPs), true negatives (TNs), false positives (FPs) & false negatives (FNs)
- Ratio of positive to negative answers by the model (Formula: (TPs + FPs) / (TNs + FNs))
- Accuracy, Precision, Recall, Specificity
- F1-Score

## Results

| LLM                    | TNs  | TPs  | FNs  | FPs | Ratio | Acc       | Prec      | Rec       | Spec      | F1        |
|------------------------| ---- | ---- | ---- | --- | ----- |-----------|-----------|-----------|-----------|-----------|
| gpt-4o-2024-08-06      | 4772 | 2616 | 2172 | 16  | 0.379 | 0.772     | **0.994** | 0.546     | **0.997** | 0.705     |
| Llama-3.3-70B-Instruct | 4737 | 4294 | 494  | 51  | 0.831 | **0.943** | 0.988     | **0.897** | 0.989     | **0.940** |
| gemma-2-27b-it         | 4621 | 4061 | 727  | 167 | 0.791 | 0.907     | 0.961     | 0.848     | 0.965     | 0.901     |

## References

[1] Z. Samadikhoshkho, K. Zareinia, and F. Janabi-Sharifi, ‘A Brief Review on Robotic Grippers Classifications’, in 2019 IEEE Canadian Conference of Electrical and Computer Engineering (CCECE), Edmonton, AB, Canada, May 2019. doi: 10.1109/CCECE.2019.8861780.

[2] N. Reimers and I. Gurevych, ‘Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks’, in Proceedings of the 2019 conference on empirical methods in natural language processing, Association for Computational Linguistics, Nov. 2019. [Online]. Available: https://arxiv.org/abs/1908.10084

[3] M. Shridhar et al., ‘ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks’, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Seattle, WA, USA, 2020, pp. 10740–10749. Accessed: Jul. 04, 2022. [Online]. Available: https://openaccess.thecvf.com/content_CVPR_2020/html/Shridhar_ALFRED_A_Benchmark_for_Interpreting_Grounded_Instructions_for_Everyday_Tasks_CVPR_2020_paper.html

[4] A. Khazatsky et al., ‘DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset’, 2024. doi: 10.48550/ARXIV.2403.12945.

[5] H.-S. Fang et al., ‘RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot’, in 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan: IEEE, May 2024, pp. 653–660. doi: 10.1109/ICRA57147.2024.10611615.

[6] X. Puig et al., ‘VirtualHome: Simulating Household Activities via Programs’, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), arXiv, 2018, pp. 8494–8502. doi: 10.48550/ARXIV.1806.07011.