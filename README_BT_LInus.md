# Bachelors Thesis - Evaluating Commonsense Capabilities Through Prompt Engineering
This thesis builds upon the [RoboCSKBench](https://ieeexplore.ieee.org/abstract/document/11078036)[^1].

The aim of this thesis is to implement different prompting techniques and
evaluate their performance on the benchmark RoboCSKBench, which cur-
rently consists of five tasks. In addition to this, the thesis explores possible
improvements to the original prompts of the RoboCSKBench using the frame-
work DSPy.
Seven different prompting techniques are implemented and their perfor-
mance on the RoboCSKBench is evaluated using the models GPT-4o-mini,
Llama 3.3 and Gemma 2. The results are then compared to the results of a
basic role-based prompt. Additionally, two tasks of the RoboCSKBench are
implemented with the framework DSPy and their prompts optimized using
the built-in optimization algorithm MIPROv2.
Most prompting techniques achieve results comparable to the basic role-
based approach, with considerable deviations only seen in specific cases.
Prompts optimized using MIPROv2 employ more descriptive language but
result in the same performance as a basic implementation of the tasks using
DSPy.
As the results obtained in this thesis were not averaged over multiple
evaluations, their significance is limited, leaving opportunities for further
research.

# Usage
The program structure is mostly the same as the [RoboCSKBench](https://ieeexplore.ieee.org/abstract/document/11078036), therefore, usage of the program is similar as well:
## Prompt with different techniques
1. Choose a list of models to prompt with. The models currently included are **GPT-4o-mini**, **Llama 3.3** and **Gemma 2**.
2. Choose a list of tasks of the [RoboCSKBench](https://ieeexplore.ieee.org/abstract/document/11078036).
3. Choose a list of prompting techniques. Current techniques are
  - Self-Generated In-Context Learning[^2] (SG-ICL)
  - Rephrase and Respond[^3] (RaR)
  - Step-Back Prompting[^4]
  - Contrastive Chain of Thought[^5]
  - Metacongnitive Prompting[^6]
  - Self-Consistency[^7]
  - Self-Refine[^8]
4. Optionally, a starting and ending point for the questions can be specified. This allows partial runs of the tasks with the techniques.

## Prompt with DSPy[^9]
For the thesis, the tasks **[Table Setting](./table_setting/README.md)** and **[Tool Usage](./tool_usage/README.md)** were implemented
using the Framework DSPy. The functionalities regarding DSPy are the following:
  - Evaluating the performance of a DSPy implementation of the basic role-based prompt from the [RoboCSKBench](https://ieeexplore.ieee.org/abstract/document/11078036).
  - Optimization and evaluation of a basic DSPy implementation with MIPROv2[^10]
  - Optimization and evaluation of a the DSPy implementation of the basic role-based prompt with MIPROv2[^10]
The DSPy program that was last evaluated is stored. This stored program can also be shown and evaluated again.

# Outputs
Many of the techniques lead to more complicated model answers, from which the final answers get extracted. These final answers along with the questions get collected and
stored in a result file. The prompts and full answers are stored in a log file. A short summary of the performances is also kept.
The following table details the different file locations:

| Task           | Link to Results                                                                                               | Log folder                        | Complete result folder                     |
|----------------|---------------------------------------------------------------------------------------------------------------|-----------------------------------|--------------------------------------------|
|Tidy Up         | [tidy_up/results_multi/model_overview.csv](tidy_up/results_multi/model_overview.csv)                          | tidy_up/logs/{model}              | tidy_up/results_multi/{model}              |
|Tool Usage      | [tool_usage/results/model_overview.csv](tool_usage/results/model_overview.csv)                                | tool_usage/logs/{model}           | tool_usage/results/{model}                 |
|Meta-Reasoning  | [meta_reasoning/results_multi/model_overview.csv](meta_reasoning/results_multi/model_overview.csv)            | meta_reasoning/logs/{model}       | meta_reasoning/results_multi/{model}       |
|Table Setting   | [table_setting/results/model_overview.csv](table_setting/results/model_overview.csv)                          | table_setting/logs/{model}        | table_setting/results/{model}              |
|Procedural Know.| [procedural_knowledge/results_multi/model_overview.csv](procedural_knowledge/results_multi/model_overview.csv)| procedural_knowledge/logs/{model} | procedural_knowledge/results_multi/{model} |

[^1]: J.-P. Töberg, S. Kenneweg, and P. Cimiano. Robocskbench: Bench-
marking embodied commonsense capabilities of large language mod-
els. In 2025 22nd International Conference on Ubiquitous Robots (UR),
pages 199–206, 2025. doi: 10.1109/UR65550.2025.11078036.

[^2]: T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. Kaplan, P. Dhariwal,
A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-
Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler,
J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray,
B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever,
and D. Amodei. Language models are few-shot learners, 2020. URL
https://arxiv.org/abs/2005.14165.

[^3]: Y. Deng, W. Zhang, Z. Chen, and Q. Gu. Rephrase and respond: Let
large language models ask better questions for themselves, 2024. URL
https://arxiv.org/abs/2311.04205.

[^4]: H. S. Zheng, S. Mishra, X. Chen, H.-T. Cheng, E. H. Chi, Q. V. Le, and
D. Zhou. Take a step back: Evoking reasoning via abstraction in large
language models, 2024. URL https://arxiv.org/abs/2310.06117.

[^5]: Y. K. Chia, G. Chen, L. A. Tuan, S. Poria, and L. Bing. Con-
trastive chain-of-thought prompting, 2023. URL https://arxiv.org/
abs/2311.09277.

[^6]: Y. Wang and Y. Zhao. Metacognitive prompting improves understand-
ing in large language models, 2024. URL https://arxiv.org/abs/
2308.05342.

[^7]: X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowd-
hery, and D. Zhou. Self-consistency improves chain of thought reasoning
in language models, 2023. URL https://arxiv.org/abs/2203.11171.

[^8]: A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe,
U. Alon, N. Dziri, S. Prabhumoye, Y. Yang, S. Gupta, B. P. Majumder,
K. Hermann, S. Welleck, A. Yazdanbakhsh, and P. Clark. Self-refine:
Iterative refinement with self-feedback, 2023. URL https://arxiv.
org/abs/2303.17651

[^9]: O. Khattab, A. Singhvi, P. Maheshwari, Z. Zhang, K. Santhanam,
S. Vardhamanan, S. Haq, A. Sharma, T. T. Joshi, H. Moazam, H. Miller,
M. Zaharia, and C. Potts. Dspy: Compiling declarative language model
calls into self-improving pipelines. 2024

[^10]: K. Opsahl-Ong, M. J. Ryan, J. Purtell, D. Broman, C. Potts, M. Za-
haria, and O. Khattab. Optimizing instructions and demonstrations for
multi-stage language model programs, 2024. URL https://arxiv.org/
abs/2406.11695
