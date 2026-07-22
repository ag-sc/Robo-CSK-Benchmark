# Table Setting Task - Results

## TS-U Task Variation (Utensils; Multiple Choice Questions w. Multiple Answers)

| LLM                    | Technique                          | Presented In | Acc       |
|------------------------|:-----------------------------------|:-------------|-----------|
| Llama-3.3-70B-Instruct | RAG (Chunking)                     | [2]          | **0.778** |
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.)                  | [2]          | 0.775     |
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)                   | [2]          | 0.765     |
| Llama-3.3-70B-Instruct | Self-Generated In-Context Learning | [3]          | 0.753     |
| Llama-3.3-70B-Instruct | Role Prompting                     | [1]          | 0.739     |
| Llama-3.3-70B-Instruct | Metacognitive Prompting            | [3]          | 0.724     |
| Llama-3.3-70B-Instruct | Self-Refine                        | [3]          | 0.723     |
| Llama-3.3-70B-Instruct | Rephrase and Respond               | [3]          | 0.722     |
| gpt-4o-2024-08-06      | Role Prompting                     | [1]          | 0.702     |
| gpt-4o-mini-2024-07-18 | Self-Generated In-Context Learning | [3]          | 0.697     |
| Llama-3.3-70B-Instruct | Contrastive Chain-of-Thought       | [3]          | 0.687     |
| gpt-4o-mini-2024-07-18 | Rephrase and Respond               | [3]          | 0.680     |
| Llama-3.3-70B-Instruct | Self-Consistency                   | [3]          | 0.674     |
| gpt-4o-mini-2024-07-18 | Metacognitive Prompting            | [3]          | 0.672     |
| gemma-2-27b-it         | Self-Generated In-Context Learning | [3]          | 0.664     |
| gemma-2-27b-it         | Rephrase and Respond               | [3]          | 0.656     |
| gpt-4o-mini-2024-07-18 | Self-Consistency                   | [3]          | 0.654     |
| gpt-4o-mini-2024-07-18 | Contrastive Chain-of-Thought       | [3]          | 0.651     |
| Llama-3.3-70B-Instruct | Step-Back Prompting                | [3]          | 0.645     |
| gpt-4o-mini-2024-07-18 | Self-Refine                        | [3]          | 0.644     |
| gemma-2-27b-it         | Role Prompting                     | [1]          | 0.642     |
| gemma-2-27b-it         | Metacognitive Prompting            | [3]          | 0.610     |
| gemma-2-27b-it         | Self-Refine                        | [3]          | 0.604     |
| gemma-2-27b-it         | Step-Back Prompting                | [3]          | 0.583     |
| gpt-4o-mini-2024-07-18 | Step-Back Prompting                | [3]          | 0.581     |
| gpt-4o-mini-2024-07-18 | Role Prompting                     | [3]          | 0.565     |

## TS-P Task Variation (Plates; Multiple Choice Questions w. Single Answer)

| LLM                    | Technique                          | Presented In | Acc       |
|------------------------|:-----------------------------------|:-------------|-----------|
| Llama-3.3-70B-Instruct | RAG (Chunking)                     | [2]          | **0.870** |
| gpt-4o-mini-2024-07-18 | Self-Refine                        | [3]          | 0.850     |
| gpt-4o-mini-2024-07-18 | Rephrase and Respond               | [3]          | 0.840     |
| gemma-2-27b-it         | Self-Generated In-Context Learning | [3]          | 0.830     |
| gpt-4o-mini-2024-07-18 | Self-Consistency                   | [3]          | 0.810     |
| gpt-4o-mini-2024-07-18 | Metacognitive Prompting            | [3]          | 0.810     |
| Llama-3.3-70B-Instruct | Metacognitive Prompting            | [3]          | 0.810     |
| gpt-4o-mini-2024-07-18 | Role Prompting                     | [3]          | 0.800     |
| Llama-3.3-70B-Instruct | Self-Generated In-Context Learning | [3]          | 0.800     |
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.)                  | [2]          | 0.800     |
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)                   | [2]          | 0.800     |
| gemma-2-27b-it         | Role Prompting                     | [1]          | 0.790     |
| gpt-4o-mini-2024-07-18 | Contrastive Chain-of-Thought       | [3]          | 0.790     |
| Llama-3.3-70B-Instruct | Role Prompting                     | [1]          | 0.790     |
| gemma-2-27b-it         | Rephrase and Respond               | [3]          | 0.780     |
| Llama-3.3-70B-Instruct | Rephrase and Respond               | [3]          | 0.780     |
| Llama-3.3-70B-Instruct | Contrastive Chain-of-Thought       | [3]          | 0.780     |
| Llama-3.3-70B-Instruct | Self-Consistency                   | [3]          | 0.780     |
| gemma-2-27b-it         | Step-Back Prompting                | [3]          | 0.760     |
| gpt-4o-mini-2024-07-18 | Self-Generated In-Context Learning | [3]          | 0.760     |
| Llama-3.3-70B-Instruct | Self-Refine                        | [3]          | 0.760     |
| gemma-2-27b-it         | Self-Refine                        | [3]          | 0.750     |
| gpt-4o-2024-08-06      | Role Prompting                     | [1]          | 0.730     |
| gpt-4o-mini-2024-07-18 | Step-Back Prompting                | [3]          | 0.650     |
| Llama-3.3-70B-Instruct | Step-Back Prompting                | [3]          | 0.640     |
| gemma-2-27b-it         | Metacognitive Prompting            | [3]          | 0.620     |

## References

[1] J.-P. Töberg, S. Kenneweg, and P. Cimiano, ‘RoboCSKBench: Benchmarking Embodied Commonsense Capabilities of Large Language Models’, in Proc. of the 22nd International Conference on Ubiquitous Robots (UR 2025), College Station, Texas, USA, 2025, pp. 199–206. doi: 10.1109/UR65550.2025.11078036.

[2] P. Frese, ‘Evaluating Retrieval-Augmented Generation as a Source of Commonsense Knowledge for Household Robots’, Project in the NWI Master, Bielefeld University, Bielefeld, Germany, 2026.

[3] L. Buiwitt, ‘Evaluating Commonsense Capabilities Through Prompt Engineering’, Bachelor Thesis, Bielefeld University, Bielefeld, Germany, 2025.