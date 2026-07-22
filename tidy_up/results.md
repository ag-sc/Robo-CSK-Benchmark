# Tidy Up Task - Results

## TU-G Task Variation (Generative Questions)

| LLM                    | Technique         | Presented In | MAP@1     | MAP@3     | MAP@5     | MAR@1     | MAR@3     | MAR@5     | MRR       |
|------------------------|:------------------|:-------------|-----------|-----------|-----------|-----------|-----------|-----------|:----------|
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.) | [2]          | **0.624** | **0.670** | **0.618** | 0.156     | 0.223     | 0.245     | **0.706** |
| Llama-3.3-70B-Instruct | Role Prompting    | [1]          | 0.608     | 0.645     | 0.584     | **0.158** | **0.225** | **0.256** | 0.680     |
| Llama-3.3-70B-Instruct | RAG (Chunking)    | [2]          | 0.595     | 0.639     | 0.570     | 0.149     | 0.218     | 0.242     | 0.673     |
| gpt-4o-2024-08-06      | Role Prompting    | [1]          | 0.580     | 0.628     | 0.580     | 0.147     | 0.221     | 0.249     | 0.661     |
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)  | [2]          | 0.486     | 0.544     | 0.502     | 0.119     | 0.176     | 0.196     | 0.574     |
| gemma-2-27b-it         | Role Prompting    | [1]          | 0.274     | 0.321     | 0.307     | 0.096     | 0.144     | 0.157     | 0.348     |

## TU-MC Task Variation (Multiple Choice Questions)

| LLM                    | Technique                          | Presented In | Acc       |
|------------------------|:-----------------------------------|:-------------|-----------|
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)                   | [2]          | **0.561** |
| gpt-4o-mini-2024-07-18 | Self-Generated In-Context Learning | [3]          | 0.559     |
| gemma-2-27b-it         | Role Prompting                     | [1]          | 0.522     |
| gpt-4o-mini-2024-07-18 | Contrastive Chain-of-Thought       | [3]          | 0.520     |
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.)                  | [2]          | 0.520     |
| gpt-4o-mini-2024-07-18 | Rephrase and Respond               | [3]          | 0.517     |
| gpt-4o-2024-08-06      | Role Prompting                     | [1]          | 0.509     |
| gemma-2-27b-it         | Metacognitive Prompting            | [3]          | 0.507     |
| gpt-4o-mini-2024-07-18 | Step-Back Prompting                | [3]          | 0.507     |
| gpt-4o-mini-2024-07-18 | Metacognitive Prompting            | [3]          | 0.507     |
| gpt-4o-mini-2024-07-18 | Role Prompting                     | [3]          | 0.504     |
| Llama-3.3-70B-Instruct | Role Prompting                     | [1]          | 0.496     |
| Llama-3.3-70B-Instruct | Self-Refine                        | [3]          | 0.496     |
| gpt-4o-mini-2024-07-18 | Self-Refine                        | [3]          | 0.488     |
| Llama-3.3-70B-Instruct | Rephrase and Respond               | [3]          | 0.488     |
| Llama-3.3-70B-Instruct | Metacognitive Prompting            | [3]          | 0.488     |
| gemma-2-27b-it         | Rephrase and Respond               | [3]          | 0.486     |
| gemma-2-27b-it         | Self-Generated In-Context Learning | [3]          | 0.486     |
| Llama-3.3-70B-Instruct | Step-Back Prompting                | [3]          | 0.486     |
| gpt-4o-mini-2024-07-18 | Self-Consistency                   | [3]          | 0.483     |
| Llama-3.3-70B-Instruct | Self-Generated In-Context Learning | [3]          | 0.480     |
| gemma-2-27b-it         | Step-Back Prompting                | [3]          | 0.473     |
| Llama-3.3-70B-Instruct | Contrastive Chain-of-Thought       | [3]          | 0.467     |
| Llama-3.3-70B-Instruct | RAG (Chunking)                     | [2]          | 0.467     |
| gemma-2-27b-it         | Self-Refine                        | [3]          | 0.441     |

## References

[1] J.-P. Töberg, S. Kenneweg, and P. Cimiano, ‘RoboCSKBench: Benchmarking Embodied Commonsense Capabilities of Large Language Models’, in Proc. of the 22nd International Conference on Ubiquitous Robots (UR 2025), College Station, Texas, USA, 2025, pp. 199–206. doi: 10.1109/UR65550.2025.11078036.

[2] P. Frese, ‘Evaluating Retrieval-Augmented Generation as a Source of Commonsense Knowledge for Household Robots’, Project in the NWI Master, Bielefeld University, Bielefeld, Germany, 2026.

[3] L. Buiwitt, ‘Evaluating Commonsense Capabilities Through Prompt Engineering’, Bachelor Thesis, Bielefeld University, Bielefeld, Germany, 2025.