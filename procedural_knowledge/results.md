# Procedural Knowledge Task - Results

## PK-S Task Variation (Binary Questions)

| LLM                    | Technique         | Presented In | TNs  | TPs  | FNs  | FPs | Ratio | Acc       | Prec      | Rec       | Spec      | F1        |
|------------------------|:------------------|:-------------|------|------|------|-----|-------|-----------|-----------|-----------|-----------|-----------|
| gpt-4o-2024-08-06      | Role Prompting    | [1]          | 1362 | 1511 | 41   | 190 | 1.212 | **0.926** | 0.888     | 0.974     | 0.878     | **0.929** |
| Llama-3.3-70B-Instruct | Role Prompting    | [1]          | 1339 | 1501 | 51   | 213 | 1.233 | 0.915     | 0.876     | 0.967     | 0.863     | 0.919     |
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.) | [2]          | 1421 | 1390 | 162  | 131 | 0.961 | 0.906     | 0.914     | 0.896     | 0.916     | 0.905     |
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)  | [2]          | 1424 | 1389 | 163  | 128 | 0.955 | 0.906     | 0.916     | 0.894     | 0.918     | 0.905     |
| gemma-2-27b-it         | Role Prompting    | [1]          | 754  | 1543 | 9    | 798 | 3.068 | 0.740     | 0.659     | **0.994** | 0.486     | 0.793     |
| Llama-3.3-70B-Instruct | RAG (Chunking)    | [2]          | 1526 | 524  | 1026 | 25  | 0.215 | 0.661     | **0.954** | 0.338     | **0.984** | 0.499     |

## PK-MC Task Variation (Multiple Choice Questions)

| LLM                    | Technique                          | Presented In | Acc       |
|------------------------|:-----------------------------------|:-------------|-----------|
| gpt-4o-2024-08-06      | Role Prompting                     | [1]          | **0.914** |
| Llama-3.3-70B-Instruct | Contrastive Chain-of-Thought       | [3]          | 0.884     |
| Llama-3.3-70B-Instruct | Rephrase and Respond               | [3]          | 0.879     |
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.)                  | [2]          | 0.873     |
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)                   | [2]          | 0.868     |
| gpt-4o-mini-2024-07-18 | Role Prompting                     | [3]          | 0.855     |
| gpt-4o-mini-2024-07-18 | Contrastive Chain-of-Thought       | [3]          | 0.854     |
| Llama-3.3-70B-Instruct | Metacognitive Prompting            | [3]          | 0.853     |
| Llama-3.3-70B-Instruct | Role Prompting                     | [1]          | 0.848     |
| Llama-3.3-70B-Instruct | Step-Back Prompting                | [3]          | 0.848     |
| Llama-3.3-70B-Instruct | Self-Consistency                   | [3]          | 0.848     |
| Llama-3.3-70B-Instruct | Self-Refine                        | [3]          | 0.848     |
| gpt-4o-mini-2024-07-18 | Rephrase and Respond               | [3]          | 0.846     |
| Llama-3.3-70B-Instruct | Self-Generated In-Context Learning | [3]          | 0.823     |
| gpt-4o-mini-2024-07-18 | Metacognitive Prompting            | [3]          | 0.815     |
| gemma-2-27b-it         | Role Prompting                     | [1]          | 0.809     |
| gpt-4o-mini-2024-07-18 | Self-Consistency                   | [3]          | 0.806     |
| gpt-4o-mini-2024-07-18 | Self-Refine                        | [3]          | 0.805     |
| gpt-4o-mini-2024-07-18 | Step-Back Prompting                | [3]          | 0.792     |
| gpt-4o-mini-2024-07-18 | Self-Generated In-Context Learning | [3]          | 0.709     |
| Llama-3.3-70B-Instruct | RAG (Chunking)                     | [2]          | 0.694     |

## References

[1] J.-P. Töberg, S. Kenneweg, and P. Cimiano, ‘RoboCSKBench: Benchmarking Embodied Commonsense Capabilities of Large Language Models’, in Proc. of the 22nd International Conference on Ubiquitous Robots (UR 2025), College Station, Texas, USA, 2025, pp. 199–206. doi: 10.1109/UR65550.2025.11078036.

[2] P. Frese, ‘Evaluating Retrieval-Augmented Generation as a Source of Commonsense Knowledge for Household Robots’, Project in the NWI Master, Bielefeld University, Bielefeld, Germany, 2026.

[3] L. Buiwitt, ‘Evaluating Commonsense Capabilities Through Prompt Engineering’, Bachelor Thesis, Bielefeld University, Bielefeld, Germany, 2025.