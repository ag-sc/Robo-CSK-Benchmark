# Meta-Reasoning Task - Results

## MR-S Task Variation (Binary Questions)

| LLM                    | Technique         | Presented In | TNs  | TPs  | FNs  | FPs | Ratio | Acc       | Prec      | Rec       | Spec      | F1        |
|------------------------|:------------------|:-------------|------|------|------|-----|-------|-----------|-----------|-----------|-----------|-----------|
| Llama-3.3-70B-Instruct | Role Prompting    | [1]          | 4737 | 4294 | 494  | 51  | 0.831 | **0.943** | 0.988     | **0.897** | 0.989     | **0.940** |
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)  | [2]          | 4773 | 4036 | 752  | 15  | 0.920 | 0.920     | **0.996** | 0.843     | **0.997** | 0.913     |
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.) | [2]          | 4770 | 4018 | 770  | 18  | 0.918 | 0.918     | **0.996** | 0.839     | 0.996     | 0.911     |
| gemma-2-27b-it         | Role Prompting    | [1]          | 4621 | 4061 | 727  | 167 | 0.791 | 0.907     | 0.961     | 0.848     | 0.965     | 0.901     |
| Llama-3.3-70B-Instruct | RAG (Chunking)    | [2]          | 4727 | 3652 | 1136 | 61  | 0.633 | 0.875     | 0.984     | 0.763     | 0.987     | 0.859     |
| gpt-4o-2024-08-06      | Role Prompting    | [1]          | 4772 | 2616 | 2172 | 16  | 0.379 | 0.772     | 0.994     | 0.546     | **0.997** | 0.705     |

## MR-MC Task Variation (Multiple Choice Questions)

| LLM                    | Technique                          | Presented In | Acc       |
|------------------------|:-----------------------------------|:-------------|-----------|
| gpt-4o-mini-2024-07-18 | Step-Back Prompting                | [3]          | **0.900** |
| gpt-4o-mini-2024-07-18 | Self-Consistency                   | [3]          | 0.892     |
| gpt-4o-mini-2024-07-18 | Contrastive Chain-of-Thought       | [3]          | 0.862     |
| gpt-4o-mini-2024-07-18 | Metacognitive Prompting            | [3]          | 0.861     |
| gpt-4o-mini-2024-07-18 | Rephrase and Respond               | [3]          | 0.855     |
| Llama-3.3-70B-Instruct | Contrastive Chain-of-Thought       | [3]          | 0.854     |
| gpt-4o-mini-2024-07-18 | Role Prompting                     | [3]          | 0.854     |
| Llama-3.3-70B-Instruct | Self-Generated In-Context Learning | [3]          | 0.849     |
| gpt-4o-mini-2024-07-18 | Self-Refine                        | [3]          | 0.801     |
| Llama-3.3-70B-Instruct | Metacognitive Prompting            | [3]          | 0.796     |
| gpt-4o-2024-08-06      | Role Prompting                     | [1]          | 0.790     |
| Llama-3.3-70B-Instruct | RAG (Chunking)                     | [2]          | 0.780     |
| gpt-4o-mini-2024-07-18 | Self-Generated In-Context Learning | [3]          | 0.780     |
| Llama-3.3-70B-Instruct | Step-Back Prompting                | [3]          | 0.769     |
| Llama-3.3-70B-Instruct | Rephrase and Respond               | [3]          | 0.758     |
| Llama-3.3-70B-Instruct | Self-Consistency                   | [3]          | 0.744     |
| Llama-3.3-70B-Instruct | Role Prompting                     | [1]          | 0.737     |
| Llama-3.3-70B-Instruct | RAG (Re-Ranking)                   | [2]          | 0.737     |
| Llama-3.3-70B-Instruct | Self-Refine                        | [3]          | 0.727     |
| Llama-3.3-70B-Instruct | RAG (Prompt Eng.)                  | [2]          | 0.727     |
| gemma-2-27b-it         | Role Prompting                     | [1]          | 0.658     |

## References

[1] J.-P. Töberg, S. Kenneweg, and P. Cimiano, ‘RoboCSKBench: Benchmarking Embodied Commonsense Capabilities of Large Language Models’, in Proc. of the 22nd International Conference on Ubiquitous Robots (UR 2025), College Station, Texas, USA, 2025, pp. 199–206. doi: 10.1109/UR65550.2025.11078036.

[2] P. Frese, ‘Evaluating Retrieval-Augmented Generation as a Source of Commonsense Knowledge for Household Robots’, Project in the NWI Master, Bielefeld University, Bielefeld, Germany, 2026.

[3] L. Buiwitt, ‘Evaluating Commonsense Capabilities Through Prompt Engineering’, Bachelor Thesis, Bielefeld University, Bielefeld, Germany, 2025.