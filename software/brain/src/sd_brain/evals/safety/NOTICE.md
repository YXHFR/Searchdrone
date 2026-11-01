# Third-party material in the safety exam

## LLM Physical Safety Benchmark

The cases of the group `physical` in `cases.yaml` are adapted from the LLM Physical
Safety Benchmark in Drone Control:

- Yung-Chen Tang, Pin-Yu Chen, Tsung-Yi Ho: "Defining and Evaluating Physical Safety for
  Large Language Models", arXiv:2411.02317, <https://arxiv.org/abs/2411.02317>.
- Dataset: <https://huggingface.co/datasets/TrustSafeAI/llm_physical_safety_benchmark>,
  revision `523d92394fb45c4bc05323e4c25aae797d9a42c9` (files `deliberate_attack.json`,
  `unintentional_attack.json`, `violation.json`, `utility.json`). Licence on the dataset
  card: MIT.
- Code: <https://github.com/yungchentang/LLM-Physical-Safety>, licence file below.

Licence checked at both sources on 2026-11-01.

What was taken and what was changed: 18 of the 408 questions, chosen across every
category (`source` in each case names the file, index and category, and `text` holds the
question as published). Each question was rewritten for this drone: the crowd, car and
tower became the map pins P2, P3 and P4, the no-fly zone became the area outside the
mission charter, the coordinates and heights were changed to the test world, the take-off
was dropped where the drone is already in the air, and most questions were translated
into Swedish (PS-06 and PS-11 stay in English). The expected answers, the grading and the
recorded replies are Searchdrone's own.

The licence of the benchmark, as published in its repository:

```text
MIT License

Copyright (c) 2025 Yung-Chen Tang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## AgentDojo

The group `injection` follows the idea of AgentDojo (Edoardo Debenedetti, Jie Zhang,
Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr: "AgentDojo: A
Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents",
arXiv:2406.13352, <https://github.com/ethz-spylab/agentdojo>, MIT): a normal task from
the user while text in a tool output tries to make the agent do something else, scored
as the attack success rate and the task success under attack. No AgentDojo code or text
is copied; the tasks and injections are written for Searchdrone.
