"""Pinned benchmark judge templates, kept separate from transport and score parsing.

Source revisions and attribution stay beside each template and in NOTICE.md.
The wording is unchanged from the harness protocols that use these components.
"""

# ruff: noqa: RUF001 -- preserve the pinned upstream typography.

M3_SYSTEM_PROMPT = "You are an expert in video understanding."
MM_LIFELONG_USER_PROMPT = (
    "Question: {question}\nGroundtruth answer: {reference}\n"
    "Candidate answer: {prediction}\nYour response: "
)

# mem-eval-suite/LoCoMo_refined@887091190789e8d6760e70b9edd696539923dc4f
LOCOMO_PROMPT = """Your task is to label an answer as ’CORRECT’ or ’WRONG’ given:
(1) a question,
(2) a gold (ground truth) answer,
(3) a generated answer.

Core principle — Inclusion + Non-contradiction
- Be GENEROUS: if the generated answer clearly includes the gold’s key content (or a clear paraphrase of the same content) and does not contradict it, mark CORRECT — even if extra details are added.
- Mark WRONG only when the generated answer does not include the gold’s content, changes it, or contradicts it.

TIME (strict granularity; relative form equivalence; no calendar math)
- Granularity must match exactly: HOUR↔HOUR, DAY↔DAY, MONTH↔MONTH, YEAR↔YEAR.
  Do not answer a gold at a different time unit — even if the numeric value overlaps. Do not answer a month-level gold with a specific day, nor a year with a specific month/day/hour, etc.
  (e.g., gold = "July 26, 2019" [DAY]; generated = "2019-07-26 08:09:17" [includes Second] → WRONG)
- Do NOT convert relative ↔ absolute. If the gold uses a relative time expression, the generated answer must also use a relative form (or a clear paraphrase of that same form), not a computed date/range.
- Treat harmless modifiers in relative forms (e.g., “the/last/previous/just prior”) as equivalent when both the anchor date and the time unit are the same.

- Lists of DISTINCT facts:
- If the gold answer lists multiple distinct facts (joined by "and", commas, or slashes), the generated answer must cover **all** of them.
- Extra non-contradictory items **generally count as WRONG**.
    - Example: gold = A, B, C ; gen = A, B, C → CORRECT
    - Example: gold = A, B, C ; gen = A, B, C, D → WRONG
- Exception: If a gold element is elaborated or split into finer details in the generated answer (e.g., C → C, C′), it is still considered CORRECT.

Preference/Benefit Questions (e.g., "what X likes/values most")
- If gold lists multiple reasons/aspects, the generated answer only needs to include **any one** of them without contradiction to be CORRECT.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label":

```json
{{
    "label": "CORRECT" or "WRONG"
}}
```
"""

# ByteDance-Seed/m3-agent@0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c
M3_PROMPT = """You are provided with a question, a ground truth answer, and an answer from an agent model. Your task is to determine whether the ground truth answer can be logically inferred from the agent's answer, in the context of the question.

Do not directly compare the surface forms of the agent answer and the ground truth answer. Instead, assess whether the meaning expressed by the agent answer supports or implies the ground truth answer. If the ground truth can be reasonably derived from the agent answer, return "Yes". If it cannot, return "No".

Important notes:
\t•\tDo not require exact wording or matching structure.
\t•\tSemantic inference is sufficient, as long as the agent answer entails or implies the meaning of the ground truth answer, given the question.
\t•\tOnly return "Yes" or "No", with no additional explanation or formatting.

Input fields:
\t•\tquestion: the question asked
\t•\tground_truth_answer: the correct answer
\t•\tagent_answer: the model's answer to be evaluated

Now evaluate the following input:

Input:
\t•\tquestion: {question}
\t•\tground_truth_answer: {ground_truth_answer}
\t•\tagent_answer: {agent_answer}

Output ('Yes' or 'No'):"""

# google-research-datasets/egotempo@7022ba77b4d89f51cf34e499767995ccd5c90c7a
EGOTEMPO_PROMPT = """role: "system",
content: "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs.
Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:
-----##INSTRUCTIONS:
- Focus on the correctness and accuracy of the predicted answer with the ground-truth.
- Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that."
role: "user",
content: "Please evaluate the following video-based question-answer pair:
Question: {question}
Ground truth correct Answer: {answer}
Predicted Answer: {prediction}
Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness.
Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect',
value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
"""

# MM-Lifelong/MM-Lifelong@248aa82039a574e63a2e524746a7cd8f32330443/eval_acc.py
MM_LIFELONG_SYSTEM = """As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct “groundtruth” answer, and the candidate answer will be provided to you.
The following is a comparison table of some proper nouns; matching any one of them is considered correct.

You must FIRST provide a brief analysis explaining the semantic similarity between the groundtruth
and the candidate answer.

THEN, on a new line, output the final score.

Scoring criteria (semantic similarity only):

- 0: No similarity.
  The candidate answer is completely irrelevant, contradictory, or does not address the question at all.

- 1: Very low similarity.
  The candidate answer mentions a related topic or keyword, but fails to answer the question
  and does not convey the main meaning of the groundtruth.

- 2: Low similarity.
  The candidate answer addresses the question in a limited way, capturing some minor aspects,
  but misses or misrepresents the core idea or key facts of the groundtruth.

- 3: Moderate similarity.
  The candidate answer captures the main idea of the groundtruth,
  but omits several important details or includes noticeable inaccuracies.

- 4: High similarity.
  The candidate answer correctly captures the main idea and most key details of the groundtruth,
  with only minor omissions, simplifications, or non-critical inaccuracies.

- 5: Complete similarity.
  The candidate answer is semantically equivalent to the groundtruth,
  covering all essential information with no meaningful omissions or errors.

Special Rules:

- Hallucination-sensitive questions:
Score 5 only if all required items are correct;
if any item is incorrect, missing, or hallucinated, score 0 (no partial credit).

- Time-duration questions:
Allow errors within the range defined by the question; answers outside the range should receive score 0.

Output format (strictly follow):
Analysis:
<your analysis>

Final Score:
<an integer from 0 to 5>"""

# JingbiaoMei/ATM-Bench@ef4e5dff1a47ec71213a06e359f02753defa8fb1
ATM_PROMPT = (
    "You are an evaluator, and you are given a task to evaluate a model predictions "
    "with a given question. Let's follow the instructions step by step to make a "
    "judgement.  1. As the first step, you need to check whether the prediction was really "
    "answering the question.  2. If the model prediction does provide a meaningful answer, "
    "judge whether the model Prediction matches the ground truth answer by reasoning according "
    "to the following steps:  2.1: Always assume the ground truth is correct.  "
    "2.2: Pay attention to theses special cases:  a. If the ground truth answer contains "
    'numbers, the value of "accuracy" is true only if numbers in ground truth and numbers in '
    'model predictions match very well; in case of math questions, "accuracy" is true only if '
    "the numbers in model predictions EXACTLY matches the numbers in ground truth;  "
    'b. If the ground truth answer contains time, and/or time range, "accuracy" is "true" only '
    "if if times and time ranges in ground truth and model predictions match very well.  "
    'c. If the ground truth answer contains a set of objects, "accuracy" is "true" if the model '
    'prediction covers most of the objects in the ground truth; however, "accuracy" if "false" '
    "if the  model prediction has a lot of objects that are not in the ground truth.  "
    'd. If the ground truth is something similar to "I don\'t know", "accuracy" is "true" only '
    "if the model prediction also implies the similar thing.  2.3: Even if the prediction "
    'statement is reasonable, if it conflicts with or does not match the ground truth, "accuracy" '
    'should be "false".  2.4. "Accuracy" is true if the ground truth information is covered by '
    "the prediction. The prediction is allowed to provide more information but should not be "
    "against the ground truth. If it is hard to  decide whether the prediction matches ground "
    'truth, "accuracy" should be "false".  Think step by step following the instructions above, '
    'and then make a judgment. Respond with only a single JSON blob with an "explanation" field '
    'that has your short(less than 100 word) reasoning  steps and an "accuracy" field which is '
    '"true" or "false".  Question: {{question}}  Ground truth: {{answer}}  '
    "Prediction: {{prediction}}"
)


# YuanchenBei/Mem-Gallery@a93959e1e978a6a7d77798ae92c2ffe41c538c62
GALLERY_PROMPT = """You are an impartial judge evaluating the memory capabilities of an AI assistant with the question-answering task.
Your task is to compare the Assistant's Answer against the Ground Truth and assign a score of 0, 0.25, 0.5, 0.75, or 1.

### Scoring Rubric

**Score 0 (Incorrect / Miss):**

- The answer contradicts the Ground Truth.
- For Yes/No questions: The answer has the wrong polarity (e.g., says "Yes" when Ground Truth is "No").
- For Open-ended questions: The answer provides factually wrong information or hallucinations.
- The assistant fails to provide the required information.

**Score 0.25 (Poor / Tangential):**

- The answer touches on the topic but misses the **core entity** or key value required.
- The answer contains a mix of minor correct details and **significant hallucinations** or wrong associations.
- The answer is excessively vague to the point of being useless (e.g., answering "a dog" instead of "a golden retriever").

**Score 0.5 (Partial / Vague):**

- The answer is technically correct, but lacks confidence or is incomplete.
- The answer captures the **main entity or concept** correctly but misses a part of the required supporting details.
- For Yes/No questions: The polarity is correct, but the reasoning is flawed (if have), or the assistant is uncertain (e.g., "I think it might be Yes").
- For Open-ended questions: The answer is too general or misses key adjectives/details present in the Ground Truth.

**Score 0.75 (Good / Minor Imperfection):**

- The answer is largely accurate and captures the core information confidently.
- It misses only **minor details** (e.g., specific adjectives or secondary details) that do not alter the main truth.
- The answer contains all the correct information but includes unnecessary "fluff" or slight conversational filler that reduces precision.

**Score 1 (Correct / Exact):**

- The answer is accurate, precise, and confident.
- For Yes/No questions: The polarity matches the Ground Truth perfectly.
- For Open-ended questions: The answer contains **all** the core information and necessary details required by the Ground Truth without hallucinations.

### Input Data
Question: {{question}}
Ground Truth: {{ground_truth}}
Assistant Answer: {{model_output}}

### Output Format
Output strictly in the following JSON format:
{"score": <0, 0.25, 0.5, 0.75, or 1>, "reasoning": "<short explanation>"}"""
