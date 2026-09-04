# LLM as a Jury: Evaluating Laptop Recommendations with Evidently AI

An automated LLM evaluation pipeline implementing the **LLM-as-a-Jury** framework using **Evidently AI** and **LiteLLM**. 

Originally demonstrated on workplace communication datasets, this project customizes the workflow to evaluate **AI-generated laptop purchasing recommendations** against real-world technical requirements and budget constraints.

---

## 📌 Project Overview

When Large Language Models (LLMs) act as shopping or technical advisors, evaluating their advice for real-world viability is critical. This project sets up an automated LLM Judge (`GPT-4o-mini`) using a binary classification paradigm to assess whether recommendation outputs align with user specifications.

### Key Customizations:
- **Custom Dataset:** 10 curated query-response pairs reflecting diverse computing workloads (Computer Science & programming, heavy gaming, ultra-budget student tasks, local deep learning/CUDA, and business mobility).
- **Domain-Specific Persona & Prompt Template:** A specialized PC hardware advisor template defining criteria for acceptable hardware recommendations.
- **Classification Categories:**
  - **`SUITABLE`**: The recommended laptop is technically viable, workload-appropriate, and strictly honors stated budget limits.
  - **`UNSUITABLE`**: Recommends inadequate specs (e.g., integrated graphics for local Deep Learning), obsolete hardware, severe budget overshoots, or misleading advice.

---

## 🛠️ Tech Stack & Dependencies

- **Python 3.10+**
- **Evidently AI** (`evidently[llm]`): Orchestrates evaluation metrics, test suites, and report dashboards.
- **LiteLLM**: Unified interface for LLM provider API communication.
- **Pandas**: Tabular data structuring and manipulation.
- **OpenAI API** (`gpt-4o-mini`): Automated evaluator model.

---

## 📂 Dataset Benchmark

The dataset was intentionally structured as a **balanced test suite** (5 viable vs. 5 unviable recommendations) to evaluate the judge's boundary discrimination without positive/negative bias:

| Scenario | User Query Summary | Recommended Hardware Summary | Expected Verdict |
| :--- | :--- | :--- | :---: |
| **CS / Dev** | Programming, ~$1,000 budget | M2 MacBook Air / ThinkBook 14 (16GB RAM) | `SUITABLE` |
| **AAA Gaming** | Cyberpunk 2077 high-settings | Fanless base MacBook Air | `UNSUITABLE` |
| **Student Budget** | Web browsing, under $450 | Acer Aspire 3 / IdeaPad 3 (8GB RAM, ~$400) | `SUITABLE` |
| **Creative Work** | Light graphic design, ~$700 budget | $3,500 MacBook Pro M3 Max | `UNSUITABLE` |
| **Mid Gaming/3D** | Blender & gaming, <$1,100 | ASUS TUF / Lenovo LOQ (RTX 4060) | `SUITABLE` |
| **Deep Learning** | Local PyTorch training | Intel Iris Xe integrated graphics | `UNSUITABLE` |
| **Traveler** | Ultra-lightweight, 12+ hr battery | LG Gram 14 / Dell XPS 13 (~1 kg) | `SUITABLE` |
| **Esports Gaming**| Competitive gaming, ~$800 budget | Obsolete 4GB RAM + entry-level iGPU | `UNSUITABLE` |
| **AI Graduate** | CUDA acceleration, 32GB RAM, <$1,800 | RTX 4070 (8GB VRAM) + 32GB DDR5 | `SUITABLE` |
| **Lecture Mobility**| Quiet, portable, long battery life | 8-lb 17" desktop-replacement (2-hr battery) | `UNSUITABLE` |

---

## 📊 Evaluation Results & Analysis

Running the pipeline through Evidently AI's `Report([TextEvals()])` yielded a **50% approval rate (5 Passed / 5 Failed)**.

### Evaluation Criteria Breakdown:
1. **Hardware & Workload Alignment:** Rejects severely underpowered hardware for demanding computing tasks (e.g., neural network training on non-CUDA architectures).
2. **Budget Constraint Adherence:** Rejects recommendations that grossly exceed target budgets.
3. **Contextual & Form-Factor Fit:** Balances weight, thermals, and battery runtimes according to mobility preferences.

The accompanying output tables capture the exact **reasoning** behind each judgment, validating model interpretability.

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/Lama-Alfaifi/SDA-Agentic-AI-Bootcamp-Challenges.git](https://github.com/Lama-Alfaifi/SDA-Agentic-AI-Bootcamp-Challenges.git)
   cd SDA-Agentic-AI-Bootcamp-Challenges

2. **Install Dependencies:**

   ```bash
    pip install "evidently[llm]" litellm pandas

3. **Set API Key:**

   ```bash
    export OPENAI_API_KEY="your-api-key-here"
(If using Google Colab, store it securely in userdata.get('OPENAI_API_KEY')).

4. **Execute the Notebook:**
Open LLM_as_a_Jury_with_Evidently_AI_(demo).ipynb and execute all cells sequentially to generate the evaluation report.
