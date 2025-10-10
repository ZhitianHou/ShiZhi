<p align="center">
  <img src="images/ShiZhi_logo.png" alt="ShiZhi Logo" width="160"/>
</p>

# ShiZhi: A Lightweight Large Model for Court View Generation

<p align="center">
  <a href="https://huggingface.co/TIM0927/ShiZhi">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFDD33?label=ShiZhi&labelColor=1a73e8" alt="Hugging Face Link"/>
  </a>
  <a href="https://huggingface.co/datasets/TIM0927/CCVG">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-FFDD33?label=CCVG&labelColor=1a73e8" alt="Hugging Face Link"/>
  </a>
</p>

**ShiZhi (释之)** is a lightweight large language model designed for **Criminal Court View Generation (CVG)** in Chinese.  
Its name comes from the historical figure **Zhang Shizhi (张释之)**, and in Chinese, “释之” also conveys the meaning of “explaining” or “interpreting,” which is particularly suitable for generating the *court view* section in legal case documents.

---

## ⚙️ Model Training

ShiZhi is fine-tuned on the **CCVG** dataset using a 0.5B-parameter instruction-tuned LLM as the base model.  
The training pipeline includes data curation, prompt construction, and instruction tuning tailored for CVG and charge prediction tasks.  

<p align="center">
  <img src="images/Pipeline.png" alt="Training Pipeline" width="70%"/>
</p>

## 📊 Model Performance

The performance of ShiZhi on **court view generation** and **charge prediction** is summarized below:

| **Models** | **ROUGE-1** | **ROUGE-2** | **ROUGE-L** | **BLEU-1** | **BLEU-2** | **BLEU-N** | **Accuracy** | **Macro-F1** |
|-------------|-------------|-------------|-------------|-------------|-------------|-------------|---------------|---------------|
| Qwen2-0.5B-Instruct | 0.0 | 0.0 | 0.0 | 11.1 | 6.3 | 2.9 | 17.3 | 29.5 |
| **ShiZhi** | **6.3** | **0.8** | **6.3** | **58.5** | **51.0** | **41.7** | **86.5** | **92.8** |

ShiZhi substantially outperforms the base model across both **BLEU/ROUGE** metrics and **charge prediction (Accuracy, Macro-F1)**.

## 📚 Dataset: CCVG

ShiZhi is trained on **CCVG**, a curated dataset of over **110K Chinese criminal cases**.  Each case includes a **fact description** and its corresponding **court view**, supporting both **court view generation** and **charge prediction** tasks. Below are some **visualizations of the dataset**, including examples of fact–court view pairs and length statistics.

<p align="center">
  <img src="images/year_statistic.png" alt="Fact Example" width="70%"/><br/>
  <img src="images/length_statistic.png" alt="Court View Example" width="70%"/><br/>
  <img src="images/charge_statistic.png" alt="Dataset Statistics" width="70%"/>
</p>

---

Feel free to explore, fine-tune, or evaluate ShiZhi on your own legal AI tasks.

---

📄 *The technical report is coming soon.*
