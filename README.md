<h1 align="center">Comparing Inferential Strategies of Humans and LLMs in Deductive Reasoning</h1>

<div style="text-align: center; width: 100%;">
  <!-- Container to align the image and the caption -->
  <div style="display: inline-block; text-align: left; width: 85%;">
    <img src="assets/images/automated_reasoning_banner.webp" style="width: 100%;" alt="automated reasoning">
    <!-- Caption below the image -->
    <p style="color: gray; font-size: small; margin: 0;">
      <em>Generated by DALL·E 3</em>
    </p>
  </div>
</div>

<br>

Deductive reasoning plays a pivotal role in the formulation of sound and cohesive arguments. It allows individuals to draw conclusions that logically follow, given the truth value of the information provided. Recent progress in the domain of large language models (LLMs) has showcased their capability in executing deductive reasoning tasks. Nonetheless, a significant portion of research primarily assesses the accuracy of LLMs in solving such tasks, often overlooking a deeper analysis of their reasoning behavior. In this project, we draw upon principles from cognitive psychology to examine inferential strategies employed by LLMs, through a detailed evaluation of their responses to propositional logic problems.

This repository contains code to evaluate the reasoning behavior of LLMs in problems of propositional logic. In particular, we assess inferential strategies employed by LLMs on 12 tasks suggested by [Van der Henst et al. (2002)](https://doi.org/10.1207/s15516709cog2604_2). The problem formulations for each task can be found in our respective [HuggingFace data repository](https://huggingface.co/datasets/mainlp/henst_prop_logic).

## Table of Contents
- [Setup](#setup)
- [Instructions](#instructions)
- [Citation](#citation)

## Setup
All code was developed and tested on Ubuntu 22.04 with Python 3.11.6.

To run the current code, we recommend to use Poetry:
```bash
poetry install                          # Install dependencies
poetry shell                            # Activate virtual environment
# Work for a while
deactivate
```

## Instructions
To run the code, you can use the following command:
```
python run.py --model <model-name> --num-samples <num-samples>
```
where `<model-name>` is the name of the LLM to be used and `<num-samples>` is the number of random seeds across which the model should be evaluated. In this project, we used the following models:

- [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)
- [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
- [meta-llama/Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)

Note that in order to use a new model, you need to add a configuration file in this [folder](reasoning_strategies/models/model_config).

## License
[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a [Creative Commons Attribution 4.0 International License][cc-by].

[cc-by]: https://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


## Citation

If you find this work helpful, please our paper as:
```
@misc{mondorf2024comparing,
      title={Comparing Inferential Strategies of Humans and Large Language Models in Deductive Reasoning}, 
      author={Philipp Mondorf and Barbara Plank},
      year={2024},
      eprint={2402.14856},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[def]: #table-of-contents