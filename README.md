# Exploring LLM Finetuning for Software Testing 

This repo serves as a high-level training and inference pipeline to train or finetune instruction-tuned language models on software testing datasets

An example of such a dataset is the 'methods2test' dataset.

## Motivation

An important aspect of any software development cycle is writing a variety of tests. These tests can be cumbersome to cover, and are often incomplete and left aside by teams. Thus, leveraging LMs for writing test cases automatically, as your code repository goes, is one of the most useful jobs that can be handed over to language models.

The current problem occurs in determining what sort of model would be considered 'good enough' for the tasks being done. This becomes an even bigger issue when you consider the lack of test-suite datasets such as 'methods2test' or 'Defects4J'. Another problem is the lack of variety languages for which a dataset might exist. Both the former datasets discussed focus on Java alone, and the spread of such datasets remains scarce.

One of the problems that can be tackled is to be able to determine the most efficient model that would be appropriate for a team. Instead of resorting to paying for huge generally-trained models such as ChatGPT, or even resorting to local, yet huge models (>13B), it would be easier if we can determine if a 1B model would be enough for the type of testing needed to be done.

A great example is the results of methods2test on a simple Llama 1B vs 3B train test. Both models achieve the same parse rate and codebleu score post-finetuning. Ideally, a team working closely with Java tests could do as well with the 1B model, saving both time and money.

## Repo Design

The pipeline uses YAML scripts to populate all the variables needing attention into a single file. 

You can locate them in the config folder, and populate your own as needed, following the same formatting for a different model. 

The train and inference sripts are in their respective folders within the 'src' folder.

Finetuned models are stored in outputs, and can be called for inference against the ground truths as needed.

Additionally, datasets are auto-sliced into train-eval-test splits so it would be recommended to pass the main dataset path only.


## Usage

1. Setup the YAML scripts as the one shown in the config folder.
2. Populate any preprocessing scripts to process the datasets. Different models can follow different formattings. An example or Llama is provided
3. Run the train script to train/fine-tune the model. You can use either 'accelerate' or 'torchrun' to achieve multi-GPU
4. You can then run the inference script and choose to save the predictions. It auto runs metrics in the end of the inference cycle.
5. Alternatively, you can run the 'run_metrics' script with the original dataset and the predictions from a model to get metrics then and there

 
