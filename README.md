# llm-project
My own project trying to reproduce a gpt model. I trained it on the CNN articles dataset found on kaggle. The dataset has been tokenize with the gpt tokenizer. 
I followed mostly the paper and I geot some help looking at Karpathy repository ( especially in the training phase)
I have trained various model with different number of parameters, trying to understand where the model starts to write something coherent. Models with few parameters (with 128 or 256 of embedding) don't get easily under a loss of 3.
