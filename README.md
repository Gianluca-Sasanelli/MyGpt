# Reproducing a GPT Model

# GPT Model Reproduction Project

This repository contains my project to reproduce a GPT model. I followed mostly Karpathy's settings of the model and the train code. The model has been trained on two datasets: the Tiny Shakespeare dataset and the DailyMail dataset.

## Datasets and Tokenization

### 1. Tiny Shakespeare Model
- **Tokenizer:** Simple tokenizer that translates each character into an integer.
- **Vocabulary Size:** 65
- **Model Size:** 3M parameters
- **Training Time:** Less than 5 minutes

### 2. DailyMail Model
- **Tokenizer:** OpenAI GPT-2 tokenizer
- **Vocabulary Size:** 50,257
- **Model Size:** 88M parameters
- **Training Time:** 7 hours (stopped before reaching maximum accuracy)

## How to Use

### Training

To train your model, use the following command:

```bash
python train.py --config $configPath
```

## Sample Output from the Tiny Shakespeare Dataset

Below is an example of the model's output when trained on the tiny Shakespeare dataset:

*CORIOLANUS:  
If should desire: see he here thoughts, as let  
most she the shame; here's it have paid too. He give  
a it poison, by the trull; I do none worse it  
notious noty, done to half dissolves, hath  
varvetulla bean you ever by an a man's loss a  
as burnm, that use if my remlioned see so.*

*LATTINGE:  
Brach, windneed, though know you.*

## Loss Plot for tiny shakespeare

The plot below visualizes the loss over time during training:

![Loss Plot](https://github.com/Gianluca-Sasanelli/mygpt/blob/main/assets/losses-tinyshakespeare.png)


## Sample output from the dailymails dataset:

*The U.S. Embassy in Washington, Virginia, said it would be a "great relief" for the government to take the country. "We are not going to make sure that we have a lot of time to do," said Rep. Michael D.C. CNN's Elise Labott contributed to this report.<|endoftext>|Two people have been arrested in the country's capital city of Raqqa, a city in the city of Raqqa, Syria, in the area. The suspected gunman, who was arrested on suspicion of being in the country, is charged with first-degree murder, aggravated murder and aggravated kidnapping. The suspect, who was caught in a car in the city's capital, is charged with first-degree murder and aggravated murder by the force. Scroll down for video*

## Loss Plot for dailymails

![Loss Plot](https://github.com/Gianluca-Sasanelli/mygpt/blob/main/assets/losses-dailymails.png)



