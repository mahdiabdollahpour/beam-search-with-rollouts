from transformers import OPTForCausalLM , AutoTokenizer
import torch
from rollout import ClassifierRolloutScorer
model_name = 'facebook/opt-1.3b'
tokenizer_name = 'facebook/opt-1.3b'


model = OPTForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)




model.to('cuda')

input_text = 'The men started swearing at me, called me'


input_len = len(input_text)
encoding =tokenizer.batch_encode_plus([input_text], return_tensors="pt").to('cuda')
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

# usuall beam search
beam_output =  model.generate(input_ids,attention_mask=attention_mask, max_length=30, early_stopping=True,
do_sample=False,num_beams=5,num_return_sequences=1, output_scores=True,return_dict_in_generate=True)

texts = tokenizer.batch_decode(beam_output['sequences'], skip_special_tokens=True)

continuations = [tt[input_len:] for tt in texts]


# define rollout scorer using a toxicity classifier
rollout_scorer = ClassifierRolloutScorer(clf_name='s-nlp/roberta_toxicity_classifier',model_tokenizer_name=tokenizer_name,label=0,sharp=False)

scores = rollout_scorer.classifier.get_scores(texts)

for s,t in zip(scores,texts):
    print('Toxicity:',1-s)
    print('text:',t)
    print('*'*50)


print('^'*50)



# beam search with rollouts
beam_output =  model.generate(input_ids,attention_mask=attention_mask,roll_out_scorer=rollout_scorer, max_length=30, 
do_sample=False,num_beams=5,num_return_sequences=1, output_scores=True,return_dict_in_generate=True,branching_factor=20,rollout_length=10)



texts = tokenizer.batch_decode(beam_output['sequences'], skip_special_tokens=True)

scores = rollout_scorer.classifier.get_scores(texts)

for s,t in zip(scores,texts):
    print('Toxicity:',1-s)
    print('text:',t)
    print('*'*50)
