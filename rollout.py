
import torch
import numpy as np
import torch.nn.functional as F
import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class BaseRolloutScorer():

    def __init__(self, model_tokenizer_name: str='gpt2'):
        self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)


    def get_scores(self,rollout_output) -> torch.FloatTensor:
        """
        doc
        """
        pass


    def decode(self,input_ids):
        decoded_sequences = self.model_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        return decoded_sequences



class ClassifierRolloutScorer(BaseRolloutScorer):


    def __init__(self, clf_name: str='BaxterAI/SentimentClassifier', model_tokenizer_name: str='gpt2',label=0, sharp=False):
        super().__init__(model_tokenizer_name)
        
        self.classifier = Classifier(clf_name,sharp=sharp,label=label)
        self.sharp = sharp

    def get_scores(self,rollout_output,original_input_ids=None) -> torch.FloatTensor:

        orig = self.decode(original_input_ids)
        # print('orig',orig)
        orig_len  = len(orig[0])
        decoded_sequences = self.decode(rollout_output['sequences'])

        decoded_sequences = [dc[orig_len:] for dc in decoded_sequences]
        scores = self.classifier.get_scores(decoded_sequences)
        # print(decoded_sequences)
        # print(scores)
        scores = torch.tensor(scores).to(rollout_output['sequences'].device)
  
        return scores



class Classifier():
    def __init__(self, clf_name: str='BaxterAI/SentimentClassifier',label=0,sharp=False):
        self.tokenizer = AutoTokenizer.from_pretrained(clf_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(clf_name).cuda()
        self.label = label
        self.sharp = sharp

    def get_scores(self,preds,batch_size = 5):
        
        results = []
        for i in range(0, len(preds), batch_size):
            batch = self.tokenizer(preds[i:i + batch_size], return_tensors='pt', padding=True)
            with torch.no_grad():
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda')
                # result = self.model(**batch)['logits'][:,1].float().data.tolist()
                probs = F.softmax(self.model(**batch)['logits'], dim=1).float().cpu().numpy()
                
            # print(probs.shape)
            if self.sharp:
                result = np.heaviside(probs[:,self.label]-0.5,0)
                
            else:
                result = probs[:,self.label]
            results.extend(result)
        return results 
