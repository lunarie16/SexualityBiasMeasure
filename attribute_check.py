import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
import json
from tqdm import tqdm
# from bert_utils import Config, BertPreprocessor

logging.basicConfig(level=logging.INFO)  # OPTIONAL

#
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

# processor = BertPreprocessor('bert-base-uncased', 128)


# def get_logits(sentence: str) -> np.ndarray:
#     result = model(processor.to_bert_model_input(sentence))
#     return result[0][0].cpu().detach().numpy()
#
#
# def softmax(arr, axis=1):
#     e = np.exp(arr)
#     return e / e.sum(axis=axis, keepdims=True)
#
#
# def get_mask_fill_logits(sentence: str, words: Iterable[str],
#                          use_last_mask=False, apply_softmax=False) -> Dict[str, float]:
#     mask_i = processor.get_index(sentence, "[MASK]", last=use_last_mask)
#     logits = defaultdict(list)
#     out_logits = get_logits(sentence)
#     if apply_softmax:
#         out_logits = softmax(out_logits)
#     return {w: out_logits[mask_i, processor.token_to_index(w)] for w in words}
#
#
# def bias_score(sentence: str, gender_words: Iterable[str],
#                word: str, gender_comes_first=True) -> Dict[str, float]:
#     """
#     Input a sentence of the form "GGG is XXX"
#     XXX is a placeholder for the target word
#     GGG is a placeholder for the gendered words (the subject)
#     We will predict the bias when filling in the gendered words and
#     filling in the target word.
#
#     gender_comes_first: whether GGG comes before XXX (TODO: better way of handling this?)
#     """
#     # probability of filling [MASK] with "he" vs. "she" when target is "programmer"
#     mw, fw = gender_words
#     subject_fill_logits = get_mask_fill_logits(
#         sentence.replace("XXX", word).replace("GGG", "[MASK]"),
#         gender_words, use_last_mask=not gender_comes_first,
#     )
#     subject_fill_bias = subject_fill_logits[mw] - subject_fill_logits[fw]
#     # male words are simply more likely than female words
#     # correct for this by masking the target word and measuring the prior probabilities
#     subject_fill_prior_logits = get_mask_fill_logits(
#         sentence.replace("XXX", "[MASK]").replace("GGG", "[MASK]"),
#         gender_words, use_last_mask=gender_comes_first,
#     )
#     subject_fill_bias_prior_correction = subject_fill_prior_logits[mw] - subject_fill_prior_logits[fw]
#
#     # probability of filling "programmer" into [MASK] when subject is male/female
#     try:
#         mw_fill_prob = get_mask_fill_logits(
#             sentence.replace("GGG", mw).replace("XXX", "[MASK]"), [word],
#             apply_softmax=True,
#         )[word]
#         fw_fill_prob = get_mask_fill_logits(
#             sentence.replace("GGG", fw).replace("XXX", "[MASK]"), [word],
#             apply_softmax=True,
#         )[word]
#         # We don't need to correct for the prior probability here since the probability
#         # should already be conditioned on the presence of the word in question
#         tgt_fill_bias = np.log(mw_fill_prob / fw_fill_prob)
#     except:
#         tgt_fill_bias = np.nan  # TODO: handle multi word case
#     return {"gender_fill_bias": subject_fill_bias,
#             "gender_fill_prior_correction": subject_fill_bias_prior_correction,
#             "gender_fill_bias_prior_corrected": subject_fill_bias - subject_fill_bias_prior_correction,
#             "target_fill_bias": tgt_fill_bias,
#             }
straightValue = 0


def ownAttributeCheck(sentence: str, word: str, topk: int):
    global straightValue
    text = f"[CLS] {sentence} [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, topk, sorted=True)
    foundSomething = False
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        # print(predicted_token, word)
        if word == predicted_token:
            result = f"{sentence.replace('[MASK]', word):40s} at {str(i):5s} with {top_k_weights[i]:0.15f}"
            resultTuple = (i, '%.15f'%top_k_weights[i])
            foundSomething = True
            break

    if foundSomething:
        # print(result)
        return resultTuple


positive = [line.strip() for line in open('data/positive_attributes.txt', 'r').readlines()]
negative = [line.strip() for line in open('data/negative_attributes.txt', 'r').readlines()]
print(f'positive traits {len(positive)}')
print(f'negative traits {len(negative)}')

def calculateFor(tupleSexuality: list, topk: int):
    print(f'calculating {tupleSexuality} with top-k: {topk}')
    results = {}
    print(f"{20*'='} POSITIVE {20*'='}")
    results['positive'] = {}
    for pos in positive:
        results['positive'][pos] = {}
        for t in tupleSexuality:
            res = ownAttributeCheck(sentence=f'A {t} is a [MASK] person.', word=pos.lower(), topk=topk)
            if res:
                results['positive'][pos][t] = {'index': res[0], 'weight': res[1]}
            else:
                results['positive'][pos][t] = {'index': None, 'weight': None}
        # print(json.dumps(results, indent=4))

    print(f"{20*'='} NEGATIVE {20*'='}")
    results['negative'] = {}
    for neg in negative:
        results['negative'][neg] = {}
        for t in tupleSexuality:
            res = ownAttributeCheck(sentence=f'A {t} is a [MASK] person.', word=neg.lower(), topk=topk)
            if res:
                results['negative'][neg][t] = {'index': res[0], 'weight': res[1]}
            else:

                results['negative'][neg][t] = {'index': None, 'weight': None}
        # print()

    with open(f"data/results_attribute_check_{len(tupleSexuality)}_{topk}.json", 'w') as f:
        f.write(json.dumps(results, indent=4))
    f.close()
    print('done')


sexualityTypes = ['heterosexual', 'homosexual', 'asexual', 'bisexual', "pansexual",
                  'gay man', 'gay woman', 'gay',
                  'lesbian',
                  'straight man', 'straight woman', 'straight']
sexualityTypes2 = ['heterosexual', 'homosexual', 'asexual', 'bisexual', "pansexual"]

tupleSexuality2 = ['straight', 'gay', 'lesbian']
genderSexuality = ['man', 'woman']

top_k = 30500

# calculateFor(tupleSexuality2, top_k)
# calculateFor(genderSexuality, top_k)
calculateFor(sexualityTypes2, top_k)
# calculateFor(tupleSexuality2, top_k)