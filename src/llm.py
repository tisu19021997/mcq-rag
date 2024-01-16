
import numpy as np
import torch

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

instruct_prompt = (
    "Bối cảnh: {context}\n\n"
    # Just use random few-shot here, it improves the stability of model output.
    "Câu hỏi: Cách nào để phòng ngừa bệnh tim mạch?\n"
    "Câu trả lời đề xuất: Tập thể dục đều đặn.\n"
    "Điểm:1\n\n"

    "Câu hỏi: {question}\n"
    "Câu trả lời đề xuất: {choice}\n"
    "Điểm:"
)


def load_llm(model_path="bigscience/bloomz-7b1-mt"):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.init_device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, config=config, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto',
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    return model, tokenizer


def is_the_answer_correct(model, tokenizer, prompt, correct_str="1", incorrect_str="0"):
    """
    Given the context, question and proposed answer in the prompt, 
    predict if the answer is correct or not based on the probability that it outputs the correct vs incorrect string.
    """
    input_ids = tokenizer(prompt, return_tensors="pt")['input_ids']
    with torch.no_grad():
        # Get the logit at the end of the sentence.
        logits = model(input_ids=input_ids).logits[0, -1]
        correct_str_token = tokenizer(correct_str).input_ids[-1]
        incorrect_str_token = tokenizer(incorrect_str).input_ids[-1]

        # Squeeze the logit on 2 tokens and put it through a softmax.
        probs = torch.nn.functional.softmax(
            torch.tensor([logits[incorrect_str_token],
                         logits[correct_str_token]]).float(),
            dim=0
        ).detach().cpu().numpy()
    return {0: incorrect_str, 1: correct_str}[np.argmax(probs)], probs


def predict(row, model, tokenizer, retrieved_context_dict, correct_str="1", incorrect_str="0"):
    """Output an array of ones and zeros where ones are believed to be correct answers."""
#     row = df.iloc[idx]
    doc_id = row['id']
    question = row['question']

    # Let's first try only closest context.
    context_list = retrieved_context_dict[doc_id][0]
    retrieved_context = '.'.join(context_list)

    # Add a period at the end of the context.
    if retrieved_context[-1] != '.':
        retrieved_context += '.'

    preds = []
    probs = []
    for i in range(1, 7):
        choice = row[f'option_{i}_pp']
        # Check because the number of answers is not fixed.
        if isinstance(choice, float):
            break

        # Add a period at the end of the proposed answer.
        if choice[-1] != '.':
            choice += '.'
        prompt = instruct_prompt.format(
            context=retrieved_context, question=row['question'], choice=choice)
        pred, prob = is_the_answer_correct(
            model, tokenizer, prompt, correct_str, incorrect_str)

        print(f'Iteration #{i}')
        print(prompt + pred + "\n")
        print('=' * 20, '\n')

        preds.append(pred)
        probs.append(prob)
    # TODO: give model information on previous answers?

    # If there is no correct answer yet, use the answer with the highest probability of outputing correct string.
    if correct_str not in preds:
        correct_idx = np.argmax([p[1] for p in probs], axis=0)
        preds[correct_idx] = correct_str

    return preds
