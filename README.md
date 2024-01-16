# How to do multiple choice question with RAG?
That's the question I asked myself when I participated on a LLM competition that ends in a week.

## The story
* I found out about the competition only a week before the competition closed. I only participated because **I wanted to know RAG better**. I knew I had no chance of winning xd.
    * I decided to *build a noob RAG system from scratch* to solve the competition.

## The competition
https://challenge.kalapa.vn/portal/vietnamese-medical-question-answering/overview
* The competition provided dataset with information (causes, symptoms, prevention method, etc.) about ~600 diseases in Vietnamese.
* Participants use the dataset to answer multiple-choice questions and there exists more than 1 correct answer.
* **Required output**: a binary string with length `n` with element `i`th is 0 if choice `i`th in the question is incorrect, 1 if it's correct.

| Question |	Expected Answer|
|--------- | ---------------|
|Đâu là triệu chứng của bệnh van tim? A. Khó thở B. Tăng cân nhanh chóng C. Vàng da D. Rụng tóc | `1100`

## The questions

### Why I have to implement the RAG from scratch?
1. At that moment, I don't know how to use the existed libraries like `langchain` and `llamaindex` (I now kinda "pro" in `llamaindex`, believe me) and it's only less than a week before the competition ended (too short for me to learn anything new).
2. I want to understand deeply each part in a RAG system (how many parts a simple RAG have, how to indexing, get embeddings, retrieve context, etc.).

### How to limit files to retrieve for each question?
* Because there are ~600 files/diseases and if there are 100 questions to answer then I'd have to do `semantic_search` for like 60000 times. God, I only had Colab's free T4s (my Mac chips suck at working with LLM, `mps` is fully supported yet).
* So I used the `question` for each answer, I compared its similarity with ~600 file/disease **names** (not the content of the file!).
    * More specific, I used `ROUGE-l` to measure their similarity. It's like matching each word in the `question` with words in the `disease_name`.
    * **Even more improvement**: I also used the `choices` to compare in case the question doesn't contain enough information.
* Then, I only takes `top-2` files as context for each question. And for each file/disease, I search for `top-3` most related chunk of texts.

### How to force the LLM to output binary string?

* For each possible choice, "ask" the model if this is a correct answer (output 1) or not (output 0).
```python
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
``` 
* Since the model only "talks" in probability and numbers, I had to find a way to know is it saying that the answer is correct or not. I did that by using the `logits` output by the model and use a `softmax` function to get the probabilities.
```python
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
            torch.tensor([
                logits[incorrect_str_token],
                logits[correct_str_token]
            ]).float(),
            dim=0
        ).detach().cpu().numpy()
    return {0: incorrect_str, 1: correct_str}[np.argmax(probs)], probs
```

# The result
* Final score: 0.6468 (rank 21/90 🫣).
* Break down:
    * 0.2744: I used a Vietnamese GPT model, 1-shot prompt, both English and Vietnamese in a lengthy, redundant prompt.
    * 0.47: switch to `bloomz-7b1-mt`, Vietnamese prompt, 1-shot prompt.
    * 0.5915: fix cases that model only output zeros (by taking the answer with highest probability of outputing the correct string).
    * 0.6118: fix some coding bugs.
    * 0.6468: 2-shot prompt, modify prompt punctuations and stuffs.
* Tried but failed (why I am writing like I am the winner??)
    * [x] increase `k` in topk files 
    * [x] increase few-shot prompt (>2)
    * [x] fine-tune the 4-bit bloomz model as a classifier to output 1 for correct answer, 0 otherwise. (the baseline score increased but I don't know how to prompt it correctly).
* Skills learned:
    * [x] use `sentence-transformer` to do semantice search and retrieval.
    * [x] implement basic `sentence-splitter` (same as `llamaindex`).
    * [x] how to do multiple choice question with LLM.
    * [x] be creative and apply small tricks to largely improve the result. 
