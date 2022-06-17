from transformers import BartTokenizer, BartForConditionalGeneration
from rouge import Rouge
import pandas as pd
import numpy as np
import math

MODEL_PATH = "./data/tst-summarization"
TEST_PATH = "./data/test.csv"
BATCH_SIZE = 20
WRITE_INTERVAL = 100
OUTPUT_FILE = "cs-results.csv"

if __name__ == "__main__":
    test_df = pd.read_csv(TEST_PATH)

    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    rouge = Rouge()

    num_splits = math.floor(test_df.shape[0] / BATCH_SIZE)
    p_df = pd.DataFrame(columns=["Title", "Prediction", "Score"])
    for counter, batch in enumerate(np.array_split(test_df, num_splits)):
        print(str(counter+1) + " / " + str(num_splits))
        abstracts = batch.iloc[:,0].tolist()
        titles = batch.iloc[:,1].tolist()
        inputs = tokenizer(abstracts, max_length=1024, truncation=True, padding=True, return_tensors="pt").input_ids
        output_ids = model.generate(inputs, min_length=0, max_length=30)
        predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        scores = list()
        for pair in zip(titles, predictions):
            scores.append(rouge.get_scores(pair[0], pair[1])[0]["rouge-1"]["p"])

        temp_df = pd.DataFrame(columns=["Title", "Prediction", "Score"])
        temp_df["Title"] = titles
        temp_df["Prediction"] = predictions
        temp_df["Score"] = scores
        p_df = pd.concat([p_df, temp_df], axis=0)
        if counter % WRITE_INTERVAL == 0:
            p_df.to_csv("data/" + OUTPUT_FILE, index=False)

    p_df.to_csv("data/" + OUTPUT_FILE, index=False)