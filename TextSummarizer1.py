
from simplet5 import SimpleT5


model =SimpleT5()
model.from_pretrained("t5","t5-base")
model.train(train_df=train_df,
            eval_df=eval_df,
            source_max_token_len=512,
            target_max_token_len=128,
            batch_size=8,
            max_epochs=5,
            use_gpu =True,
            outputdir ="outputs",
            early_stopping_patience_epochs =0)
import pandas as pd
from sklearn.model_selection import train_test_split

path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
df = pd.read_csv(path)
df.head()
# let's load the trained model for inferencing:
model.load_model("t5","outputs/SimpleT5-epoch-2-train-loss-0.9526", use_gpu=True)

text_to_summarize="""summarize: Rahul Gandhi has replied to Goa CM Manohar Parrikar's letter,
which accused the Congress President of using his "visit to an ailing man for political gains".
"He's under immense pressure from the PM after our meeting and needs to demonstrate his loyalty by attacking me,"
Gandhi wrote in his letter. Parrikar had clarified he didn't discuss Rafale deal with Rahul.
"""
model.predict(text_to_summarize)

# --> model quantization & ONNX support

# for faster inference on cpu, quantization, onnx support:
model.convert_and_load_onnx_model(model_dir="outputs/SimpleT5-epoch-2-train-loss-0.9526")
model.onnx_predict(text_to_summarize)