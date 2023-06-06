
#%%
# import packages

from pathlib import Path
import pandas as pd



#%%
# Load data

# Loop through all folders in output/scores
files = []
for folder in sorted(list(Path('output/scores/').glob('*'))):

    # Skip if not a folder
    if not folder.is_dir():
        continue

    
    # if folder.name == "blip2_opt_coco_game_finetuned":
    #     continue

    file = folder / f"scores_{folder.name}.csv"

    data = pd.read_csv(file, sep=' ', header=0)
    data.insert(0, "Name", folder.name, True)
    # data.columns=[folder.name]

    files.append(data)


#%%
# Concatenate all dataframes
df = pd.concat(files, axis=0, ignore_index=True)

# %%
# Save to csv
print(df)
df.to_csv('output/scores.csv', index=False)


# %%
# Convert to latex table

names = {
    "blip2_opt_caption_coco_opt2.7b": "OPT COCO 2.7b",
    "blip2_opt_caption_coco_opt6.7b": "OPT COCO 6.7b",
    "blip2_opt_pretrain_opt2.7b": "OPT 2.7b",
    "blip2_opt_pretrain_opt6.7b": "OPT 6.7b",
    "blip2_t5_caption_coco_flant5xl": "T5 COCO XL",
    "blip2_t5_pretrain_flant5xl": "T5 XL",
    "blip2_t5_pretrain_flant5xxl": "T5 XXL"
    # "blip2_opt_coco_game_finetuned": "OPT 6.7b + FT",


}

new_df = df.copy()
new_df = new_df.drop(columns=['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4'])
new_df['Name'] = new_df['Name'].replace(names)

print(new_df.to_latex(index=False, float_format="%.3f", caption="Puntuación de cada modelo.", label="table:scores_all", column_format="lcccccc"))



# %%
