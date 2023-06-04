
#%%
# import packages

from pathlib import Path
import pandas as pd



#%%
# Load data

# Loop through all folders in output/scores
files = []
for folder in Path('output/scores/').iterdir():

    # Skip if not a folder
    if not folder.is_dir():
        continue

    
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
print(df.to_latex(index=False, float_format="%.3f", caption="Puntuación de cada modelo."))



# %%
