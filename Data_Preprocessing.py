## You can substitute below data preprocessing code with the data preprocessing cell in "SFR_mistral_IMDB.ipynb" to swich the dataset to run.

import csv
from struct import pack, unpack
import lzma
import numpy as np
import pickle
import torch

name_tt = {}
nconst_name = []
nconst_profession = []
nconst_embedding = []
objects = []
prompts = []

device = torch.device("cuda")

# set 'dataset' as one of ["IMDB_name.basics", "nobel-prize-laureates", "basketball_players", "books"]
with open("Data/"+dataset+".csv", mode='r', encoding='utf-8') as fd:
    rd = csv.reader(fd)
    counter = 0
    
    for row in rd:
        if counter > 1000:
            break
        if counter > 0: 
            if dataset == "IMDB_name.basics":
                prompt = row[1] +" was born in "+row[2]+", "+"and died in "+row[3]+". He/She's primary professions are "+', '.join(map(str, row[4].split(",")))+"."
                tts = row[5].split(",")
                prompt += " He/She is known for movies:"
                for t in tts:
                    if t == tts[-1]:
                        prompt += " '" + tconst_title[t]+"'."
                    else:
                        prompt += " '" + tconst_title[t]+"',"

            elif dataset == "nobel-prize-laureates":
                lst = row[0].split(";")
                if len(lst) < 10:
                    continue
                if len(lst) < 17:
                    if lst[11] == "male":
                        prompt = lst[1] + " " + lst[2] + " was born in " + lst[5] + ", " + lst[3] + ". And died in " + lst[6] + ", " + lst[4] + ". He won Nobel prize in " + lst[13] + ", " + lst[12] + ", " + lst[15][3:-3] + "."
                    else:
                        prompt = lst[1] + " " + lst[2] + " was born in " + lst[5] + ", " + lst[3] + ". And died in " + lst[6] + ", " + lst[4] + ". She won Nobel prize in " + lst[13] + ", " + lst[12] + ", " + lst[15][3:-3] + "."
                    
                else:
                    if lst[11] == "male":
                        prompt = lst[1] + " " + lst[2] + " was born in " + lst[5] + ", " + lst[3] + ". And died in " + lst[6] + ", " + lst[4] + ". He won Nobel prize in " + lst[13] + ", " + lst[12] + ", " + lst[15][3:-3] + ". He work in " + lst[16] + ", " + lst[17] + " " + lst[18] + "."
                    else:
                        prompt = lst[1] + " " + lst[2] + " was born in " + lst[5] + ", " + lst[3] + ". And died in " + lst[6] + ", " + lst[4] + ". She won Nobel prize in " + lst[13] + ", " + lst[12] + ", " + lst[15][3:-3] + ". She work in " + lst[16] + ", " + lst[17] + " " + lst[18] + "."
                        
            elif dataset == "basketball_players":
                prompt = row[1] +", born on "+row[2]+", and played for "+row[3]+". "+row[1]+" has been honored with the " + row[4] + "."
            elif dataset == "books":
                prompt = "'" + row[1] + "'" + " was publicated at " + row[2] + ", writen by " + row[3] + ". Overall, it is a " + row[4] + ", and it can be accessed at this URL: " + row[0] 
                
            max_length = 526
            input_texts = prompt
            batch_dict = tok(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**batch_dict)
                embedding = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).to("cpu")
        
            nconst_embedding.append(embedding[0])
            prompts.append(prompt)
            objects.append(lst[1] + " " + lst[2])
                                    
        counter += 1