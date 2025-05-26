import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import faiss
import random
import os
import sys
import torch
import torch.nn.functional as F

device = torch.device("cuda")

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def get_embedding(sentence, model, tokenizer):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    # print("tokens: ", len(tokens["input_ids"][0]))
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
    # Assuming we're using the mean of the last hidden state as the embedding representation
    embedding = output.hidden_states[-1][0].mean(dim=0, keepdim=True)
    return embedding.to(device)

def perturb_sentence(sentence, index):
    words = sentence.split()
    if 0 <= index < len(words):
        words[index] = "[MASK]"
    return " ".join(words)

def rank_tokens_by_importance(sentence, model, tokenizer):
    original_embedding = get_embedding(sentence, model, tokenizer)
    distances = []
    
    for i in range(len(sentence.split())):
        perturbed = perturb_sentence(sentence, i)
        perturbed_embedding = get_embedding(perturbed, model, tokenizer)
        
        # Computing cosine distance
        # distance = cosine_distances(original_embedding, perturbed_embedding)[0][0]
        # Computing MSE distance
        distance = torch.nn.MSELoss()(original_embedding, perturbed_embedding).item()
        distances.append(distance)
        
    ranked_indices = np.argsort(distances)[::-1]
    words = sentence.split()
    ranked_words = [words[i] for i in ranked_indices]
    
    return ranked_words

def get_initial_ids(important_tokens, sentence, firstk, MODEL_NAME, model, tok):
    top_k_tokens = important_tokens[:firstk]
    
    # Sort them by their order in the original sentence
    sorted_top_k_tokens = sorted(top_k_tokens, key=lambda x: sentence.split().index(x))

    intital_sentence = ""
    for token in sorted_top_k_tokens:
        intital_sentence += (token + " ")

    if MODEL_NAME == "mistralai":
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"][:, 1:-1].to(device)[0]
    else:
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0]
    return initial_ids[:firstk]

def get_initial_ids_logits(MODEL_NAME, user_prompt, desired_token, firstk, model, tok):
    name = user_prompt.split("was")[0].split("(")[0]
    if MODEL_NAME == "EleutherAI/gpt-j-6b":
        intital_sentence = name + desired_token[1:]
        print("intital_sentence: ", intital_sentence)
        
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0].tolist()
        print("initial_ids length: ", len(initial_ids))
        if firstk > len(initial_ids):
            pad = initial_ids[-1]
            while len(initial_ids) < firstk:
                initial_ids.append(pad)
    elif MODEL_NAME == "mistralai":
        intital_sentence =  desired_token + name
        print("intital_sentence: ", intital_sentence)
        
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0].tolist()[1:]
        print("initial_ids length: ", len(initial_ids))
        if firstk > len(initial_ids):
            while len(initial_ids) < firstk:
                initial_ids.append(918)
    else:
        intital_sentence = name + desired_token
        print("intital_sentence: ", intital_sentence)
        
        initial_ids = tok(intital_sentence, return_tensors="pt")["input_ids"].to(device)[0].tolist()[1:]
        print("initial_ids length: ", len(initial_ids))
        if firstk > len(initial_ids):
            while len(initial_ids) < firstk:
                initial_ids.append(918)
    
    return initial_ids[:firstk]

def user_prompt_generation(prompt_text, object, dataset, query_choose):
    if dataset == "imdb":
        if query_choose == "for_subjects":
            user_prompts = "According to IMDB dataset and your knowledges, who is known" + prompt_text.split("is known")[-1][:-1] + "?"
        elif query_choose == "for_objects":  
            surname = prompt_text.split()[1]
            lifespan = prompt_text.split(". ")[0].split(surname)[-1]
            movies = prompt_text.split(" is known for movies: ")[-1]
            user_prompts = "According to IMDB dataset and your knowledges, what movies " + prompt_text.split()[0] + " " + prompt_text.split()[1] + " has worked on and what were her/his roles?"
    elif dataset == "basketball":
        user_prompts = "According to wikidata (basketball players) dataset and your knowledges, what teams did " + object + "play for and what did she/he accomplish with them?"
    elif dataset == "book_query":
        user_prompts = "According to wikidata (book query) dataset and your knowledges, who wrote this book '" + object + "'?  And when was it published?"
    elif dataset == "Nobel_prize":
        user_prompts = "According to Nobel winner dataset and your knowledges, why and when did " + object + " win the Nobel prize?"
    return user_prompts


def check_success_2(user_prompt_id, target_prompt_id, prompt_text, token_ids, topk, model, tok, nconst_embedding):
    class SuppressStdout:
        def __enter__(self):
            self.original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self.original_stdout

    with SuppressStdout():
        with torch.no_grad():
            input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
            lst = token_ids + input_ids.tolist()[0]
            input_ids = torch.tensor([lst], dtype=torch.int).to(device)
            with torch.no_grad():
                out = model(input_ids, output_hidden_states=True)
            data = np.array(nconst_embedding).astype('float32')
        
            # Generate some random queries
            query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
            queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
            
            dimension = 4096
            # Build the HNSW index in faiss
            index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
            index.hnsw.efConstruction = 40
            index.verbose = True
        
            faiss.normalize_L2(data)
            index.add(data)
    
            # Search using the index
            faiss.normalize_L2(queries)
            # index.hnsw.efSearch = 1000000
            distances, indices = index.search(queries, topk)
            
            Neighbors = []
            for j in range(topk):
                Neighbors.append(indices[0][j])
            
            # print("Neighbors: ", Neighbors)
            # print("user_prompt_id: ", user_prompt_id)
            # print("target_prompt_id: ", target_prompt_id)
            if user_prompt_id not in Neighbors:
                # if Neighbors[0] == target_prompt_id:
                #     return 2
                return 2
            elif user_prompt_id not in Neighbors:
                return -1
            else:
                return 0

def check_success(target_prompt_id, user_prompt_id, Neighbors):
    if target_prompt_id in Neighbors and user_prompt_id not in Neighbors:
        if Neighbors[0] == target_prompt_id:
            return 2
        return 1
    elif user_prompt_id in Neighbors:
        if Neighbors[0] == target_prompt_id:
            return -2
        else:
            return 0
    elif user_prompt_id not in Neighbors:
        return -1
    else:
        return 0
        

def check_success_target_origin(user_prompt_id, target_prompt_id, prompt_text, token_ids, topk, model, tok, nconst_embedding):
    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        data = np.array(nconst_embedding).astype('float32')
    
        # Generate some random queries
        query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
        queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
        
        print("query shape: ", queries.shape)
        
        dimension = 4096
        # Build the HNSW index in faiss
        index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
        index.hnsw.efConstruction = 40
        index.verbose = True
    
        faiss.normalize_L2(data)
        index.add(data)

        # Search using the index
        faiss.normalize_L2(queries)
        # index.hnsw.efSearch = 1000000
        distances, indices = index.search(queries, topk)
        
        Neighbors = []
        for j in range(topk):
            Neighbors.append(indices[0][j])
        
        print("Neighbors: ", Neighbors)
        print("user_prompt_id: ", user_prompt_id)
        print("target_prompt_id: ", target_prompt_id)
        if target_prompt_id in Neighbors:
            if Neighbors[0] == target_prompt_id:
                return 2
            return 1
        else:
            return 0

def get_retrieved_result(user_prompt_id, prompt_text, topk, model, tok, nconst_embedding):
    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        data = np.array(nconst_embedding).astype('float32')
    
        query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
        queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
        
        print("query shape: ", queries.shape)
        
        dimension = 4096
        # Build the HNSW index in faiss
        index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
        index.hnsw.efConstruction = 40
        index.verbose = True
    
        faiss.normalize_L2(data)
        index.add(data)

        # Search using the index
        faiss.normalize_L2(queries)
        # index.hnsw.efSearch = 1000000
        distances, indices = index.search(queries, topk)
        
        
        # print(f"Query {user_prompt_id}:")
        Neighbors = []
        for j in range(topk):
            
            Neighbors.append(indices[0][j])
        
        return Neighbors

def construct_prompt(topk_passage_ids, prompts):
    composed_passage = ""
    for id in topk_passage_ids:
        composed_passage += prompts[id]
        composed_passage += " "
    return composed_passage

def get_neighbor_ids(user_prompt_id, prompt_text, topk, model, tok, nconst_embedding):
    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        # input_ids = tok(prompt_text, padding='max_length', max_length=100, return_tensors="pt")["input_ids"].to(device)
        lst = input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
        data = np.array(nconst_embedding).astype('float32')
    
        # Generate some random queries
        query_vector = np.array(out.hidden_states[-1].to(torch.device('cpu')).tolist())
        queries = np.array([np.array(query_vector[0]).mean(axis=0).astype('float32')])
        
        print("query shape: ", queries.shape)
        
        dimension = 4096
        
        index = faiss.IndexFlatL2(dimension)
        index.add(data)
        distances, indices = index.search(queries, topk)
        target_id = indices[0][-1]  # Last 10 indices will be the farthest
        
        return target_id

def check_success_logits(prompt_text, token_ids, desired_token, original_token, model, tok):

    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            first_output_token_id = out.logits[0, -1].argmax().item()
            first_output_token = tok.decode(first_output_token_id).strip()

        print("new first_output_token: ", first_output_token)
        print("desired_token: ", desired_token)
        
        if first_output_token != original_token:
            if first_output_token == desired_token.strip():
                return 2, first_output_token
            return 1, first_output_token
        else:
            return 0, first_output_token

def check_success_logits_to_origin(prompt_text, token_ids, desired_token, original_token, model, tok):

    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            first_output_token_id = out.logits[0, -1].argmax().item()
            first_output_token = tok.decode(first_output_token_id).strip()

        print("new first_output_token: ", first_output_token)
        print("desired_token: ", desired_token)
        
        
        if first_output_token == desired_token.strip():
            return 1, first_output_token
        else:
            return 0, first_output_token
            

def check_success_logits_Instructed(Instructed_embeddings, prompt_text, token_ids, desired_token, original_token, model, tok):

    with torch.no_grad():
        instructed_ids = tok(Instructed_embeddings, return_tensors="pt")["input_ids"].to(device)
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = instructed_ids.tolist()[0] + token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            first_output_token_id = out.logits[0, -1].argmax().item()
            first_output_token = tok.decode(first_output_token_id).strip()

        print("new first_output_token: ", first_output_token)
        print("desired_token: ", desired_token)
        
        if first_output_token != original_token:
            if first_output_token == desired_token.strip():
                return 2, first_output_token
            return 1, first_output_token
        else:
            return 0, first_output_token

def check_MSEloss(target_prompt, prompt_text, token_ids, model, tok):
    
    with torch.no_grad():
        input_ids = tok(prompt_text, return_tensors="pt")["input_ids"].to(device)
        lst = token_ids + input_ids.tolist()[0]
        input_ids = torch.tensor([lst], dtype=torch.int).to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            user_tokens = out.hidden_states[-1] 
        user_emb = user_tokens[0].mean(dim=0, keepdim=True)

        input_ids = tok(target_prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids, output_hidden_states=True)
            target_prompt_tokens = out.hidden_states[-1]
        
        target_emb = target_prompt_tokens[0].mean(dim=0, keepdim=True)
        
        mse_loss = loss_function(user_emb, target_emb).item()
    return mse_loss

def get_first_output_token(prompt, model, tok):
    input_ids = tok(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
        first_output_token_id = out.logits[0, -1].argmax().item()
        first_output_token = tok.decode(first_output_token_id)
    return first_output_token

# SFR mistral
def get_relevant_documents(nconst_embeddings, query, topK, model, tok):
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    task_query = get_detailed_instruct(task, query)
    print("task_query: ", task_query)
    batch_dict = tok(task_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state.to("cpu"), batch_dict['attention_mask'].to(torch.device('cpu'))).to("cpu")
    query_embedding = np.array(embeddings[0])
    queries = np.array([query_embedding])
   
    docs_embeddings = np.array(F.normalize(nconst_embeddings, p=2, dim=1))
    
    dimension = 4096
    # Build the HNSW index in faiss
    index = faiss.IndexHNSWFlat(dimension, 16)  # 16 is the number of links per object
    
    index.add(docs_embeddings)

    index.hnsw.efSearch = 1000000
    distances, indices = index.search(queries, topK)
    
    print("top_k: ", indices[0])

    return indices[0]

def last_token_pool(last_hidden_states,
                 attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]