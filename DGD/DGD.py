# functional test

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from livelossplot import PlotLosses
from DGD.help_functions import check_success, get_relevant_documents, last_token_pool, get_detailed_instruct
import numpy as np
from torch.nn.functional import cosine_similarity

# Define device
device = torch.device("cuda")

mseLoss = torch.nn.MSELoss()
sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
triplet_loss = torch.nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)

import numpy as np



# Loss function
def loss_function(new_logits, new_target_tokens, orginal_emb):
    loss_for_origin = mseLoss(new_logits, orginal_emb)
    loss_for_target = torch.sqrt(mseLoss(new_logits, new_target_tokens))
    l2_distance_norm = torch.norm(new_target_tokens - new_logits, p=2)
    tripletLoss = triplet_loss(new_logits, new_target_tokens, orginal_emb)
    l1_distance = torch.abs(new_logits - new_target_tokens).sum()
    cos_sim_positive = cosine_similarity(new_logits.unsqueeze(0), new_target_tokens.unsqueeze(0))
    cos_distance = 1 - cos_sim_positive
    cos_sim_negative = cosine_similarity(new_logits.unsqueeze(0), orginal_emb.unsqueeze(0))
    cos_sim = cos_distance + cos_sim_negative
    
    return cos_sim, loss_for_target, loss_for_origin

def initialize_one_hot_from_target(model_name, tok, initial_ids, vocab_size, device):
    # Tokenize the target text to get token IDs
    prefix_length = len(initial_ids)
    print("prefix_length: ", prefix_length)
    # Create an empty one-hot tensor
    if model_name == "Qwen/Qwen-7B":
        one_hot = torch.zeros(1, prefix_length, vocab_size, device=device, dtype=torch.bfloat16)
    else:
        one_hot = torch.zeros(1, prefix_length, vocab_size, device=device, dtype=torch.float16)
    
    # Fill the initial rows of the one-hot tensor using the target token IDs
    for idx in range(min(prefix_length, len(initial_ids))):
        one_hot[0, idx, :] = 0.0
        one_hot[0, idx, initial_ids[idx]] = 1.0

    return one_hot.requires_grad_(), prefix_length

# 1. Integrated Gradients method
def integrated_gradients(model, baseline, one_hot, embed_weights, input_embeddings, original_tokens, steps=10):
    # List to store the integrated gradients
    integrated_grads = []
    
    # Compute the path integral
    for step in range(steps):
        # Calculate the interpolated input
        alpha = (step + 1) / steps
#         alpha = step / steps
        one_hot.requires_grad_(False)
        baseline.requires_grad_(False)
        interpolated_input = (baseline * (1 - alpha)) + (one_hot * alpha)
        interpolated_input.requires_grad_(True)
        # Convert one-hot prefix to embeddings
        prefix_embeddings = torch.matmul(interpolated_input, embed_weights)
        combined_embeddings = torch.cat([prefix_embeddings.squeeze(0), input_embeddings.squeeze(0)], dim=0)
        
        # Forward pass and compute gradients
        outputs = model(inputs_embeds=combined_embeddings.unsqueeze(0), output_hidden_states=True)
        logits = outputs.hidden_states[-1]
        new_logits = logits[0].mean(dim=0, keepdim=True)
        reverse_loss = 1 - loss_function(new_logits, original_tokens)
        total_temp = reverse_loss 
            
        total_temp.backward()
        
        # Store the gradient
        integrated_grads.append(interpolated_input.grad.clone())
        interpolated_input.grad.zero_()
    
    # Average the gradients across all steps
    avg_grads = torch.mean(torch.stack(integrated_grads), dim=0)
    
    return avg_grads

def random_except(k, exclusion_list):
    valid_nums = [i for i in range(k) if i not in exclusion_list]
    index = torch.randint(0, len(valid_nums), (1,))()
    return valid_nums[index]

def normalize_L2_torch(tensor):
    norm = torch.norm(tensor, dim=1, keepdim=True)
    return tensor / norm.clamp(min=1e-10)

def get_optimized_prefix_embedding(model_name, orginal_emb_logits, new_logits, user_prompt, prompt_text, target_text, initial_ids, iterations, user_prompt_id, target_prompt_id, topk_result, model, tok, nconst_embedding, thredhold):
    print("prompt_text: ", prompt_text)
    new_target_tokens = new_logits.to(device)
    orginal_emb = orginal_emb_logits.to(device)

    prompt_tokens = tok.encode(user_prompt, return_tensors="pt")[:, 1:].to(device)

    Instructed_Prompting_tokens = torch.tensor([[1]]).to(device)
    
    if model_name == "EleutherAI/gpt-j-6b":
        vocab_size = 50400
        count_limit = 50
        input_embeddings = model.transformer.wte(prompt_tokens)
        # Instructed_embeddings = model.transformer.wte(Instructed_Prompting_tokens)
        embed_weights = model.transformer.wte.weight
        k = 32
    elif model_name == "mistralai":
        vocab_size = 32000
        count_limit = 100
        input_embeddings = model.embed_tokens(prompt_tokens)
        Instructed_embeddings = model.embed_tokens(Instructed_Prompting_tokens)
        
        embed_weights = model.embed_tokens.weight
        # k = 20
        k = 128
    else:
        vocab_size = 151936
        count_limit = 50
        input_embeddings = model.transformer.wte(prompt_tokens)
        embed_weights = model.transformer.wte.weight
        k = 32

    one_hot, prefix_length = initialize_one_hot_from_target(model_name, tok, initial_ids, vocab_size, device)
    # Training loop
    num_epochs = 1
    temperature = 0.8
    
    # Training loop
    for epoch in range(num_epochs):

        # Apply the GCG step to get the new prefix
        one_hot, loss_list = greedy_coordinate_gradient_step(model, tok, one_hot, embed_weights, input_embeddings, new_target_tokens, iterations, user_prompt_id, target_prompt_id, user_prompt, topk_result, nconst_embedding, count_limit, orginal_emb, k, Instructed_embeddings, user_prompt, thredhold)

        token_ids = torch.argmax(one_hot, dim=-1)
        optimized_prefix = tok.decode(token_ids[0].tolist(), skip_special_tokens=True)

    return token_ids, optimized_prefix, loss_list

# Define the GCG step
def greedy_coordinate_gradient_step(model, tok, one_hot, embed_weights, input_embeddings, new_target_tokens, iterations, user_prompt_id, target_prompt_id, user_prompt, topk_result, nconst_embedding, count_limit, orginal_emb, k, Instructed_embeddings, user_query, thredhold):
    # Greedy Coordinate Gradient loop
    loss_record = float('inf')
    global_best_loss = float('inf')
    global_best_one_hot = None
    one_hot_return =  None
    loss_list = {}
    loss_list["loss"] = []
    loss_list["target_loss"] = []
    loss_list["original_loss"] = []
    for epoch in range(iterations):
        # Convert one-hot prefix to embeddings
        one_hot = one_hot.to(dtype=torch.float16)
        embed_weights = embed_weights.to(dtype=torch.float16)
        
        prefix_embeddings = torch.matmul(one_hot, embed_weights)
        # Concatenate prefix embeddings with embeddings of input_ids
        combined_embeddings = torch.cat([prefix_embeddings.squeeze(0), input_embeddings.squeeze(0)], dim=0)
        combined_embeddings = torch.cat([Instructed_embeddings.squeeze(0), combined_embeddings.squeeze(0)], dim=0)

        # Forward pass: compute the output
        outputs = model(inputs_embeds=combined_embeddings.unsqueeze(0), output_hidden_states=True)
        token_ids = torch.argmax(one_hot, dim=-1)
        temp_prefix = tok.decode(token_ids[0].tolist(), skip_special_tokens=True)
        full_query = temp_prefix+user_query
        batch_dict = tok(full_query, return_tensors="pt").to(device)
        new_logits = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        mse_loss, loss_for_target, loss_for_origin = loss_function(new_logits[0], new_target_tokens, orginal_emb)

        current_loss = mse_loss
        
        # Compute gradients
        current_loss.backward()
        grad = one_hot.grad.clone()
        grad = grad / grad.norm(dim=-1, keepdim=True)

        # best_loss = float('inf')
        best_loss = current_loss
        best_loss_for_target = loss_for_target
        best_loss_for_origin = loss_for_origin

        loss_list["loss"].append(best_loss)
        loss_list["target_loss"].append(best_loss_for_target)
        loss_list["original_loss"].append(best_loss_for_origin)
        
        with torch.no_grad():
            top_indices = (-grad).topk(k, dim=2).indices
            best_one_hot = one_hot.clone()
            all_random_numbers = set()

            while len(all_random_numbers) < 15:
                random_number = random.randint(1, 3)
                current_random_numbers = [random.randint(0, one_hot.size(1)-1) for _ in range(random_number)]
                # Convert list to tuple so it can be added to a set of lists
                all_random_numbers.add(tuple(current_random_numbers))
            
            # Convert back to list of lists
            unique_random_number_lists = [list(t) for t in all_random_numbers]

            for random_numbers in unique_random_number_lists:
                candidate_one_hot = one_hot.clone()
                for pos in random_numbers:
                    # Randomly select a token substitution from top-k
                    element = torch.randint(0, k, (1,))
                    tk = top_indices[0, pos, element]
                    candidate_one_hot[0, pos, :] = 0.0
                    candidate_one_hot[0, pos, tk] = 1.0

                prefix_embeddings = torch.matmul(candidate_one_hot, embed_weights)
                combined_embeddings = torch.cat([prefix_embeddings.squeeze(0), input_embeddings.squeeze(0)], dim=0)
                combined_embeddings = torch.cat([Instructed_embeddings.squeeze(0), combined_embeddings.squeeze(0)], dim=0)
    
                # Forward pass: compute the output
                outputs = model(inputs_embeds=combined_embeddings.unsqueeze(0), output_hidden_states=True)
                token_ids = torch.argmax(candidate_one_hot, dim=-1)
                temp_prefix = tok.decode(token_ids[0].tolist(), skip_special_tokens=True)
                full_query = temp_prefix+user_query
                batch_dict = tok(full_query, return_tensors="pt").to(device)
                new_logits = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                mse_loss, loss_for_target, loss_for_origin = loss_function(new_logits[0], new_target_tokens, orginal_emb)
            
                loss_temp = mse_loss
                # print("loss_temp: ", loss_temp)
                if loss_temp < best_loss:
                    best_loss = loss_temp
                    best_loss_for_target = loss_for_target
                    best_loss_for_origin = loss_for_origin
                    best_one_hot = candidate_one_hot
        
            # Update one-hot tensor
            one_hot = best_one_hot.detach().requires_grad_()
                        
            # Jump out local minimum
            if epoch % count_limit == 0:
                if loss_record == best_loss:
                    # Remove less influence tokens
                    best_loss = float('inf') 
                    numbers = list(range(0, one_hot.size(1)))
                    for pos in numbers:
                        candidate_one_hot = one_hot.clone()
                        candidate_one_hot[0, pos, :] = 0.0
    
                        # Compute loss for this substitution
                        prefix_embeddings = torch.matmul(candidate_one_hot, embed_weights)
                        combined_embeddings = torch.cat([prefix_embeddings.squeeze(0), input_embeddings.squeeze(0)], dim=0)
                        combined_embeddings = torch.cat([Instructed_embeddings.squeeze(0), combined_embeddings.squeeze(0)], dim=0)
    
                        # Forward pass: compute the output
                        outputs = model(inputs_embeds=combined_embeddings.unsqueeze(0), output_hidden_states=True)
                        token_ids = torch.argmax(candidate_one_hot, dim=-1)
                        temp_prefix = tok.decode(token_ids[0].tolist(), skip_special_tokens=True)
                        full_query = temp_prefix+user_query
                        batch_dict = tok(full_query, return_tensors="pt").to(device)
                        new_logits = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                        # new_logits = F.normalize(new_logits, p=2, dim=1)[0]
                        mse_loss, loss_for_target, loss_for_origin = loss_function(new_logits[0], new_target_tokens, orginal_emb)
                        loss_temp = mse_loss
                        
                        if loss_temp < best_loss:
                            best_loss = loss_temp
                            best_loss_for_target = loss_for_target
                            best_loss_for_origin = loss_for_origin
                            best_one_hot = candidate_one_hot
                loss_record = best_loss

        
            # Update one-hot tensor
            one_hot = best_one_hot.detach().requires_grad_()

            if global_best_loss > best_loss:
                global_best_loss = best_loss
                global_best_one_hot = one_hot.detach()
            
            if epoch % 1 == 0:
                print(f'Epoch [{epoch}/{iterations}], Best Loss: {best_loss.item():.4f}')
                loss_list["loss"].append(best_loss)
                loss_list["target_loss"].append(best_loss_for_target)
                loss_list["original_loss"].append(best_loss_for_origin)
            
            # Print progress
            if epoch % 20 == 0:
                # Decode the optimized prefix
                token_ids = torch.argmax(one_hot, dim=-1)
                optimized_prefix = tok.decode(token_ids[0].tolist(), skip_special_tokens=True)
                top_k_ids = get_relevant_documents(torch.stack(nconst_embedding), optimized_prefix+user_prompt, topk_result, model, tok)
                is_success = check_success(target_prompt_id, user_prompt_id, top_k_ids)

                print("token_ids: ", token_ids)
                print("Optimized Prefix:", optimized_prefix)
                if is_success > thredhold:
                    print("!!!--------> Successfully point to target entry <-------!!!")
                    return one_hot, loss_list
                elif is_success > 0:
                    one_hot_return = one_hot

    if one_hot_return == None:
        one_hot_return = global_best_one_hot

    return one_hot_return, loss_list