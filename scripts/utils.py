import copy
import torch
import numpy as np
import torchtext as text
from state import Token, ParseState, DependencyEdge, shift, left_arc, right_arc, is_final_state

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def get_word2ix(path = "data/pos_set.txt"):
    """ Generates a mapping given a vocabulary file. 
    Input
    -------------
    path: str or pathlib.Path. Relative path to the vocabulary file. 

    Output
    -------------
    word2ix: dict. Dictionary mapping words to unique IDs. Keys are words and 
                values are the indices.
    """
    word2ix = {}
    with open(path, encoding="utf8") as f:
        for index, word in enumerate(f):
            word2ix[word.strip()] = int(index)
    return word2ix


def get_glove_emb_from_stack_buff_words(stack_buff_lists, name = '6B', dim = 50):
    vec = text.vocab.GloVe(name=name, dim=dim)
    stack_buffs = [vec.get_vecs_by_tokens(t).mean(0) for t in stack_buff_lists]
    return torch.stack(stack_buffs)

# def get_glove_emb_from_stack_buff_words_test_data(stack_buff_list, name = '6B', dim = 50):
#     vec = text.vocab.GloVe(name=name, dim=dim)
#     # stack_buffs = []
#     # for stack_buff_list in stack_buff_lists:
#     stack_buff = vec.get_vecs_by_tokens(stack_buff_list, lower_case_backup=True)
#         # stack_buffs.append(stack_buff)
#     return stack_buff

def get_ix_from_token(lists, word2ix):
    data = []
    for list in lists:
        file_data = []
        for l in list:
            file_data.append(word2ix[l.strip()])
        data.append(file_data.copy())
    return torch.tensor(data)


def get_ix_from_token_test(list, word2ix):
    data = []
    # for list in lists:
    file_data = []
    for l in list:
        file_data.append(word2ix[l.strip()])
    data.append(file_data.copy())
    return torch.tensor(data)


def parse_file(file_path):
    word_lists = []
    pos_lists = []
    actions_lists = []

    with open(file_path, 'r', encoding="utf8") as file:
        for line in file:
            parts = line.strip().split("|||")

            word_list = [word.strip() for word in parts[0].split()]
            pos_list = [pos.strip() for pos in parts[1].split()]
            actions_list = [action.strip() for action in parts[2].split()]

            word_lists.append(word_list)
            pos_lists.append(pos_list)
            actions_lists.append(actions_list)

    return word_lists, pos_lists, actions_lists

# def parsestate_to_lists(parse_state):
#     """
#     Converts a ParseState object to lists.
    
#     Args:
#     parse_state (ParseState): The parse state to convert.
    
#     Returns:
#     tuple: A tuple containing three lists (stack, buffer, dependencies).
#     """
#     stack = [(token.idx, token.word, token.pos) for token in parse_state.stack]
#     parse_buffer = [(token.idx, token.word, token.pos) for token in parse_state.parse_buffer]
#     dependencies = [((edge.source.idx, edge.source.word, edge.source.pos), 
#                      (edge.target.idx, edge.target.word, edge.target.pos), 
#                       edge.label) for edge in parse_state.dependencies]
    
#     return stack, parse_buffer, dependencies

# def lists_to_parsestate(stack_list, buffer_list, dependencies_list):
#     """
#     Converts lists to a ParseState object.
    
#     Args:
#     stack_list (list): A list of stack tokens.
#     buffer_list (list): A list of buffer tokens.
#     dependencies_list (list): A list of dependencies.
    
#     Returns:
#     ParseState: A ParseState object.
#     """
#     stack = [Token(idx, word, pos) for idx, word, pos in stack_list]
#     parse_buffer = [Token(idx, word, pos) for idx, word, pos in buffer_list]
#     dependencies = [DependencyEdge(Token(src_idx, src_word, src_pos), 
#                                    Token(tgt_idx, tgt_word, tgt_pos), 
#                                    label) 
#                     for ((src_idx, src_word, src_pos), 
#                          (tgt_idx, tgt_word, tgt_pos), 
#                          label) in dependencies_list]
    
#     return ParseState(stack, parse_buffer, dependencies)

def get_deps_training(file_path, cwindow):
    """ Computes all the dependencies set for all the sentences according to 
    actions provided
    Inputs
    -----------
    words_lists: List[List[str]].  This is a list of lists. Each inner list is a list of words in a sentence,
    poss: List[List[str]]. This is a list of lists. Each inner list is a list of POS tags for the corresponding sentence,
    actions: List[List[str]]. This is a list of lists where each inner list is the sequence of actions
                Note that the elements should be valid actions as in `tagset.txt`
    cwindow: int. Context window. Default=2
    """

    words_lists, poss, actions = parse_file(file_path)

    stack_buff_lists = []   # List of List of words in the stack and buffer
    pos_lists = []  # List of List of POS tags
    actions_lists = []  # List of List of actions
    # stack_buffer_count = []  # List of number of elements in stack and buffer

    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Intialize stack and buffer appropriately
        stack = [Token(idx=-i-1, word="[PAD]", pos="NULL") for i in range(cwindow)]
        parser_buff = []
        for ix in range(len(words_list)):
            parser_buff.append(Token(idx=ix, word=words_list[ix], pos=poss[w_ix][ix]))
        parser_buff.extend([Token(idx=ix+i+1, word="[PAD]",pos="NULL") for i in range(cwindow)])
        # Initilaze the parse state
        state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])

        # Iterate over the actions and do the necessary state changes
        for action in actions[w_ix]:

            stack_buff_list = []
            pos_list = []

            stack_w = state.stack[-cwindow:]
            buffer_w = state.parse_buffer[:cwindow]

            stack_buff_list.extend(w.word for w in stack_w)
            pos_list.extend(w.pos for w in stack_w)
            stack_buff_list.extend(w.word for w in buffer_w)
            pos_list.extend(w.pos for w in buffer_w)
            actions_list = [action]
            # stack_buffer_count.append([len(state.stack), len(state.parse_buffer)])

            stack_buff_lists.append(stack_buff_list)
            pos_lists.append(pos_list)
            actions_lists.append(actions_list)
            
            if action == "SHIFT":
                shift(state)
            elif action[:8] == "REDUCE_L":
                left_arc(state, action[9:])
            else:
                right_arc(state, action[9:])
        assert is_final_state(state,cwindow)    # Check to see that the parse is complete
        right_arc(state, "root")    # Add te root dependency for the remaining element on stack
    return stack_buff_lists, pos_lists, actions_lists

def process_data(file_path, path_to_pos_set = 'data/pos_set.txt', path_to_tag_set = 'data/tagset.txt', name_glove = '6B', dim_glove = 50, rep_type = 'mean'):
    
    stack_buff_lists, pos_lists, actions_lists = get_deps_training(file_path, 2)

    vec = text.vocab.GloVe(name=name_glove, dim=dim_glove)

    if rep_type == 'mean':
        stack_buffs = [vec.get_vecs_by_tokens(t).mean(0) for t in stack_buff_lists]
    elif rep_type == 'concat':
        stack_buffs = [torch.flatten(vec.get_vecs_by_tokens(t)) for t in stack_buff_lists]
        
    
    pos2ix = get_word2ix(path_to_pos_set)
    act2ix = get_word2ix(path_to_tag_set)

    feature_pos = get_ix_from_token(pos_lists, pos2ix)
    target = get_ix_from_token(actions_lists, act2ix)
    
    # print("process data")
    # print(torch.tensor(feature_pos).shape)
    return torch.stack(stack_buffs), feature_pos, target


def process_data_test(words_list, pos_list, path_to_pos_set = 'data/pos_set.txt', name_glove = '6B', dim_glove = 50, rep_type = 'mean'):
    # print("words_list", words_list)
    # print("pos_list", pos_list)
    # stack_buff_lists, pos_lists, actions_lists = get_deps_training(file_path, 2)

    vec = text.vocab.GloVe(name=name_glove, dim=dim_glove)

    if rep_type == 'mean':
        stack_buffs = [vec.get_vecs_by_tokens(words_list).mean(0)]
    elif rep_type == 'concat':
        stack_buffs = [torch.flatten(vec.get_vecs_by_tokens(words_list))]
        
    
    pos2ix = get_word2ix(path_to_pos_set)
    # print(pos2ix)

    feature_pos = get_ix_from_token_test(pos_list, pos2ix)

    # print("process data test")
    # print(feature_pos.shape)
    # print(feature_pos)
    
    return torch.stack(stack_buffs), feature_pos
    

def act_pred(model, file_path, cwindow, rep_type, name_glove, dim_glove):

    words_lists, poss, actions = parse_file(file_path)

    stack_buff_lists = []   # List of List of words in the stack and buffer
    pos_lists = []  # List of List of POS tags
    actions_lists = []  # List of List of actions
    # stack_buffer_count = []  # List of number of elements in stack and buffer

    act_preds = []

    # Iterate over sentences
    for w_ix, words_list in enumerate(words_lists):
        # Intialize stack and buffer appropriately
        stack = [Token(idx=-i-1, word="[PAD]", pos="NULL") for i in range(cwindow)]
        parser_buff = []
        for ix in range(len(words_list)):
            parser_buff.append(Token(idx=ix, word=words_list[ix], pos=poss[w_ix][ix]))
        parser_buff.extend([Token(idx=ix+i+1, word="[PAD]",pos="NULL") for i in range(cwindow)])
        # Initilaze the parse state
        state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])
        act_s = []

        x = 0
        while is_final_state(state,cwindow) == False:
        # while x<1:
        #     x=x+1

            word_r = []
            pos_r = []

            stack_w = state.stack[-cwindow:]
            buffer_w = state.parse_buffer[0:cwindow]

            word_r.extend([w.word for w in stack_w])
            word_r.extend([w.word for w in buffer_w])
            # print(word_r)

            pos_r.extend([w.pos for w in stack_w])
            pos_r.extend([w.pos for w in buffer_w])
            # print(pos_r)

            feature_stack_buff, feature_pos = process_data_test(word_r, pos_r,name_glove = name_glove, dim_glove = dim_glove, rep_type = rep_type)
            # print("from predict action")
            # print(feature_stack_buff.shape, feature_pos.shape)

            out = model(feature_stack_buff.to(device), feature_pos.to(device, dtype=torch.long)).to(device)
            # print(out)
            _, predicted = torch.max(out, 1)
            # print(predicted)

            action_to_ix = get_word2ix("data/tagset.txt")
            ix_to_action = {ix: action for action, ix in action_to_ix.items()}

            action = ix_to_action[predicted[0].item()]
            # print(action)

            if len(state.stack) <= cwindow + 1:
                action = "SHIFT"
            if len(state.parse_buffer) == cwindow and action == 'SHIFT':
                action = "REDUCE_R_fixed"
            if action == "SHIFT":
                shift(state)
            if action[:8] == "REDUCE_L":
                left_arc(state, action[9:])
            if action[:8] == "REDUCE_R":
                right_arc(state, action[9:])
            
            act_s.append(action)
        act_preds.append(act_s)


            # print(out)
            # print(feature_stack_buff.shape, feature_pos.shape)



            # Iterate over the actions and do the necessary state changes
            # for action in actions[w_ix]:
            #     print("before passsing to model test")
            #     print(cwindow)
            #     print(feature_stack_buff.shape, feature_pos.shape)

            #     out = model(feature_stack_buff.to(device), feature_pos.to(device, dtype=torch.long)).to(device)
            #     print(out)
                
            #     if action == "SHIFT":
            #         shift(state)
            #     elif action[:8] == "REDUCE_L":
            #         left_arc(state, action[9:])
            #     else:
            #         right_arc(state, action[9:])
            # assert is_final_state(state,cwindow)    # Check to see that the parse is complete
            # right_arc(state, "root")    # Add te root dependency for the remaining element on stack
    return act_preds

# def illegal_actions_func(stack_count, buffer_count):
#     """
#     Args:
#     - state: Contains the information necessary to determine illegal actions, e.g., 
#              the number of words left in the buffer and stack.

#     Returns:
#     - illegal_actions (list): A list of illegal actions given the state.
#     """
#     illegal_actions = []

#     # Read actions from a file and store them in ALL_ACTIONS
#     with open("data/tagset.txt", "r") as file:
#         ALL_ACTIONS = [line.strip() for line in file]
    
#     # 1. When there is only 2 words left in the buffer 'SHIFT' is illogical
#     # if buffer_count == 2:
#     #     illegal_actions.append('SHIFT')

#     # 2. When there is only 3 words left in the stack 'action[:8] == "REDUCE_L"' and 
#     #    'action[:8] == "REDUCE_R"' is illogical
#     if stack_count <= stack_count+buffer_count-5:
#         illegal_actions.extend([action for action in ALL_ACTIONS 
#                                 if action[:8] == "REDUCE_L" or action[:8] == "REDUCE_R"])
    
#     return illegal_actions

# def correcting_illegal_actions(file_path, predicted_actions_tmp, cwindow):
#     """ Computes all the dependencies set for all the sentences according to 
#     actions provided
#     Inputs
#     -----------
#     words_lists: List[List[str]].  This is a list of lists. Each inner list is a list of words in a sentence,
#     actions: List[List[str]]. This is a list of lists where each inner list is the sequence of actions
#                 Note that the elements should be valid actions as in `tagset.txt`
#     cwindow: int. Context window. Default=2
#     """

#     words_lists, poss, actions = parse_file(file_path)

#     all_deps = []   # List of List of dependencies
#     # Iterate over sentences
#     for w_ix, words_list in enumerate(words_lists):
#         # Intialize stack and buffer appropriately
#         stack = [Token(idx=-i-1, word="[NULL]", pos="NULL") for i in range(cwindow)]
#         parser_buff = []
#         for ix in range(len(words_list)):
#             parser_buff.append(Token(idx=ix, word=words_list[ix], pos="NULL"))
#         parser_buff.extend([Token(idx=ix+i+1, word="[NULL]",pos="NULL") for i in range(cwindow)])
#         # Initilaze the parse state
#         state = ParseState(stack=stack, parse_buffer=parser_buff, dependencies=[])

#         # Iterate over the actions and do the necessary state changes
#         for action in predicted_actions_tmp[w_ix]:
#             print(f"Applying action: {action}")
#             action_replaced = False

#             while not action_replaced:
#                 action_replaced = True

#                 if action == "SHIFT":
#                     try:
#                         shift(state)
#                     except IndexError:
#                         action = "REDUCE_R_punct" # porer ta add koro
#                         action_replaced = False
#                 elif action[:8] == "REDUCE_L":
#                     try:
#                         left_arc(state, action[9:])
#                     except IndexError:
#                         action = "SHIFT"
#                         action_replaced = False
#                 else:
#                     try:
#                         right_arc(state, action[9:])
#                     except IndexError:
#                         action = "SHIFT"
#                         action_replaced = False
#         print(f"After action, stack: {[token.word for token in state.stack]}")
#         print(f"After action, buffer: {[token.word for token in state.parse_buffer]}")
#         print(f"After action, dependencies: {state.dependencies}")
#         try:
#             assert is_final_state(state,cwindow)    # Check to see that the parse is complete
#         except AssertionError:
#             print(f"Assertion Error for sentence {w_ix}: {words_list}")
#             print(f"Final stack: {state.stack}")
#             print(f"Final buffer: {state.parse_buffer}")
#             print(f"Final dependencies: {state.dependencies}")
#             print(f"Final actions: {predicted_actions_tmp[w_ix]}")
#             raise
#         right_arc(state, "root")    # Add te root dependency for the remaining element on stack
#         all_deps.append(state.dependencies.copy())  # Copy over the dependenices found
#     return all_deps

# def model_out_to_actions(out, gold_actions, stack_buffer_count):
#     """
#     Args:
#     - out (torch.Tensor): The raw output from the model.
#     - gold_actions (list of lists): The correct actions.
#     - legal_actions_func (function): A function that takes the current state and returns a list
#                                      of legal actions.

#     Returns:
#     - predicted_actions_nested (list of lists): The predicted actions.
#     """

#     action_to_ix = get_word2ix("data/tagset.txt")
#     ix_to_action = {ix: action for action, ix in action_to_ix.items()}

#     probabilities = torch.softmax(out, dim=-1)
#     sorted_predicted_indices = torch.argsort(probabilities, dim=-1, descending=True)
#     # sorted_predicted_indices_2 = np.argmax(probabilities)
#     print(sorted_predicted_indices[0][0])
#     # print(sorted_predicted_indices_2)

#     predicted_actions = []
#     for i, actions in enumerate(sorted_predicted_indices.cpu().numpy()):
#         # Assuming the state is available, or compute it here
#         stack_count, buffer_count = stack_buffer_count[i]
#         illegal_actions = illegal_actions_func(stack_count, buffer_count)
#         illegal_actions_ix = [action_to_ix[action] for action in illegal_actions]


#         # Find the most probable legal action
#         for action_ix in actions:
#             if action_ix not in illegal_actions_ix:
#                 predicted_actions.append(ix_to_action[action_ix])
#                 continue
                

#     action_lengths = [len(inner_list) for inner_list in gold_actions]

#     predicted_actions_nested = []
#     start = 0

#     for length in action_lengths:
#         predicted_actions_nested.append(predicted_actions[start:start + length])
#         start += length

#     return predicted_actions_nested

# def model_out_to_actions(out, gold_actions):

#     action_to_ix = get_word2ix("data/tagset.txt")
#     ix_to_action = {ix: action for action, ix in action_to_ix.items()}

#     probabilities = torch.softmax(out, dim=-1)
#     predicted_indices = torch.argmax(probabilities, dim=-1)

#     predicted_actions = [ix_to_action[ix] for ix in predicted_indices.cpu().numpy()]

#     action_lengths = [len(inner_list) for inner_list in gold_actions]
    
#     predicted_actions_nested = []
#     start = 0

#     for length in action_lengths:
#         predicted_actions_nested.append(predicted_actions[start:start + length])
#         start += length

#     return predicted_actions_nested