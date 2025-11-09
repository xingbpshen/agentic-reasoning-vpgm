import re
import numpy as np
import string


def parse_response_text(response, reasoning_model, option_list):
    """
    Only parse the answer option and numerical values, it will not handle summation to 1,
    option_list should be the option indicator in the text, such as ['A', 'B', 'C', ...]
    """
    if reasoning_model in ['cot', 'chameleon', 'chameleon-plus']:
        # Extract the first uppercase letter following "answer is "
        answer_match = re.search(rf'answer is\s+({option_list})', response)
        answer = answer_match.group(1) if answer_match else None
        # Extract the first numerical probability following "Probability"
        prob_match = re.search(r'Probability.*?(\d+\.\d+)', response)
        probability = float(prob_match.group(1)) if prob_match else None
        return answer, probability
    elif reasoning_model in ['vpgm-n2']:
        # Create a dict of options with prob 0
        prob_dict = {opt: 0.0 for opt in option_list}
        pattern = r"P\(Z3_\{Option ([A-Z])\} \| Z1, Z2\)=(\d+\.\d+)"
        matches = re.findall(pattern, response)
        # Convert to dict
        seen = {opt: float(prob) for opt, prob in matches if opt in option_list}
        # Copy the key's value in seen to prob_dict
        prob_dict.update(seen)
        # Output just the list of probabilities in alphabetical order of options
        probs = [prob_dict[opt] for opt in sorted(option_list) if opt in prob_dict]
        return probs
    elif reasoning_model in ['vpgm-n3', 'vpgm-n4']:
        # Extract all option-probability pairs
        matches = re.findall(rf'P\(Y=({option_list})\s*\|[^)]*\)=([\d.]+)', response)
        # Create a dict of options with prob 0
        prob_dict = {opt: 0.0 for opt in option_list}
        # Track first occurrence only
        seen = {}
        for opt, prob in matches:
            if opt in option_list and opt not in seen:
                seen[opt] = float(prob)
        # Copy the key's value in seen to prob_dict
        prob_dict.update(seen)
        # Output just the list of probabilities in alphabetical order of options
        probs = [prob_dict[opt] for opt in sorted(option_list) if opt in prob_dict]
        return probs
    else:
        assert False, f"Unsupported reasoning model: {reasoning_model}"


def get_random_answer_prob(num_choices):
    # create list of letters of length num_choices, elements are A, B, C, D, E, ...
    assert num_choices <= 26
    choices = list(string.ascii_uppercase[:num_choices])
    # random choice from num_choices
    return np.random.choice(choices), 1 / num_choices


def get_options(num_choices):
    # create list of letters of length num_choices, elements are A, B, C, D, E, ...
    return list(string.ascii_uppercase[:num_choices])


def options_are_equal(option1, option2):
    # check if they are strings
    if isinstance(option1, str) and isinstance(option2, str):
        # check if they are the same
        return option1.lower() == option2.lower()
    # check if they are int or float
    elif isinstance(option1, (int, float)) and isinstance(option2, (int, float)):
        # check if they are the same
        return int(option1) == int(option2)
    else:
        assert False, f"Unsupported type for option comparison: {type(option1)} and {type(option2)}"


def get_lm_answer_prob(lm_response_list, option_list, reasoning_model):
    """
    Get the answer and probability from the language model response, and it will handle summation to 1
    :param lm_response_list: list of language model responses for a single question, it can be multiple responses or just one
    :param option_list: list, all possible answer options
    :param reasoning_model: str, reasoning model
    :return: a dictionary with the answer/option as key and the probability as value, the length of the dictionary depends on the reasoning model
    """
    num_choices = len(option_list)
    if reasoning_model in ['cot', 'chameleon']:
        # len(lm_response_list) should be 1
        assert len(lm_response_list) == 1
        answer, probability = parse_response_text(lm_response_list[0], reasoning_model, option_list)
        if answer is None or probability is None:
            answer, probability = get_random_answer_prob(num_choices)
        return {answer: probability}
    elif reasoning_model in ['chameleon-plus']:
        assert len(lm_response_list) > 1
        first_answer = None
        subsequent_answers = []
        subsequent_probabilities = []
        # compute avg-conf
        for i in range(len(lm_response_list)):
            lm_response = lm_response_list[i]
            answer, probability = parse_response_text(lm_response, reasoning_model, option_list)
            if answer is None or probability is None or probability == 0:
                answer, probability = get_random_answer_prob(num_choices)
            if i == 0:
                first_answer = answer
            else:
                subsequent_answers.append(answer)
                subsequent_probabilities.append(probability)
        # denominator is the sum of subsequent probabilities
        denominator = sum(subsequent_probabilities)
        # numerator is the sum of subsequent probabilities where the answer is the same as the first answer
        numerator = sum([subsequent_probabilities[i] for i in range(len(subsequent_answers)) if options_are_equal(subsequent_answers[i], first_answer)])
        avg_conf = numerator / denominator
        return {first_answer: avg_conf}
    elif reasoning_model in ['vpgm-n2', 'vpgm-n3', 'vpgm-n4']:
        assert len(lm_response_list) > 1
        avg_probabilities = np.zeros(num_choices)
        cnt = 0
        for lm_response in lm_response_list:
            probabilities = parse_response_text(lm_response, reasoning_model, option_list)
            if len(probabilities) != num_choices:
                if len(probabilities) == 0:
                    avg_probabilities += np.array([1 / num_choices] * num_choices)
                    cnt += 1
                else:
                    continue
            else:
                # apply renormalization, we use prob/sum(probs), if sum(probs) is 0, we use uniform distribution
                probabilities = np.array(probabilities)
                sum_probs = np.sum(probabilities)
                if sum_probs == 0:
                    probabilities = np.array([1 / num_choices] * num_choices)
                else:
                    probabilities /= sum_probs

                avg_probabilities += probabilities
                cnt += 1
        if cnt == 0:
            # if no cnt, return uniform distribution over options
            return {option_list[i]: 1 / num_choices for i in range(num_choices)}
        else:
            avg_probabilities /= cnt
            return {option_list[i]: avg_probabilities[i] for i in range(num_choices)}


def pid_seqs_to_icl_exemplars_text(pid_seq, reasoning_model, exemplars):
    """
    Convert pid_seq to a text consists of ICL exemplers
    :param pid_seq: list of str, each element is a pid in str format
    :param reasoning_model: str, reasoning model
    :param exemplars: list of dict, each dict is an exemplar, loaded from the json file
    :return: a single text for all ICL exemplars and the list of ICL exemplars
    """
    assert reasoning_model == 'vpgm-n2'

    output_text = ""
    output_list = []
    # create a dict of keys pid and value None from pid_seq
    selected_exemplar_seq = {pid: None for pid in pid_seq}
    # iterate over exemplars
    for exemplar in exemplars:
        # if pid in pid_seq_dict
        if exemplar['pid'] in selected_exemplar_seq:
            # append the exemplar to selected_exemplars
            selected_exemplar_seq[exemplar['pid']] = exemplar
    # transform the dict to a list of exemplars
    selected_exemplar_seq = [v for k, v in selected_exemplar_seq.items() if v is not None]
    # iterate over selected_exemplar_seq
    for selected_exemplar in selected_exemplar_seq:
        question = selected_exemplar['question']
        choices = selected_exemplar['choices']
        # parse options
        inds = ["A", "B", "C", "D", "E"]
        choice_list = [f"({inds[i]}) {choices[i]}" for i in range(len(choices))]
        options = " ".join(choice_list)
        metadata = selected_exemplar['metadata']
        retrieved_knowledge = selected_exemplar['knowledge']
        hint = selected_exemplar['hint']
        image_caption = selected_exemplar['caption']
        lm_response = selected_exemplar['lm_response'][reasoning_model]
        curr_text = f'Question: {question}\n\nContext: Select the better answer. {hint}\n\nOptions: {options}\n\nMetadata: {metadata}\n\nImage caption: {image_caption}\n\nRetrieved knowledge: {retrieved_knowledge}\n\nSolution:\n{lm_response}'
        output_text = output_text + curr_text + '\n\n'
        output_list.append(curr_text)
    return output_text, output_list
