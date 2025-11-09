from prompts.scienceqa import prompt_cot, prompt_chameleon, prompt_vpgm_n2, inst_vpgm_n2
from prompts.scienceqa_new import prompt_vpgm_n3, prompt_vpgm_n4


def build_lm_input(data_point, reasoning_model, use_these_exemplars=None):
    """

    :param data_point:
    :param reasoning_model:
    :param use_these_exemplars: a single text containing exemplars to use for the reasoning model
    :return:
    """
    question = data_point['question']
    choices = data_point['choices']

    # parse options
    inds = ["A", "B", "C", "D", "E"]
    choice_list = [f"({inds[i]}) {choices[i]}" for i in range(len(choices))]
    options = " ".join(choice_list)

    metadata = data_point['metadata']
    retrieved_knowledge = data_point['knowledge']
    hint = data_point['hint']
    image_caption = data_point['caption']

    if use_these_exemplars is None:
        if reasoning_model == 'cot':
            return f'{prompt_cot}\n\nQuestion: {question}\n\nContext: Select the better answer. {hint}\n\nOptions: {options}\n\nSolution:\n'
        elif reasoning_model in ['chameleon', 'chameleon-plus']:
            return f'{prompt_chameleon}\n\nQuestion: {question}\n\nContext: Select the better answer. {hint}\n\nOptions: {options}\n\nMetadata: {metadata}\n\nImage caption: {image_caption}\n\nRetrieved knowledge: {retrieved_knowledge}\n\nSolution:\n'
        elif reasoning_model == 'vpgm-n2':
            return f'{prompt_vpgm_n2}\n\nQuestion: {question}\n\nContext: Select the better answer. {hint}\n\nOptions: {options}\n\nMetadata: {metadata}\n\nImage caption: {image_caption}\n\nRetrieved knowledge: {retrieved_knowledge}\n\nSolution:\n'
        elif reasoning_model == 'vpgm-n3':
            return f'{prompt_vpgm_n3}\n\nQuestion: {question}\n\nContext: Select the better answer. {hint}\n\nOptions: {options}\n\nMetadata: {metadata}\n\nImage caption: {image_caption}\n\nRetrieved knowledge: {retrieved_knowledge}\n\nSolution:\n'
        elif reasoning_model == 'vpgm-n4':
            return f'{prompt_vpgm_n4}\n\nQuestion: {question}\n\nContext: Select the better answer. {hint}\n\nOptions: {options}\n\nMetadata: {metadata}\n\nImage caption: {image_caption}\n\nRetrieved knowledge: {retrieved_knowledge}\n\nSolution:\n'
        else:
            raise ValueError('Invalid reasoning model:', reasoning_model)
    else:
        assert reasoning_model == 'vpgm-n2'
        return f'{inst_vpgm_n2}\n\n{use_these_exemplars}Question: {question}\n\nContext: Select the better answer. {hint}\n\nOptions: {options}\n\nMetadata: {metadata}\n\nImage caption: {image_caption}\n\nRetrieved knowledge: {retrieved_knowledge}\n\nSolution:\n'
