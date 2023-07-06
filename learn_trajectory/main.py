import random
import json
import numpy as np
from markov_model import MarkovModel
import time


def last_location_prompt(prompt, trajectory):
    beginning = trajectory[:-1]
    return prompt.format(beginning), trajectory


def start_end_prompt(prompt, trajectory):
    beginning, end = trajectory[0], trajectory[-1]
    return prompt.format(beginning, end), trajectory


def start_end_multiple_prompt(prompt, trajectory):
    """format(beginning sub-trajectory, end sub-trajectory), middle sub-trajectory"""
    pass


def modified_trajectory_prompt(prompt, trajectory):
    """choose a point in the trajectory and try to modify it to something that is an impossible transition"""
    """ return modified trajectory, "no" or original trajectory, "yes" as a result"""
    pass


def hide_and_seek_prompt(prompt, trajectory):
    """ given start, end, and maybe intermediate location with some restrictions, generate a possible trajectory"""
    if len(trajectory) < 3:
        raise ValueError("Length must be at least 3.")
    middle_index = random.randint(1, len(trajectory) - 2)
    beginning, middle, end = trajectory[0], trajectory[middle_index], trajectory[-1]
    return prompt.format(beginning, middle, (middle_index + 1), end), trajectory


def create_prompt_from_trajectories(prompt_type, trajectories):
    input_mapping = {
        "Past locations in the trajectory were {0}. The last location is unknown. What is the entire trajectory?": last_location_prompt,
        "Given start location {0} and end location {1}, what is the entire trajectory?": start_end_prompt,
        "Given start locations {0} and end locations {2}. What are locations in between?": start_end_multiple_prompt,
        "Is example trajectory {0} consistent with the behavior of a traveler?": modified_trajectory_prompt,
        "Given start location {0}, intermediate location {1} at step {2}, and end location {3}, "
        "what is the entire trajectory?": hide_and_seek_prompt
    }
    print("num keys:", len(input_mapping.keys()))
    prompt_types = list(input_mapping.keys())
    for trajectory in trajectories:
        instruction = "The task is to predict some unknown locations visited by a traveler along" \
                      f" a trajectory of length {len(trajectory)}."
        function_input = prompt_types[prompt_type]
        yield instruction, *input_mapping[function_input](function_input, trajectory)


def get_model():
    states = ['start', 'A', 'B', 'C', 'D', 'E']
    model = MarkovModel(states)
    # Define transition probabilities
    model.add_transition('start', 'A', 0.2)
    model.add_transition('start', 'B', 0.2)
    model.add_transition('start', 'C', 0.2)
    model.add_transition('start', 'D', 0.2)
    model.add_transition('start', 'E', 0.2)
    model.add_transition('A', 'A', 0.1)
    model.add_transition('A', 'B', 0.4)
    model.add_transition('A', 'C', 0.5)
    model.add_transition('B', 'C', 0.7)
    model.add_transition('B', 'D', 0.3)
    model.add_transition('C', 'E', 1.0)
    model.add_transition('D', 'A', 0.2)
    model.add_transition('D', 'E', 0.8)
    model.add_transition('E', 'B', 0.9)
    model.add_transition('E', 'D', 0.1)
    return model


def main():
    print("Hello TrajectoryGenerator")
    # Example usage

    model = get_model()
    print(f"Transition probabilities from state A: {model.get_transition_probabilities('A')}")
    print(f"Transition probabilities from state B: {model.get_transition_probabilities('B')}")
    print(f"Transition probabilities from state C: {model.get_transition_probabilities('C')}")
    print(f"Transition probabilities from state D: {model.get_transition_probabilities('D')}")
    print(f"Transition probabilities from state E: {model.get_transition_probabilities('E')}")

    # Generate a sequence of states
    N = 25000
    chain_length = 10
    chains = model.get_chains('start', chain_length, N, include_start_state=False)
    num_prompt_types = 5
    for prompt_type in range(num_prompt_types):
        if prompt_type != 1:
            continue
        prompts = create_prompt_from_trajectories(prompt_type, chains)
        start_time = time.time()
        prompts_list = [prompt for prompt in prompts]
        print(len(prompts_list))
        end_time = time.time()
        print(f'Elapsed time: {end_time - start_time} seconds')
        with open(f'prompt_type_{prompt_type}_{N}.json', 'w') as file:
            outputs = []
            for (instruction, input_, response) in prompts_list:
                response = str(response).replace('[', '').replace(']', '').replace('\'', '')
                output = {
                    'instruction': instruction,
                    'input': input_,
                    'output': response
                }
                outputs.append(output)
            json.dump(outputs, file)
            # file.write(f'{instruction}\n{input_}\n{response}\n')


if __name__ == '__main__':
    # main()
    model_ = get_model()
    chain = model_.get_chain('start', 10, include_start_state=False)
    next_chain = model_.get_chain('start', 10, include_start_state=False)
    print(model_.get_log_probability_for_sequence(chain))
    print(model_.get_log_probability_for_sequence(next_chain))
    factor = 0.8
    chain_prob = model_.get_log_probability_for_sequence(chain)
    next_chain_prob = model_.get_log_probability_for_sequence(next_chain)
    print("probability of chain 1:", np.exp(chain_prob))
    print("probability of chain 2:", np.exp(next_chain_prob))
    print("chain prob tolerably > chain 2 prob:", chain_prob > next_chain_prob + np.log(factor))
    print("chain 2 prob tolerably > chain prob:", next_chain_prob > chain_prob + np.log(factor))
    print("chain prob > chain 2 prob:", chain_prob > next_chain_prob)
    print("chain 2 prob > chain prob:", next_chain_prob > chain_prob)

