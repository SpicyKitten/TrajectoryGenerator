import random
import json
import numpy as np
from markov_model import MarkovModel
import time


def trajectory_prompt(prompt, trajectory):
    """choose a point in the trajectory and try to modify it to something that is an impossible transition"""
    """ return modified trajectory, "no" or original trajectory, "yes" as a result"""
    return prompt.format(trajectory), len(trajectory)


def create_prompt_from_trajectories(prompt_type, trajectories):
    input_mapping = {
        "The trajectory is {0}. What is the trajectory's length?": trajectory_prompt,
    }
    print("num keys:", len(input_mapping.keys()))
    prompt_types = list(input_mapping.keys())
    for trajectory in trajectories:
        instruction = "The task is to identify the length of a trajectory visited by a traveler."
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
    N = 35000
    min_length = 1
    max_length = 30
    chains = model.get_chains('start', min_length, max_length, N, include_start_state=False)
    num_prompt_types = 1
    for prompt_type in range(num_prompt_types):
        if prompt_type != 0:
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
    main()
