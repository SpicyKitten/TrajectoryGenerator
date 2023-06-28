import numpy as np


class MarkovModel:
    def __init__(self, states):
        self.states = states
        self.transition_matrix = np.zeros((len(states), len(states)))
        print(f"Transition matrix dimensions: {self.transition_matrix.shape}")

    def get_transition_probabilities(self, current_state):
        current_index = self.states.index(current_state)
        if np.sum(self.transition_matrix[current_index, :]) == 0:
            raise AssertionError(f'state {current_state} doesn\'t transition')
            # self.transition_matrix[current_index, current_index] = 1
        return self.transition_matrix[current_index, :]

    def add_transition(self, source, destination, probability):
        source_index = self.states.index(source)
        dest_index = self.states.index(destination)
        self.transition_matrix[source_index, dest_index] = probability

    def get_next_state(self, current_state):
        transition_probabilities = self.get_transition_probabilities(current_state)
        next_state_index = np.random.choice(
            a=len(self.states),
            p=transition_probabilities
        )
        next_state = self.states[next_state_index]
        return next_state

    def get_chains(self, start_state, sequence_length, num_chains, include_start_state=True):
        for _ in range(num_chains):
            yield self.get_chain(start_state, sequence_length, include_start_state)

    def get_chain(self, start_state, sequence_length, include_start_state=True):
        sequence = [start_state] if include_start_state else []
        sequence_length = sequence_length if include_start_state else sequence_length + 1
        current_state = start_state
        for _ in range(sequence_length - 1):
            next_state = self.get_next_state(current_state)
            sequence.append(next_state)
            current_state = next_state
        return sequence
