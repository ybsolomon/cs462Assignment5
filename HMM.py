

import random
import argparse
import codecs
import os
import numpy
import random


def getObservations(filename):
    observations = []
    file = open(filename, 'r')
    line = file.readline()

    while line:
        if len(line) > 1:
            observations.append(['#'] + line.split())

        line = file.readline()

    return observations


class HMM:
    def __init__(self, transitions={}, emissions={}):
        self.transitions = transitions
        self.emissions = emissions

    def load(self, basename):
        trans = open(basename + '.trans', 'r')
        line = trans.readline()
        while line:
            words = line.split()
            if not words[0] in self.transitions:
                self.transitions[words[0]] = {'#': 0.0}

            self.transitions[words[0]][words[1]] = float(words[2])
            line = trans.readline()

        trans.close()

        self.emissions['#'] = {}
        emit = open(basename + '.emit', 'r')
        line = emit.readline()
        while line:
            words = line.split()
            if not words[0] in self.emissions:
                self.emissions[words[0]] = {}

            self.emissions[words[0]][words[1]] = float(words[2])
            self.emissions['#'][words[1]] = 0.0
            line = emit.readline()

        emit.close()

    def generate(self, n):
        states = ['#']
        emissions = ['#']
        current = '#'

        for i in range(n):
            random_state = random.choices(population=list(self.transitions[current].keys()),
                                          weights=list(self.transitions[current].values()))[0]

            emission = random.choices(population=list(self.emissions[random_state].keys()),
                                          weights=list(self.emissions[random_state].values()))[0]
            states.append(random_state)
            emissions.append(emission)

            current = random_state

        return states, emissions

    def forward(self, observation):
        rows, cols = (len(self.transitions), len(observation))
        matrix = [[0.0 for i in range(cols)] for j in range(rows)]

        matrix[0][0] = 1.0

        state_num = 0
        for state in self.transitions.keys():
            trans = self.transitions[observation[0]]

            if observation[1] not in self.emissions[state]:
                emission = 0
            else:
                emission = self.emissions[state][observation[1]]

            matrix[state_num][1] = trans[state] * emission

            state_num += 1

        for i in range(2, len(observation)):
            state_num = 0
            for state in self.transitions.keys():
                sum = 0
                s2_num = 0
                for s2 in self.transitions.keys():
                    if observation[i] not in self.emissions[state]:
                        emission_prob = 0
                    else:
                        emission_prob = self.emissions[state][observation[i]]

                    transition_prob = self.transitions[s2][state]

                    sum += matrix[s2_num][i-1]*transition_prob*emission_prob
                    s2_num += 1

                matrix[state_num][i] = sum

                state_num += 1

        final_vals = [row[-1] for row in matrix]
        states = list(self.transitions.keys())
        max_val, max_state = final_vals[0], states[0]
        for i in range(1, len(final_vals)):
            if max_val < final_vals[i]:
                max_val = final_vals[i]
                max_state = states[i]

        return max_state

    # you do this: Implement the Viterbi algorithm. Given an Observation (a list of outputs or emissions)
    # determine the most likely sequence of states.
    def viterbi(self, observation):
        rows, cols = (len(self.transitions), len(observation))
        matrix = [[0.0 for i in range(cols)] for j in range(rows)]
        back_pointers = [[0 for i in range(cols)] for j in range(rows)]

        matrix[0][0] = 1.0

        state_num = 0
        for state in self.transitions.keys():
            trans = self.transitions[observation[0]]

            if observation[1] not in self.emissions[state]:
                emission = 0
            else:
                emission = self.emissions[state][observation[1]]

            matrix[state_num][1] = trans[state] * emission

            state_num += 1

        for i in range(2, len(observation)):
            state_num = 0
            for state in self.transitions.keys():
                s2_num = 0
                max_val = -1
                max_idx = 0
                for s2 in self.transitions.keys():
                    if observation[i] not in self.emissions[state]:
                        emission_prob = 0
                    else:
                        emission_prob = self.emissions[state][observation[i]]

                    transition_prob = self.transitions[s2][state]

                    curr_val = matrix[s2_num][i - 1] * transition_prob * emission_prob
                    if curr_val > max_val:
                        max_val = curr_val
                        max_idx = s2_num

                    s2_num += 1

                matrix[state_num][i] = max_val
                back_pointers[state_num][i] = int(max_idx)

                state_num += 1

        final_vals = [row[-1] for row in matrix]
        states = list(self.transitions.keys())
        max_val, max_state, max_idx = final_vals[0], states[0], 0
        for i in range(1, len(final_vals)):
            if max_val < final_vals[i]:
                max_val = final_vals[i]
                max_idx = i

        state_order = []
        current_idx = max_idx

        for i in reversed(range(1, len(observation))):
            next_point = back_pointers[current_idx][i]

            state_order.append(states[current_idx])
            current_idx = next_point

        state_order.reverse()

        return state_order

    def printMatrix(self, matrix):
        keys = list(self.transitions.keys())
        for i in range(len(self.transitions)):
            print(keys[i], matrix[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get word frequencies from a file.")
    parser.add_argument('file', type=str)

    parser.add_argument('--generate', type=int)
    parser.add_argument('--forward', type=str)
    parser.add_argument('--viterbi', type=str)

    args = parser.parse_args()

    model = HMM()
    model.load(args.file)

    if args.generate:
        generated = model.generate(args.generate)
        print(generated[0])
        print(generated[1])

    if args.forward:
        observations = getObservations(args.forward)
        for observation in observations:
            print(observation)
            print('predicted', model.forward(observation))

    if args.viterbi:
        observations = getObservations(args.viterbi)
        for observation in observations:
            print(observation)
            print('predicted', model.viterbi(observation))
