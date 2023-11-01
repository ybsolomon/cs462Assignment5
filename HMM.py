

import random
import argparse
import codecs
import os
import numpy
import random

# hmm model
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
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
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

        print(self.transitions.keys())


   ## you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
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

        return emissions

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


    ## you do this: Implement the Viterbi algorithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.
    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """

    def printMatrix(self, matrix):
        keys = list(self.transitions.keys())
        for i in range(len(self.transitions)):
            print(keys[i], matrix[i])


if __name__ == "__main__":
    model = HMM()
    model.load('partofspeech.browntags.trained')

    observ = ['#', 'i', 'shot', 'the', 'elephant', '.']
    observations = getObservations('ambiguous_sents.obs')

    print(observations)
    for observation in observations:
        print('final state:', model.forward(observation))

    # print('final state:', model.forward(observation))



