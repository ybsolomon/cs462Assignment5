from HMM import HMM


def getObservations(filename):
    observations = []
    file = open(filename, 'r')
    line = file.readline()

    while line:
        if len(line) > 1:
            observations.append(['#'] + line.split())

        line = file.readline()

    return observations


if __name__ == "__main__":
    model = HMM()
    model.load('partofspeech.browntags.trained')

    observ = ['#', 'i', 'shot', 'the', 'elephant', '.']
    observations = getObservations('ambiguous_sents.obs')
    for observation in observations:
        print(observation)
        print(model.viterbi(observation))
        print('predicted final state:', model.forward(observation))
        print()
