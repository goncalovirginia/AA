import random

def flipCoin(headsProbability, numTosses):
    return [1 if random.random() <= headsProbability else 0 for _ in range(numTosses)]

def headsEstimator(coinFlipResults) :
    return coinFlipResults.count(1) / len(coinFlipResults)

print(headsEstimator(flipCoin(0.6, 10000)))