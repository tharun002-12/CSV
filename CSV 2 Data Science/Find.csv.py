import pandas as pd
import numpy as np
data = pd.read_csv("C:\CC EXP\Find.csv")
print(data, "\n")
d = np.array(data.iloc[:, :-1])
print("\nThe attributes are:", d)
target = np.array(data.iloc[:, -1])
print("\nThe target is:", target)

def train(c, t):
    specific_hypothesis = [""] * len(c[0])
    for i, val in enumerate(t):
        if val == "Yes":
            specific_hypothesis = c[i].copy()
            break
    for i, val in enumerate(c):
        if t[i] == "Yes":
             for x in range(len(specific_hypothesis)):
                 if val[x] != specific_hypothesis[x]:
                     specific_hypothesis[x] = '?'
                 else:
                     pass
    return specific_hypothesis
print("\nThe final hypothesis is:", train(d, target))
