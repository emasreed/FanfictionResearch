import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)
df = pd.read_csv("fanfics_metadata.csv")
for col in df.columns:
    print(col)


plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
people = df['work_id'][:10]
print(people)
y_pos = np.arange(len(people))
print(y_pos)
performance = df['kudos'][:10]
print(performance)
error = np.random.rand(len(people))
print(error)

ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')

plt.savefig('img1.png')
print(df)
