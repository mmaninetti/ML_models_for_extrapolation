import pandas as pd
import numpy as np 

# Create a DataFrame with random values
df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

# Save the DataFrame to a CSV file
df.to_csv('sasaprovaprova.csv', index=False)