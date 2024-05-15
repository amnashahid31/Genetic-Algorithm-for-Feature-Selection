# Importing necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Defining folder paths
data_dir = r'C:\Users\Downloads\AI-Assignment2'  
input_dir = os.path.join(data_dir, 'dataset')  
output_dir = os.path.join(data_dir, 'filtered_dataset')  

# Ensuring output folder existence
os.makedirs(output_dir, exist_ok=True)

# Columns to be removed
cols_to_drop = list(range(1, 3)) + list(range(298, 433))

# Loading and preprocessing data
filtered_data = pd.DataFrame()
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        parts = filename.split("-")
        mod = parts[0]
        emotion = parts[2]
        intensity = parts[3]
        actor = int(parts[-1].split(".")[0])

        if mod == "01" and (emotion == "03" or emotion == "04") and intensity == "01" and actor <= 10:
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path)

            # Adding emotion column
            df['emotion'] = 'happy' if emotion == "03" else 'sad'
            
            # Dropping specified columns
            df.drop(df.columns[cols_to_drop], axis=1, inplace=True)

            # Appending to main dataframe
            filtered_data = pd.concat([filtered_data, df], ignore_index=True)

            # Saving modified data
            output_file = os.path.join(output_dir, filename)
            df.to_csv(output_file, index=False)

# Retrieving filtered dataframe shape
num_rows, num_cols = filtered_data.shape

print("Data filtering completed.")
print("Filtered dataset loaded and processed.")
print("Filtered dataframe shape:", num_rows, "rows,", num_cols, "columns")

# Function to calculate accuracy using a basic neural network
def calculate_accuracy(selected_cols):
    # Preparing data
    X = filtered_data.drop(columns=['emotion'])
    X_selected = X.iloc[:, selected_cols]
    y = filtered_data['emotion']
    
    # Encoding labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_categorical, test_size=0.2, random_state=42)
    
    # Building neural network model
    n_outputs = 2
    model = Sequential()
    model.add(Dense(120, input_dim=X_train.shape[1], activation='relu'))  # Define input layer with 120 neurons
    model.add(Dense(80, activation='relu'))  # Define hidden layer 1 with 80 neurons
    model.add(Dense(80, activation='relu'))  # Define hidden layer 2 with 80 neurons
    model.add(Dense(n_outputs, activation='softmax'))  # Define output layer with softmax activation
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  # Compile model
    
    # Training the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Train model
    
    # Evaluating the model
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)  # Evaluate model
    
    return accuracy

# Genetic Algorithm Parameters
pop_size = 10  # Population size
chrom_length = len(filtered_data.columns) - 1  # Chromosome length
generations = 3  # Number of generations

# Initializing population
population = np.random.randint(2, size=(pop_size, chrom_length))  # Initialize population

# Genetic Algorithm
for gen in range(generations):  # Iterate through generations
    print("\nProcessing Generation:", gen + 1)  # Print generation number
    fitness = []  # Initialize fitness list
    
    # Calculating fitness
    for i, chromosome in enumerate(population):  # Iterate through population
        acc = calculate_accuracy(np.nonzero(chromosome)[0])  # Calculate accuracy
        fitness.append(acc)  # Append accuracy to fitness list
        print(f"Accuracy of Chromosome {i+1}: {acc:.4f}")  # Print accuracy
    
    # Selection: Roulette Wheel
    total_fit = sum(fitness)  # Calculate total fitness
    probs = [fit / total_fit for fit in fitness]  # Calculate probabilities
    selected_idx = np.random.choice(range(pop_size), size=pop_size, p=probs)  # Select indices based on probabilities
    selected_pop = population[selected_idx]  # Select population based on indices
    
    # Crossover: Single-point
    for i in range(0, pop_size, 2):  # Iterate through population by twos
        cross_point = np.random.randint(1, chrom_length)  # Choose crossover point
        temp = selected_pop[i, cross_point:].copy()  # Copy genes after crossover point
        selected_pop[i, cross_point:] = selected_pop[i+1, cross_point:]  # Swap genes after crossover point
        selected_pop[i+1, cross_point:] = temp  # Swap genes after crossover point
    
    # Mutation
    mutation_rate = 0.1  # Mutation rate
    for i in range(pop_size):  # Iterate through population
        for j in range(chrom_length):  # Iterate through genes
            if np.random.rand() < mutation_rate:  # Check if mutation occurs
                selected_pop[i, j] = 1 if selected_pop[i, j] == 0 else 0  # Perform mutation
    
    population = selected_pop  # Update population

# Retrieving best chromosome and accuracy
best_idx = np.argmax(fitness)  # Find index of best chromosome
best_chrom = population[best_idx]  # Get best chromosome
best_acc = fitness[best_idx]  # Get accuracy of best chromosome

print("\nBest chromosome:", best_chrom)  # Print best chromosome
print("\nBest accuracy achieved:", best_acc)  # Print best accuracy
