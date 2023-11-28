import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
dataset_path = 'simulated_metaverse_user_data.csv'  # Replace with the actual file path
df = pd.read_csv(dataset_path)

# Preprocess the dataset
def preprocess_dataframe(df):
    scaler = MinMaxScaler()
    df['Preference_Score'] = scaler.fit_transform(df[['Preference_Score']])  # Normalize preference scores
    return df

df = preprocess_dataframe(df)

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2)

# Convert DataFrame to TensorFlow Dataset
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Preference_Score')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

train_dataset = df_to_dataset(train_df)
test_dataset = df_to_dataset(test_df, shuffle=False)

# Define the model architecture
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)),  # Input layer with ReLU activation
        tf.keras.layers.Dense(1, activation='linear')  # Output layer for predicting preference score
    ])
    return model

# Convert the Keras model to be compatible with TensorFlow Federated
def model_fn():
    keras_model = create_model()
    # Convert the Keras model into a TFF model
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,  # Input specification from example dataset
        loss=tf.keras.losses.MeanSquaredError(),  # Loss function for regression
        metrics=[tf.keras.metrics.MeanAbsoluteError()])  # Metric to evaluate model performance

# Build the federated learning process using federated averaging
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()  # Initialize the federated learning process

# Training loop for federated learning
NUM_ROUNDS = 10
for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, federated_train_data)  # Perform one round of training
    print(f'Round {round_num}, Metrics: {metrics}')  # Print metrics after each round

# Evaluation process for the federated learning model
evaluation = tff.learning.build_federated_evaluation(model_fn)
test_metrics = evaluation(state.model, federated_test_data)  # Evaluate the model on test data
print(test_metrics)  # Print the evaluation metrics
