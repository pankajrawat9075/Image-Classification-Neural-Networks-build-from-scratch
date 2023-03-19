# CS6910_Assignment_1 
# roll no. CS22M062 Pankaj Singh Rawat

**Instructions to train and evaluate the neural network models:**

1.Install the required libraries in your environment using this command:
pip install -r requirements.txt

2. To train the neural network on the Fashion-MNIST dataset using cross-entropy loss, use the notebook: Deep_Learning_wandb.ipynb.
  a. In this notebook, to train using the best values for hyperparameters obtained from our use of the wandb sweeps functionality, do not run cells in the section titled "Wandb Train and Hyperparameter tuning". Run all the other cells of the notebook to train the model. Testing is done on the best parameter model.
  
  b. In order to run the hyperparameter search on your own, run the full notebook.
  
3. To train the model for the mean square error, just call the train_wandb function with parameter loss_fuction parameter as "MSE"
 
4. To run the 3 recommendations just run the cells for 3 recommendations given at the end.
