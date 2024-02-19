import numpy as np
import torch
import tqdm.auto as tqdm

#### Define early stopping function
class EarlyStopping:
    def __init__(self, patience=40, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

#### Define train function
def train(model, criterion, loss_Adam, optimizer, training_iterations, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, early_stopping):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    n_epochs=0
    for _ in iterator:
        n_epochs += 1
        # making a prediction in forward pass
        y_train_hat = model(X_train_tensor).reshape(-1,)
        # calculating the loss between original and predicted data points
        loss = criterion(y_train_hat, y_train_tensor)
        # store loss into list
        loss_Adam.append(loss.item())
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()
        iterator.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()

        # validate the model 
        y_val_hat = model(X_val_tensor).reshape(-1,)
        val_loss = criterion(y_val_hat, y_val_tensor)

        # check if early stopping condition is met
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return n_epochs


#### Define train function for transformer
def train_trans(model, criterion, loss_Adam, optimizer, training_iterations, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, early_stopping):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    n_epochs=0
    for _ in iterator:
        n_epochs += 1
        # making a pridiction in forward pass
        y_train_hat = model(X_train_tensor, None).reshape(-1,)
        # calculating the loss between original and predicted data points
        loss = criterion(y_train_hat, y_train_tensor)
        # store loss into list
        loss_Adam.append(loss.item())
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()
        iterator.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()

        # validate the model 
        y_val_hat = model(X_val_tensor, None).reshape(-1,)
        val_loss = criterion(y_val_hat, y_val_tensor)

        # check if early stopping condition is met
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
        return n_epochs

def train_no_early_stopping(model, criterion, loss_Adam, optimizer, training_iterations, X_train_tensor, y_train_tensor):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    for _ in iterator:
        # making a prediction in forward pass
        y_train_hat = model(X_train_tensor).reshape(-1,)
        # calculating the loss between original and predicted data points
        loss = criterion(y_train_hat, y_train_tensor)
        # store loss into list
        loss_Adam.append(loss.item())
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()
        iterator.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()

#### Define train function for transformer
def train_trans_no_early_stopping(model, criterion, loss_Adam, optimizer, training_iterations, X_train_tensor, y_train_tensor):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    for _ in iterator:
        # making a pridiction in forward pass
        y_train_hat = model(X_train_tensor, None).reshape(-1,)
        # calculating the loss between original and predicted data points
        loss = criterion(y_train_hat, y_train_tensor)
        # store loss into list
        loss_Adam.append(loss.item())
        # zeroing gradients after each iteration
        optimizer.zero_grad()
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        # updating the parameters after each iteration
        optimizer.step()
        iterator.set_postfix(loss=loss.item())
        torch.cuda.empty_cache()

def train_GP(model,X_train_tensor,y_train_tensor,training_iterations,mll,optimizer):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    for _ in iterator:
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(X_train_tensor)
        # Calc loss and backprop derivatives
        loss = -mll(output, y_train_tensor)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        torch.cuda.empty_cache()


