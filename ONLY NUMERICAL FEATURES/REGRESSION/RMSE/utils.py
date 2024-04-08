import numpy as np
import torch
import tqdm.auto as tqdm

#### Define early stopping function
class EarlyStopping():
    def __init__(self, patience=40, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

#### Define train function
def train(model, criterion, optimizer, training_iterations, train_loader, val_loader, early_stopping, checkpoint_path):

    n_epochs=0
    for _ in range(training_iterations):
        n_epochs += 1
        for batch_X, batch_y in train_loader:
            # Move batch to device
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X).reshape(-1,)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            for batch_X, batch_y in val_loader:
                # Move batch to device
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()

                # Forward pass and calculate loss
                outputs = model(batch_X).reshape(-1,)
                batch_loss = criterion(outputs, batch_y)

                # Accumulate batch loss
                val_loss += batch_loss.item()
                num_batches += 1

            # Calculate average validation loss
            val_loss /= num_batches

        # Check if early stopping condition is met
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            # Load the best model parameters
            model.load_state_dict(torch.load(checkpoint_path))
            n_epochs=n_epochs-early_stopping.patience
            break

    return n_epochs


def train_trans(model, criterion, optimizer, training_iterations, train_loader, val_loader, early_stopping, checkpoint_path):

    n_epochs=0
    for _ in range(training_iterations):
        n_epochs += 1
        for batch_X, batch_y in train_loader:
            # Move batch to device
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X, None).reshape(-1,)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()


        # Validation
        with torch.no_grad():
            val_loss = 0
            num_batches = 0
            for batch_X, batch_y in val_loader:
                # Move batch to device
                if torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.cuda()

                # Forward pass and calculate loss
                outputs = model(batch_X, None).reshape(-1,)
                batch_loss = criterion(outputs, batch_y)

                # Accumulate batch loss
                val_loss += batch_loss.item()
                num_batches += 1

            # Calculate average validation loss
            val_loss /= num_batches

        # Check if early stopping condition is met
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            # Load the best model parameters
            model.load_state_dict(torch.load(checkpoint_path))
            n_epochs=n_epochs-early_stopping.patience
            break

    return n_epochs

def train_no_early_stopping(model, criterion, optimizer, training_iterations, train_loader):

    n_epochs=0
    for _ in range(training_iterations):
        n_epochs += 1
        for batch_X, batch_y in train_loader:
            # Move batch to device
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X).reshape(-1,)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

#### Define train function for transformer
def train_trans_no_early_stopping(model, criterion, optimizer, training_iterations, train_loader):

    n_epochs=0
    for _ in range(training_iterations):
        n_epochs += 1
        for batch_X, batch_y in train_loader:
            # Move batch to device
            if torch.cuda.is_available():
                batch_X = batch_X.cuda()
                batch_y = batch_y.cuda()

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X,None).reshape(-1,)
            loss = criterion(outputs, batch_y)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()


