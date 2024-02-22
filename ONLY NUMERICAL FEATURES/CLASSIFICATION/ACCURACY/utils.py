import numpy as np
import torch
import tqdm.auto as tqdm
import gpytorch

#### Define early stopping function
class EarlyStopping:
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
def train(model, criterion, optimizer, training_iterations, train_loader, val_loader, early_stopping):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    n_epochs=0
    for _ in iterator:
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

            iterator.set_postfix(loss=loss.item())

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
            model.load_state_dict(torch.load('checkpoint.pt'))
            n_epochs=n_epochs-early_stopping.patience
            break

    return n_epochs


def train_trans(model, criterion, optimizer, training_iterations, train_loader, val_loader, early_stopping):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    n_epochs=0
    for _ in iterator:
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

            iterator.set_postfix(loss=loss.item())

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
            model.load_state_dict(torch.load('checkpoint.pt'))
            n_epochs=n_epochs-early_stopping.patience
            break

    return n_epochs

def train_no_early_stopping(model, criterion, optimizer, training_iterations, train_loader):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    n_epochs=0
    for _ in iterator:
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

            iterator.set_postfix(loss=loss.item())

#### Define train function for transformer
def train_trans_no_early_stopping(model, criterion, optimizer, training_iterations, train_loader):
    iterator = tqdm.tqdm(range(training_iterations), desc="Train")

    n_epochs=0
    for _ in iterator:
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

            iterator.set_postfix(loss=loss.item())


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


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, train_x, kernel):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(self, train_x, variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred