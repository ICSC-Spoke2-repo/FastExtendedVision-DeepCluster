import torch

# Checkpoints (to save model parameters during training)
# this is implemented by writing a python class that uses the torch.save method
class SaveBestModel:
    def __init__(
        self, 
        model_name: str = 'best_model',
        best_valid_loss=float('inf')
    ): #object initialized with best_loss = +infinite
        self.best_valid_loss = best_valid_loss
        self.model_name = model_name
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            # method to save a model (the state_dict: a python dictionary object that 
            # maps each layer to its parameter tensor) and other useful parametrers
            # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'{self.model_name}.pth')
            
            
class SaveBestOptunedModel:
    def __init__(
        self, 
        model_name: str = 'best_model'
    ): #object initialized with best_loss = +infinite
        self.model_name = model_name
        
    def __call__(
        self, 
        trial, model, optimizer, criterion
    ):
        # method to save a model (the state_dict: a python dictionary object that 
        # maps each layer to its parameter tensor) and other useful parametrers
        # see: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save({
            'trial': trial,
            'model_state_dict': model.state_dict(),
            }, f'{self.model_name}.pth')