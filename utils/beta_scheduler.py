class BetaScheduler:
    def __init__(
        self,
        β_min: float, 
        β_max: float,
        number_of_epochs: int,
        starting_epoch: int, 
        ending_epoch: int     
    ): #object initialized with best_loss = +infinite
        if β_min >= β_max:
            print(f"Selected costant beta of value: {β_min}")
            #raise Exception("Invalid β_min β_max")
            
        if starting_epoch >= ending_epoch or starting_epoch >= number_of_epochs or ending_epoch >= number_of_epochs or starting_epoch < 0 or ending_epoch  < 0 or number_of_epochs <= 0:
            raise Exception("Invalid epochs")
            
        self.β_min = β_min
        self.β_max = β_max
        
        self.number_of_epochs = number_of_epochs
        self.starting_epoch   = starting_epoch
        self.ending_epoch     = ending_epoch
        
    def __call__(
        self, current_epoch: int
    ):

        try:
            # check if 
            if self.β_min >= self.β_max:
                raise Exception("Return beta min") # this sends to except and retirn  self.β_min
            # Before starting epocjh
            if current_epoch < self.starting_epoch:
                return self.β_min
            # After ending epoch
            elif current_epoch > self.ending_epoch:
                return self.β_max
            # adiabatic process
            elif  current_epoch >= self.starting_epoch and current_epoch <= self.ending_epoch:
                return self.adiabatic_function(current_epoch)
        except:
            return self.β_min
        
    def adiabatic_function(self, current_epoch):
        return self.β_min + (self.β_max - self.β_min)*(current_epoch - self.starting_epoch)/(self.ending_epoch - self.starting_epoch)
    