class MultiplyActivation(nn.Module):
    def __init__(self):
        super(MultiplyActivation, self).__init__()

    def forward(self, input_set1, input_set2):
        # Element-wise multiplication of the two sets of neurons
        result = input_set1 * input_set2

        return result
        
mat_size= size*size
mul = 7

# Define the neural network architecture
class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.input_a = nn.Linear(mat_size, mul,bias=False)  # Input layer to hidden layer
        self.input_b = nn.Linear(mat_size, mul,bias=False)  # Input layer to hidden layer

        self.output_c = nn.Linear(mul, mat_size,bias=False)  # Hidden layer to output layer

        
        self.multiply_activation = MultiplyActivation()
        
        nn.init.xavier_uniform_(self.input_a.weight)
        nn.init.xavier_uniform_(self.input_b.weight)
        nn.init.xavier_uniform_(self.output_c.weight)


    def forward(self, a,b):
        a,b = self.input_a(a), self.input_b(b)
        
        x = self.multiply_activation(a,b)
        
        x = self.output_c(x)

        return x
    

model = SimpleANN()

criterion = nn.MSELoss() 

optimizer = optim.Adam(model.parameters(), lr=0.05)

