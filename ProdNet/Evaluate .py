# Evaluate the model on the test set
model.eval() 
test_loss = 0.0

with torch.no_grad():
    for inputs,inputs_, targets in test_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            inputs_ = inputs_.cuda()
            targets = targets.cuda()
        
        
        outputs = model(inputs.float(),inputs_.float())  
        test_loss += criterion(outputs, targets.float()).item()

# Calculate the average test loss
average_test_loss = test_loss / len(test_loader)
print(f"Average Test Loss: {average_test_loss}")

for param_tensor in model.state_dict():
    model.state_dict()[param_tensor].data.round_()
    
for name, param in model.state_dict().items():
    print(f"Parameter name: {name}, Size: {param.size()}")
    print(f"Parameter values:\n{param.data}\n")

