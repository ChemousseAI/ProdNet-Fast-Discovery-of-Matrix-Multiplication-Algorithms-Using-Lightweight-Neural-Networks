def regularisation(model, alpha):
    somme =0

    for param_tensor in model.state_dict():

        #L0 regularisation for each weight
        l0 = abs(model.state_dict()[param_tensor].data)
        

        #L1 regularisation for each weight
        l1 = abs(model.state_dict()[param_tensor].data - 1)

        #L -1 regularisation for each weight
        l_neg1 = abs(model.state_dict()[param_tensor].data + 1)
        
        min_values,_ = torch.min(torch.stack((alpha *l0,l1,l_neg1),dim = 2), dim=2)
        somme += torch.sum(min_values)


    total = sum(p.numel() for p in model.parameters())
    return somme * (torch.tensor(1/total))
    

def zeroVectors(model, mul):

    ac = np.ones(mul) 

    for i in range (mul):
        if( ((torch.all(torch.abs(model.state_dict()['input_a.weight'][i]))) <= 0.5) or ((torch.all(abs(model.state_dict()['input_b.weight'][i]))) <= 0.5)):
            ac[i] = 0

    return (torch.tensor((sum(ac))/mul))
    
# Flatten parameters
flat_parameters = torch.cat([param.view(-1) for param in model.parameters()])
print(flat_parameters)

# Now, reflatten the modified parameters
start_idx = 0
for param in model.parameters():
    end_idx = start_idx + param.numel()  # Get the index of the end of the parameter
    param.data = flat_parameters[start_idx:end_idx].view(param.shape)  # Reshape the flattened parameters
    start_idx = end_idx  # Update the start index for the next parameter


for name, param in model.named_parameters():
    print("Parameter:", name)
    print("Values:", param.data)
    
    
epochs = 10000   # Adjust the number of epochs as needed
alpha = 1/2


for epoch in range(epochs):
    total_loss = 0.0

    for inputs,inputs_, targets in train_loader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            inputs_ = inputs_.cuda()
            targets = targets.cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs.float(),inputs_.float())  # Assuming your inputs are floats

        # Compute the loss
        loss = criterion(outputs, targets.float()) + 2 * regularisation(model, alpha)
        print(f'MSE= {criterion(outputs, targets.float())} and regularisation= {regularisation(model,alpha)}')
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()


    # Print the training loss for each epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(train_loader)}\n")
    epo.append(epoch)
    lo.append(total_loss/len(train_loader))
    
    
    if ((epoch+1) % 500 == 0):
        save_dir = '/kaggle/working/'
        torch.save(model.state_dict(),os.path.join(save_dir, f'2*2_8perceptrons_2MinReg_Alpha0.5_{epoch+1}epochs.pth'))
        



