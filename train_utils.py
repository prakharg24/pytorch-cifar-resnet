import torch

def train_loop(model, trainloader, optimizer, loss_criterion, device, scheduler=None):
    model.train()
    train_loss = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    if scheduler:
        scheduler.step()
    return train_loss/len(trainloader)

def eval_loop(model, testloader, loss_criterion, device):
    model.eval()
    test_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = loss_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100.*correct/total
    return test_loss/len(testloader), acc
