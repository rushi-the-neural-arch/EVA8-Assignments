import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(epoch, optimizer, model, trainloader, criterion, train_loss_list):

  running_loss = 0.0
  epoch_loss = 0.0
  for i, data in enumerate(trainloader, 0):
      # get the inputs
      inputs, labels = data
      inputs = inputs.to(device)
      labels = labels.to(device)
      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = model(inputs)
      outputs = outputs.to(device)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      epoch_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
          print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
          running_loss = 0.0
  epoch_loss = epoch_loss / 50000
  print("Training Loss of the network on the 50000 train images:", epoch_loss)
  train_loss_list.append(epoch_loss)

def test(model, testloader, criterion, test_loss_list):
  correct = 0
  total = 0
  epoch_loss = 0
  with torch.no_grad():
      for data in testloader:
          images, labels = data
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          loss = criterion(outputs, labels)
          epoch_loss += loss.item()
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  epoch_loss = epoch_loss / 10000
  test_loss_list.append(epoch_loss)
  print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
  print('Loss of the network on the 10000 test images:', epoch_loss)