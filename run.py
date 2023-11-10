from tqdm import tqdm

def train(epoch, model, iterator, optimizer, scheduler, device):
    model.train()
    with tqdm(total=len(iterator)) as pbar:
        for data, label in iterator:
            print(data)
            data, label = data.to(device), label.to(device)
            pbar.update(1)
            hat = model(data)
