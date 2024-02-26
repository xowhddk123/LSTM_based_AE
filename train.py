import numpy as np

import torch
import torch.optim as optim

import tqdm

def trainer(config, model, train_loader, valid_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)  # adam
    loss_fn = torch.nn.MSELoss()  # MSE
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)  # ReduceLROnPlateau : loss 변화량에 기반해 lr 조절
    
    epochs = tqdm(range(1, config.epochs+1))    
    
    ## 학습하기
    best_loss = np.inf
    
    # loss 값 저장
    train_loss_arr = []
    valid_loss_arr = []
    
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        train_iterators = enumerate(train_loader)
        
        loss_arr = []
        
        for i, batch_data in train_iterators:
            

            batch_data = batch_data.to(config.device)
            predict_values = model(batch_data)
            loss = loss_fn(predict_values, batch_data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_arr.append(loss.item())
            
    
        
        model.eval()
        eval_loss = 0
        valid_iterators = enumerate(valid_loader)
        
        with torch.no_grad():
            for i, batch_data in valid_iterators:
                
                batch_data = batch_data.to(config.device)
                predict_values = model(batch_data)
                loss = loss_fn(predict_values, batch_data)
                
                eval_loss += loss.mean().item()
                

        eval_loss /= len(valid_loader)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_model = model
            print('save model')
        else:
            if config.early_stop:
                print("Early Stopping")
                break
        
        scheduler.step(eval_loss)
        
        train_loss_arr.append(np.mean(loss_arr))
        valid_loss_arr.append(eval_loss)
        print(f"epoch : {epoch}, train_loss : {np.mean(loss_arr)}, valid_loss : {eval_loss}")
    
    return best_model, train_loss_arr, valid_loss_arr