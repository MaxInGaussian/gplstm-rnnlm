import torch

N, D_in, H, D_out = 64, 1000, 100, 10

device = torch.device('cpu')
x = torch.randn(N, D_in, device=device)
y = torch.randn(N, D_out, device=device)

# w1 = torch.randn(D_in, H, device=device, requires_grad=True)
# w2 = torch.randn(H, D_out, device=device, requires_grad=True)
w = []
w.append(torch.randn(D_in, H, device=device, requires_grad=True))
w.append(torch.randn(H, D_out, device=device, requires_grad=True))

learning_rate = 1e-6

for t in range (100):
    # y_pred = x.mm(w[0]).clamp(min=0.).mm(w[1])
    # y_opred = torch.empty(N, D_out, requires_grad=True)
    y_preds = []
    y_mid = x.mm(w[0]).clamp(min=0.)
    for i in range(D_out):
        wgt = w[1][:, i].unsqueeze(1)
        # y_pred[:, i]  = torch.mm(y_mid, wgt).squeeze()
        a = torch.mm(y_mid, wgt).squeeze()
        y_preds.append(a)
    y_pred = torch.stack(y_preds, 1)
        

    loss = (y_pred-y).pow(2).sum()
    print (t, loss.item())
    loss.backward()
    with torch.no_grad():
        w[0] -= learning_rate*w[0].grad
        w[1] -= learning_rate*w[1].grad
        w[0].grad.zero_()
        w[1].grad.zero_()
