import torch

x = torch.tensor([1.0], requires_grad=True)
y = x
z = y

print(x)
print(y)
print(z)

w = torch.tensor([1.0])

loss = torch.nn.functional.binary_cross_entropy_with_logits(z, w)

print(loss)

loss.backward()

print(x.grad)