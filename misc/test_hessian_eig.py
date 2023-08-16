import torch as t
from torch.autograd import grad
from scipy.sparse.linalg import LinearOperator, eigsh

class QuadModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = t.nn.Parameter(t.tensor(0.))
        self.b = t.nn.Parameter(t.tensor(0.))

    def forward(self):
        return 2 * self.a ** 2 + 3 * self.b ** 2 + 2 * self.a * self.b

def get_hessian_eigenvalues(model, device="cpu"):
    
    num_params = sum(p.numel() for p in model.parameters())

    def compute_loss():
        return model()

    def hessian_vector_product(vector):
        model.zero_grad()
        grad_params = grad(compute_loss(), model.parameters(), create_graph=True)
        flat_grad = t.cat([g.view(-1) for g in grad_params])
        grad_vector_product = t.sum(flat_grad * vector)
        hvp = grad(grad_vector_product, model.parameters(), retain_graph=True)
        return t.cat([g.contiguous().view(-1) for g in hvp])
    
    def matvec(v):
        v_tensor = t.tensor(v, dtype=t.float32, device=device)
        return hessian_vector_product(v_tensor).cpu().detach().numpy()

    linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
    eigenvalues, _ = eigsh(linear_operator, k=num_params-1, which='SM')
    # print hessian by multiplying the linear operator with unit vectors
    for i in range(num_params):
        v = t.zeros(num_params, dtype=t.float32, device=device)
        v[i] = 1.0
        v = v.cpu().detach().numpy()
        print(matvec(v))
    for e in eigenvalues:
        print("{:.2f}".format(e))

if __name__ == "__main__":
    model = QuadModel().to("cpu")
    get_hessian_eigenvalues(model)
    # [4. 0.]
    # [0. 6.]
    # 6.00
