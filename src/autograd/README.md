**What is Autograd?**

In short, automatic differentiation (or AD).
Here, AD is achieved through backpropagation, which is baked into the *scalar* data types, like `Float64`.
By using these data types, the performed ops are automatically recorded, resulting in a DAG by tracing which we can automatically compute the gradients as needed.

## references

https://pytorch.org/docs/stable/notes/autograd.html
