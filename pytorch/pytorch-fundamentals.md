# Pytorch fundamentals

| **Introduction to tensors**          | Tensors are the basic building block of all of machine learning and deep learning.                                                                                                                                                                  |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Creating tensors**                 | Tensors can represent almost any kind of data (images, words, tables of numbers).                                                                                                                                                                   |
| **Getting information from tensors** | If you can put information into a tensor, you'll want to get it out too.                                                                                                                                                                            |
| **Manipulating tensors**             | Machine learning algorithms (like neural networks) involve manipulating tensors in many different ways such as adding, multiplying, combining.                                                                                                      |
| **Dealing with tensor shapes**       | One of the most common issues in machine learning is dealing with shape mismatches (trying to mixed wrong shaped tensors with other tensors).                                                                                                       |
| **Indexing on tensors**              | If you've indexed on a Python list or NumPy array, it's very similar with tensors, except they can have far more dimensions.                                                                                                                        |
| **Mixing PyTorch tensors and NumPy** | PyTorch plays with tensors ([`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html)), NumPy likes arrays ([`np.ndarray`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html)) sometimes you'll want to mix and match these. |
| **Reproducibility**                  | Machine learning is very experimental and since it uses a lot of _randomness_ to work, sometimes you'll want that _randomness_ to not be so random.                                                                                                 |
| **Running tensors on GPU**           | GPUs (Graphics Processing Units) make your code faster, PyTorch makes it easy to run your code on GPUs.                                                                                                                                             |

> Introduction to tensors&#x20;

* Fundamental building block of ML, job is to represent data in numerical way.
  *   Creating tensors

      * can use torch.tensor()
      *   **Scalar**&#x20;

          * &#x20;a single number and in tensor-speak it is a zero dimension tensor.
          * Although scalar is a single number it is of type torch.tensor

          <pre class="language-python" data-overflow="wrap" data-full-width="true"><code class="lang-python">scalar=torch.tensor(7)  # initialize a scalar tensor as 7
          <strong>scalar.ndim  # gives the number of dimension of scalar    
          </strong></code></pre>



      *   **Vector**

          * Single dimension tensor that can contain many numbers

          ```python
          vector = torch.tensor([7, 7])   #creates the vector tensor 
          vector.ndim #gets the number of dimension of vector
          ```



      *   **Matrix**

          * Matrices are flexible as vectors except they got an extra dimension

          `matrix= torch.tensor([[7,8],[9,10]]) # initialize the matrix matrix.ndim # get the number of dimension of matrix`\
          `matrix.shape # shape gives us number of rows and columns`



      * Random tensors

      `#Create a random tensor of size (3, 4)`

      `random_tensor = torch.rand(size=(3, 4)) random_tensor,                 random_tensor.dtype`&#x20;
  * ```python
    # Create a random tensor of size (224, 224, 3)
    random_image_size_tensor = torch.rand(size=(224, 224, 3))
    random_image_size_tensor.shape, random_image_size_tensor.ndim
    ```



* Zeros and ones&#x20;
