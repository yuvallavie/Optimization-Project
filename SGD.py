import numpy as np
#%%
def Minimize(psi,psi_grad,alpha,max_iterations,batch_size,sample_set):

    # First iterate is the zero vector
    input_dimension = len(sample_set[0][0]);

    W = np.zeros(input_dimension);
    # Define the gradient holder
    gradient = np.zeros(input_dimension);

    for i in range(max_iterations):


        # Sample a batch from the sample_set
        indices = np.random.randint(0,len(sample_set[0]),batch_size)
        batch = [];

        for j in indices:
            batch.append((sample_set[0][j],sample_set[1][j]));


        # loss = [psi(W,sample_set[0][k],sample_set[1][k]) for k in range(len(sample_set))];
        # print(np.mean(loss))


        # Approximate the gradient of phi using the batch and psi
        for sample in batch:
            gradient += psi_grad(W,sample[0],sample[1]);

        gradient = gradient / batch_size;

        # Calculate the direction
        direction = (-1) * gradient;

        # Update the iterate
        W = W + alpha * direction;

    return W;










