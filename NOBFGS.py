import numpy as np
#%%


def Minimize(psi,psi_grad,max_iterations,batch_size,sample_set,alpha=1,gamma=0.2,tau=10,mu=0.1):
    # First iterate is the zero vector
    input_dimension = len(sample_set[0][0]);

    # Initialize the first iterate to 0
    W = np.zeros(input_dimension);

    # Initialize the Inverse Hessian Approximation
    H = np.eye(input_dimension);

    # Define the gradient holder
    gradient = np.zeros(input_dimension);

    # Create the place holder for the iterates
    W = np.zeros((max_iterations,input_dimension));

    # Initialize the velocity
    V = np.zeros(input_dimension);

    for i in range(max_iterations-1):
        # Sample a batch from the sample_set
        indices = np.random.randint(-1,len(sample_set[0]),batch_size)
        batch = [];

        for j in indices:
            batch.append((sample_set[0][j],sample_set[1][j]));

        # Approximate the gradient of psi using the batch
        for sample in batch:
            gradient += psi_grad(W[i] + mu*V,sample[0],sample[1]);

        gradient = gradient / batch_size;

        # Calculate the direction
        direction = (-1) * np.dot(H,gradient);

        # Normalize the direction
        direction = direction / np.linalg.norm(direction);

        # Set the step rate
        rate = ( tau / (tau + i) ) * alpha;

        V = mu*V + rate*direction;

        # Update the iterate
        W[i+1] = W[i] + V;

        # Define the next gradient holder
        next_gradient = np.zeros(input_dimension);

        # Approximate the gradient of phi using the batch and psi
        for sample in batch:
            next_gradient += psi_grad(W[i+1],sample[0],sample[1]);

        next_gradient = next_gradient / batch_size;

        step_dif = W[i+1] - (W[i] + mu*V);
        grad_dif = next_gradient - gradient + gamma*step_dif;
        rho = 1 / np.inner(grad_dif,step_dif);

        # Identity matrix
        I = np.eye(input_dimension);

        P1 = I - rho*np.outer(step_dif,grad_dif);
        P2 = I - rho*np.outer(grad_dif,step_dif);

        # New Approximation
        H = P1@H@P2 + rho*np.outer(step_dif,step_dif);


    return W[-1];
