Summary of implemented (or to be implemented) Gradient Descent based algorithms
All these algorithms can be adapted to use parallel gradient computation to speed up the optimization, especially for problems where the gradient computation is a bottleneck.


1. Gradient Descent (GD)

    Description: Updates parameters in the opposite direction of the gradient of the loss function. Typically uses the entire dataset for each update.
    Advantages:
        Simple and easy to implement.
        Works well for convex optimization problems.
    Disadvantages:
        Requires careful tuning of the learning rate.
        Slow for large datasets.
        Can get stuck in local minima or saddle points.


2. Momentum

    Description: Enhances SGD by adding a fraction of the previous update vector to the current update, accelerating convergence.
    Advantages:
        Speeds up convergence, especially in the presence of shallow, narrow valleys.
        Helps in smoothing out noisy gradients.
    Disadvantages:
        Adds an extra hyperparameter (momentum term).
        Can overshoot if not tuned properly.


3. Nesterov Accelerated Gradient (NAG)

    Description: A variation of momentum that computes gradients at the lookahead position, providing a correction factor.
    Advantages:
        More accurate update direction than standard momentum.
        Can lead to faster convergence.
    Disadvantages:
        More complex to implement.
        Hyperparameter tuning is required.


4. Adagrad -> Has an issue, does not converge

    Description: Adapts learning rates by dividing by the square root of the sum of all past squared gradients, providing smaller updates for frequently updated parameters.
    Advantages:
        Eliminates the need to manually tune the learning rate.
        Well-suited for sparse data.
    Disadvantages:
        Learning rates decay too aggressively, leading to premature convergence.
        May stop learning altogether if the learning rate becomes too small.


5. Adadelta -> Has an issue, does not converge

    Description: An extension of Adagrad that limits the accumulation of past gradients by using a moving window of squared gradients, addressing the issue of diminishing learning rates.
    Advantages:
        No need to set a default learning rate.
        Learning rates remain adaptable without diminishing too much.
    Disadvantages:
        Adds additional hyperparameters to control the window size.
        Slightly more complex to implement.


6. RMSProp

    Description: Similar to Adadelta, it uses a moving average of squared gradients to normalize the gradient, preventing the learning rate from decaying too quickly.
    Advantages:
        Adapts learning rates dynamically based on gradient history.
        Suitable for non-stationary objectives.
    Disadvantages:
        Requires hyperparameter tuning (decay rate and learning rate).
        Can oscillate around the optimum.


7.0 Adam (Adaptive Moment Estimation)

    Description: Combines RMSProp and momentum by maintaining moving averages of both the gradients and their squares.
    Advantages:
        Adaptive learning rates for each parameter.
        Handles sparse gradients well.
        Robust to noisy gradients.
    Disadvantages:
        Requires tuning of multiple hyperparameters (learning rate, beta1, beta2).
        Can converge to suboptimal solutions in some cases due to aggressive updates.


7.1 NAdam ((Nesterov Accelerated) Adaptive Moment Estimation)

    Description: Combines RMSProp and momentum by maintaining moving averages of both the gradients and their squares.
    Advantages:
        Improves convergence speed and stability
    Disadvantages:
        Requires tuning of multiple hyperparameters (learning rate, beta1, beta2).
        Can converge to suboptimal solutions in some cases due to aggressive updates.


7.2 RAdam (Rectified Adaptive Moment Estimation)

    Description: Rectified Adam
    Advantages:
        Stabilized Early training
        Adaptive Learning Rate
    Disadvantages:
        Requires tuning of multiple hyperparameters (learning rate, beta1, beta2).
        Can converge to suboptimal solutions in some cases due to aggressive updates.


8. Adamax

    Description: A variant of Adam based on the infinity norm, making it more stable when dealing with large gradients.
    Advantages:
        Handles large gradients more effectively than Adam.
        Maintains most of the benefits of Adam (adaptive learning rates, sparse gradients).
    Disadvantages:
        Still requires tuning of hyperparameters.
        Slightly more memory intensive due to additional parameter storage.


9. ADMM (Alternating Direction Method of Multipliers)

    Description: An optimization algorithm that splits the problem into smaller subproblems, solved iteratively with dual variable updates.
    Advantages:
        Effective for problems with complex constraints or separable structures.
        Can handle large-scale optimization problems.
    Disadvantages:
        More complex to implement.
        May require reformulation of the optimization problem.
        Slower convergence for some types of problems compared to other gradient-based methods.
    