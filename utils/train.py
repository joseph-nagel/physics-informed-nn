'''PINN training.'''

from warnings import warn


def test_pinn(pinn, colloc_dict):
    '''
    Test PINN physics loss.

    Summary
    -------
    The physics loss of a PINN is computed for given collocation points.
    It is remarked that, due to the occurrence of the partial derivatives
    in the loss function, the autograd machinery needs to be enabled.

    Parameters
    ----------
    pinn : PINN module
        PINN model with a physics loss method.
    colloc_dict : dict
        Dict of collocation points.

    '''

    pinn.eval()

    loss = pinn.physics_loss(**colloc_dict)
    loss = loss.detach().item()

    return loss


def train_pinn(pinn,
               optimizer,
               num_epochs,
               train_colloc,
               val_colloc=None,
               print_every=1):
    '''
    Train PINN by minimizing the physics loss.

    Summary
    -------
    A CPU-based non-batched training scheme for PINNs is provided.
    The physics loss is minimized for a given set of collocation points.
    It is assumed that no observational data is available,
    such that the regression loss can be omitted.

    Parameters
    ----------
    pinn : PINN module
        PINN model with a physics loss method.
    num_epochs : int
        Number of training epochs.
    train_colloc : dict
        Dict of collocation points for training.
    val_colloc : dict
        Dict of collocation points for validation.
    print_every : int
        Determines when losses are printed.

    '''

    num_epochs = abs(int(num_epochs))
    print_every = abs(int(print_every))

    train_losses = []
    val_losses = []

    # perform initial test
    train_loss = test_pinn(pinn, train_colloc)
    train_losses.append(train_loss)

    if val_colloc is not None:
        val_loss = test_pinn(pinn, val_colloc)
        val_losses.append(val_loss)
    else:
        warn('Collocation points for validation are missing')

    # print losses
    print_str = 'Before training, train loss : {:.2e}'.format(train_loss)

    if val_colloc is not None:
        print_str += ', val. loss: {:.2e}'.format(val_loss)

    print(print_str)

    # loop over training epochs
    for epoch_idx in range(num_epochs):
        pinn.train()

        loss = pinn.physics_loss(**train_colloc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.detach().item())

        # compute val. loss
        if val_colloc is not None:
            val_loss = test_pinn(pinn, val_colloc)
            val_losses.append(val_loss)

        # print losses
        if (epoch_idx + 1) % print_every == 0 or (epoch_idx + 1) == num_epochs:
            print_str = 'Epoch: {:d}, train loss: {:.2e}'.format(epoch_idx + 1, loss.detach().item())

            if val_colloc is not None:
                print_str += ', val. loss: {:.2e}'.format(val_loss)

            print(print_str)

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses
    }

    return history

