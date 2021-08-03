import torch as th


def flag(
    fwd,
    n_perturb_shape,
    y,
    step_size,
    m,
    device,
    criterion,
):
    n_perturb = (
        th.FloatTensor(*n_perturb_shape).uniform_(-step_size, step_size).to(device)
    )
    n_perturb.requires_grad_()
    out = fwd(n_perturb)
    losses = []
    loss = criterion(out, y)
    loss /= m
    losses.append(loss)
    for _ in range(m - 1):
        loss.backward()
        n_perturb_data = n_perturb.detach() + step_size * th.sign(
            n_perturb.grad.detach()
        )

        n_perturb.data = n_perturb_data.data
        n_perturb.grad[:] = 0

        out = fwd(n_perturb)
        loss = criterion(out, y)
        loss /= m
        losses.append(loss)

    return loss, losses


def flag_pret(
    fwd,
    n_perturb_shape,
    e_perturb_shape,
    n_y,
    e_y,
    step_size,
    m,
    device,
    criterion,
):
    n_perturb = (
        th.FloatTensor(*n_perturb_shape).uniform_(-step_size, step_size).to(device)
    )
    e_perturb = (
        th.FloatTensor(*e_perturb_shape).uniform_(-step_size, step_size).to(device)
    )
    n_perturb.requires_grad_()
    e_perturb.requires_grad_()
    pred_n, pred_e, mask_n, mask_e = fwd(n_perturb, e_perturb)

    losses = []
    n_loss = criterion(pred_n, n_y) / m
    e_loss = criterion(pred_e, e_y) / m
    loss = n_loss + e_loss
    losses.append(loss)

    for _ in range(m - 1):
        loss.backward()

        n_perturb_data = n_perturb.detach() + step_size * th.sign(
            n_perturb.grad.detach()
        )
        e_perturb_data = e_perturb.detach() + step_size * th.sign(
            e_perturb.grad.detach()
        )
        n_perturb.data = n_perturb_data.data
        e_perturb.data = e_perturb_data.data
        n_perturb.grad[:] = 0
        e_perturb.grad[:] = 0

        pred_n, pred_e, mask_n, mask_e = fwd(n_perturb, e_perturb)
        n_loss = criterion(pred_n, n_y) / m
        e_loss = criterion(pred_e, e_y) / m
        loss = n_loss + e_loss
        losses.append(loss)

    acc_node = th.sum(th.argmax(pred_n, dim=1) == n_y) / len(pred_n)
    acc_edge = th.sum(th.argmax(pred_e, dim=1) == e_y) / len(pred_e)

    return loss, losses, acc_node, acc_edge
